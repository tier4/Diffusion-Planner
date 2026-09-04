"""Smoke-test that a v1-style dataset can still train every pipeline.

Point this at a dataset built by ``build_small_dataset.py`` and it runs a tiny,
fast pass of each training pipeline — warmstarted from ``--base_model`` (IL may also
run **from scratch** when no base is given) — reporting PASS/FAIL per pipeline, EMITTING
each pipeline's trained model, and showing a base-vs-trained regression check:

* **IL**    - imitation learning: ``diffusion_planner/train_predictor.py`` on the frame
  set. This is the README's primary supervised predictor training (the base/"pretraining"
  stage), NOT a fine-tune-after-pretraining step. Warmstarted from ``--base_model`` when
  given (``train_epochs = base_epoch + N``, computed from the checkpoint), or **from
  scratch** when ``--base_model`` is omitted (the README cold-start recipe;
  ``train_epochs = N``). PASS = finite train+val loss + a saved checkpoint. With a base it
  reports **base->trained L2** (ego/neighbor via ``valid_predictor``) as the regression
  signal; from scratch there is no base to regress against.
* **RSFT**  - ``rlvr.autoresearch.run_experiment`` (K-sample ranked SFT), warmstarted.
  PASS = finite per-epoch loss + reward + a LoRA save. Reports **base->trained reward**
  and merges the LoRA into a full model.
* **R2LPL** - ``rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds``: one full lifelong
  round (mine the model's mistakes on the contiguous corpus -> repair targets -> replay
  memory -> train), warmstarted. PASS = a trained model is produced. Reports
  **base->trained L2**. NOTE: R2LPL learns from MISTAKES, so the contiguous corpus MUST
  contain scenes the base model fails on (moving_collision / road_border_crossing /
  expert_disagreement); a clean corpus mines zero events and R2LPL fails loud. The
  reward config is an internal asset — set ``reward_config`` in the R2LPL config.

Each pipeline emits its trained model to ``<out_dir>/models/`` (``--emit_models``,
default on; ``--no-emit-models`` to skip). The base-vs-trained L2 check is on by
default (``--no_l2`` to skip); a >2x base ego-L2 blowup FAILs the pipeline.

The intent is a fast "did I break training?" gate. Each pipeline can be skipped
independently (all run by default). No paths are hard-coded; the only assumption is
the dataset layout produced by ``build_small_dataset.py``::

    <dataset_root>/
      frame/train.json          # IL + RSFT train scene list
      frame/val.json            # IL + RSFT + L2 val scene list
      contiguous/window_scenes.json     # R2LPL ordered contiguous scene list
      contiguous/reproducer_chunks.jsonl  # R2LPL chunk manifest (failure corpus)
      normalization.json        # (or pass --normalization)

The base model (``--base_model``) must have an ``args.json`` beside it (the standard
deploy-dir layout). It is required whenever RSFT or R2LPL runs (they warmstart from it);
an IL-only run may omit it to train from scratch (README cold-start).

Usage:
    # warmstart smoke of all three pipelines
    python -m rlvr.autoresearch.tools.verify_dataset_training \
        --dataset_root <v1_dir> --base_model <best_model.pth> \
        [--skip_il] [--skip_rsft] [--skip_r2lpl] [--no-emit-models] [--no_l2] \
        [--device auto]

    # from-scratch IL (no base model): omit --base_model and skip the warmstart-only pipelines
    python -m rlvr.autoresearch.tools.verify_dataset_training \
        --dataset_root <v1_dir> --skip_rsft --skip_r2lpl [--device auto]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from rlvr.autoresearch.scene_features import validate_canonical_scene

# Repo root = three levels up from rlvr/autoresearch/tools/this_file.py. Subprocesses
# run from here so both ``rlvr`` and ``diffusion_planner`` import cleanly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parent / "dataset_smoke_configs"

# smoke-only keys the RSFT config carries but run_experiment must not receive as
# config fields (they are applied by this script instead).
_RSFT_SCRIPT_KEYS = {"_doc", "n_train_cap", "n_val_cap", "sft_batch_size", "train_epochs"}


def _fmt_dur(seconds: float) -> str:
    """Human-readable wall-clock: raw seconds plus mm:ss once past a minute
    (e.g. ``42s`` / ``156s (2m36s)``) — so each pipeline's cost is legible at a glance."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    return f"{s}s ({s // 60}m{s % 60:02d}s)"


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _require(path: Path, what: str) -> Path:
    # `path is None` means the value was not provided (e.g. a null config field) —
    # surface the intended "what" message instead of an opaque AttributeError.
    if path is None:
        raise FileNotFoundError(f"{what} not provided (no default permitted)")
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _subsample(scene_list: Path, cap: int, seed: int, out_path: Path) -> int:
    """Write the scene list to ``out_path``, optionally capped to a seeded ``cap``-scene
    subset. ``cap`` falsy (0/None — the default) means **use the FULL dataset** — no
    subsampling. A positive ``cap`` is an explicit opt-in (only the ``*_ci`` quick-check
    configs set one). Returns the kept count."""
    scenes = _load_json(scene_list)
    if not isinstance(scenes, list):
        raise ValueError(f"{scene_list} must be a JSON list of NPZ paths")
    if cap and len(scenes) > cap:
        scenes = random.Random(seed).sample(scenes, cap)
    with open(out_path, "w") as f:
        json.dump(scenes, f)
    return len(scenes)


def _run(cmd: list[str], log_path: Path, env: dict) -> tuple[int, str]:
    """Run a subprocess from the repo root, teeing combined output to a log."""
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT
        )
    text = log_path.read_text(errors="replace")
    return proc.returncode, text


def resolve_device(requested: str) -> str:
    """Resolve the requested device to a concrete 'cuda' or 'cpu' using ACTUAL CUDA
    availability. 'auto' must check availability (hiding CUDA via an env var does not
    make it available), and 'cuda' on a CPU-only host fails loudly rather than
    launching subprocesses that then crash."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        avail = torch.cuda.is_available()
    except Exception:
        avail = False
    if requested == "cuda":
        if not avail:
            raise SystemExit("--device cuda requested but no CUDA device is available")
        return "cuda"
    return "cuda" if avail else "cpu"  # auto


def _env_for_device(device: str) -> dict:
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:  # cuda (already resolved concrete)
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Put the repo root on PYTHONPATH so repo-root packages (planner_metrics, rlvr,
    # diffusion_planner) resolve even when they are not pip-installed. train_predictor
    # runs as a script path (sys.path[0] = diffusion_planner/, not the repo root), so
    # without this a source-only planner_metrics is not importable.
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + prev if prev else "")
    return env


# Model-architecture flags that must match the warmstart checkpoint for strict
# state_dict loading. train_predictor takes these as CLI flags and otherwise uses parser
# DEFAULTS, which silently diverge for a non-default base -> resume load fails. We
# reconstruct them from the base model's args.json.
_ARCH_KEYS = (
    "future_len", "time_len", "ego_prediction_horizon", "agent_state_dim", "agent_num",
    "static_objects_state_dim", "static_objects_num", "lane_num", "lane_len", "route_num",
    "route_len", "polygon_num", "polygon_len", "line_string_num", "line_string_len",
    "encoder_mixer_depth", "encoder_fusion_depth", "decoder_depth", "num_heads",
    "hidden_dim", "predicted_neighbor_num", "use_ego_history", "use_turn_indicators",
    "diffusion_model_type", "use_velocity_representation",
)  # fmt: skip


def _base_arch_args(args_json: Path) -> list[str]:
    """CLI flags reconstructing the base model's architecture from its args.json so a
    warmstart train_predictor builds a model matching the checkpoint (its parser
    defaults would otherwise diverge for a non-default base and fail strict loading)."""
    with open(args_json) as f:
        a = json.load(f)
    out: list[str] = []
    for k in _ARCH_KEYS:
        if k in a:
            v = a[k]
            out += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    return out


# --------------------------------------------------------------------------- #
# warmstart / model-output / L2 helpers (the script does all of this itself)
# --------------------------------------------------------------------------- #
def _ckpt_epoch(model_path: Path) -> int:
    """The saved epoch in a checkpoint. train_predictor RESUMES at this epoch (its loop
    is ``range(init_epoch, train_epochs)``), so to warmstart from this model and train
    N more epochs the script passes ``train_epochs = epoch + N`` — never a hardcoded
    number, so it is correct for any warmstart model."""
    import torch

    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    return int(ck.get("epoch", 0))


def _deployable_ckpt(src: Path, dst: Path) -> Path:
    """Write a valid_predictor-ready checkpoint next to a copied args.json. Uses the
    EMA weights when present: train_predictor stores BOTH raw ``model`` (optimizer
    iterates) and ``ema_state_dict`` (the deployable weights) — the EMA is what to
    evaluate/ship (raw shows large fake L2 drift)."""
    import shutil

    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    weights = ck["ema_state_dict"] if "ema_state_dict" in ck else ck["model"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": weights}, dst)
    aj = src.parent / "args.json"
    if aj.exists():
        shutil.copy(aj, dst.parent / "args.json")
    return dst


def _l2_eval(model_path: Path, args_json: Path, val_list: Path, device: str, env: dict, log: Path):
    """Ego/neighbor L2 on ``val_list`` via the canonical valid_predictor. Returns
    (ego, neighbor) floats or None on failure. Runs single-process with ``--ddp false``
    (numerically identical to DDP on one GPU): its ``resume_model`` path applies
    ``strip_module_prefix``, so it loads BOTH a DDP-trained base (``module.`` prefix)
    and a single-process-trained smoke model (no prefix) uniformly."""
    cmd = [
        sys.executable, "diffusion_planner/valid_predictor.py", "--ddp", "false",
        "--resume_model_path", str(model_path), "--args_json_path", str(args_json),
        "--valid_set_list", str(val_list), "--batch_size", "32",
        # valid_predictor defaults to cuda; pass the resolved device so --device cpu
        # (and auto on a CPU-only host) actually runs on CPU instead of crashing.
        "--device", device,
    ]  # fmt: skip
    rc, text = _run(cmd, log, env)
    ego = re.findall(r"avg_loss_ego=([0-9.eE+-]+)", text)
    nbr = re.findall(r"avg_loss_neighbor=([0-9.eE+-]+)", text)
    if rc == 0 and ego and nbr:
        return float(ego[-1]), float(nbr[-1])
    return None


def _emit_model(src: Path, name: str, args) -> str:
    """Copy a trained (deployable/EMA) checkpoint to <out_dir>/models/<name>/<name>.pth
    with its OWN args.json beside it (a per-pipeline subdir so one pipeline's args.json
    can't overwrite another's) so each pipeline leaves one self-contained, loadable
    model output. Returns a short display path."""
    dst = args.out_dir / "models" / name / f"{name}.pth"
    _deployable_ckpt(src, dst)
    return str(dst.relative_to(args.out_dir))


def _pct(new: float, base: float) -> str:
    return f"{(new - base) / base * 100:+.1f}%" if base else "n/a"


def _l2_line(base_l2, trained_l2) -> str:
    """A compact 'base -> trained (Δ%)' L2 summary for both ego and neighbor."""
    if not base_l2 or not trained_l2:
        return f"L2=BASE{'?' if not base_l2 else ''}/TRAINED{'?' if not trained_l2 else ''}"
    be, bn = base_l2
    te, tn = trained_l2
    return f"L2_ego {be:.3f}->{te:.3f} ({_pct(te, be)}) L2_nbr {bn:.3f}->{tn:.3f} ({_pct(tn, bn)})"


def _regression_l2(base_model: Path, trained_ckpt: Path, val_list: Path, work: Path, args, env):
    """Base-vs-trained L2 regression check. Returns (note, ok).

    BOTH models are evaluated on their EMA (deployable) weights via ``_deployable_ckpt``
    — ``valid_predictor`` scores ``ckpt["model"]``, so comparing the base's RAW ``model``
    against the trained EMA would be apples-to-oranges (raw shows large fake drift), and
    an inflated base would defeat the regression gate. ``ok`` is False if the check was
    REQUESTED but either eval failed (a failed-but-requested L2 must not silently PASS —
    distinct from ``--no_l2``) or ego L2 blew up >2x base."""
    base_dep = _deployable_ckpt(base_model, work / "eval_base" / "base.pth")
    trained_dep = _deployable_ckpt(trained_ckpt, work / "eval_trained" / "trained.pth")
    base_l2 = _l2_eval(
        base_dep, base_dep.parent / "args.json", val_list, args.device, env, work / "l2_base.log"
    )
    trained_l2 = _l2_eval(
        trained_dep, trained_dep.parent / "args.json", val_list, args.device, env,
        work / "l2_trained.log",
    )  # fmt: skip
    if not base_l2 or not trained_l2:
        return f" L2 eval FAILED ({_l2_line(base_l2, trained_l2)})", False
    note = " " + _l2_line(base_l2, trained_l2)
    if trained_l2[0] > 2 * base_l2[0]:
        return note + " [REGRESSION: ego L2 >2x base]", False
    return note, True


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #
def run_il(ds, cfg, args, env) -> tuple[bool, str]:
    train_list = _require(ds["frame_train"], "frame/train.json (IL)")
    val_list = _require(ds["frame_val"], "frame/val.json (IL)")
    norm = _require(ds["normalization"], "normalization.json (IL)")
    work = args.out_dir / "il"
    work.mkdir(parents=True, exist_ok=True)
    # No cap in the config => FULL dataset (both lists used in their entirety). Only the
    # explicit *_ci quick-check configs set a positive cap.
    n_tr = _subsample(train_list, int(cfg.get("n_train_cap", 0)), args.seed, work / "train.json")
    n_va = _subsample(val_list, int(cfg.get("n_val_cap", 0)), args.seed, work / "val.json")
    n_epochs = int(cfg.get("train_epochs", 1))
    # Warmstart from --base_model when given: train_predictor RESUMES at the base
    # checkpoint's epoch, so train for n_epochs MORE => train_epochs = base_epoch +
    # n_epochs (computed from the model, never hardcoded). warm_up_epoch=0 so a good
    # warmstart isn't LR-warmed from ~0. Without a base model it trains from scratch.
    resume = []
    if args.base_model:
        base_args = _require(
            args.base_model.parent / "args.json", "args.json beside --base_model (IL)"
        )
        train_epochs = _ckpt_epoch(args.base_model) + n_epochs
        # pass the base ARCHITECTURE args so the resumed model matches the checkpoint
        resume = ["--resume_model_path", str(args.base_model), *_base_arch_args(base_args)]
    else:
        train_epochs = n_epochs
    cmd = [
        sys.executable, "diffusion_planner/train_predictor.py",
        "--exp_name", "verify_il",
        "--save_dir", str(work / "out"),
        "--train_set_list", str(work / "train.json"),
        "--valid_set_list", str(work / "val.json"),
        "--normalization_file_path", str(norm),
        "--train_epochs", str(train_epochs),
        "--warm_up_epoch", str(int(cfg.get("warm_up_epoch", 0))),
        "--batch_size", str(int(cfg.get("batch_size", 8))),
        "--learning_rate", str(float(cfg.get("learning_rate", 1e-4))),
        "--ddp", "false",
        # train_predictor defaults to --device cuda; pass the resolved concrete device
        # so --device cpu (and auto on a CPU-only host) actually runs on CPU.
        "--device", args.device,
        "--use_data_augment", str(bool(cfg.get("use_data_augment", True))).lower(),
        "--augment_prob", str(float(cfg.get("augment_prob", 0.5))),
        *resume,
    ]  # fmt: skip
    print(
        f"[IL] warmstart={'yes' if resume else 'no'} train {n_tr} / val {n_va} "
        f"-> train_epochs={train_epochs} -> {work / 'il.log'}",
        flush=True,
    )
    rc, text = _run(cmd, work / "il.log", env)
    num = r"(-?inf|nan|[0-9.eE+-]+)"
    tr_loss = re.findall(rf"epoch_mean_loss\['loss'\]={num}", text)
    va_loss = re.findall(rf"valid_loss_ego={num}", text)

    def _finite(xs):
        if not xs:
            return False
        try:
            return all(math.isfinite(float(x)) for x in xs)
        except ValueError:
            return False

    # locate the saved checkpoint FIRST — a trained checkpoint is part of the PASS
    # contract (finite logs + rc=0 alone can be true while no model was written).
    best = work / "out" / "best_model" / "best_model.pth"
    if not best.exists():
        best = work / "out" / "latest.pth"
    ok = rc == 0 and _finite(tr_loss) and _finite(va_loss) and best.exists()
    l2_note = ""
    if ok and not args.no_l2:
        if args.base_model:
            l2_note, l2_ok = _regression_l2(
                args.base_model, best, work / "val.json", work, args, env
            )
            ok = ok and l2_ok
        else:
            # From-scratch IL: no base checkpoint exists, so a base-vs-trained L2
            # regression is undefined. The PASS contract is finite train/val loss + a
            # written checkpoint (reported below). Flag the gate as N/A loudly rather
            # than silently skipping it.
            l2_note = " [from-scratch: no base for L2 regression]"
    saved = _emit_model(best, "il", args) if (ok and best.exists() and args.emit_models) else None
    detail = (
        f"rc={rc} train_loss={tr_loss[-1] if tr_loss else 'NONE'} "
        f"valid_loss_ego={va_loss[-1] if va_loss else 'NONE'} "
        f"ckpt={'yes' if best.exists() else 'MISSING'}{l2_note}"
        f"{f' model={saved}' if saved else ''}"
    )
    return ok, detail


def run_rsft(ds, cfg, args, env) -> tuple[bool, str]:
    train_list = _require(ds["frame_train"], "frame/train.json (RSFT)")
    val_list = _require(ds["frame_val"], "frame/val.json (RSFT)")
    base_model = _require(args.base_model, "--base_model (RSFT)")
    _require(base_model.parent / "args.json", "args.json beside --base_model (RSFT)")
    work = args.out_dir / "rsft"
    work.mkdir(parents=True, exist_ok=True)
    # No cap in the config => FULL dataset. Only the explicit *_ci config caps.
    n_tr = _subsample(train_list, int(cfg.get("n_train_cap", 0)), args.seed, work / "train.json")
    n_va = _subsample(val_list, int(cfg.get("n_val_cap", 0)), args.seed, work / "val.json")
    # hand run_experiment a clean config (strip the smoke-only keys)
    clean = {k: v for k, v in cfg.items() if k not in _RSFT_SCRIPT_KEYS}
    clean_path = work / "run_experiment_config.json"
    with open(clean_path, "w") as f:
        json.dump(clean, f, indent=2)
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.run_experiment",
        "--config",
        str(clean_path),
        "--name",
        "verify_rsft",
        "--model_path",
        str(base_model),
        "--train_scenes",
        str(work / "train.json"),
        "--val_scenes",
        str(work / "val.json"),
        "--output_dir",
        str(work / "out"),
        "--train_epochs",
        str(int(cfg.get("train_epochs", 1))),
        "--sft_batch_size",
        str(int(cfg.get("sft_batch_size", 8))),
    ]
    # NOTE: --skip_baseline is deliberately NOT passed so run_experiment evaluates the
    # BASE model reward ("Eval [base-val]: ... reward="), giving the base-vs-trained
    # reward comparison (RSFT's native regression signal).
    print(f"[RSFT] warmstart train {n_tr} / val {n_va} scenes -> {work / 'rsft.log'}", flush=True)
    rc, text = _run(cmd, work / "rsft.log", env)
    # PASS requires evidence that TRAINING actually happened, not just rc=0 + a saved
    # adapter + a promoted summary reward:
    #  - a per-epoch training-loss line was emitted (`trained`) — this is the real
    #    guard: the trainer swallows train-load failures to {} for N=0, but then
    #    log_metrics raises (rc!=0) AND no per-epoch Loss= line is printed, so N=0
    #    can't satisfy `trained`. (kept_train below is informational only: the
    #    skip-filter line prints only when skip_filtered_scenes is enabled, so gating
    #    on it would false-FAIL a healthy run that disables that feature.)
    #  - a FINITE per-epoch validation reward (the "Eval [epochN-val]: reward=" line,
    #    which exists regardless of the model-quality promotion threshold; the summary
    #    "val_reward" stays -inf with --skip_baseline when the reward is <= -5, so
    #    gating on it would false-FAIL a real run).
    kept = re.findall(r"train:\s*kept\s+(\d+)\s*/", text)
    kept_train = int(kept[-1]) if kept else None
    # require a FINITE per-epoch TOTAL training loss (a Loss=nan/inf line means training
    # ran but diverged — not evidence of a healthy pipeline). Anchor to the first
    # standalone ``Loss=`` field: the negative lookbehind ``(?<![A-Za-z])`` skips
    # sub-fields like ``EgoLoss=`` (real line: ``Epoch 1 ...: Loss=nan, EgoLoss=0``),
    # and the non-greedy prefix prevents matching a later sub-loss instead of the total.
    tl = re.findall(r"Epoch\s+\d+[^\n]*?(?<![A-Za-z])Loss=(-?(?:inf|nan|[0-9.eE+-]+))", text)
    trained = False
    if tl:
        try:
            trained = math.isfinite(float(tl[-1]))
        except ValueError:
            trained = False
    ev = re.findall(r"Eval \[epoch\d+-val\][^\n]*reward=([+-]?(?:inf|nan|[0-9.eE]+))", text)
    ev_val = None
    if ev:
        try:
            v = float(ev[-1])
            ev_val = v if math.isfinite(v) else None
        except ValueError:
            ev_val = None
    # base model reward (from the non-skipped baseline eval) -> reward comparison
    bv = re.findall(r"Eval \[base-val\][^\n]*reward=([+-]?(?:inf|nan|[0-9.eE]+))", text)
    base_reward = None
    if bv:
        try:
            b = float(bv[-1])
            base_reward = b if math.isfinite(b) else None
        except ValueError:
            base_reward = None
    lora = list((work / "out").rglob("adapter_model.safetensors"))
    # base_reward is required: we drop --skip_baseline specifically to compute the
    # base-vs-trained reward comparison, so a missing base reward is a failed contract.
    ok = rc == 0 and trained and ev_val is not None and base_reward is not None and bool(lora)
    # merge the LoRA into the base and emit a usable full model (own subdir + args.json,
    # matching IL/R2LPL so the emitted model is self-contained and loadable)
    saved = None
    if ok and lora and args.emit_models:
        merged = args.out_dir / "models" / "rsft" / "rsft.pth"
        merged.parent.mkdir(parents=True, exist_ok=True)
        base_args = base_model.parent / "args.json"
        if base_args.exists():
            import shutil

            shutil.copy(base_args, merged.parent / "args.json")
        mrc, _ = _run(
            [
                sys.executable,
                "-m",
                "preference_optimization.merge_lora",
                "--model_path",
                str(base_model),
                "--lora_dir",
                str(lora[-1].parent),
                "--output",
                str(merged),
            ],  # fmt: skip
            work / "merge.log",
            env,
        )
        if mrc == 0 and merged.exists():
            saved = str(merged.relative_to(args.out_dir))
        else:
            # a broken merge means the emitted model is unusable -> FAIL loudly (not
            # just a note in the detail string)
            saved = "MERGE_FAILED"
            ok = False
    reward_cmp = (
        f"reward {base_reward:+.2f}->{ev_val:+.2f} ({ev_val - base_reward:+.2f})"
        if base_reward is not None and ev_val is not None
        else f"reward base={bv[-1] if bv else 'NONE'} trained={ev[-1] if ev else 'NONE'}"
    )
    detail = (
        f"rc={rc} kept_train={kept_train} trained={trained} {reward_cmp} "
        f"lora_saved={'yes' if lora else 'no'}{f' model={saved}' if saved else ''}"
    )
    return ok, detail


def _resolve_cfg_path(value) -> Path | None:
    """Resolve a config-file reference: absolute as-is, else relative to the repo."""
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _scenes_loadable(scene_list: Path, k: int = 8) -> tuple[int, int]:
    """Sample k EVENLY-SPACED scenes across the whole list (incl. first + last) and
    enforce the FULL canonical schema on each, then run the model-input loader.
    Returns (checked, loadable). ``load_npz_data`` alone is insufficient — it only
    tensorizes keys that exist and effectively requires just ``ego_shape``, so an NPZ
    carrying only ``ego_shape`` loads fine yet the real pipeline cannot consume it. So
    each sampled scene must FIRST pass ``validate_canonical_scene`` (every required
    field present with a compatible shape) and THEN load. This makes the R2LPL check
    genuinely CONSUME the dataset (the miner's plan-only branch only parses path
    strings). Even spacing means a corpus valid only at the start still fails."""
    scenes = _load_json(scene_list)
    if not isinstance(scenes, list) or not scenes:
        return 0, 0
    import numpy as np
    import torch

    from preference_optimization.utils import load_npz_data

    n = len(scenes)
    # Spacing denominator is (samples-1), where samples = min(k, n) — so the largest
    # index always lands on n-1 (the last scene) even when the corpus is smaller than k.
    m = min(k, n)
    idx = sorted({round(i * (n - 1) / max(m - 1, 1)) for i in range(m)}) if n > 1 else [0]
    checked = loadable = 0
    for i in idx:
        checked += 1
        try:
            # 1) full schema: every required field present with a compatible shape;
            # 2) the canonical loader builds the tensors + heading_to_cos_sin — so a
            #    scene the real pipeline cannot consume raises at one of these steps.
            validate_canonical_scene(np.load(scenes[i], allow_pickle=True), ctx=scenes[i])
            load_npz_data(scenes[i], torch.device("cpu"))
            loadable += 1
        except Exception:
            pass
    return checked, loadable


def run_r2lpl(ds, cfg, args, env) -> tuple[bool, str]:
    scene_list = _require(ds["contig_scene_list"], "contiguous/window_scenes.json (R2LPL)")
    base_model = _require(args.base_model, "--base_model (R2LPL)")
    _require(base_model.parent / "args.json", "args.json beside --base_model (R2LPL)")
    val_list = _require(ds["frame_val"], "frame/val.json (R2LPL)")
    work = args.out_dir / "r2lpl"
    work.mkdir(parents=True, exist_ok=True)
    # No cap in the config => FULL val set. Only the explicit *_ci config caps.
    n_va = _subsample(val_list, int(cfg.get("n_val_cap", 0)), args.seed, work / "val.json")
    # The lifelong round mines corrective chunks from the contiguous corpus, builds
    # repair targets, updates replay memory, and TRAINS — warmstarting from base_model.
    # The reward config is an internal asset (not committed): fail loud if absent
    # rather than silently degrading (a real closed-loop reward is required to judge
    # failures). threshold/credit ship in-repo.
    reward_cfg = _require(
        _resolve_cfg_path(cfg.get("reward_config")),
        "reward_config (R2LPL lifelong training; internal asset — set it in r2lpl.json)",
    )
    thr = _require(_resolve_cfg_path(cfg["threshold_config"]), "threshold_config")
    crd = _require(_resolve_cfg_path(cfg["credit_window_config"]), "credit_window_config")
    # output_dir must contain an "auto_research" path component (runner enforces this).
    out_dir = work / "auto_research" / "rounds"
    labels = cfg.get("enabled_labels", ["moving_collision", "road_border_crossing"])
    # The runner consumes the high-level "workflow contract": a workflow_config (judge/
    # repair/replay/rounds/reproducer sections) + a training_config, from which it
    # builds the full round config. Warmstart is --model_path.
    workflow = {
        "judgement": {
            "reward_config": str(reward_cfg),
            "threshold_config": str(thr),
            "credit_window_config": str(crd),
            "enabled_labels": labels,
            "repair_labels": labels,
        },
        "repair_generation": {
            "ego_shape": cfg.get("ego_shape", "from_npz"),
            "min_margin": float(cfg.get("min_margin", 0.3)),
            # thread the resolved device so repair generation honors --device cpu
            "device": args.device,
        },
        "replay_memory": {"capacity": int(cfg.get("replay_capacity", 200))},
        "training": {"val_scenes": str(work / "val.json")},
        "rounds": {
            "rounds": int(cfg.get("rounds", 1)),
            "epochs_per_round": int(cfg.get("epochs_per_round", 1)),
        },
        "perception_reproducer": {
            "device": args.device,  # mining rollout honors --device cpu
            **{
                k: cfg[k]
                for k in (
                    "chunk_len",
                    "start_stride",
                    "expected_frame_step",
                    "tracker_mode",
                    "timeline_progress_mode",
                    "neighbor_history_mode",
                    "batch_size",
                )
                if cfg.get(k) is not None
            },
        },  # fmt: skip
        "event_mining": {k: cfg[k] for k in ("max_chunks",) if cfg.get(k) is not None},
    }
    training_cfg = cfg.get("training_config", {"backend": "base_sft", "train_args": {}})
    # ensure the base-SFT round training also honors the resolved device
    training_cfg.setdefault("train_args", {}).setdefault("device", args.device)
    wf_path = work / "workflow_config.json"
    tc_path = work / "training_config.json"
    with open(wf_path, "w") as f:
        json.dump(workflow, f, indent=2)
    with open(tc_path, "w") as f:
        json.dump(training_cfg, f, indent=2)
    manifest = ds.get("contig_chunk_manifest")
    source = (
        ["--chunk_manifest", str(manifest)]
        if manifest and manifest.exists()
        else ["--scene_list", str(scene_list)]
    )
    print(f"[R2LPL] lifelong round (warmstart) val {n_va} -> {work / 'r2lpl.log'}", flush=True)
    rc, text = _run(
        [
            sys.executable,
            "-m",
            "rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds",
            "--model_path",
            str(base_model),
            "--output_dir",
            str(out_dir),
            "--workflow_config",
            str(wf_path),
            "--training_config",
            str(tc_path),
            *source,
        ],  # fmt: skip
        work / "r2lpl.log",
        env,
    )
    # locate the round's trained checkpoint — newest by mtime (a multi-round/epoch run
    # writes several; alphabetical order does not track round/epoch across the mixed
    # latest.pth / best_model.pth / epoch_*.pth names).
    models = list(out_dir.rglob("best_model.pth")) or list(out_dir.rglob("*.pth"))
    trained = max(models, key=lambda p: p.stat().st_mtime) if models else None
    ok = rc == 0 and trained is not None
    l2_note = ""
    if ok and not args.no_l2:
        l2_note, l2_ok = _regression_l2(base_model, trained, work / "val.json", work, args, env)
        ok = ok and l2_ok
    saved = _emit_model(trained, "r2lpl", args) if (trained and args.emit_models) else None
    detail = (
        f"rc={rc} trained_model={'yes' if trained else 'NONE'}{l2_note}"
        f"{f' model={saved}' if saved else ''}"
    )
    return ok, detail


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset_root", type=Path, required=True, help="dataset dir (build_small_dataset layout)"
    )
    ap.add_argument(
        "--base_model",
        type=Path,
        help="base model .pth (with args.json beside it); required unless IL is the only pipeline",
    )
    ap.add_argument(
        "--normalization",
        type=Path,
        default=None,
        help="override normalization.json (default <dataset_root>/normalization.json)",
    )
    ap.add_argument("--il_config", type=Path, default=_CONFIG_DIR / "il.json")
    ap.add_argument("--rsft_config", type=Path, default=_CONFIG_DIR / "rsft.json")
    ap.add_argument("--r2lpl_config", type=Path, default=_CONFIG_DIR / "r2lpl.json")
    ap.add_argument("--skip_il", action="store_true")
    ap.add_argument("--skip_rsft", action="store_true")
    ap.add_argument("--skip_r2lpl", action="store_true")
    ap.add_argument(
        "--emit_models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write each pipeline's trained model to <out_dir>/models/ (default on; "
        "--no-emit-models to skip)",
    )
    ap.add_argument(
        "--no_l2",
        action="store_true",
        help="skip the base-vs-trained L2 regression check (on by default)",
    )
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="scratch dir for logs+artifacts (default <dataset_root>/_verify_<ts>)",
    )
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--dataset_root not a directory: {root}")
    if args.out_dir is None:
        args.out_dir = root / f"_verify_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = {
        "frame_train": root / "frame" / "train.json",
        "frame_val": root / "frame" / "val.json",
        "contig_scene_list": root / "contiguous" / "window_scenes.json",
        "contig_chunk_manifest": root / "contiguous" / "reproducer_chunks.jsonl",
        "normalization": args.normalization or (root / "normalization.json"),
    }
    args.device = resolve_device(args.device)  # 'auto' -> concrete cuda/cpu by availability
    env = _env_for_device(args.device)

    pipelines = []
    if not args.skip_il:
        pipelines.append(("IL", run_il, _load_json(args.il_config)))
    if not args.skip_rsft:
        pipelines.append(("RSFT", run_rsft, _load_json(args.rsft_config)))
    if not args.skip_r2lpl:
        pipelines.append(("R2LPL", run_r2lpl, _load_json(args.r2lpl_config)))
    if not pipelines:
        print("Nothing to do: all pipelines skipped.")
        return
    # RSFT and R2LPL always warmstart from the base model (and report base-vs-trained
    # metrics against it), so a base is required whenever either is enabled. IL (the base
    # supervised training) can also train from scratch (no --base_model) — the README's
    # cold-start recipe — so an IL-only run may omit it. Fail loud if a warmstart-only
    # pipeline has no base.
    needs_base = any(name in ("RSFT", "R2LPL") for name, _, _ in pipelines)
    if needs_base and not args.base_model:
        raise SystemExit(
            "--base_model is required for RSFT/R2LPL (they warmstart from it). "
            "For from-scratch IL, run with --skip_rsft --skip_r2lpl and omit --base_model."
        )

    print(f"dataset_root: {root}")
    print(f"out_dir:      {args.out_dir}")
    print(f"device:       {args.device}\n")

    results = []
    for name, fn, cfg in pipelines:
        print(f"===== {name} =====", flush=True)
        t0 = time.time()
        try:
            ok, detail = fn(ds, cfg, args, env)
        except Exception as e:  # a setup failure is a pipeline failure, reported not raised
            ok, detail = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        print(f"[{name}] {status}  (took {_fmt_dur(dt)})  {detail}\n", flush=True)
        results.append((name, ok, detail, dt))

    total = sum(dt for _, _, _, dt in results)
    print("===== SUMMARY =====")
    print(f"  {'PIPE':6s} {'RES':4s} {'TIME':>10s}  DETAIL")
    for name, ok, detail, dt in results:
        print(f"  {name:6s} {'PASS' if ok else 'FAIL':4s} {_fmt_dur(dt):>10s}  {detail}")
    print(f"  {'TOTAL':6s} {'':4s} {_fmt_dur(total):>10s}")
    all_ok = all(ok for _, ok, _, _ in results)
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
