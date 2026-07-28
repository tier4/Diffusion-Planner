"""Stratified sampler + assembler for the small pipeline-test dataset.

Operates on the **sharded parquet** feature table emitted by
``scene_feature_index.py`` (re-opens the NPZs only for the final selected set and
for renders). Seeded / deterministic. No silent fallbacks: missing required args
or NPZ fields raise.

Two modes:

* **Frame mode** (default): bucket = the ``label`` triple. Selection reuses the
  3-pass quota algorithm ``lifelong_replay_memory.build_memory`` (per-label reserve,
  >=1 per bucket, fill by priority). Pre-dedup with ``search_scenes.group_sequences``
  + per-sequence stride caps so adjacent frames of one event don't dominate. Emits a
  stratified ``train.json`` + a disjoint stratified ``val.json`` from the SAME source.
* **Contiguous mode** (``--contiguous``): select K windows x W frames from a
  sidecar-bearing corpus (verifies frame contiguity + ``neighbor_ids`` presence, fails
  loudly if missing), preferring stop/turn/dense segments. Emits an R2LPL-compatible
  ``reproducer_chunks.jsonl`` (``mine_direct_reproducer_chunks._chunk_row`` schema).

Both modes canonicalize every selected NPZ on output to 320 slots / 4-col neighbor
futures (``future_to_4col``), pass ``ego_shape`` through (fail if absent), copy
sidecars, and write a README with the bucket histogram before/after sampling plus
stratified per-bucket renders (canonical ``save_step_figure`` path).

Usage (frame mode):
    python -m rlvr.autoresearch.tools.build_small_dataset \
        --parquet_dir <feature_parquet_dir> --out_dir <dataset_dir> \
        --n_train 20000 --n_val 2000 --seed 0 \
        [--label_quotas "stopped|turn-L|has-ped=0.03,..."] [--render_frac 0.02]

Usage (contiguous mode):
    python -m rlvr.autoresearch.tools.build_small_dataset --contiguous \
        --parquet_dir <contiguous_parquet_dir> --out_dir <repro_dir> \
        --n_windows 4 --window_len 400 --frame_step 1 --seed 0

Dataset layout for verify_dataset_training.py:
    That verifier expects the frame set at ``<root>/frame/`` and the contiguous
    corpus at ``<root>/contiguous/``, so build the two modes into those subdirs:
    ``--out_dir <root>/frame`` (frame mode) and ``--out_dir <root>/contiguous``
    (contiguous mode). Each mode writes its own ``train.json`` / ``val.json`` /
    ``window_scenes.json`` at the root of its ``--out_dir``.

    ``normalization.json`` is NOT produced here — it is an external input (the
    normalization stats the model was trained with). Copy the repo's
    ``diffusion_planner/normalization.json`` (or your model's) to ``<root>/`` for
    the verifier / SFT to consume.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
from diffusion_planner.dimensions import (
    MAX_NUM_NEIGHBORS as N_SLOTS,  # canonical neighbor slot count
)

from planner_metrics.scene_format import future_to_4col
from rlvr.autoresearch.scene_features import (
    _load_util_script,
    canonical_required_fields,
    session_key,
    validate_canonical_scene,
)
from rlvr.autoresearch.tools.lifelong_replay_memory import _parse_label_quotas, build_memory

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Every field the model consumes at train time, sourced from the shared canonical
# schema (scene_features.canonical_required_fields) so the builder's required-field
# list can never drift from the verifier's. Every output NPZ must carry all of these
# with a compatible shape or SFT crashes mid-training.
_REQUIRED_FIELDS = canonical_required_fields()


def _require_fields(d, ctx: str) -> None:
    """Fail loud if any required model-input field NAME is absent, checked early on the
    INPUT (before transforms touch tensors). Full shape validation happens on the
    OUTPUT via the shared ``validate_canonical_scene`` — see ``_validate_output``."""
    missing = [k for k in _REQUIRED_FIELDS if k not in (d.files if hasattr(d, "files") else d)]
    if missing:
        raise ValueError(
            f"{ctx}: missing required training field(s) {missing} (no default permitted)"
        )


def _validate_output(path: str) -> None:
    """Full per-output schema+shape validation, with NO hardcoded shape literals: the
    shared ``validate_canonical_scene`` asserts every required field's shape against
    the canonical ``diffusion_planner.dimensions`` contract (so a scalar/mis-shaped
    tensor is rejected, not just the neighbor axes), then the canonical model-input
    loader (``load_npz_data`` — builds the tensors + heading_to_cos_sin the model
    consumes) is run so a scene that would crash the model is rejected at build time."""
    import torch

    from preference_optimization.utils import load_npz_data

    d = np.load(path, allow_pickle=True)
    validate_canonical_scene(d, ctx=path)
    load_npz_data(path, torch.device("cpu"))


def output_basename(src: str) -> str:
    """Collision-free output filename for a source NPZ. A bare basename is NOT
    unique across a multi-bag/multi-date corpus (e.g. the same ``HH-MM-SS_...npz``
    recorded on different dates), which would silently overwrite scenes and leak
    the same output path into both train and val. Prefixing a short hash of the
    FULL source path makes distinct sources map to distinct outputs, and is stable
    (same src -> same name) so canonicalization and rendering agree."""
    h = hashlib.md5(src.encode()).hexdigest()[:10]
    return f"{h}_{os.path.basename(src)}"


def contig_relpath(src: str) -> str:
    """Output path (relative to ``npz/``) for CONTIGUOUS mode. The frame keeps its
    ORIGINAL dataset filename under a per-session subdirectory. The subdir is
    ``scene_features.session_key`` (``<tail>_<hash8>`` over the full source dir) —
    the SAME injective key used to group contiguous windows in ``bag_and_frame``, so
    a selected window maps to exactly one output dir and distinct source dirs never
    collide (a plain ``_``-join of the last two path parts is not injective:
    ``/A/a_b/c`` and ``/B/a/b_c`` both -> ``a_b_c``). Consecutive frames of a bag then
    share a parent dir + prefix + a +1 trailing index, which is what the reproducer's
    lineage check (``mine_direct_reproducer_chunks._is_next_contiguous``) requires."""
    return os.path.join(session_key(src), os.path.basename(src))


# --------------------------------------------------------------------------- #
# parquet load
# --------------------------------------------------------------------------- #
def load_feature_table(parquet_dir: str) -> list[dict]:
    """Load all ``shard_*.parquet`` rows into a list of dicts."""
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(parquet_dir, "shard_*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.parquet in {parquet_dir}")
    rows: list[dict] = []
    for s in shards:
        rows.extend(pq.read_table(s).to_pylist())
    print(f"[build] loaded {len(rows)} feature rows from {len(shards)} shards", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# frame-mode dedup + selection
# --------------------------------------------------------------------------- #
def dedup_sequences(rows: list[dict], *, gap: int, max_per_seq: int, stride: int) -> list[dict]:
    """Collapse temporally-adjacent frames of one event.

    Reuses ``search_scenes.group_sequences`` (groups by bag, splits on frame gap
    > ``gap``); then within each sequence keeps a strided subset capped at
    ``max_per_seq`` (the per-bag cap pattern from curate_curve_scenes.curate_scenes:
    ``step = max(1, len // n); picked = seq[::step][:n]``)."""
    group_sequences = _load_util_script(
        "search_scenes", "diffusion_planner/util_scripts/search_scenes.py"
    ).group_sequences
    seqs = group_sequences(rows, max_gap_frames=gap)
    kept: list[dict] = []
    for seq in seqs:
        n = min(max_per_seq, len(seq))
        if n <= 0:
            continue
        step = max(stride, len(seq) // n)
        kept.extend(seq[::step][:n])
    print(
        f"[build] dedup: {len(rows)} rows -> {len(seqs)} sequences -> {len(kept)} kept "
        f"(gap={gap}, max_per_seq={max_per_seq}, stride={stride})",
        flush=True,
    )
    return kept


def _assign_difficulty(rows: list[dict], uniform: bool) -> None:
    """Set row['difficulty'] = a composite 'interestingness' so within-bucket
    priority favors more dynamic / interactive scenes. build_memory normalizes it.
    uniform=True -> flat 1.0 (pure coverage sampling)."""
    if uniform:
        for r in rows:
            r["difficulty"] = 1.0
        return
    for r in rows:
        yaw = abs(float(r.get("gt_yaw_deg") or 0.0))
        la = float(r.get("peak_lat_accel") or 0.0)
        nc = float(r.get("neighbor_count") or 0.0)
        bonus = 0.0
        if r.get("has_lead"):
            bonus += 1.0
        if r.get("has_pedestrian"):
            bonus += 1.0
        if (r.get("static_count") or 0) > 0:
            bonus += 0.5
        # scale-mixed composite; absolute scale irrelevant (build_memory min-max norm)
        r["difficulty"] = yaw / 30.0 + la * 2.0 + nc / 6.0 + bonus


def select_frame(
    rows: list[dict], capacity: int, *, label_quotas: dict, alpha: float, beta: float
) -> list[dict]:
    """Reuse build_memory's 3-pass quota selection. frame_index is deliberately
    NOT provided, so _bucket collapses arc to 0 and bucket == the label triple."""
    mem = build_memory(
        rows,
        previous_memory={},
        per_scene_loss={},
        capacity=capacity,
        alpha=alpha,
        beta=beta,
        arc_bin_m=1e12,  # collapse any arc bucketing; bucket := label triple
        label_quotas=label_quotas or None,
    )
    return mem["entries"]


# --------------------------------------------------------------------------- #
# contiguous-mode window selection
# --------------------------------------------------------------------------- #
def _read_sidecar_json(npz_path: str) -> dict | None:
    side = npz_path[:-4] + ".json"
    if not os.path.exists(side):
        return None
    with open(side) as f:
        return json.load(f)


def select_contiguous(
    rows: list[dict], *, n_windows: int, window_len: int, frame_step: int, prefer_labels: set[str]
) -> list[list[dict]]:
    """Pick K non-overlapping contiguous windows of W frames, preferring windows
    rich in ``prefer_labels``. Fails loudly if a chosen window is not
    frame-contiguous or lacks neighbor_ids sidecars."""
    by_bag: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bag[str(r["bag"])].append(r)
    # candidate windows: every contiguous run start, scored by preferred-label count
    candidates: list[tuple[float, list[dict]]] = []
    for bag, brows in by_bag.items():
        brows.sort(key=lambda r: int(r["frame"]))
        for i in range(0, len(brows) - window_len + 1):
            win = brows[i : i + window_len]
            frames = [int(r["frame"]) for r in win]
            # must be exactly frame_step-contiguous
            if any(frames[j + 1] - frames[j] != frame_step for j in range(len(frames) - 1)):
                continue
            score = sum(
                1
                for r in win
                if str(r.get("interaction")) in prefer_labels
                or str(r.get("maneuver")) in prefer_labels
                or str(r.get("speed_bin")) in prefer_labels
            )
            candidates.append((score, win))
    if not candidates:
        raise ValueError(
            f"no {window_len}-frame contiguous (step={frame_step}) windows found; "
            "corpus is not contiguous at this step"
        )
    candidates.sort(key=lambda t: (-t[0], str(t[1][0]["npz_path"])))
    chosen: list[list[dict]] = []
    used: set[tuple[str, int]] = set()
    for _score, win in candidates:
        span = {(str(r["bag"]), int(r["frame"])) for r in win}
        if span & used:
            continue
        # verify neighbor_ids sidecar presence (fail loud)
        for r in win:
            sc = _read_sidecar_json(str(r["npz_path"]))
            if sc is None or "neighbor_ids" not in sc:
                raise ValueError(
                    f"contiguous mode requires neighbor_ids sidecars; missing for {r['npz_path']}"
                )
        chosen.append(win)
        used |= span
        if len(chosen) >= n_windows:
            break
    if len(chosen) < n_windows:
        raise ValueError(f"only found {len(chosen)} non-overlapping windows, requested {n_windows}")
    return chosen


# --------------------------------------------------------------------------- #
# canonicalization
# --------------------------------------------------------------------------- #
def _pad_slots(arr: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad (or truncate) the leading agent-slot axis to exactly n."""
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] > n:
        return arr[:n]
    pad = np.zeros((n - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _to_raw3_pose(arr: np.ndarray) -> np.ndarray:
    """Normalize an ego/goal pose tensor to the canonical 3-col on-disk width
    [x, y, heading]: a 4-col [x, y, cos, sin] is collapsed to heading=atan2(sin, cos);
    3-col passes through. Works for both a [T, C] history/future and a [C] goal pose."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] == 4:
        heading = np.arctan2(arr[..., 3], arr[..., 2])
        arr = np.concatenate([arr[..., :2], heading[..., None]], axis=-1)
    return arr.astype(np.float32)


def canonicalize_npz(in_path: str, out_path: str) -> None:
    """Write a 320-slot / 4-col-neighbor-future / float32 copy of in_path.

    ego_shape must be present (fail loud). Sidecar copied alongside if present.
    version stamped to 3 (native 4-col)."""
    d = dict(np.load(in_path, allow_pickle=True))
    # Fail loud early if a required field is missing (before transforms touch them).
    _require_fields(d, in_path)

    d["neighbor_agents_past"] = _pad_slots(
        np.asarray(d["neighbor_agents_past"], dtype=np.float32), N_SLOTS
    )
    naf = future_to_4col(
        np.asarray(d["neighbor_agents_future"], dtype=np.float32),
        zero_rows_are_padding=True,
    )
    d["neighbor_agents_future"] = _pad_slots(naf, N_SLOTS)

    # Normalize ego/goal poses to ONE canonical on-disk width (3-col [x, y, heading]).
    # The loader's heading_to_cos_sin accepts 3- OR 4-col, but a corpus that MIXES widths
    # crashes at default DataLoader collation (before that transform): [31,3] vs [31,4].
    # Collapsing any 4-col [x, y, cos, sin] to heading here keeps every output uniform.
    for k in ("ego_agent_past", "ego_agent_future", "goal_pose"):
        if k in d:
            d[k] = _to_raw3_pose(d[k])

    # enforce float32 on all floating arrays (ego_agent_future stays [80,3])
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            d[k] = v.astype(np.float32)
    d["version"] = np.int64(3)  # npz FORMAT version (native 4-col); read by loaders

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **d)
    side = in_path[:-4] + ".json"
    if os.path.exists(side):
        shutil.copy(side, out_path[:-4] + ".json")

    # Validate every output against the canonical schema (names + N_SLOTS/POSE_DIM +
    # the real model-input loader) — a malformed tensor fails here, not mid-training.
    _validate_output(out_path)


def _verify_canonical(out_path: str) -> None:
    _validate_output(out_path)


# --------------------------------------------------------------------------- #
# renders
# --------------------------------------------------------------------------- #
def render_selected(
    selected: list[dict],
    out_root: str,
    render_dir: str,
    *,
    frac: float,
    seed: int,
    namer=output_basename,
) -> None:
    """Stratified per-bucket titled renders + one collage grid per bucket.
    Reuses from_npz + _ego_future_to_predictions + save_step_figure (no route ->
    no lanelet2). Renders the CANONICALIZED output NPZ (also validates it loads)."""
    if frac <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scenario_generation.npz_loader import from_npz
    from scenario_generation.render_npz_dir import _ego_future_to_predictions
    from scenario_generation.replay import save_step_figure

    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        by_bucket[str(r["label"])].append(r)

    os.makedirs(render_dir, exist_ok=True)
    for bucket, brows in sorted(by_bucket.items()):
        k = max(1, int(round(frac * len(brows))))
        picks = rng.sample(brows, min(k, len(brows)))
        safe = bucket.replace("|", "__").replace("/", "-")
        bdir = os.path.join(render_dir, safe)
        os.makedirs(bdir, exist_ok=True)
        pngs = []
        for r in picks:
            out_npz = os.path.join(out_root, "npz", namer(str(r["npz_path"])))
            if not os.path.exists(out_npz):
                continue
            title = (
                f"{bucket}  v={r.get('ego_speed_ms'):.1f} yaw={r.get('gt_yaw_deg'):.0f} "
                f"nb={r.get('neighbor_count')} lead={r.get('has_lead')}"
            )
            png = os.path.join(bdir, os.path.basename(out_npz)[:-4] + ".png")
            try:
                scene = from_npz(out_npz)
                save_step_figure(
                    scene=scene,
                    agent_predictions=_ego_future_to_predictions(scene),
                    output_path=png,
                    step=0,
                    n_steps=0,
                    title_prefix=title,
                )
                pngs.append(png)
            except Exception as e:
                print(f"[build] render FAIL {out_npz}: {type(e).__name__}: {e}", flush=True)
        # collage grid — cosmetic only; never let a matplotlib hiccup abort a
        # build whose NPZs + manifests are already written.
        if pngs:
            fig = None
            try:
                ncol = min(4, len(pngs))
                nrow = (len(pngs) + ncol - 1) // ncol
                fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
                axes = np.atleast_1d(axes).ravel()
                for ax in axes:
                    ax.axis("off")
                for ax, png in zip(axes, pngs):
                    ax.imshow(plt.imread(png))
                fig.suptitle(bucket)
                fig.tight_layout()
                fig.savefig(os.path.join(render_dir, f"collage__{safe}.png"), dpi=80)
            except Exception as e:
                print(f"[build] collage FAIL {bucket}: {type(e).__name__}: {e}", flush=True)
            finally:
                if fig is not None:
                    plt.close(fig)
    print(f"[build] renders -> {render_dir}", flush=True)


# --------------------------------------------------------------------------- #
# output assembly
# --------------------------------------------------------------------------- #
def _histogram(rows: list[dict]) -> Counter:
    return Counter(str(r["label"]) for r in rows)


def _write_readme(path: str, *, mode: str, before: Counter, after: Counter, meta: dict) -> None:
    lines = [f"# pipeline-test dataset ({mode} mode)\n"]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Bucket histogram (label triple: speed|maneuver|interaction)\n")
    lines.append("| bucket | before | after |")
    lines.append("|---|---:|---:|")
    for b in sorted(set(before) | set(after)):
        lines.append(f"| {b} | {before.get(b, 0)} | {after.get(b, 0)} |")
    lines.append(f"\n**Totals**: before={sum(before.values())} after={sum(after.values())}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def canonicalize_all(rows: list[dict], out_root: str, namer=output_basename) -> list[str]:
    npz_out_dir = os.path.join(out_root, "npz")
    os.makedirs(npz_out_dir, exist_ok=True)
    out_paths = []
    for i, r in enumerate(rows):
        src = str(r["npz_path"])
        dst = os.path.join(npz_out_dir, namer(src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)  # namer may include a bag subdir
        canonicalize_npz(src, dst)
        out_paths.append(dst)
        if (i + 1) % 500 == 0:
            print(f"[build] canonicalized {i + 1}/{len(rows)}", flush=True)
    # verify a sample (first + last)
    for p in {out_paths[0], out_paths[-1]} if out_paths else set():
        _verify_canonical(p)
    return out_paths


def run_frame_mode(rows: list[dict], args) -> None:
    before = _histogram(rows)
    _assign_difficulty(rows, args.uniform_priority)
    kept = dedup_sequences(
        rows, gap=args.dedup_gap, max_per_seq=args.max_per_seq, stride=args.seq_stride
    )
    label_quotas = _parse_label_quotas(args.label_quotas)

    train_rows = select_frame(
        kept, args.n_train, label_quotas=label_quotas, alpha=args.alpha, beta=args.beta
    )
    train_paths_set = {str(r["scene_path"]) for r in train_rows}
    remainder = [r for r in kept if str(r.get("npz_path")) not in train_paths_set]
    val_rows = select_frame(
        remainder, args.n_val, label_quotas=label_quotas, alpha=args.alpha, beta=args.beta
    )

    print(f"[build] selected train={len(train_rows)} val={len(val_rows)}", flush=True)
    train_out = canonicalize_all(train_rows, args.out_dir)
    val_out = canonicalize_all(val_rows, args.out_dir)

    with open(os.path.join(args.out_dir, "train.json"), "w") as f:
        json.dump(train_out, f, indent=0)
    with open(os.path.join(args.out_dir, "val.json"), "w") as f:
        json.dump(val_out, f, indent=0)

    _write_readme(
        os.path.join(args.out_dir, "README.md"),
        mode="frame",
        before=before,
        after=_histogram(train_rows),
        meta={
            "n_train": len(train_out),
            "n_val": len(val_out),
            "source_rows": len(rows),
            "after_dedup": len(kept),
            "label_quotas": label_quotas or "none",
            "seed": args.seed,
            "slots": N_SLOTS,
        },
    )
    render_selected(
        train_rows,
        args.out_dir,
        os.path.join(args.out_dir, "renders"),
        frac=args.render_frac,
        seed=args.seed,
    )
    print(f"[build] frame-mode DONE -> {args.out_dir}", flush=True)


def run_contiguous_mode(rows: list[dict], args) -> None:
    before = _histogram(rows)
    prefer = set(s.strip() for s in args.contiguous_prefer.split(",") if s.strip())
    windows = select_contiguous(
        rows,
        n_windows=args.n_windows,
        window_len=args.window_len,
        frame_step=args.frame_step,
        prefer_labels=prefer,
    )
    all_rows = [r for win in windows for r in win]
    # Contiguous mode keeps each frame's ORIGINAL dataset filename under a per-bag
    # subdir (contig_relpath) so a bag's frames stay one detectable rollout lineage
    # for the reproducer (frame mode's per-file hash would give every frame a
    # different prefix and break it).
    out_paths = canonicalize_all(all_rows, args.out_dir, namer=contig_relpath)

    # index output paths by (bag, frame) for manifest reconstruction
    chunk_lines = []
    gidx = 0
    for wi, win in enumerate(windows):
        win_out = [
            os.path.join(args.out_dir, "npz", contig_relpath(str(r["npz_path"]))) for r in win
        ]
        start_frame = int(win[0]["frame"])
        end_frame = int(win[-1]["frame"])
        chunk_lines.append(
            {
                "chunk_key": f"g{gidx:012d}_{win[0]['bag']}_{start_frame:08d}",
                "global_start_index": gidx,
                "global_end_index": gidx + len(win),
                "n_frames": len(win),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "end_reason": "chunk_len",
                "is_full_length": True,
                "start_scene_path": win_out[0],
                "end_scene_path": win_out[-1],
                "paths": win_out,
            }
        )
        gidx += len(win)

    with open(os.path.join(args.out_dir, "reproducer_chunks.jsonl"), "w") as f:
        for line in chunk_lines:
            f.write(json.dumps(line) + "\n")
    with open(os.path.join(args.out_dir, "window_scenes.json"), "w") as f:
        json.dump(out_paths, f, indent=0)
    # empty train/val for interface uniformity (contiguous corpus is the deliverable)
    with open(os.path.join(args.out_dir, "train.json"), "w") as f:
        json.dump(out_paths, f, indent=0)
    with open(os.path.join(args.out_dir, "val.json"), "w") as f:
        json.dump([], f)

    _write_readme(
        os.path.join(args.out_dir, "README.md"),
        mode="contiguous",
        before=before,
        after=_histogram(all_rows),
        meta={
            "n_windows": len(windows),
            "window_len": args.window_len,
            "frame_step": args.frame_step,
            "total_frames": len(all_rows),
            "prefer_labels": sorted(prefer),
            "seed": args.seed,
            "slots": N_SLOTS,
        },
    )
    # 1 render per window (first frame)
    firsts = [win[0] for win in windows]
    for r in firsts:
        r["label"] = f"window_{r['bag']}_{int(r['frame'])}"
    render_selected(
        firsts,
        args.out_dir,
        os.path.join(args.out_dir, "renders"),
        frac=1.0,
        seed=args.seed,
        namer=contig_relpath,  # contiguous frames live in per-bag subdirs, not hash-flat
    )
    print(f"[build] contiguous-mode DONE: {len(windows)} windows -> {args.out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--parquet_dir", required=True, help="dir of shard_*.parquet from scene_feature_index"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render_frac", type=float, default=0.02)
    ap.add_argument("--contiguous", action="store_true", help="contiguous-window mode")
    # frame mode
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument(
        "--label_quotas", type=str, default=None, help='"label=frac,..." rare-class reserves'
    )
    ap.add_argument("--alpha", type=float, default=1.0, help="difficulty (interestingness) weight")
    ap.add_argument(
        "--beta", type=float, default=0.0, help="utility (loss) weight; 0 = no loss available"
    )
    ap.add_argument(
        "--uniform_priority", action="store_true", help="flat difficulty (pure coverage)"
    )
    ap.add_argument("--dedup_gap", type=int, default=5, help="group_sequences max frame gap")
    ap.add_argument(
        "--max_per_seq", type=int, default=8, help="cap frames kept per contiguous event"
    )
    ap.add_argument("--seq_stride", type=int, default=1, help="min stride within a sequence")
    # contiguous mode
    ap.add_argument("--n_windows", type=int, default=4)
    ap.add_argument("--window_len", type=int, default=400)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument(
        "--contiguous_prefer", type=str, default="stopped,turn-L,turn-R,dense,has-lead,has-ped"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_feature_table(args.parquet_dir)
    # Deduplicate by canonical (realpath'd) npz_path BEFORE sequence grouping / caps.
    # extract_scene_features maps path aliases to one identity, but the parquet can
    # still hold multiple rows for the same physical scene (same path listed twice, or
    # two aliases pre-realpath in an older index); consuming them as distinct frames
    # underfills the requested capacity and skews stride caps.
    seen: set[str] = set()
    deduped = []
    for r in rows:
        # realpath so relative/absolute/symlink aliases of one physical scene (which an
        # older pre-realpath index can still carry) collapse to a single identity —
        # otherwise the same NPZ can land in both train and val.
        p = os.path.realpath(str(r.get("npz_path")))
        if p not in seen:
            seen.add(p)
            deduped.append(r)
    if len(deduped) != len(rows):
        print(f"[build] dedup npz_path identity: {len(rows)} -> {len(deduped)} rows", flush=True)
    rows = deduped
    if args.contiguous:
        run_contiguous_mode(rows, args)
    else:
        run_frame_mode(rows, args)


if __name__ == "__main__":
    main()
