"""Run closed-loop evaluation for all groups.

Accepts the same ``ClosedLoopConfig`` fields as train.py. Example::

    python diffusion_planner/run_all_groups_closed_loop.py \\
        --closed_loop_npz_root override.json site.json \\
        --model_path /media/.../best_model.pth \\
        --out_root /media/.../cl_results
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from diffusion_planner.config.closed_loop_config import (
    ClosedLoopConfig,
    ClosedLoopPassCondition,
)
from diffusion_planner.config.config_cli import build_config, build_parser, resolve_paths
from diffusion_planner.config.config_utils import save_config
from diffusion_planner.utils import ddp

from scenario_generation.wandb_closed_loop import (
    log_closed_loop_to_wandb,
)
from tag_toolkit.store import TagStore


def resolve_closed_loop_inputs(
    inputs: str | list[str],
    modes: list[str] | None = None,
) -> list[dict]:
    """Resolve input paths to a list of ``{"name", "groups", "mode", "tag_store"}`` entries.

    ``tag_store`` is built via ``TagStore.from_source``: if a same-named ``.tags.db``
    exists next to the input it is loaded; otherwise an in-memory index is built.
    The caller holds this store for the lifetime of the evaluation.

    ``modes`` is zipped positionally with the *input list* (1-to-1):

    - Omit ``modes``  ⇒ every entry gets mode ``"objects"``.
    - Provide ``modes`` ⇒ len(modes) MUST equal len(inputs); otherwise a
      ValueError is raised so a mismatched CLI argument is never silently swallowed.

    Return shape:

        [{"name": "sites", "groups": {"all": [...]}, "mode": "objects", "tag_store": TagStore}, ...]
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    if not modes:
        modes = ["objects"] * len(inputs)
    if len(modes) != len(inputs):
        raise ValueError(
            f"--closed_loop_object_modes ({len(modes)}: {modes}) must have the same length as "
            f"the number of JSON/folder inputs ({len(inputs)}: {list(inputs)})."
        )

    entries: list[dict] = []
    for input_path, mode in zip(inputs, modes):
        p = Path(input_path)

        if not p.exists():
            print(f"Warning: {input_path} does not exist, skipping", file=sys.stderr)
            continue

        groups: dict[str, list[str]] = {}

        if p.is_dir():
            groups.setdefault("all", []).append(str(p))
            tag_store = TagStore.from_source(str(p))

        elif p.suffix == ".json":
            with open(p, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for group_name, paths in data.items():
                    groups.setdefault(group_name, []).extend(str(Path(item)) for item in paths)
            elif isinstance(data, list):
                groups.setdefault("all", []).extend(str(Path(item)) for item in data)
            tag_store = TagStore.from_source(str(p))

        else:
            print(f"Warning: {input_path} has unsupported extension, skipping", file=sys.stderr)
            continue

        entries.append(
            {
                "name": p.stem if p.suffix else p.name,
                "groups": groups,
                "mode": mode,
                "tag_store": tag_store,
            }
        )

    return entries


def run_one_group(
    model,
    model_args,
    npz_root_list: list[str],
    out_dir: str | Path,
    cfg: ClosedLoopConfig,
    mode: str | None = None,
    render_media: bool = True,
    pass_condition: ClosedLoopPassCondition | None = None,
    tag_store=None,
) -> None:
    """Run closed-loop evaluation for a single group; writes ``summary.json`` + ``segments.jsonl``
    under ``out_dir``. Wandb logging is left to the caller.

    ``mode`` is passed explicitly so it doesn't have to be inferred from ``out_dir``
    (a json_name containing ``__noobj`` would silently misfer).

    ``pass_condition`` (when set) is plumbed through to ``ClosedLoopEvalConfig`` so that
    per-segment ``passed`` flags and the summary's ``pass_count`` / ``pass_rate`` /
    ``pass_condition`` block are produced at the metric-aggregation source — no
    post-processing pass over the file system needed here.
    """
    from scenario_generation.closed_loop_evaluation import (
        ClosedLoopEvalConfig,
        FullRouteClosedLoopEvaluation,
        RolloutParams,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode not in (None, "objects", "noobj"):
        raise ValueError(f"mode must be 'objects' or 'noobj', got {mode!r}")
    drop_objects = mode == "noobj"

    ddp_rank = ddp.get_rank()
    ddp_world_size = ddp.get_world_size()

    if len(npz_root_list) > 1:
        npz_root_arg = out_dir / "_npz_roots.json"
        if ddp_rank == 0:
            npz_root_arg.write_text(json.dumps([str(p) for p in npz_root_list]))
        if ddp_world_size > 1:
            torch.distributed.barrier()
        npz_root_arg = str(npz_root_arg)
    else:
        npz_root_arg = npz_root_list[0]

    evaluator = FullRouteClosedLoopEvaluation(
        model,
        model_args,
        ClosedLoopEvalConfig(
            out_dir=out_dir,
            params=RolloutParams(
                device=cfg.device,
                near_miss_thresh=cfg.closed_loop_near_miss_thresh,
                search_radius=cfg.closed_loop_search_radius,
                warmup_steps=cfg.closed_loop_warmup_steps,
                unstick_after=cfg.closed_loop_unstick_after,
                unstick_advance_m=cfg.closed_loop_unstick_advance_m,
                unstick_radius_mult=cfg.closed_loop_unstick_radius_mult,
                unstick_teleport_after=cfg.closed_loop_unstick_teleport_after,
                draw_every=cfg.closed_loop_draw_every if render_media else None,
                draw_workers=cfg.closed_loop_draw_workers,
                replan_interval=cfg.closed_loop_replan_interval,
                tracker_mode=cfg.closed_loop_tracker_mode,
                neighbor_history_mode=cfg.closed_loop_neighbor_history_mode,
                yaw_gate=cfg.closed_loop_yaw_gate,
                strong_brake_mps2=cfg.closed_loop_strong_brake_mps2,
                abort_deviation_m=cfg.closed_loop_abort_deviation_m,
                abort_after=cfg.closed_loop_abort_after,
                abort_max_snaps=cfg.closed_loop_abort_max_snaps,
                drop_objects=drop_objects,
                goal_mode=cfg.closed_loop_goal_mode,
                title_prefix=cfg.closed_loop_title_prefix,
                distance_label_offset_m=cfg.closed_loop_distance_label_offset_m,
                view_half_m=cfg.closed_loop_view_half_m,
                max_stuck_steps=cfg.closed_loop_max_stuck_steps,
                goal_reach_m=cfg.closed_loop_goal_reach_m,
                interpolate=cfg.closed_loop_interpolate,
                color_by_uuid=cfg.closed_loop_color_by_uuid,
                window=cfg.closed_loop_window,
                max_steps=cfg.closed_loop_max_steps,
                timeline_progress_mode=cfg.closed_loop_timeline_progress_mode,  # replay mode
            ),
            fps=float(cfg.closed_loop_fps),
            verbose=False,
            profile=False,
            max_jobs=None,
            pass_condition=pass_condition,
        ),
        npz_root_arg,
        seg_len=cfg.closed_loop_seg_len,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        tag_store=tag_store,
    )

    evaluator.run_distributed()


def _make_summary_key(json_name: str, group_name: str) -> str:
    """Build the summary key, e.g. 'override/departure' or 'site/all'."""
    if "/" in group_name or "/" in json_name:
        raise ValueError(
            f"json_name and group_name must not contain '/': got {json_name!r}, {group_name!r}"
        )
    return f"{json_name}/{group_name}"


def _load_group_results(
    out_dir: Path | str,
) -> dict[str, dict]:
    """Reload per-group results from ``out_dir`` (each ``<group_dir>/summary.json``
    augmented with rows from the matching ``segments.jsonl``). Groups missing
    ``segments.jsonl`` (partial run, manual deletion) are skipped with a warning.

    Pass/fail flags (``passed``) and the summary's ``pass_count`` / ``pass_rate`` /
    ``pass_condition`` block are produced at the metric-aggregation source
    (see ``aggregate()`` and ``run_job()`` in ``scenario_generation/closed_loop_eval{ation}.py``),
    so this loader just reads them — no second pass is needed.
    """
    out_dir = Path(out_dir)
    summaries: dict[str, dict] = {}
    for summary_file in out_dir.rglob("summary.json"):
        key = "/".join(summary_file.parent.relative_to(out_dir).parts)
        try:
            summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError as exc:
            print(
                f"Warning: skipping malformed summary at {summary_file}: {exc}",
                file=sys.stderr,
            )
            continue
        segments_jsonl = summary_file.parent / "segments.jsonl"
        if not segments_jsonl.is_file():
            print(
                f"Warning: {key} has no segments.jsonl, skipping",
                file=sys.stderr,
            )
            continue
        summary["segments"] = [
            json.loads(line) for line in segments_jsonl.read_text().splitlines() if line.strip()
        ]
        summaries[key] = summary
    return summaries


def _write_groups_manifest(out_dir: Path | str, summaries: dict[str, dict]) -> None:
    """Write ``<out_dir>/groups.json`` aggregating ``summaries``.

    Aggregate keys are bare (``route_completion``, ``total_curb_hits``, ...) and
    the file shape matches what ``scenario_generation.wandb_closed_loop``'s
    ``_aggregate`` would produce, so the two paths can't drift apart.
    """
    agg: dict = {}
    if summaries:
        objects_only_values = [s for k, s in summaries.items() if "__noobj/" not in k]
        n_segments = sum(int(s.get("n_segments", 0) or 0) for s in summaries.values())
        route_num = sum(
            float(s.get("mean_route_completion", 0.0) or 0.0) * int(s.get("n_segments", 0) or 0)
            for s in summaries.values()
        )

        dev_num = 0.0
        dev_steps = 0
        for v in summaries.values():
            dev = v.get("mean_gt_deviation_m", None)
            steps = int(v.get("total_steps", 0) or 0)
            if dev is not None and math.isfinite(dev) and steps > 0:
                dev_num += float(dev) * steps
                dev_steps += steps

        agg = {
            "n_groups": len(summaries),
            "n_segments": n_segments,
            "total_steps": sum(int(s.get("total_steps", 0) or 0) for s in summaries.values()),
            "mean_route_completion": (route_num / n_segments) if n_segments else 0.0,
            "mean_gt_deviation_m": (dev_num / dev_steps) if dev_steps else float("inf"),
            "total_curb_hits": sum(
                int(s.get("road_border", {}).get("collision_count", 0) or 0)
                for s in summaries.values()
            ),
            "total_snaps": sum(
                int(s.get("reproducer", {}).get("snap_count", 0) or 0) for s in summaries.values()
            ),
            "total_red_light_violations": sum(
                int(s.get("red_light_violation", {}).get("count", 0) or 0)
                for s in summaries.values()
            ),
            "total_strong_brakes": sum(
                int(s.get("strong_brake", {}).get("count", 0) or 0) for s in summaries.values()
            ),
            "n_segments_diverged": sum(
                int(s.get("n_segments_diverged", 0) or 0) for s in summaries.values()
            ),
            "total_collision_events": sum(
                int(s.get("object", {}).get("collision_count", 0) or 0) for s in objects_only_values
            ),
        }

        total_pass = sum(int(s.get("pass_count", 0) or 0) for s in summaries.values())
        total_fail = sum(int(s.get("fail_count", 0) or 0) for s in summaries.values())
        agg["n_pass_segments"] = total_pass
        agg["n_fail_segments"] = total_fail
        agg["pass_rate"] = (total_pass / n_segments) if n_segments else 0.0

    Path(out_dir, "groups.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))


def run_closed_loop_main(
    model,  # always provided by the caller
    model_args,  # model args from load_model (for closed-loop rollout)
    cfg: ClosedLoopConfig,
    out_root: str | Path | None,
    *,
    wandb_run: "wandb.sdk.wandb_run.Run"
    | None,  # Optional wandb.Run instance; if None, creates own session
    only_json: list[str] | None,
    render_media: bool | None,
) -> bool:
    """Unified entry point for closed-loop evaluation.

    Works both as:
    - Direct API call from train.py (model provided, wandb_run provided)
    - CLI entry point via main() (model provided directly)

    Output directory structure:
        <out_root>/<json_name>/<group_name>          (objects mode)
        <out_root>/<json_name>__noobj/<group_name>   (noobj mode)

    Multi-GPU: launched via ``torch.distributed.run``. DDP rank is read from ``RANK``.
    All ranks evaluate all groups; per-group ``run_distributed()`` does its own barrier
    + rank-0 merge, so by the time we reach aggregation every rank's shards are flushed.
    Rank 0 then writes the manifest and logs to wandb.

    Writes:
        - ``<out_root>/groups.json`` (root aggregate)
        - ``<out_root>/<json_name>/groups.json`` and ``<out_root>/<json_name>__noobj/groups.json``
          (per-JSON aggregates)
        - W&B log payload if a run is provided (or one is created here)

    Returns True on success, False if no inputs were resolved (so CLI wrappers
    can map that to their own exit code).
    """
    if not cfg.closed_loop_npz_root:
        print("No closed_loop_npz_root set, skipping closed-loop evaluation", file=sys.stderr)
        return False

    # ``cfg.pass_conditions`` is auto-loaded at construction time (see ``__post_init__``).
    pass_conditions = cfg.pass_conditions

    entries = resolve_closed_loop_inputs(
        cfg.closed_loop_npz_root, modes=cfg.closed_loop_object_modes
    )
    if only_json:
        entries = [e for e in entries if e["name"] in only_json]
    if not entries:
        print(f"No inputs found under {cfg.closed_loop_npz_root}", file=sys.stderr)
        return False

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if ddp.get_rank() == 0:
        save_config(cfg, out_root, "closed_loop_config.json")

    if render_media is None:
        render_media = cfg.render_media

    written_json_labels: set[str] = set()
    for entry in entries:
        json_name, mode, groups = entry["name"], entry["mode"], entry["groups"]
        json_label = json_name if mode == "objects" else f"{json_name}__noobj"
        json_out_dir = out_root / json_label
        json_out_dir.mkdir(parents=True, exist_ok=True)

        tag_store = entry.get("tag_store")

        for group_name, npz_paths in groups.items():
            mode_out_dir = json_out_dir / group_name
            summary_key = _make_summary_key(json_label, group_name)
            # Per-group condition (falls back to the loaded default if not overridden in YAML).
            group_condition = pass_conditions.get_condition(group_name)
            run_one_group(
                model,
                model_args,
                npz_paths,
                mode_out_dir,
                cfg,
                mode=mode,
                render_media=render_media,
                pass_condition=group_condition,
                tag_store=tag_store,
            )

        written_json_labels.add(json_label)

    # rank-0 reads the per-group summaries; run_one_group already barriered + merged internally.
    if ddp.get_rank() == 0:
        all_summaries = _load_group_results(out_root)
        all_group_names = sorted(all_summaries.keys())

        _write_groups_manifest(out_root, all_summaries)

        for json_label in written_json_labels:
            json_prefix = f"{json_label}/"
            per_json_summaries = {
                k: v for k, v in all_summaries.items() if k.startswith(json_prefix)
            }
            json_out_dir = out_root / json_label
            if json_out_dir.exists() and per_json_summaries:
                _write_groups_manifest(json_out_dir, per_json_summaries)

        if all_group_names:
            log_closed_loop_to_wandb(cfg, all_group_names, all_summaries, run=wandb_run)

    return True


def _build_parser() -> "argparse.ArgumentParser":
    parser = build_parser(ClosedLoopConfig, description=__doc__)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument(
        "--only_json",
        nargs="*",
        default=None,
        help="run only these JSON/folder names (e.g. override site)",
    )
    return parser


def main() -> int:
    import torch
    from diffusion_planner.utils import ddp

    from scenario_generation.closed_loop_evaluation import FullRouteClosedLoopEvaluation

    parser = _build_parser()
    args = parser.parse_args()
    resolve_paths(args, ClosedLoopConfig)

    cfg = build_config(ClosedLoopConfig, args)
    base_out_root = args.out_root

    # Init DDP when launched under torchrun (RANK/WORLD_SIZE present) so that
    # ddp.get_rank()/get_world_size() below reflect the real process group; without
    # this every rank would fall back to (0, 1) and silently re-run every route.
    # verbose=True installs setup_for_distributed(rank == 0) so non-master prints
    # are silenced under torchrun, matching train.py / valid_predictor.py.
    global_rank, local_rank, world_size = ddp.ddp_setup_universal(True, cfg)
    print(f"{global_rank=}, {local_rank=}, {world_size=}")

    if cfg.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"

    model, model_args = FullRouteClosedLoopEvaluation.load_model_pair(args.model_path, cfg.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_root = base_out_root / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    ok = run_closed_loop_main(
        model=model,
        model_args=model_args,
        cfg=cfg,
        out_root=out_root,
        wandb_run=None,
        only_json=args.only_json,
        render_media=None,
    )
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())
