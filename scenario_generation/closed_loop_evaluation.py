"""Class-based closed-loop evaluation framework.

``ClosedLoopEvaluation`` defines the shared workflow:

  1. ``discover_jobs`` — what to evaluate (routes, bags, classification dates, …)
  2. ``shard_jobs`` — DDP round-robin assignment (override for custom sharding)
  3. ``run_job`` — rollout for one job unit (segments via ``render_segment``, etc.)
  4. ``build_summary`` / ``write_artifacts`` / ``print_summary`` — serialize + terminal output

``FullRouteClosedLoopEvaluation`` is the segment-wise evaluator (``segments.jsonl``) over every
route under an npz_root.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scenario_generation.closed_loop_ddp import shard_items
from scenario_generation.closed_loop_eval import (
    aggregate,
    build_mp4,
    enumerate_multi_root_routes,
    format_summary_lines,
    load_segment_rows_with_tdigests,
    segment_row_for_json,
    tdigest_sidecar_row,
)
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import render_segment
from scenario_generation.route_timeline import RouteTimeline


def _collective_device():
    """Device the small control-plane collectives must live on for the active backend."""
    import torch
    import torch.distributed as dist

    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _raise_if_any_rank_failed(failed_detail: str, rank: int) -> None:
    """Turn one rank's exception into a failure on every rank. Collective; call after a barrier.

    Both halves matter, and each covers a distinct failure:

    * The caller must not re-raise before the barrier -- a rank that dies early leaves the others
      parked there until the process-group timeout (10000s as configured by
      ``ddp.ddp_setup_universal``), turning a crash into a multi-hour wedge.
    * Reaching the barrier is not enough on its own, because ``execute_jobs`` writes
      ``segments_{rank}.jsonl`` inside a ``with`` block: when a rank raises, its partial shard is
      still flushed and closed, so the file *exists* and ``collect_ddp_shards``'s missing-file
      check passes. Rank-0 would then merge a truncated shard and publish it as a clean summary.

    Every rank must call this, including rank-0 and including the ranks that succeeded.
    """
    import torch
    import torch.distributed as dist

    flag = torch.tensor(
        [1.0 if failed_detail else 0.0], dtype=torch.float64, device=_collective_device()
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    n_failed = int(flag.item())
    if n_failed:
        raise RuntimeError(
            f"closed-loop DDP: {n_failed} rank(s) failed during evaluation; refusing to merge a "
            f"partial result. This rank: {failed_detail or f'rank {rank} ok'}"
        )


@dataclass
class RolloutParams:
    """Rollout knobs for full-route segment evaluation."""

    device: str = "cuda"
    near_miss_thresh: float = 0.5
    search_radius: float = 1.5
    warmup_steps: int = 0
    unstick_after: int = 300
    unstick_advance_m: float = 2.5
    unstick_radius_mult: float = 10.0
    unstick_teleport_after: int = 300
    # write a PNG only every N steps; None skips the per-step render entirely (no PNGs, no
    # video, no colormap images -- see build_full_closed_loop_wandb_log's matching
    # render_media) while metrics/wandb scalars are unaffected -- lets a caller (e.g.
    # train.py, most epochs) skip the dominant per-epoch cost and only pay it on the one call
    # that actually needs media (e.g. the final epoch).
    draw_every: int | None = 8
    replan_interval: int = 10
    tracker_mode: str = "mpc"
    neighbor_history_mode: str = "recorded"
    yaw_gate: bool = True
    strong_brake_mps2: float = -2.5
    # Early-abort a badly-diverged segment instead of burning the full step budget on a
    # rollout that will never recover (e.g. an undertrained model driving off-lane). Set well
    # above the unstick_* knobs above: unstick snaps the ego back onto GT to let a merely-stuck
    # rollout continue; abort gives up on a rollout unstick can't save. 0 = disabled for either
    # knob. See ``reproducer_rollout.render_segment`` for the exact trigger condition.
    abort_deviation_m: float = 50.0
    abort_after: int = 30
    abort_max_snaps: int = 0
    # Empty-world ablation: zero neighbor_agents_past + static_objects each step (map kept) —
    # separates "reacts badly to traffic" from "can't follow the route/map". collision/near-miss
    # are 0 by construction when this is set.
    drop_objects: bool = False

    def render_kwargs(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "near_miss_thresh": self.near_miss_thresh,
            "search_radius": self.search_radius,
            "warmup_steps": self.warmup_steps,
            "unstick_after": self.unstick_after,
            "unstick_advance_m": self.unstick_advance_m,
            "unstick_radius_mult": self.unstick_radius_mult,
            "unstick_teleport_after": self.unstick_teleport_after,
            "draw_every": self.draw_every,
            "replan_interval": self.replan_interval,
            "tracker_mode": self.tracker_mode,
            "neighbor_history_mode": self.neighbor_history_mode,
            "yaw_gate": self.yaw_gate,
            "strong_brake_mps2": self.strong_brake_mps2,
            "abort_deviation_m": self.abort_deviation_m,
            "abort_after": self.abort_after,
            "abort_max_snaps": self.abort_max_snaps,
            "drop_objects": self.drop_objects,
        }


@dataclass
class ClosedLoopEvalConfig:
    """Shared configuration for any closed-loop evaluation mode."""

    out_dir: Path
    params: RolloutParams
    fps: float = 10.0
    verbose: bool = True
    profile: bool = False
    max_jobs: int | None = None


@dataclass
class ClosedLoopJob:
    """One schedulable unit (route, bag, classification date, …)."""

    job_id: str


@dataclass
class JobRunResult:
    """Accumulated rows + artifacts from one or more jobs on this rank."""

    rows: list[dict] = field(default_factory=list)
    video_mp4s: list[Path] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


class ClosedLoopEvaluation(ABC):
    """Template-method closed-loop evaluation workflow."""

    mode: str = "base"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        *,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        self.model = model
        self.model_args = model_args
        self.config = config
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_model_pair(model_path: Path | str, device: str):
        """Load a torch checkpoint or ONNX export (``args.json`` next to the file)."""
        from scenario_generation.simulate import load_model, load_onnx_model

        model_path = Path(model_path)
        if model_path.suffix == ".onnx":
            return load_onnx_model(model_path, device)
        return load_model(model_path, device)

    def run(self) -> dict:
        """Discover jobs, optionally shard for DDP, execute, summarize, and persist."""
        t0 = time.perf_counter()
        jobs = self.discover_jobs()
        if self.config.max_jobs is not None:
            jobs = jobs[: self.config.max_jobs]
        jobs = self.shard_jobs(jobs)
        if self.config.verbose and jobs and self.ddp_world_size > 1:
            print(
                f"DDP rank {self.ddp_rank}/{self.ddp_world_size}: "
                f"{len(jobs)} job(s) -> {[j.job_id for j in jobs]}"
            )

        result = self.execute_jobs(jobs)
        elapsed_sec = time.perf_counter() - t0

        if self.ddp_world_size > 1:
            return self.finalize_ddp(result, elapsed_sec=elapsed_sec)

        summary = self.build_summary(result, elapsed_sec=elapsed_sec)
        summary["mode"] = self.mode
        summary["ddp_rank"] = self.ddp_rank
        summary["ddp_world_size"] = self.ddp_world_size
        self.write_artifacts(summary, result)
        if self.config.verbose:
            self.print_summary(summary)
        return summary

    def run_distributed(self) -> dict:
        """Run evaluation; under DDP, barrier then rank-0 ``merge_ddp_shards``.

        Requires an initialized process group whenever ``ddp_world_size > 1``: skipping the
        barrier (e.g. because torch.distributed was never initialized) would let rank-0 start
        merging before every other rank has finished writing its ``segments_{rank}.jsonl``
        shard, silently producing a summary built from a partial/incomplete set of ranks. That
        check runs *before* the rollouts, so a misconfigured launch fails in a second rather
        than after the full evaluation.

        A rank that raises still reaches the barrier, and the failure is then propagated to
        every rank -- see ``_raise_if_any_rank_failed`` for why neither half is optional.
        """
        t0 = time.perf_counter()
        if self.ddp_world_size <= 1:
            return self.run()
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                f"closed-loop run_distributed: ddp_world_size={self.ddp_world_size} > 1 but "
                "torch.distributed is not initialized -- refusing to merge without a barrier "
                "to guarantee every rank finished writing its shard first."
            )

        partial: dict = {}
        failed_detail = ""
        try:
            partial = self.run()
        except Exception as exc:  # noqa: BLE001 - re-raised on every rank below
            # Deliberately swallowed *here*: an immediate re-raise would skip the barrier and
            # strand every other rank on it until the process-group timeout.
            traceback.print_exc()
            failed_detail = f"rank {self.ddp_rank} raised: {exc!r}"

        dist.barrier()
        _raise_if_any_rank_failed(failed_detail, self.ddp_rank)

        if self.ddp_rank != 0:
            return {}
        if partial.get("ddp_shard"):
            return self.merge_ddp_shards(self.ddp_world_size, elapsed_sec=time.perf_counter() - t0)
        return partial

    def shard_jobs(self, jobs: list[ClosedLoopJob]) -> list[ClosedLoopJob]:
        return shard_items(jobs, self.ddp_rank, self.ddp_world_size)

    def initial_job_extras(self) -> dict[str, Any]:
        return {}

    def merge_job_result(self, merged: JobRunResult, partial: JobRunResult) -> None:
        merged.rows.extend(partial.rows)
        merged.video_mp4s.extend(partial.video_mp4s)
        for key, value in partial.extras.items():
            if key in merged.extras and isinstance(merged.extras[key], list):
                merged.extras[key].extend(value)
            elif key in merged.extras and isinstance(merged.extras[key], dict):
                merged.extras[key].update(value)
            elif key in merged.extras and isinstance(merged.extras[key], int):
                merged.extras[key] = int(merged.extras[key]) + int(value)
            else:
                merged.extras[key] = value

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        """Hook for per-job progress (subclasses print route/bag summaries here)."""

    def execute_jobs(self, jobs: list[ClosedLoopJob]) -> JobRunResult:
        merged = JobRunResult(extras=self.initial_job_extras())
        for ri, job in enumerate(jobs):
            partial = self.run_job(job)
            self.merge_job_result(merged, partial)
            self.on_job_complete(job, partial, ri, len(jobs))
        return merged

    def finalize_ddp(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        """Signal rank-0 to merge once every rank reaches the barrier.

        Per-rank rows are already durably persisted by ``execute_jobs`` writing
        ``segments_{rank}.jsonl`` (+ tdigest sidecar) directly -- no separate shard blob to write
        here, so a crashed rank can't leave stale/inconsistent shard state behind.
        """
        return {
            "mode": self.mode,
            "ddp_shard": True,
            "ddp_rank": self.ddp_rank,
            "ddp_world_size": self.ddp_world_size,
            "n_rows": len(result.rows),
            "elapsed_sec": elapsed_sec,
        }

    def collect_ddp_shards(self, world_size: int) -> JobRunResult:
        """Load every rank's ``segments_{rank}.jsonl`` (+ tdigest sidecar), the same convention
        ``closed_loop_eval.run_closed_loop_eval`` uses for its own multi-GPU shards.

        Fails loudly if any rank's file is missing (crashed before writing, or never reached
        ``finalize_ddp``) instead of silently merging a partial result as a "successful" summary.
        Videos are found by globbing the shared ``out_dir`` directly -- every rank renders into
        the same directory with globally-unique route-keyed filenames, so there's nothing to
        track per-shard.
        """
        missing = [
            r for r in range(world_size) if not (self.out_dir / f"segments_{r}.jsonl").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"closed-loop DDP merge: missing segments_<rank>.jsonl for rank(s) {missing} "
                f"under {self.out_dir} (world_size={world_size}) -- one or more ranks likely "
                "crashed or never reached finalize_ddp; refusing to merge a partial result."
            )
        rows = load_segment_rows_with_tdigests(self.out_dir, shard_glob="segments_*.jsonl")
        video_mp4s = sorted(self.out_dir.glob("*.mp4"))
        return JobRunResult(rows=rows, video_mp4s=video_mp4s)

    def merge_ddp_shards(self, world_size: int, *, elapsed_sec: float = 0.0) -> dict:
        """Rank-0: merge shard files, persist artifacts, and return the final summary."""
        result = self.collect_ddp_shards(world_size)
        self.prepare_ddp_merge_artifacts(result)
        summary = self.build_summary(result, elapsed_sec=elapsed_sec)
        summary["mode"] = self.mode
        summary["ddp_shard"] = True
        summary["ddp_world_size"] = world_size
        self.write_artifacts(summary, result)
        if self.config.verbose:
            self.print_summary(summary)
        return summary

    def prepare_ddp_merge_artifacts(self, result: JobRunResult) -> None:
        """Hook for mode-specific files written during rank-0 DDP merge."""

    @abstractmethod
    def discover_jobs(self) -> list[ClosedLoopJob]:
        raise NotImplementedError

    @abstractmethod
    def run_job(self, job: ClosedLoopJob) -> JobRunResult:
        raise NotImplementedError

    @abstractmethod
    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        raise NotImplementedError

    @abstractmethod
    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def print_summary(self, summary: dict) -> None:
        raise NotImplementedError


@dataclass
class FullRouteRouteJob(ClosedLoopJob):
    """One route (bag-prefix group) under an NPZ root."""

    npz_root: Path = field(default_factory=Path)
    route_key: str = ""
    route_paths: list[Path] = field(default_factory=list)
    seg_len: int = 100_000


class FullRouteClosedLoopEvaluation(ClosedLoopEvaluation):
    """Segment-wise full-route closed-loop over every route under ``npz_root``.

    ``npz_root`` accepts anything ``closed_loop_eval.resolve_npz_roots`` does: a single
    directory tree, a ``.json`` path-list file, or an already-resolved list of root paths
    (e.g. one site's several curated date/time entries from a JSON-manifest-based site
    discovery) -- ``discover_jobs`` merges routes across all of them.
    """

    mode = "full"

    def __init__(
        self,
        model,
        model_args,
        config: ClosedLoopEvalConfig,
        npz_root: Path | str | list[Path | str],
        *,
        seg_len: int = 100_000,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        super().__init__(
            model,
            model_args,
            config,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        self.npz_root = npz_root
        self.seg_len = seg_len

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path | str,
        npz_root: Path | str | list[Path | str],
        config: ClosedLoopEvalConfig,
        *,
        seg_len: int = 100_000,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> FullRouteClosedLoopEvaluation:
        model, model_args = cls.load_model_pair(model_path, config.params.device)
        return cls(
            model,
            model_args,
            config,
            npz_root,
            seg_len=seg_len,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

    def discover_jobs(self) -> list[FullRouteRouteJob]:
        routes, route_sidecar_dir = enumerate_multi_root_routes(self.npz_root)
        return [
            FullRouteRouteJob(
                job_id=key,
                npz_root=route_sidecar_dir[key],
                route_key=key,
                route_paths=routes[key],
                seg_len=self.seg_len,
            )
            for key in sorted(routes)
        ]

    def initial_job_extras(self) -> dict[str, Any]:
        return {"route_keys": []}

    def execute_jobs(self, jobs: list[ClosedLoopJob]) -> JobRunResult:
        """Stream per-segment rows to ``segments.jsonl`` (``segments_{rank}.jsonl`` + a tdigest
        sidecar under DDP sharding -- the same convention ``run_closed_loop_eval`` uses), while
        jobs run."""
        merged = JobRunResult(extras={"route_keys": []})
        suffix = f"_{self.ddp_rank}" if self.ddp_world_size > 1 else ""
        segments_path = self.out_dir / f"segments{suffix}.jsonl"
        digests_path = self.out_dir / f"tdigests{suffix}.jsonl"
        with (
            segments_path.open("w", encoding="utf-8") as fout,
            digests_path.open("w", encoding="utf-8") as fdigest,
        ):
            for ri, job in enumerate(jobs):
                assert isinstance(job, FullRouteRouteJob)
                partial = self.run_job(job, segments_file=fout, digest_file=fdigest)
                merged.rows.extend(partial.rows)
                merged.video_mp4s.extend(partial.video_mp4s)
                merged.extras["route_keys"].append(job.route_key)
                self.on_job_complete(job, partial, ri, len(jobs))
        return merged

    def run_job(
        self,
        job: ClosedLoopJob,
        *,
        segments_file=None,
        digest_file=None,
    ) -> JobRunResult:
        assert isinstance(job, FullRouteRouteJob)
        params = self.config.params
        timers = Timers() if self.config.profile else None
        tl = RouteTimeline(job.route_paths, sidecar_dir=job.npz_root, timers=timers)
        rows: list[dict] = []
        video_mp4s: list[Path] = []

        for start, end in tl.iter_segments(job.seg_len):
            png_dir = self.out_dir / f"{job.route_key}_{start}_{end}"
            metrics = render_segment(
                self.model,
                self.model_args,
                tl,
                start,
                end,
                png_dir,
                **params.render_kwargs(),
            )
            row = {"route": job.route_key, **metrics}
            if segments_file is not None:
                # Human-readable segments.jsonl never carries the raw _tdigest blobs; those go to
                # the sidecar so a later DDP merge (or a re-load of this run) can still pool an
                # approximate global clearance percentile without the full per-step samples.
                segments_file.write(json.dumps(segment_row_for_json(row), default=float) + "\n")
                segments_file.flush()
                if digest_file is not None:
                    side = tdigest_sidecar_row(row)
                    if side is not None:
                        digest_file.write(json.dumps(side, default=float) + "\n")
                        digest_file.flush()
            rows.append(row)

            if not any(png_dir.glob("*.png")):
                if self.config.verbose:
                    print(f"  [{job.route_key}] segment [{start},{end}] -> 0 frames, no video")
                continue
            seg_mp4 = self.out_dir / f"{job.route_key}_{start}_{end}.mp4"
            build_mp4(png_dir, seg_mp4, self.config.fps)
            video_mp4s.append(seg_mp4)
            if self.config.verbose:
                obj = metrics["object"]
                print(
                    f"  [{job.route_key}] segment [{start},{end}] -> {seg_mp4.name}  "
                    f"coll={obj['collision_steps']} near={obj['miss_steps']} "
                    f"min_clr={obj['clearance_min_m']:.3f}"
                )

        extras: dict[str, Any] = {}
        if timers is not None:
            extras["timers"] = timers
        return JobRunResult(rows=rows, video_mp4s=video_mp4s, extras=extras)

    def on_job_complete(
        self,
        job: ClosedLoopJob,
        partial: JobRunResult,
        index: int,
        total: int,
    ) -> None:
        assert isinstance(job, FullRouteRouteJob)
        if self.config.verbose:
            print(
                f"[{index + 1}/{total}] {job.route_key}: {len(partial.video_mp4s)} segment video(s)"
            )

    def build_summary(self, result: JobRunResult, *, elapsed_sec: float) -> dict:
        summary = aggregate(
            result.rows,
            self.config.params.near_miss_thresh,
            strong_brake_mps2=self.config.params.strong_brake_mps2,
        )
        summary["npz_root"] = str(self.npz_root)
        # Derived straight from the merged rows (each carries its own "route") rather than
        # extras["route_keys"] -- extras aren't reconstructed across a DDP merge (see
        # collect_ddp_shards), so this stays correct for both the single-process and merged path.
        summary["n_routes"] = len({r["route"] for r in result.rows})
        summary["elapsed_sec"] = elapsed_sec
        summary["video_mp4s"] = result.video_mp4s
        summary["segments"] = result.rows
        return summary

    def write_artifacts(self, summary: dict, result: JobRunResult) -> None:
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in summary.items() if k not in ("video_mp4s", "segments")},
                f,
                indent=4,
            )

    def print_summary(self, summary: dict) -> None:
        n_seg = summary["n_segments"]
        print(
            f"\n=== closed-loop validation: {n_seg} segments in {summary['elapsed_sec']:.1f}s ==="
        )
        for line in format_summary_lines(summary):
            print(line)
        print(f"videos: per-segment <route>_<start>_<end>.mp4 in {self.out_dir}")

    def prepare_ddp_merge_artifacts(self, result: JobRunResult) -> None:
        """Rewrite ``segments.jsonl`` as the single merged, human-readable view across every
        rank's ``segments_{rank}.jsonl`` (digests stripped, same convention ``execute_jobs`` uses
        for the non-sharded path)."""
        with open(self.out_dir / "segments.jsonl", "w", encoding="utf-8") as fout:
            for row in result.rows:
                fout.write(json.dumps(segment_row_for_json(row), default=float) + "\n")
