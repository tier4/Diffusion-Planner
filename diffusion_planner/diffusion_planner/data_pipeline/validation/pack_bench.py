"""Pack-throughput bench (spec §5). Measurement only — it asserts no thresholds."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import duckdb

from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline import partition as P
from diffusion_planner.data_pipeline.defaults import SHARD_SIZE_BYTES
from diffusion_planner.data_pipeline.errors import PipelineError


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def _rss_kb() -> tuple[int, int]:
    """(self, children) `ru_maxrss` high-water marks, in kB.

    Both are process-lifetime watermarks that never reset, so across a whole `bench()`
    run they are monotone non-decreasing and cumulative -- a later, lower-memory config
    cannot un-inflate an earlier row's number, and `RUSAGE_CHILDREN` only reflects
    workers that have already been waited on (which, since `pack()` spawns and joins its
    own pool before returning, is every worker from every row seen so far). These are
    kept as two separate fields rather than combined by `max()` (which would mix a
    process-lifetime child watermark with the current-row parent RSS as if they were the
    same kind of quantity) and labelled cumulative in `render()`.

    The `/proc/<pid>/status` VmHWM approach that
    `validation.throughput_bench._peak_worker_rss_mb` uses for its DataLoader workers is
    not reusable here: that function samples *live* worker processes while the caller
    still holds an iterator handle to them. `pack()`'s worker pool is entirely internal
    -- spawned and joined inside that one call -- so by the time `bench()` regains
    control there is no live worker PID left to read `/proc` from.
    """
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )


def bench(
    source: Path,
    dest_root: Path,
    worker_counts: list[int],
    rule: P.PartitionRule,
    path_list: list[str] | None,
    shard_size_bytes: int = SHARD_SIZE_BYTES,
) -> list[dict]:
    rows: list[dict] = []
    baseline: float | None = None
    baseline_w = worker_counts[0]
    # One tag per bench() call (not per row: each worker count already packs into its own
    # `dest_root/w{N}` sibling, so within a call there is no collision either way) so that
    # re-running this bench against the same --dest-root -- the runbook calls for at least
    # three runs -- gets a fresh tag instead of tripping VersionExistsError against a
    # previous run's "bench" version.
    run_tag = f"bench-{time.time_ns()}"
    for w in worker_counts:
        dest = Path(dest_root) / f"w{w}"
        t0 = time.monotonic()
        version = PK.pack(
            PK.PackOptions(
                source=Path(source),
                dest=dest,
                base="none",
                tag=run_tag,
                rule=rule,
                path_list=path_list,
                shard_size_bytes=shard_size_bytes,
                workers=w,
                progress=False,
            )
        )
        wall = time.monotonic() - t0
        samples = sum(e.sample_count for e in version.partitions.values())
        rate = samples / wall if wall > 0 else 0.0
        if baseline is None:
            baseline = rate
        # Efficiency is relative to the FIRST worker count actually measured, not to
        # workers=1 specifically -- a sweep given as e.g. --workers 2,8 never measures a
        # workers=1 row at all, so scaling every row's ideal rate by raw `w` would silently
        # assume a baseline that was never run.
        ideal = baseline * w / baseline_w
        rss_self_kb, rss_children_kb = _rss_kb()
        rows.append(
            {
                "workers": w,
                "partitions": len(version.partitions),
                "samples": samples,
                "wall_s": round(wall, 3),
                "samples_per_s": round(rate, 2),
                "efficiency": round(rate / ideal, 3) if ideal > 0 else 0.0,
                "bytes_written": _tree_bytes(dest),
                "rss_self_kb": rss_self_kb,
                "rss_children_kb": rss_children_kb,
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    notes = [
        "# rss_self/rss_children: cumulative ru_maxrss watermarks over this whole bench "
        "run, not a per-row measurement -- see _rss_kb().",
        "# cache: row order is measurement order; the source tree is cold for the first "
        "row and progressively OS-page-cache-warm for later rows, which inflates their "
        "samples/s and efficiency relative to a cold run at that worker count.",
    ]
    head = (
        f"{'workers':>7} {'samples':>9} {'wall_s':>9} {'samples/s':>10} "
        f"{'eff':>6} {'written_MB':>11} {'rss_self_MB':>12} {'rss_children_MB':>16}"
    )
    lines = [*notes, head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workers']:>7} {r['samples']:>9} {r['wall_s']:>9.2f} "
            f"{r['samples_per_s']:>10.2f} {r['efficiency']:>6.2f} "
            f"{r['bytes_written'] / 2**20:>11.1f} "
            f"{r['rss_self_kb'] / 1024:>12.1f} {r['rss_children_kb'] / 1024:>16.1f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pack_bench")
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--dest-root", required=True, type=Path)
    ap.add_argument("--workers", default="1,8", help="comma-separated worker counts")
    ap.add_argument("--path-list", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--partition-depth", type=int, help="partition = first N path components (no default)"
    )
    g.add_argument(
        "--partition-regex",
        help="partition = named group 'partition' (or group 1) matched on the key",
    )
    ap.add_argument("--shard-size-gb", type=float, default=SHARD_SIZE_BYTES / 2**30)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args(argv)
    try:
        rows = bench(
            source=a.source,
            dest_root=a.dest_root,
            worker_counts=[int(x) for x in a.workers.split(",")],
            rule=P.PartitionRule(depth=a.partition_depth, regex=a.partition_regex),
            path_list=json.loads(a.path_list.read_text()) if a.path_list else None,
            shard_size_bytes=max(int(a.shard_size_gb * 2**30), 1),
        )
    except (
        PipelineError,
        ValueError,
        FileNotFoundError,
        KeyError,
        duckdb.Error,
        TimeoutError,
    ) as e:
        # Same contract as pack_shards: `bench()` calls straight into `PK.pack()`, which can
        # raise any of these (a bad --workers, a bad path-list entry, a broken pool, ...), and
        # until now nothing here caught them -- they escaped as raw tracebacks instead of the
        # "error: ..." / exit-1 shape every other CLI in this package gives an operator.
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(render(rows))
    if a.json_out:
        a.json_out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
