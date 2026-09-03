"""Pack-throughput bench (spec §5). Measurement only — it asserts no thresholds."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline import partition as P
from diffusion_planner.data_pipeline.defaults import SHARD_SIZE_BYTES


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def _peak_rss_kb() -> int:
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(me, kids)


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
    for w in worker_counts:
        dest = Path(dest_root) / f"w{w}"
        t0 = time.monotonic()
        version = PK.pack(
            PK.PackOptions(
                source=Path(source),
                dest=dest,
                base="none",
                tag="bench",
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
        rows.append(
            {
                "workers": w,
                "partitions": len(version.partitions),
                "samples": samples,
                "wall_s": round(wall, 3),
                "samples_per_s": round(rate, 2),
                "efficiency": round(rate / (baseline * w), 3) if baseline > 0 else 0.0,
                "bytes_written": _tree_bytes(dest),
                "peak_rss_kb": _peak_rss_kb(),
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    head = (
        f"{'workers':>7} {'samples':>9} {'wall_s':>9} {'samples/s':>10} "
        f"{'eff':>6} {'written_MB':>11} {'peak_rss_MB':>12}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workers']:>7} {r['samples']:>9} {r['wall_s']:>9.2f} "
            f"{r['samples_per_s']:>10.2f} {r['efficiency']:>6.2f} "
            f"{r['bytes_written'] / 2**20:>11.1f} {r['peak_rss_kb'] / 1024:>12.1f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pack_bench")
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--dest-root", required=True, type=Path)
    ap.add_argument("--workers", default="1,8", help="comma-separated worker counts")
    ap.add_argument("--path-list", type=Path)
    ap.add_argument("--partition-depth", type=int)
    ap.add_argument("--partition-regex")
    ap.add_argument("--shard-size-gb", type=float, default=SHARD_SIZE_BYTES / 2**30)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args(argv)
    rows = bench(
        source=a.source,
        dest_root=a.dest_root,
        worker_counts=[int(x) for x in a.workers.split(",")],
        rule=P.PartitionRule(depth=a.partition_depth, regex=a.partition_regex),
        path_list=json.loads(a.path_list.read_text()) if a.path_list else None,
        shard_size_bytes=max(int(a.shard_size_gb * 2**30), 1),
    )
    print(render(rows))
    if a.json_out:
        a.json_out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
