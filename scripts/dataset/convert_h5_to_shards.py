"""Convert an H5 frame index into a versioned tar-shard dataset for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import argcomplete

from diffusion_planner.data.h5_to_shards import convert


def main() -> None:
    """Stage, pack, key-set, scrub and verify the frames of one Parquet index."""
    parser = argparse.ArgumentParser(
        description="Convert the frames of an H5 Parquet index into tar shards"
    )
    parser.add_argument(
        "index", type=Path, help="Parquet index, e.g. <h5 root>/indexes/train.parquet"
    )
    parser.add_argument("dest", type=Path, help="shard dataset root to write or extend")
    parser.add_argument(
        "--tag", required=True, help="version tag to publish, e.g. train-v1"
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="scratch directory for one chunk of per-frame npz files (about 90 KB per frame); "
        "use a scratch disk or tmpfs, never the root disk",
    )
    parser.add_argument(
        "--h5-root",
        type=Path,
        default=None,
        help="directory the bag hierarchy is relative to (default: parent of the index directory)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
        help="bags are staged and packed in this many chunks",
    )
    parser.add_argument(
        "--stage-workers", type=int, default=32, help="processes reading H5 files"
    )
    parser.add_argument(
        "--pack-workers", type=int, default=32, help="parallel partition builders"
    )
    parser.add_argument("--shard-size-gb", type=float, default=1.0)
    parser.add_argument(
        "--every", type=int, default=1, help="convert every Nth bag (slices for tests)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="first bag index when using --every"
    )
    parser.add_argument("--max-bags", type=int, default=None)
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=200,
        help="members checked bit-exact against H5; 0 disables",
    )
    parser.add_argument(
        "--keyset-out",
        type=Path,
        default=None,
        help="default: <dest>/keysets/<tag>.parquet",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.chunks < 1:
        parser.error("--chunks must be positive")
    if args.stage_workers < 1 or args.pack_workers < 1:
        parser.error("worker counts must be positive")
    if args.every < 1 or args.offset < 0:
        parser.error("--every must be positive and --offset non-negative")
    result = convert(
        args.index,
        args.dest,
        tag=args.tag,
        staging_dir=args.staging_dir,
        h5_root=args.h5_root,
        chunks=args.chunks,
        stage_workers=args.stage_workers,
        pack_workers=args.pack_workers,
        shard_size_gb=args.shard_size_gb,
        every=args.every,
        offset=args.offset,
        max_bags=args.max_bags,
        verify_samples=args.verify_samples,
        keyset_out=args.keyset_out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
