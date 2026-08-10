"""Parallel rsync fetch — split path list into N chunks, run N rsync processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def split_paths_for_fetch(all_paths: list[str], dest: str, num_chunks: int) -> list[list[str]]:
    """Exclude already-present files, then round-robin into num_chunks lists."""
    dest_path = Path(dest)
    to_fetch = [p for p in all_paths if not (dest_path / p.lstrip("/")).exists()]
    chunks: list[list[str]] = [[] for _ in range(num_chunks)]
    for i, p in enumerate(to_fetch):
        chunks[i % num_chunks].append(p)
    return chunks


def run_parallel_rsync(
    path_list_json: str,
    dest: str,
    host: str = "sakurab",
    num_connections: int = 4,
    bwlimit: int = 10000,
) -> None:
    """Fetch NPZs via num_connections parallel rsync processes."""
    with open(path_list_json) as f:
        all_paths = json.load(f)

    chunks = split_paths_for_fetch(all_paths, dest, num_connections)
    total = sum(len(c) for c in chunks)
    if total == 0:
        print(f"All {len(all_paths)} files already present.")
        return

    print(f"Fetching {total} files in {num_connections} parallel streams...")
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        listfile = dest_path / f".rsync_chunk_{i}.txt"
        listfile.write_text("\n".join(p.lstrip("/") for p in chunk) + "\n")
        cmd = [
            "rsync",
            "-a",
            "--info=progress2",
            f"--bwlimit={bwlimit}",
            f"--files-from={listfile}",
            f"{host}:/",
            str(dest_path),
        ]
        print(f"  Stream {i}: {len(chunk)} files")
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        procs.append(proc)

    failures = 0
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            print(f"  Stream {i} exited with code {rc}")
            failures += 1

    if failures:
        print(f"WARNING: {failures}/{len(procs)} streams had errors")
    else:
        print(f"All {total} files fetched successfully.")


def main():
    p = argparse.ArgumentParser(description="Parallel rsync fetch from sakurab.")
    p.add_argument("--path_list", required=True, help="JSON path list")
    p.add_argument("--dest", default="data/per_scene_eval/mirror")
    p.add_argument("--host", default="sakurab")
    p.add_argument("--num_connections", type=int, default=4)
    p.add_argument("--bwlimit", type=int, default=10000)
    args = p.parse_args()
    run_parallel_rsync(args.path_list, args.dest, args.host, args.num_connections, args.bwlimit)


if __name__ == "__main__":
    main()
