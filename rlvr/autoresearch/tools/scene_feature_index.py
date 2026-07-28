"""Headless, shardable per-scene feature extractor over an NPZ ``path_list.json``.

Emits a **sharded parquet** feature table (one parquet file per shard) that
``build_small_dataset.py`` consumes for stratified sampling. Designed to scale to
tens of millions of NPZs:

* the path list is streamed in fixed-size shards (never fully materialized as rows),
* each shard is processed by a ``ProcessPoolExecutor`` and written to its own
  ``shard_NNNNN.parquet``,
* per-shard resume: a shard whose parquet already exists is skipped,
* optional ``--prefilter_frac`` seeded random pre-subsample of the path list.

The per-scene feature/label logic lives in the standalone
``rlvr.autoresearch.scene_features`` library (usable on its own to score/label a
single scene); this module is just the shard/parquet/CLI orchestration around it.

Usage:
    python -m rlvr.autoresearch.tools.scene_feature_index \
        --path_list <path_list.json> --out_dir <parquet_dir> \
        --shard_size 5000 --workers 8 [--prefilter_frac 0.1] [--seed 0]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rlvr.autoresearch.scene_features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    extract_scene_features,
)

# Populated per worker process by ``_init_worker`` (so heavy library imports are
# paid once per worker, not per NPZ).
_WORKER_CONFIG = FeatureConfig()


def _worker(npz_path: str):
    try:
        return extract_scene_features(npz_path, _WORKER_CONFIG)
    except Exception as e:  # per-NPZ error: return sentinel, don't crash the shard
        return {"__error__": f"{npz_path}: {type(e).__name__}: {e}"}


def _init_worker(config: FeatureConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _process_shard_cpu(shard_paths, config, workers):
    """CPU fallback: ProcessPool of per-scene extractors. Returns (rows, err_msgs)."""
    rows, err_msgs = [], []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(config,)
    ) as ex:
        for r in ex.map(_worker, shard_paths, chunksize=8):
            if r is None:
                err_msgs.append("unknown")
            elif "__error__" in r:
                err_msgs.append(r["__error__"])
            else:
                rows.append(r)
    return rows, err_msgs


def _gpu_load(npz_path):
    """Thread worker: load NPZ + extract the active-neighbor positions / ego path
    needed to batch the geometry (reuses scene_features.active_neighbor_info).

    Materializes every array into a plain dict inside the ``with`` block and closes
    the NpzFile before returning: ``np.load`` holds an open zip handle, and the
    shard-level list retains one entry per scene, so returning the live NpzFile
    leaked a file descriptor per scene and hit the open-files limit at shard scale."""
    from rlvr.autoresearch.scene_features import active_neighbor_info

    with np.load(npz_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    pos, _ped, _veh, _do = active_neighbor_info(d["neighbor_agents_past"])
    path = np.asarray(d["ego_agent_future"], dtype=np.float32)[:, :2]
    return npz_path, d, pos, path


def _process_shard_gpu(shard_paths, config, device, gpu_batch_size, load_workers):
    """GPU path: load scenes (threads), batch the point→ego-path geometry on the
    GPU in one kernel per mini-batch (scene_features.batch_min_dist_to_paths), then
    finalize each row with the precomputed distances. Numerically identical to the
    CPU path (shared finalize) — validated in tests. Returns (rows, err_msgs)."""
    import torch

    from rlvr.autoresearch.scene_features import batch_min_dist_to_paths, extract_scene_features

    rows, err_msgs = [], []
    from concurrent.futures import ThreadPoolExecutor

    def _safe_load(p):
        try:
            return _gpu_load(p)
        except Exception as e:
            return ("__error__", f"{p}: {type(e).__name__}: {e}")

    # Load + process ONE gpu_batch_size chunk at a time (thread-load just that chunk,
    # batch the geometry, extract, release) so peak memory is bounded by
    # gpu_batch_size scenes — NOT the whole shard. Materializing a full default
    # 5000-scene shard of 320-slot NPZs is multiple GB and can OOM before batch 1.
    with ThreadPoolExecutor(max_workers=load_workers) as tex:
        for start in range(0, len(shard_paths), gpu_batch_size):
            loaded = list(tex.map(_safe_load, shard_paths[start : start + gpu_batch_size]))
            chunk = [it for it in loaded if it[0] != "__error__"]
            err_msgs += [it[1] for it in loaded if it[0] == "__error__"]
            if not chunk:
                continue
            b = len(chunk)
            qmax = max((it[2].shape[0] for it in chunk), default=0)
            pmax = max(it[3].shape[0] for it in chunk)
            pts = np.zeros((b, max(qmax, 1), 2), np.float32)
            mask = np.zeros((b, max(qmax, 1)), bool)
            paths = np.zeros((b, pmax, 2), np.float32)
            for i, (_p, _d, pos, path) in enumerate(chunk):
                q = pos.shape[0]
                if q:
                    pts[i, :q] = pos
                    mask[i, :q] = True
                # edge-repeat pad the path to pmax (repeated last point => zero-length
                # segments AT a real path vertex, which never lowers the true min dist)
                paths[i, : path.shape[0]] = path
                if path.shape[0] < pmax:
                    paths[i, path.shape[0] :] = path[-1]
            t_pts = torch.from_numpy(pts).to(device)
            t_p1 = torch.from_numpy(paths[:, :-1]).to(device)
            t_p2 = torch.from_numpy(paths[:, 1:]).to(device)
            t_mask = torch.from_numpy(mask).to(device)
            dpath = batch_min_dist_to_paths(t_pts, t_p1, t_p2, t_mask).cpu().numpy()  # (b, qmax)
            for i, (p, d, pos, _path) in enumerate(chunk):
                q = pos.shape[0]
                try:
                    rows.append(
                        extract_scene_features(p, config, loaded=d, precomputed_dpath=dpath[i, :q])
                    )
                except Exception as e:
                    err_msgs.append(f"{p}: {type(e).__name__}: {e}")
    return rows, err_msgs


def _resolve_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if arg == "cuda":
        if not cuda:
            raise RuntimeError("--device cuda requested but no CUDA device is available")
        return "cuda"
    return "cuda" if cuda else "cpu"  # auto


def _write_parquet(rows: list[dict], path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {c: [r.get(c) for r in rows] for c in FEATURE_COLUMNS}
    pq.write_table(pa.table(cols), path)


def main() -> None:
    _def = FeatureConfig()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--path_list", required=True, help="path_list.json (list of NPZ paths)")
    ap.add_argument("--out_dir", required=True, help="output dir for shard_NNNNN.parquet")
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument(
        "--prefilter_frac",
        type=float,
        default=None,
        help="random pre-subsample fraction of the path list (0<frac<=1)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap total NPZs (testing); 0=all")
    ap.add_argument(
        "--overwrite", action="store_true", help="rewrite shards even if parquet exists"
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="auto = GPU if available else CPU. GPU batches the point->ego-path "
        "geometry (fast for bulk data); CPU uses a ProcessPool of per-scene workers.",
    )
    ap.add_argument(
        "--gpu_batch_size", type=int, default=1024, help="scenes per GPU geometry batch"
    )
    # label-taxonomy thresholds (defaults from FeatureConfig)
    ap.add_argument("--speed_bins", type=float, nargs=3, default=list(_def.speed_bins))
    ap.add_argument("--straight_yaw_deg", type=float, default=_def.straight_yaw_deg)
    ap.add_argument("--dense_count", type=int, default=_def.dense_count)
    ap.add_argument("--neighbor_radius_m", type=float, default=_def.neighbor_radius_m)
    ap.add_argument("--lead_range_m", type=float, default=_def.lead_range_m)
    ap.add_argument("--lead_half_width_m", type=float, default=_def.lead_half_width_m)
    ap.add_argument("--interaction_corridor_m", type=float, default=_def.interaction_corridor_m)
    ap.add_argument("--ped_corridor_m", type=float, default=_def.ped_corridor_m)
    args = ap.parse_args()

    if args.prefilter_frac is not None and not (0.0 < args.prefilter_frac <= 1.0):
        raise ValueError(f"--prefilter_frac must be in (0,1], got {args.prefilter_frac}")

    with open(args.path_list) as f:
        paths = json.load(f)
    if not isinstance(paths, list):
        raise ValueError(f"{args.path_list} must be a JSON list of NPZ paths")
    n_total = len(paths)

    if args.prefilter_frac is not None:
        rng = random.Random(args.seed)
        k = max(1, int(round(args.prefilter_frac * n_total)))
        paths = rng.sample(paths, k)
    if args.limit:
        paths = paths[: args.limit]
    print(f"[index] {n_total} paths in list -> {len(paths)} after prefilter/limit", flush=True)

    config = FeatureConfig(
        speed_bins=tuple(args.speed_bins),
        straight_yaw_deg=args.straight_yaw_deg,
        dense_count=args.dense_count,
        neighbor_radius_m=args.neighbor_radius_m,
        lead_range_m=args.lead_range_m,
        lead_half_width_m=args.lead_half_width_m,
        interaction_corridor_m=args.interaction_corridor_m,
        ped_corridor_m=args.ped_corridor_m,
    )

    device = _resolve_device(args.device)
    print(f"[index] device={device} (requested {args.device})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.overwrite:
        # Wipe existing shards so a re-run with a different --shard_size can't leave
        # stale higher-numbered shards behind (which would silently mix old + new
        # rows). Without --overwrite, existing shards are kept (resume).
        stale = sorted(glob.glob(os.path.join(args.out_dir, "shard_*.parquet")))
        for f in stale:
            os.remove(f)
        if stale:
            print(f"[index] --overwrite: cleared {len(stale)} existing shard(s)", flush=True)

    # Job fingerprint: shard skip-on-resume is by filename only, so re-running into the
    # same dir with a different path list/order, prefilter seed/fraction, limit, shard
    # size, or label thresholds would silently mix old + new rows. Bind resume to the
    # exact job. (path list/order + prefilter/seed/limit are all baked into `paths`.)
    fingerprint = {
        # Bumped when the feature/label extractor semantics change, so a partial index
        # built by an older extractor can't be silently resumed and have new-semantic
        # shards appended into the same dataset.
        "schema_version": FEATURE_SCHEMA_VERSION,
        "paths_sha1": hashlib.sha1("\n".join(paths).encode()).hexdigest(),
        "n_paths": len(paths),
        "shard_size": args.shard_size,
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(config).items()
        },
    }
    manifest_path = os.path.join(args.out_dir, "_index_manifest.json")
    existing_shards = sorted(glob.glob(os.path.join(args.out_dir, "shard_*.parquet")))
    if not args.overwrite:
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                prev = json.load(f)
            if prev != fingerprint:
                changed = [k for k in fingerprint if prev.get(k) != fingerprint[k]]
                raise SystemExit(
                    f"[index] resume aborted: job fingerprint changed {changed} vs "
                    f"{manifest_path}; pass --overwrite to rebuild {args.out_dir}"
                )
        elif existing_shards:
            # Shards but no manifest = an index written before the fingerprint existed
            # (or a partial/corrupt dir). We cannot prove they match THIS job, so we
            # must not silently bless + skip them. Fail loud; --overwrite rebuilds.
            raise SystemExit(
                f"[index] found {len(existing_shards)} existing shard(s) but no "
                f"{os.path.basename(manifest_path)} to prove they match this job; "
                f"pass --overwrite to rebuild {args.out_dir}"
            )
    with open(manifest_path, "w") as f:
        json.dump(fingerprint, f, indent=2)

    n_shards = math.ceil(len(paths) / args.shard_size)
    err_log = os.path.join(args.out_dir, "errors.log")
    total_ok = 0
    total_err = 0
    for si in range(n_shards):
        shard_paths = paths[si * args.shard_size : (si + 1) * args.shard_size]
        out_pq = os.path.join(args.out_dir, f"shard_{si:05d}.parquet")
        if os.path.exists(out_pq) and not args.overwrite:
            print(f"[index] shard {si + 1}/{n_shards}: exists, skip (resume)", flush=True)
            continue
        t0 = time.time()
        if device == "cuda":
            rows, err_msgs = _process_shard_gpu(
                shard_paths, config, device, args.gpu_batch_size, args.workers
            )
        else:
            rows, err_msgs = _process_shard_cpu(shard_paths, config, args.workers)
        if err_msgs:
            with open(err_log, "a") as ef:
                ef.write("\n".join(err_msgs) + "\n")
        tmp = out_pq + ".tmp"
        _write_parquet(rows, tmp)
        os.replace(tmp, out_pq)
        total_ok += len(rows)
        total_err += len(err_msgs)
        dt = time.time() - t0
        rate = len(shard_paths) / dt if dt > 0 else 0.0
        print(
            f"[index] shard {si + 1}/{n_shards}: {len(rows)} rows, {len(err_msgs)} err, "
            f"{dt:.1f}s ({rate:.1f} npz/s) -> {os.path.basename(out_pq)}",
            flush=True,
        )

    print(
        f"[index] DONE: {total_ok} rows, {total_err} errors across {n_shards} shards -> {args.out_dir}",
        flush=True,
    )
    if total_err:
        print(f"[index] see {err_log}", flush=True)


if __name__ == "__main__":
    main()
