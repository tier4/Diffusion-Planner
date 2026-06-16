"""Build curated GRAFT-CL targets = a competent model's deterministic trajectory.

GRAFT-CL (HEAL Mechanism B) trains the wounded model toward a KNOWN-GOOD explicit
target instead of ranking its own samples. The target is the deterministic
prediction of a competent model (typically the baseline / a model that keeps the
line well exactly where the grafted model drifted) on each scene. This tool runs
that model's det inference on a scene list and writes the result into each NPZ's
``ego_agent_future`` so the scenes can be fed through ``ranked_sft_mode=curated``.

Det output convention (verified): the model prediction is (T, 4) = [x, y, cos, sin]
(cos²+sin² ≈ 1 already); this tool unit-normalizes the (cos, sin) columns to be
exactly unit before writing, matching the canonical 4-col future format. All other
NPZ fields are copied verbatim; ``ego_shape`` is injected. Reuses
``eval_det_avoidance.{load_model, det_inference_batched}`` — no new inference path.

Usage:
    python -m rlvr.autoresearch.tools.build_baseline_det_target \
        --model <competent_model.pth> --scenes <scenes.json> \
        --ego_shape WB,L,W --out_dir <dir> --out_list <out.json> [--batch_size 16]

Notes:
- ``--model`` is the TARGET source (e.g. the baseline that keeps centerline at the
  wounded arc), NOT the model being trained. Train (curated low-LR ranked-SFT, e.g.
  lr 5e-5) warm-started from the wounded model toward these targets.
- Scenes must already carry ``ego_shape`` OR pass --ego_shape (asserted to match if present).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from preference_optimization.utils import load_npz_data  # canonical NPZ loader
from rlvr.autoresearch.tools.eval_det_avoidance import det_inference_batched, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", required=True, help="competent target-source model .pth (e.g. baseline)"
    )
    ap.add_argument("--scenes", required=True, help="JSON list of scene NPZ paths")
    ap.add_argument("--ego_shape", required=True, help="WB,L,W")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    parts = [float(v) for v in args.ego_shape.split(",")]
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise ValueError(
            f"--ego_shape must be 'WB,L,W' with 3 positive values; got {args.ego_shape!r}"
        )
    ego_shape = np.array(parts, dtype=np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.model, dev)
    scenes = json.load(open(args.scenes))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for start in range(0, len(scenes), args.batch_size):
        batch = scenes[start : start + args.batch_size]
        datas = [load_npz_data(p, dev) for p in batch]
        trajs = det_inference_batched(model, margs, datas, dev)  # (B, T, 4) [x,y,cos,sin]
        for i, sp in enumerate(batch):
            tr = trajs[i].cpu().numpy().astype(np.float32)  # (T, 4)
            # unit-normalize the (cos, sin) heading columns
            norm = np.sqrt(tr[:, 2] ** 2 + tr[:, 3] ** 2)
            norm = np.where(norm < 1e-6, 1.0, norm)
            tr[:, 2] /= norm
            tr[:, 3] /= norm
            d = dict(np.load(sp, allow_pickle=True))
            if "ego_shape" in d:
                es = np.asarray(d["ego_shape"]).reshape(-1)[:3]
                if not np.allclose(es, ego_shape, atol=1e-2):
                    raise ValueError(
                        f"{sp} ego_shape {es.tolist()} != --ego_shape {ego_shape.tolist()}"
                    )
            d["ego_shape"] = ego_shape
            d["ego_agent_future"] = tr  # curated SFT target = competent model's det line
            op = out_dir / Path(sp).name
            np.savez(op, **d)
            written.append(str(op))

    json.dump(written, open(args.out_list, "w"), indent=2)
    print(f"wrote {len(written)} curated baseline-det-target scenes -> {args.out_dir}")
    print(f"  scene list -> {args.out_list}")
    print(
        f"  target = det of {args.model}; train curated (lr 5e-5) warm-started from the wounded model."
    )


if __name__ == "__main__":
    main()
