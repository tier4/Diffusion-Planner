"""Token-group importance via input ablation (feature-importance analog).

Zeroing raw rows removes variable input information. Variable-size classes
(neighbors/maps) become padding and are masked by the encoder. Fixed-size
classes (goal pose/turn indicators) remain present as constant-valued tokens,
so their results are input-information ablations rather than token-removal
experiments. Groups: input class (drop:*), nearest-K sweeps for neighbors /
lanes / line_strings, and distance cutoffs.

Usage (CPU, sshfs checkpoint):
  uv run python scripts/token_importance.py \
    --run_dir /path/to/run_dir \
    --valid_set_list /path/to/path_list_valid.json \
    --n_samples 128 --batch_size 8 --out_tsv token_importance_vtail.tsv
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from token_analysis_common import init_distributed, load_model, prepare_inputs
from torch.utils.data import DataLoader, Subset

METRIC_KEYS = ("fde_top", "ade_top", "min_fde", "min_ade")


def select_moving_indices(dataset, n_samples, move_min_m):
    """Select moving scenes across the full list; run on rank 0 only."""
    stride = max(1, len(dataset) // (n_samples * 2))
    # Check one evenly spaced pass first, then the remaining offsets as needed.
    # This preserves broad dataset coverage without silently stopping after only
    # about 2 * n_samples candidates when moving scenes are sparse.
    cand = [index for offset in range(stride) for index in range(offset, len(dataset), stride)]
    probe = DataLoader(Subset(dataset, cand), batch_size=1, shuffle=False)
    idxs = []
    for index, sample in zip(cand, probe):
        if float(np.linalg.norm(sample["ego_agent_future"][0, -1, :2])) >= move_min_m:
            idxs.append(index)
        if len(idxs) >= n_samples:
            break
    return idxs, stride


def broadcast_indices(indices, world_size):
    if world_size == 1:
        return indices
    payload = [indices]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


ONNX_TYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(bool)": np.bool_,
    "tensor(int64)": np.int64,
    "tensor(double)": np.float64,
}


class OnnxModel:
    """Callable mimicking Diffusion_Planner(inputs) -> (_, outputs) on an ORT session.

    The ONNX graph contains the encoder's zeroing/masking logic, so raw-input
    ablation works identically to the PyTorch path. Normalization is NOT
    embedded in the ONNX; the caller must pass normalized inputs (same as the
    C++ node), which prepare_inputs already does.
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_specs = self.sess.get_inputs()
        self.device = device
        print(f"onnxruntime providers: {self.sess.get_providers()}", flush=True)

    def __call__(self, feed: dict):
        B = feed["ego_current_state"].shape[0]
        onnx_feed = {}
        for si in self.input_specs:
            dtype = ONNX_TYPE_MAP.get(si.type, np.float32)
            shp = [B if not isinstance(d, int) or d <= 0 else d for d in si.shape]
            if si.name in feed:
                arr = feed[si.name].cpu().numpy().astype(dtype)
                if list(arr.shape) != shp and arr.size == int(np.prod(shp)):
                    arr = arr.reshape(shp)  # e.g. delay [B] -> [B,1]
                onnx_feed[si.name] = arr
            else:
                onnx_feed[si.name] = np.zeros(shp, dtype=dtype)
        pred = self.sess.run(["prediction"], onnx_feed)[0]
        return None, {"prediction": torch.from_numpy(np.asarray(pred)).to(self.device)}


# ---------------------------------------------------------------- ablations


def _neighbor_dist(nbr: torch.Tensor) -> torch.Tensor:
    """Distance at the current step; inf for slots masked by NeighborEncoder.

    NeighborEncoder discards all but the last six history rows before building
    its padding mask, so the analysis must use the same temporal window.
    """
    valid = (nbr[:, :, -6:, :8] != 0).any(dim=(2, 3))  # [B, N]
    d = nbr[:, :, -1, :2].norm(dim=-1)  # [B, N]
    return torch.where(valid, d, torch.full_like(d, float("inf")))


def _polyline_dist(x: torch.Tensor, geom_dims: int | None) -> torch.Tensor:
    """x: [B, P, V, D] raw. Min distance over valid points; inf for padding."""
    pt_valid = (x != 0).any(dim=-1) if geom_dims is None else (x[..., :geom_dims] != 0).any(dim=-1)
    d = x[..., :2].norm(dim=-1)  # [B, P, V]
    d = torch.where(pt_valid, d, torch.full_like(d, float("inf")))
    return d.min(dim=-1).values  # [B, P]


def _keep_topk(x: torch.Tensor, dist: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all element slots except the k nearest valid ones."""
    x = x.clone()
    B = x.shape[0]
    for b in range(B):
        order = torch.argsort(dist[b])  # inf (padding) sorts last
        drop = order[k:]
        finite = torch.isfinite(dist[b][drop])
        x[b, drop[finite]] = 0.0
    return x


def _keep_within(x: torch.Tensor, dist: torch.Tensor, radius: float) -> torch.Tensor:
    x = x.clone()
    far = torch.isfinite(dist) & (dist > radius)
    x[far] = 0.0
    return x


def apply_ablation(inputs: dict, name: str) -> dict:
    out = {k: v.clone() for k, v in inputs.items()}
    if name == "baseline":
        return out
    kind, _, arg = name.partition(":")
    if kind == "drop":
        key = {
            "neighbors": "neighbor_agents_past",
            "lanes": "lanes",
            "route": "route_lanes",
            "line_strings": "line_strings",
            "polygons": "polygons",
            "goal_pose": "goal_pose",
            "turn_indicators": "turn_indicators",
        }[arg]
        out[key] = torch.zeros_like(out[key])
        if key in ("lanes", "route_lanes"):
            out[f"{key}_speed_limit"] = torch.zeros_like(out[f"{key}_speed_limit"])
            out[f"{key}_has_speed_limit"] = torch.zeros_like(out[f"{key}_has_speed_limit"])
        return out
    if kind == "nbr_top":
        d = _neighbor_dist(out["neighbor_agents_past"])
        out["neighbor_agents_past"] = _keep_topk(out["neighbor_agents_past"], d, int(arg))
        return out
    if kind == "nbr_within":
        d = _neighbor_dist(out["neighbor_agents_past"])
        out["neighbor_agents_past"] = _keep_within(out["neighbor_agents_past"], d, float(arg))
        return out
    if kind == "lane_top":
        d = _polyline_dist(out["lanes"], 8)
        out["lanes"] = _keep_topk(out["lanes"], d, int(arg))
        return out
    if kind == "lane_within":
        d = _polyline_dist(out["lanes"], 8)
        out["lanes"] = _keep_within(out["lanes"], d, float(arg))
        return out
    if kind == "ls_top":
        d = _polyline_dist(out["line_strings"], None)
        out["line_strings"] = _keep_topk(out["line_strings"], d, int(arg))
        return out
    raise ValueError(name)


CONFIGS = [
    "baseline",
    # class-level removal (permutation-importance analog)
    "drop:neighbors",
    "drop:lanes",
    "drop:route",
    "drop:line_strings",
    "drop:polygons",
    "drop:goal_pose",
    "drop:turn_indicators",
    # nearest-K sweeps (token-count optimization curves)
    "nbr_top:2",
    "nbr_top:4",
    "nbr_top:8",
    "nbr_top:16",
    "nbr_within:50",
    "nbr_within:100",
    "lane_top:10",
    "lane_top:20",
    "lane_top:40",
    "lane_top:70",
    "lane_top:100",
    "lane_within:50",
    "lane_within:100",
    "ls_top:10",
    "ls_top:20",
    "ls_top:40",
]


# ---------------------------------------------------------------- metrics


@torch.no_grad()
def eval_config(model, cfg, batches, device, name):
    totals = {key: 0.0 for key in METRIC_KEYS}
    count = 0
    for raw in batches:
        inputs = apply_ablation(raw, name)
        inputs_n, ego_future = prepare_inputs(inputs, cfg, device, include_future=True)
        _, outputs = model(inputs_n)
        gt = ego_future[..., :2]  # [B,T,2] metres
        if "trajectory" in outputs and "probability" in outputs:
            traj = outputs["trajectory"]  # [B,K,T,4] normalized
            std = cfg.state_normalizer.std[0].to(device)
            mean = cfg.state_normalizer.mean[0].to(device)
            xy = (traj * std + mean)[..., :2]  # [B,K,T,2]
            err = (xy - gt[:, None]).norm(dim=-1)  # [B,K,T]
            ade_k = err.mean(dim=-1)  # [B,K]
            fde_k = err[..., -1]  # [B,K]
            top = outputs["probability"].argmax(dim=-1)  # [B]
            bidx = torch.arange(gt.shape[0], device=fde_k.device)
            totals["fde_top"] += float(fde_k[bidx, top].sum())
            totals["ade_top"] += float(ade_k[bidx, top].sum())
            totals["min_fde"] += float(fde_k.min(dim=-1).values.sum())
            totals["min_ade"] += float(ade_k.min(dim=-1).values.sum())
        else:
            pred = outputs["prediction"][:, 0, :, :2]  # [B,T,2] metres
            err = (pred - gt).norm(dim=-1)
            totals["fde_top"] += float(err[:, -1].sum())
            totals["ade_top"] += float(err.mean(dim=-1).sum())
            totals["min_fde"] += float(err[:, -1].sum())
            totals["min_ade"] += float(err.mean(dim=-1).sum())
        count += gt.shape[0]
    return totals, count


def reduce_metric_totals(totals, count, device, world_size):
    values = [totals[key] for key in METRIC_KEYS] + [float(count)]
    reduced = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    global_count = int(reduced[-1].item())
    if global_count == 0:
        raise RuntimeError("no evaluation samples were assigned")
    return {
        key: float(reduced[index].item() / global_count) for index, key in enumerate(METRIC_KEYS)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", help="PyTorch run dir (args.json + epochXXXX/best_model.pth)")
    p.add_argument("--onnx", help="Path to diffusion_planner.onnx (forward-only backend)")
    p.add_argument("--args_json", help="args.json for --onnx (default: sibling of the onnx)")
    p.add_argument("--valid_set_list", required=True)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--move_min_m", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_tsv", default="token_importance.tsv")
    args = p.parse_args()

    args.device, rank, local_rank, world_size = init_distributed(args.device)
    is_main = rank == 0
    if args.onnx:
        if world_size > 1:
            raise ValueError("distributed execution is supported only for the PyTorch backend")
        args_json = args.args_json or str(Path(args.onnx).parent / "args.json")
        cfg = Config(args_json)
        cfg.device = args.device
        cfg.ddp = False
        model = OnnxModel(args.onnx, args.device)
        if is_main:
            print(f"loaded ONNX: {args.onnx} (normalizer from {args_json})", flush=True)
    else:
        model, cfg, ckpt = load_model(Path(args.run_dir), args.device)
        if is_main:
            print(f"loaded: {ckpt}")
            print(
                f"data parallel: world_size={world_size}, per-rank batch={args.batch_size}, "
                f"workers/rank={args.num_workers}",
                flush=True,
            )

    # Spread sample selection across the whole valid set (avoid one clip),
    # keeping only "moving" scenes so map/agent inputs actually matter.
    dataset = DiffusionPlannerData(args.valid_set_list)
    if is_main:
        idxs, stride = select_moving_indices(dataset, args.n_samples, args.move_min_m)
    else:
        idxs, stride = None, None
    idxs = broadcast_indices(idxs, world_size)
    if not idxs:
        if world_size > 1:
            dist.destroy_process_group()
        raise RuntimeError(
            f"no samples moved at least {args.move_min_m} m in the evaluation dataset"
        )
    local_idxs = idxs[rank::world_size]
    if is_main:
        print(
            f"selected {len(idxs)} moving samples (stride={stride}, dataset={len(dataset)})",
            flush=True,
        )

    loader = DataLoader(
        Subset(dataset, local_idxs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    batches = [{k: v for k, v in b.items()} for b in loader]
    print(
        f"[rank {rank}/{world_size}, local_rank={local_rank}] cached {len(local_idxs)} samples",
        flush=True,
    )

    rows = []
    base = None
    for name in CONFIGS:
        t0 = time.time()
        totals, count = eval_config(model, cfg, batches, args.device, name)
        m = reduce_metric_totals(totals, count, args.device, world_size)
        if name == "baseline":
            base = m
        d = {f"d_{k}": m[k] - base[k] for k in m}
        if is_main:
            rows.append({"config": name, **m, **d})
            print(
                f"{name:<22} fde_top={m['fde_top']:6.2f} ({d['d_fde_top']:+5.2f})  "
                f"ade_top={m['ade_top']:5.2f} ({d['d_ade_top']:+5.2f})  "
                f"min_fde={m['min_fde']:6.2f} ({d['d_min_fde']:+5.2f})  "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )

    if is_main:
        cols = list(rows[0].keys())
        with open(args.out_tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for row in rows:
                f.write("\t".join(str(row[column]) for column in cols) + "\n")
        print(f"wrote {args.out_tsv}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
