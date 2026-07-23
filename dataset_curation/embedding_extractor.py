from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _pool_embedding(enc: torch.Tensor) -> torch.Tensor:
    if enc.dim() == 3 and enc.shape[0] == 1:
        return enc[0].mean(dim=0)
    elif enc.dim() == 3:
        return enc.mean(dim=1)
    return enc


def extract_embeddings(
    model_path: str,
    scene_paths: list[str],
    device: str = "cuda",
    batch_size: int = 1,
) -> np.ndarray:
    from diffusion_planner.train_config import TrainConfig
    from diffusion_planner.utils.train_utils import openjson

    from exploration_policy.utils import get_frozen_encoder, run_frozen_encoder
    from preference_optimization.utils import load_npz_data

    args_path = Path(model_path).parent / "args.json"
    if args_path.exists():
        raw = openjson(str(args_path))
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        config = TrainConfig(**filtered)
    else:
        config = TrainConfig()

    from diffusion_planner.model.diffusion_planner import Diffusion_Planner

    model = Diffusion_Planner(config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    encoder = get_frozen_encoder(model)
    embeddings = []
    dev = torch.device(device)

    for i, sp in enumerate(scene_paths):
        try:
            data = load_npz_data(sp, dev)
            enc = run_frozen_encoder(model, data)
            pooled = _pool_embedding(enc).cpu().numpy()
            embeddings.append(pooled)
        except Exception as e:
            print(f"[WARN] Failed on {sp}: {e}")
            embeddings.append(None)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(scene_paths)}")

    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = np.stack([embeddings[i] for i in valid_indices])
    if len(valid_indices) < len(scene_paths):
        print(f"[WARN] {len(scene_paths) - len(valid_indices)} scenes failed embedding extraction")

    return valid_embeddings, [scene_paths[i] for i in valid_indices]


def main():
    parser = argparse.ArgumentParser(description="Extract frozen encoder embeddings")
    parser.add_argument("--model_path", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--scenes", required=True, help="JSON list of NPZ paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenes) as f:
        scene_paths = json.load(f)

    print(f"Extracting embeddings from {len(scene_paths)} scenes...")
    embeddings, valid_paths = extract_embeddings(
        args.model_path,
        scene_paths,
        device=args.device,
    )
    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "embedding_paths.json", "w") as f:
        json.dump(valid_paths, f)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
