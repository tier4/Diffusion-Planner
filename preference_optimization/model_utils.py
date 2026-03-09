"""Model loading and management utilities."""

from pathlib import Path

import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.guidance.guidance_wrapper import GuidanceWrapper
from diffusion_planner.utils.config import Config


def _migrate_fused_attention(state_dict: dict) -> dict:
    """Convert DiT decoder nn.MultiheadAttention weights to UnfusedMultiheadAttention format.

    Splits in_proj_weight (shape [3D, D]) into q/k/v_proj.weight (shape [D, D] each)
    for keys under decoder.blocks.* (DiT blocks only). Applied in-memory during loading
    — no file is modified. No-op if the state dict already uses the unfused format.

    The encoder still uses nn.MultiheadAttention (fused), so encoder keys are left
    unchanged. Only paths containing "decoder" and ending with in_proj_weight/bias
    are converted.
    """
    new_sd = {}
    for key, val in state_dict.items():
        if "decoder" in key and key.endswith(".in_proj_weight"):
            prefix = key[: -len(".in_proj_weight")]
            D = val.shape[0] // 3
            new_sd[prefix + ".q_proj.weight"] = val[:D].clone()
            new_sd[prefix + ".k_proj.weight"] = val[D: 2 * D].clone()
            new_sd[prefix + ".v_proj.weight"] = val[2 * D:].clone()
        elif "decoder" in key and key.endswith(".in_proj_bias"):
            prefix = key[: -len(".in_proj_bias")]
            D = val.shape[0] // 3
            new_sd[prefix + ".q_proj.bias"] = val[:D].clone()
            new_sd[prefix + ".k_proj.bias"] = val[D: 2 * D].clone()
            new_sd[prefix + ".v_proj.bias"] = val[2 * D:].clone()
        else:
            new_sd[key] = val
    return new_sd


def load_model(
    model_path: Path,
    device: torch.device,
    use_collision: bool = False,
    use_route_following: bool = False,
    use_lane_keeping: bool = False,
    use_centerline_following: bool = False,
    guidance_scale: float = 0.5,
) -> tuple[Diffusion_Planner, Config]:
    """Load Diffusion Planner model and its configuration.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model onto
        use_collision: Enable collision-avoidance guidance during inference.
        use_route_following: Enable route-following guidance during inference.
        use_lane_keeping: Enable lane-keeping guidance during inference.
        use_centerline_following: Enable centerline-following guidance during inference.
        guidance_scale: Classifier guidance scale (higher = stronger guidance).

    Returns:
        Tuple of (model, model_args)

    Raises:
        FileNotFoundError: If model or args.json not found
    """
    print(f"Loading model from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load model configuration
    model_dir = model_path.parent
    args_path = model_dir / "args.json"

    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in model directory: {args_path}")

    use_any_guidance = use_collision or use_route_following or use_lane_keeping or use_centerline_following
    guidance_fn = (
        GuidanceWrapper(
            use_collision=use_collision,
            use_route_following=use_route_following,
            use_lane_keeping=use_lane_keeping,
            use_centerline_following=use_centerline_following,
        )
        if use_any_guidance
        else None
    )
    model_args = Config(str(args_path), guidance_fn=guidance_fn)
    model_args.guidance_scale = guidance_scale

    # Create model
    model = Diffusion_Planner(model_args)

    # Load weights (handle different checkpoint formats).
    # _migrate_fused_attention is a no-op if the checkpoint already uses the
    # unfused q/k/v format; it transparently handles old checkpoints.
    if "model" in checkpoint:
        # Distributed training checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
        state_dict = _migrate_fused_attention(state_dict)
        model.load_state_dict(state_dict, strict=False)
    elif "ema_state_dict" in checkpoint:
        # EMA checkpoint
        print("Loading EMA weights")
        state_dict = _migrate_fused_attention(checkpoint["ema_state_dict"])
        model.load_state_dict(state_dict, strict=False)
    else:
        # Direct state dict
        state_dict = _migrate_fused_attention(checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    print(f"Model loaded successfully on {device}")

    return model, model_args
