"""Model loading and management utilities."""

from pathlib import Path

import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config


def load_model(
    model_path: Path, device: torch.device
) -> tuple[Diffusion_Planner, Config]:
    """Load Diffusion Planner model and its configuration.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model onto

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

    model_args = Config(str(args_path), guidance_fn=None)

    # Create model
    model = Diffusion_Planner(model_args)

    # Load weights (handle different checkpoint formats)
    if "model" in checkpoint:
        # Distributed training checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state_dict, strict=False)
    elif "ema_state_dict" in checkpoint:
        # EMA checkpoint
        print("Loading EMA weights")
        model.load_state_dict(checkpoint["ema_state_dict"], strict=False)
    else:
        # Direct state dict
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    print(f"Model loaded successfully on {device}")

    return model, model_args
