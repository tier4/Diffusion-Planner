
def verify_lora_loaded(model, model_args, scene_path, device, label=""):
    """Verify LoRA adapter is active and changes model output.

    Uses a real scene (not dummy data) to avoid missing-key errors.
    Call after loading LoRA to ensure it's not silently inactive.

    Args:
        model: Model with LoRA adapter loaded.
        model_args: Config from load_model.
        scene_path: Path to a real NPZ scene file.
        device: Torch device.
        label: Label for log messages.

    Returns:
        True if LoRA has measurable effect, False otherwise.
    """
    import copy

    import numpy as np
    import torch

    from guidance_gui.generate_samples import generate_samples
    from preference_optimization.utils import load_npz_data

    data = load_npz_data(scene_path, device)
    if "delay" not in data:
        data["delay"] = torch.zeros(1, dtype=torch.long, device=device)

    model.eval()
    inner = model.module if hasattr(model, "module") else model
    has_adapter = hasattr(inner, "disable_adapter")

    if not has_adapter:
        print(f"  [{label}] WARNING: model has no LoRA adapter")
        return False

    with torch.no_grad():
        norm1 = copy.deepcopy(model_args.observation_normalizer)(
            {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
        traj_lora = generate_samples(model, model_args, norm1, 0.0, 1, None, device)[0]

        with inner.disable_adapter():
            norm2 = copy.deepcopy(model_args.observation_normalizer)(
                {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()})
            traj_base = generate_samples(model, model_args, norm2, 0.0, 1, None, device)[0]

    diff = np.abs(traj_lora - traj_base).max()
    fde = np.linalg.norm(traj_lora[-1, :2] - traj_base[-1, :2])
    print(f"  [{label}] LoRA effect: max_diff={diff:.4f}m FDE={fde:.4f}m")

    if diff < 1e-5:
        print(f"  [{label}] WARNING: LoRA has ZERO effect! Check loading method.")
        print(f"  [{label}] Use load_lora_checkpoint(), NOT PeftModel.from_pretrained()")
        return False
    return True
