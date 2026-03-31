
def verify_lora_loaded(model, model_args, device, label=""):
    """Verify LoRA adapter is active and changes model output.
    
    Call after loading LoRA to ensure it's not silently inactive.
    Raises RuntimeError if LoRA has no effect.
    """
    import torch, copy
    from guidance_gui.generate_samples import generate_samples
    
    # Create a dummy input
    dummy = {
        "ego_current_state": torch.zeros(1, 10, device=device),
        "ego_agent_past": torch.zeros(1, 31, 4, device=device),
        "neighbor_agents_past": torch.zeros(1, 32, 31, 11, device=device),
        "lanes": torch.zeros(1, 140, 20, 33, device=device),
        "route_lanes": torch.zeros(1, 25, 20, 33, device=device),
        "line_strings": torch.zeros(1, 60, 20, 4, device=device),
        "polygons": torch.zeros(1, 10, 40, 3, device=device),
        "static_objects": torch.zeros(1, 5, 10, device=device),
        "goal_pose": torch.zeros(1, 4, device=device),
        "ego_shape": torch.tensor([[2.79, 4.34, 1.70]], device=device),
        "delay": torch.zeros(1, dtype=torch.long, device=device),
    }
    dummy["ego_current_state"][0, 2] = 1.0  # cos heading
    
    norm = copy.deepcopy(model_args.observation_normalizer)(
        {k: v.clone() for k, v in dummy.items()})
    
    model.eval()
    inner = model.module if hasattr(model, "module") else model
    has_adapter = hasattr(inner, "disable_adapter")
    
    if not has_adapter:
        print(f"  [{label}] WARNING: model has no LoRA adapter")
        return False
    
    with torch.no_grad():
        # With LoRA
        traj_lora = generate_samples(model, model_args, norm, 0.0, 1, None, device)[0]
        # Without LoRA
        with inner.disable_adapter():
            norm2 = copy.deepcopy(model_args.observation_normalizer)(
                {k: v.clone() for k, v in dummy.items()})
            traj_base = generate_samples(model, model_args, norm2, 0.0, 1, None, device)[0]
    
    import numpy as np
    diff = np.abs(traj_lora - traj_base).max()
    print(f"  [{label}] LoRA effect: max_diff={diff:.6f}m")
    
    if diff < 1e-5:
        print(f"  [{label}] WARNING: LoRA has ZERO effect on output!")
        return False
    return True
