"""Merge LoRA adapter weights into a base model checkpoint.

Produces a single .pth file with the LoRA deltas baked into the base weights,
ready for ONNX export or deployment without any PEFT dependency.

Usage:
    python3 -m preference_optimization.merge_lora \
        --model_path <base_model.pth> \
        --lora_dir <lora_epoch_NNN or lora_latest> \
        --output <merged_model.pth>

If --lora_dir is omitted, the script searches the directory containing
model_path for lora_latest/ or the highest-numbered lora_epoch_NNN/.

If --output is omitted, writes to <model_dir>/merged.pth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from preference_optimization.lora_utils import (
    load_lora_checkpoint,
    merge_lora_and_unload,
)
from preference_optimization.model_utils import load_model


def find_lora_dir(search_dir: Path) -> Path | None:
    """Return the most recent LoRA adapter directory, or None."""
    lora_latest = search_dir / "lora_latest"
    if lora_latest.exists() and (lora_latest / "adapter_config.json").exists():
        return lora_latest.resolve()
    for d in reversed(sorted(search_dir.glob("lora_epoch_*"))):
        if (d / "adapter_config.json").exists():
            return d
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--model_path", type=Path, required=True, help="Base model .pth")
    parser.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        help="LoRA adapter directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for merged .pth (default: <model_dir>/merged.pth)",
    )
    args = parser.parse_args()

    model_path: Path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_dir = model_path.parent
    args_json = model_dir / "args.json"
    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found in {model_dir}")

    # Find LoRA adapter
    lora_dir = args.lora_dir
    if lora_dir is None:
        lora_dir = find_lora_dir(model_dir)
    if lora_dir is None:
        raise FileNotFoundError(f"No LoRA adapter found in {model_dir}")
    if not (lora_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found in {lora_dir}")

    output_path: Path = args.output or (model_dir / "merged.pth")

    print(f"Base model:    {model_path}")
    print(f"LoRA adapter:  {lora_dir}")
    print(f"Output:        {output_path}")

    # Load base model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_model(model_path, device)

    # Detect adapter format: new adapters target q/k/v/out_proj Linear sub-layers;
    # old adapters target the MHA module directly (before UnfusedMHA migration).
    with open(lora_dir / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    target = adapter_cfg.get("target_modules", "")
    is_new_format = isinstance(target, str) and "q_proj" in target

    if is_new_format:
        model = load_lora_checkpoint(model, str(lora_dir), is_trainable=False)
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_dir), is_trainable=False)

    # Merge LoRA deltas into base weights and remove PEFT scaffolding
    model = merge_lora_and_unload(model)
    model.eval()

    # After merge_and_unload, DiT blocks still contain UnfusedMHA (with separate
    # q/k/v_proj Linear layers). Re-fuse them into the in_proj_weight/in_proj_bias
    # format expected by nn.MultiheadAttention so the checkpoint is loadable by
    # the unmodified Diffusion_Planner model (and torch2onnx.py).
    state_dict = model.state_dict()
    fused_state_dict = {}
    handled = set()
    for key in state_dict:
        if ".q_proj.weight" in key:
            prefix = key.rsplit(".q_proj.weight", 1)[0]
            fused_state_dict[f"{prefix}.in_proj_weight"] = torch.cat(
                [state_dict[f"{prefix}.q_proj.weight"],
                 state_dict[f"{prefix}.k_proj.weight"],
                 state_dict[f"{prefix}.v_proj.weight"]], dim=0,
            )
            handled.update([f"{prefix}.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")])
            if f"{prefix}.q_proj.bias" in state_dict:
                fused_state_dict[f"{prefix}.in_proj_bias"] = torch.cat(
                    [state_dict[f"{prefix}.q_proj.bias"],
                     state_dict[f"{prefix}.k_proj.bias"],
                     state_dict[f"{prefix}.v_proj.bias"]], dim=0,
                )
                handled.update([f"{prefix}.{p}.bias" for p in ("q_proj", "k_proj", "v_proj")])
    for key, val in state_dict.items():
        if key not in handled:
            fused_state_dict[key] = val

    # Save as a standard checkpoint compatible with both DDP and non-DDP loading.
    # Include both module.-prefixed keys (for DDP / torchrun) and bare keys (for
    # single-GPU valid_predictor.py --ddp false).  The resume_model helper tries
    # ckpt["model"] first; with DDP the model expects module.* keys, without DDP
    # it expects bare keys.  We save the module.* version (matching trainer output)
    # so torchrun works, and also stash bare keys under "model_no_ddp".
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": {f"module.{k}": v for k, v in fused_state_dict.items()},
        "model_no_ddp": dict(fused_state_dict),
    }
    torch.save(checkpoint, output_path)

    # Copy args.json alongside the merged checkpoint if saving elsewhere
    output_args = output_path.parent / "args.json"
    if not output_args.exists():
        shutil.copy2(args_json, output_args)

    print(f"\nMerged model saved: {output_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nYou can now run: python3 ros_scripts/torch2onnx.py {output_path.parent}")


if __name__ == "__main__":
    main()
