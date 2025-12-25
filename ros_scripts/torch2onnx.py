import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config

torch.backends.mha.set_fastpath_enabled(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_npz", type=Path, default=None)
    parser.add_argument("root_dir", type=Path)
    args = parser.parse_args()
    return args


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        sampled_trajectories,
        ego_agent_past,
        ego_current_state,
        neighbor_agents_past,
        static_objects,
        lanes,
        lanes_speed_limit,
        lanes_has_speed_limit,
        route_lanes,
        route_lanes_speed_limit,
        route_lanes_has_speed_limit,
        polygons,
        line_strings,
        goal_pose,
        ego_shape,
        turn_indicators,
    ):
        inputs = {
            "sampled_trajectories": sampled_trajectories,
            "ego_agent_past": ego_agent_past,
            "ego_current_state": ego_current_state,
            "neighbor_agents_past": neighbor_agents_past,
            "static_objects": static_objects,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "polygons": polygons,
            "line_strings": line_strings,
            "goal_pose": goal_pose,
            "ego_shape": ego_shape,
            "turn_indicators": turn_indicators,
        }
        encoder_outputs, decoder_outputs = self.model(inputs)
        return decoder_outputs["prediction"], decoder_outputs["turn_indicator_logit"]


def compare_outputs(torch_output, onnx_output):
    torch_prediction, torch_turn_indicator = torch_output
    onnx_prediction, onnx_turn_indicator = onnx_output

    print(f"Prediction comparison:")
    print(f"torch prediction, with shape {torch_prediction.shape}:")
    print(f"onnx prediction, with shape {onnx_prediction.shape}:")
    abs_diff_pred = np.abs(torch_prediction - onnx_prediction)
    print(f"Max diff: {abs_diff_pred.max()}")
    print(f"Mean diff: {abs_diff_pred.mean()}")
    print(f"Close? {np.allclose(torch_prediction, onnx_prediction, rtol=1e-03, atol=1e-05)}")

    print(f"\nTurn indicator comparison:")
    print(f"torch turn_indicator, with shape {torch_turn_indicator.shape}:")
    print(f"onnx turn_indicator, with shape {onnx_turn_indicator.shape}:")
    abs_diff_turn = np.abs(torch_turn_indicator - onnx_turn_indicator)
    print(f"Max diff: {abs_diff_turn.max()}")
    print(f"Mean diff: {abs_diff_turn.mean()}")
    print(
        f"Close? {np.allclose(torch_turn_indicator, onnx_turn_indicator, rtol=1e-03, atol=1e-05)}"
    )


def build_inputs_from_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    inputs = {}
    inputs["sampled_trajectories"] = 0.5 * torch.randn(
        1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32
    )
    inputs["ego_agent_past"] = heading_to_cos_sin(
        torch.tensor(data["ego_agent_past"], dtype=torch.float32).unsqueeze(0)
    )
    inputs["ego_current_state"] = torch.tensor(
        data["ego_current_state"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["neighbor_agents_past"] = torch.tensor(
        data["neighbor_agents_past"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["static_objects"] = torch.tensor(data["static_objects"], dtype=torch.float32).unsqueeze(
        0
    )
    inputs["lanes"] = torch.tensor(data["lanes"], dtype=torch.float32).unsqueeze(0)
    inputs["lanes_speed_limit"] = torch.tensor(
        data["lanes_speed_limit"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["lanes_has_speed_limit"] = torch.tensor(
        data["lanes_has_speed_limit"], dtype=torch.bool
    ).unsqueeze(0)
    inputs["route_lanes"] = torch.tensor(data["route_lanes"], dtype=torch.float32).unsqueeze(0)
    inputs["route_lanes_speed_limit"] = torch.tensor(
        data["route_lanes_speed_limit"], dtype=torch.float32
    ).unsqueeze(0)
    inputs["route_lanes_has_speed_limit"] = torch.tensor(
        data["route_lanes_has_speed_limit"], dtype=torch.bool
    ).unsqueeze(0)
    inputs["polygons"] = torch.tensor(data["polygons"], dtype=torch.float32).unsqueeze(0)
    inputs["line_strings"] = torch.tensor(data["line_strings"], dtype=torch.float32).unsqueeze(0)
    goal_pose = torch.tensor(data["goal_pose"], dtype=torch.float32).unsqueeze(0)
    if goal_pose.shape[-1] == 3:
        goal_pose = heading_to_cos_sin(goal_pose)
    inputs["goal_pose"] = goal_pose
    inputs["ego_shape"] = torch.tensor([[2.75, 4.34, 1.70]], dtype=torch.float32)
    inputs["turn_indicators"] = torch.tensor(
        data["turn_indicators"], dtype=torch.float32
    ).unsqueeze(0)
    return inputs


def convert_model(
    config_json_path: str, ckpt_path: str, onnx_path: str, eval_npz_path: Path | None
):
    """Convert a single PyTorch model to ONNX format."""
    print(f"\n{'=' * 80}")
    print(f"Converting: {ckpt_path}")
    print(f"Config: {config_json_path}")
    print(f"Output: {onnx_path}")
    print(f"{'=' * 80}\n")

    # Load config
    config_obj = Config(config_json_path)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    inputs = {}
    inputs["sampled_trajectories"] = torch.ones(
        1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32
    )
    inputs["ego_agent_past"] = torch.randn(1, INPUT_T + 1, POSE_DIM, dtype=torch.float32)
    inputs["ego_current_state"] = torch.randn(1, 10, dtype=torch.float32)
    inputs["neighbor_agents_past"] = torch.randn(
        1, MAX_NUM_NEIGHBORS, INPUT_T + 1, 11, dtype=torch.float32
    )
    inputs["static_objects"] = torch.randn(1, 5, 10, dtype=torch.float32)
    inputs["lanes"] = torch.randn(
        1, NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM, dtype=torch.float32
    )
    inputs["lanes_speed_limit"] = torch.randn(1, NUM_SEGMENTS_IN_LANE, 1, dtype=torch.float32)
    inputs["lanes_has_speed_limit"] = torch.ones(1, NUM_SEGMENTS_IN_LANE, 1, dtype=torch.bool)
    inputs["route_lanes"] = torch.randn(
        1, NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM, dtype=torch.float32
    )
    inputs["route_lanes_speed_limit"] = torch.randn(
        1, NUM_SEGMENTS_IN_ROUTE, 1, dtype=torch.float32
    )
    inputs["route_lanes_has_speed_limit"] = torch.ones(
        1, NUM_SEGMENTS_IN_ROUTE, 1, dtype=torch.bool
    )
    inputs["polygons"] = torch.randn(1, NUM_POLYGONS, POINTS_PER_POLYGON, 2, dtype=torch.float32)
    inputs["line_strings"] = torch.randn(
        1, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2, dtype=torch.float32
    )
    inputs["goal_pose"] = torch.randn(1, POSE_DIM, dtype=torch.float32)
    inputs["ego_shape"] = torch.tensor([[2.75, 4.34, 1.70]], dtype=torch.float32)
    inputs["turn_indicators"] = torch.randint(0, 3, (1, INPUT_T + 1), dtype=torch.float32)

    for key in inputs.keys():
        print(f"{key}: {inputs[key].shape}, {inputs[key].dtype}")

    input_names = list(inputs.keys())

    # Export
    # Init model
    model = Diffusion_Planner(config_obj)
    model.eval()
    model.encoder.eval()
    model.decoder.eval()
    model.decoder.training = False

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Wrap model for onnx compatibility
    wrapper = ONNXWrapper(model).eval()

    # Prepare input
    torch_input_tuple = tuple(inputs.values())
    print(f"{len(torch_input_tuple)=}")
    print(f"{input_names=}")
    onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items() if k in input_names}

    print(f"creating a new onnx model: {onnx_path}")
    # Define dynamic axes for both inputs and outputs
    dynamic_axes = {}
    # Add dynamic batch dimension for inputs
    for name in input_names:
        dynamic_axes[name] = {0: "batch"}
    # Add dynamic batch dimension for outputs
    dynamic_axes["prediction"] = {0: "batch"}
    dynamic_axes["turn_indicator_logit"] = {0: "batch"}

    onnx_model = torch.onnx.export(
        wrapper,
        torch_input_tuple,
        onnx_path,
        input_names=input_names,
        output_names=["prediction", "turn_indicator_logit"],
        dynamic_axes=dynamic_axes,
        opset_version=20,
        dynamo=False,
    )

    sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    ort_session = ort.InferenceSession(
        onnx_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    with torch.no_grad():
        output = wrapper(*torch_input_tuple)
        torch_output = (output[0].cpu().numpy(), output[1].cpu().numpy())
    onnx_output = ort_session.run(None, onnx_inputs)
    print("Compare outputs using the creation input")
    compare_outputs(torch_output, onnx_output)

    # TEST WITH NORMALIZED INPUT
    normalized_inputs = config_obj.observation_normalizer(inputs)
    torch_input_tuple = tuple(normalized_inputs.values())
    onnx_inputs = {k: v.cpu().numpy() for k, v in normalized_inputs.items() if k in input_names}

    for i in range(3):
        print(f"\nTest {i + 1} with normalized random input")
        with torch.no_grad():
            output = wrapper(*torch_input_tuple)
            torch_output = (output[0].cpu().numpy(), output[1].cpu().numpy())
        onnx_output = ort_session.run(None, onnx_inputs)
        compare_outputs(torch_output, onnx_output)

    if eval_npz_path and eval_npz_path.exists():
        print(f"\nTest with eval NPZ input: {eval_npz_path}")
        eval_inputs = build_inputs_from_npz(eval_npz_path)
        torch_eval_tuple = tuple(eval_inputs[name] for name in input_names)
        onnx_eval_inputs = {k: v.cpu().numpy() for k, v in eval_inputs.items() if k in input_names}
        with torch.no_grad():
            output = wrapper(*torch_eval_tuple)
            torch_output = (output[0].cpu().numpy(), output[1].cpu().numpy())
        onnx_output = ort_session.run(None, onnx_eval_inputs)
        compare_outputs(torch_output, onnx_output)
    elif eval_npz_path:
        print(f"\n⚠ Eval NPZ not found, skipped: {eval_npz_path}")

    print(f"\n✓ Successfully converted to ONNX: {onnx_path}\n")


if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root_dir)

    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' does not exist")
        exit(1)

    if not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a directory")
        exit(1)

    # Find all .pth files recursively
    pth_files = list(root_dir.rglob("*.pth"))

    print(f"Found {len(pth_files)} .pth file(s) in '{root_dir}'")

    skipped_count = 0

    for pth_file in pth_files:
        pth_dir = pth_file.parent
        config_file = pth_dir / "args.json"
        onnx_file = pth_dir / "model.onnx"

        print(f"\n{'#' * 80}")
        print(f"Processing: {pth_file.relative_to(root_dir)}")

        # Check if args.json exists in the same directory
        if not config_file.exists():
            print(f"⚠ Skipping: args.json not found in {pth_dir}")
            skipped_count += 1
            continue

        # Convert the model
        convert_model(
            config_json_path=str(config_file),
            ckpt_path=str(pth_file),
            onnx_path=str(onnx_file),
            eval_npz_path=args.eval_npz,
        )

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Conversion Summary:")
    print(f"  Total found: {len(pth_files)}")
    print(f"  Skipped (no args.json): {skipped_count}")
    print(f"{'=' * 80}")
