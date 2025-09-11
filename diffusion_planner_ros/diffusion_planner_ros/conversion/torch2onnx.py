import argparse
import json
import random

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

torch.backends.mha.set_fastpath_enabled(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert torch model to onnx")
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="If set, only test the output of the existing ONNX model",
    )

    parser.add_argument("--config", type=str, default="args.json", help="Config file path")
    parser.add_argument("--ckpt", type=str, default="latest.pth", help="Checkpoint file path")
    parser.add_argument("--onnx_path", type=str, default="model.onnx", help="ONNX model file path")
    parser.add_argument(
        "--wrap_with_onnx_functions",
        action="store_true",
        help="Wether to replace some original functions with onnx-friendly ones",
    )
    args = parser.parse_args()
    return args


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))
    """
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


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
        goal_pose,
        ego_shape,
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
            "goal_pose": goal_pose,
            "ego_shape": ego_shape,
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


if __name__ == "__main__":
    args = parse_args()
    config_json_path = args.config
    ckpt_path = args.ckpt
    onnx_path = args.onnx_path
    wrap_with_onnx = args.wrap_with_onnx_functions
    test_only = args.test_only

    # Load config
    with open(config_json_path, "r") as f:
        config_json = json.load(f)
    config_obj = Config(config_json_path)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    inputs = {}
    inputs["sampled_trajectories"] = torch.ones(1, 33, 81, 4, dtype=torch.float32)
    inputs["ego_agent_past"] = torch.randn(1, 21, 4, dtype=torch.float32)
    inputs["ego_current_state"] = torch.randn(1, 10, dtype=torch.float32)
    inputs["neighbor_agents_past"] = torch.randn(1, 32, 21, 11, dtype=torch.float32)
    inputs["static_objects"] = torch.randn(1, 5, 10, dtype=torch.float32)
    inputs["lanes"] = torch.randn(1, 70, 20, 33, dtype=torch.float32)
    inputs["lanes_speed_limit"] = torch.randn(1, 70, 1, dtype=torch.float32)
    inputs["lanes_has_speed_limit"] = torch.ones(1, 70, 1, dtype=torch.bool)
    inputs["route_lanes"] = torch.randn(1, 25, 20, 33, dtype=torch.float32)
    inputs["route_lanes_speed_limit"] = torch.randn(1, 25, 1, dtype=torch.float32)
    inputs["route_lanes_has_speed_limit"] = torch.ones(1, 25, 1, dtype=torch.bool)
    inputs["goal_pose"] = torch.randn(1, 4, dtype=torch.float32)
    inputs["ego_shape"] = torch.tensor([[2.75, 4.34, 1.70]], dtype=torch.float32)

    for key in inputs.keys():
        print(f"{key}: {inputs[key].shape}, {inputs[key].dtype}")

    input_names = list(inputs.keys())

    # Export
    # Init model
    model = Diffusion_Planner(config_obj)
    model.eval()
    model.encoder.encoder.eval()
    model.decoder.decoder.eval()
    model.decoder.decoder.training = False

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

    if not test_only:
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
