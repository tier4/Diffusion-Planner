import argparse
import random
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusion_planner.dimensions import *
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.mha.set_fastpath_enabled(False)


FULL_INPUT_NAMES = [
    "sampled_trajectories",
    "ego_agent_past",
    "ego_current_state",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "polygons",
    "line_strings",
    "goal_pose",
    "ego_shape",
    "turn_indicators",
    "delay",
]

ENCODER_INPUT_NAMES = [
    "ego_agent_past",
    "neighbor_agents_past",
    "static_objects",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "polygons",
    "line_strings",
    "goal_pose",
    "ego_shape",
    "turn_indicators",
]

DECODER_INPUT_NAMES = [
    "encoding",
    "sampled_trajectories",
    "diffusion_time",
    "neighbor_agents_past",
]

TURN_INDICATOR_INPUT_NAMES = ["encoding", "final_x0"]

FULL_OUTPUT_NAMES = ["prediction", "turn_indicator_logit"]
ENCODER_OUTPUT_NAMES = ["encoding"]
DECODER_OUTPUT_NAMES = ["model_output"]
TURN_INDICATOR_OUTPUT_NAMES = ["turn_indicator_logit"]

TensorDict = dict[str, torch.Tensor]
NumpyDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class ModelWrappers:
    full: nn.Module
    encoder: nn.Module
    decoder: nn.Module
    turn_indicator: nn.Module


@dataclass(frozen=True)
class ExportSpec:
    wrapper: nn.Module
    inputs: TensorDict
    input_names: list[str]
    output_names: list[str]
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--eval_npz", type=Path, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="diffusion_planner",
        help="Prefix for output ONNX files (default: diffusion_planner)",
    )
    parser.add_argument(
        "--output-name",
        dest="output_prefix",
        type=str,
        default=argparse.SUPPRESS,
        help="Alias for --output-prefix, kept for torch2onnx.py compatibility",
    )
    parser.add_argument(
        "--use-simplify",
        action="store_true",
        help="Run onnxsim to produce simplified ONNX models",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=20,
        help="ONNX opset version used for export (default: 20)",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Export ONNX with external data (weights stored as separate files)",
    )
    parser.add_argument(
        "--no_ego_from_control",
        action="store_true",
        help="Disable control->trajectory conversion for ego (trajectory_and_control models)",
    )
    return parser.parse_args()


class EncoderONNXWrapper(nn.Module):
    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.encoder = model.encoder

    def forward(
        self,
        ego_agent_past: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        static_objects: torch.Tensor,
        lanes: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lanes_has_speed_limit: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_lanes_has_speed_limit: torch.Tensor,
        polygons: torch.Tensor,
        line_strings: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
        turn_indicators: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "ego_agent_past": ego_agent_past,
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
        return self.encoder(inputs)


class DecoderONNXWrapper(nn.Module):
    """One denoising-network evaluation.

    This wrapper intentionally does not call a sampler. An external denoising loop should
    update x_t and timesteps, then call this ONNX model once per model evaluation.
    """

    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        encoding: torch.Tensor,
        sampled_trajectories: torch.Tensor,
        diffusion_time: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
    ) -> torch.Tensor:
        neighbors_current = neighbor_agents_past[:, : self.decoder._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current, 0), dim=-1) == 0
        batch_size = encoding.shape[0]
        agent_num = 1 + self.decoder._predicted_neighbor_num

        # ONNX input is always 4D (POSE_DIM). Pad to D if trajectory_and_control.
        D = self.decoder._D
        sampled_trajectories = sampled_trajectories.reshape(
            batch_size, agent_num, 1 + self.decoder._future_len, POSE_DIM
        )
        if D > POSE_DIM:
            pad = torch.zeros(
                *sampled_trajectories.shape[:-1],
                D - POSE_DIM,
                device=sampled_trajectories.device,
                dtype=sampled_trajectories.dtype,
            )
            sampled_trajectories = torch.cat([sampled_trajectories, pad], dim=-1)

        model_output = self.decoder.dit(
            sampled_trajectories,
            diffusion_time,
            encoding,
            neighbor_current_mask,
        ).reshape(batch_size, agent_num, 1 + self.decoder._future_len, D)

        return model_output


class TurnIndicatorONNXWrapper(nn.Module):
    """Turn-indicator head evaluated once after the external denoising loop."""

    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, encoding: torch.Tensor, final_x0: torch.Tensor) -> torch.Tensor:
        batch_size = encoding.shape[0]
        agent_num = 1 + self.decoder._predicted_neighbor_num
        D = self.decoder._D
        final_x0 = final_x0.reshape(batch_size, agent_num, 1 + self.decoder._future_len, D)

        encoding_pooled = torch.mean(encoding, dim=1)
        ego_trajectory = final_x0[:, 0, 1::10, :2].reshape(
            batch_size, 2 * (self.decoder._future_len // 10)
        )
        return self.decoder._compute_turn_indicator(ego_trajectory, encoding_pooled)


class FullONNXWrapper(nn.Module):
    """Original all-in-one planner export."""

    def __init__(self, model: Diffusion_Planner):
        super().__init__()
        self.model = model

    def forward(
        self,
        sampled_trajectories: torch.Tensor,
        ego_agent_past: torch.Tensor,
        ego_current_state: torch.Tensor,
        neighbor_agents_past: torch.Tensor,
        static_objects: torch.Tensor,
        lanes: torch.Tensor,
        lanes_speed_limit: torch.Tensor,
        lanes_has_speed_limit: torch.Tensor,
        route_lanes: torch.Tensor,
        route_lanes_speed_limit: torch.Tensor,
        route_lanes_has_speed_limit: torch.Tensor,
        polygons: torch.Tensor,
        line_strings: torch.Tensor,
        goal_pose: torch.Tensor,
        ego_shape: torch.Tensor,
        turn_indicators: torch.Tensor,
        delay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ONNX input is always 4D (POSE_DIM). Pad to D if trajectory_and_control.
        D = self.model.decoder._D
        if D > POSE_DIM:
            pad = torch.zeros(
                *sampled_trajectories.shape[:-1],
                D - POSE_DIM,
                device=sampled_trajectories.device,
                dtype=sampled_trajectories.dtype,
            )
            sampled_trajectories = torch.cat([sampled_trajectories, pad], dim=-1)
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
            "delay": delay,
        }
        _, decoder_outputs = self.model(inputs)
        return decoder_outputs["prediction"], decoder_outputs["turn_indicator_logit"]


def build_inputs_from_npz(npz_path: Path) -> TensorDict:
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
    inputs["delay"] = torch.zeros(1, 1, dtype=torch.float32)
    return inputs


def build_dummy_inputs() -> TensorDict:
    inputs = {}
    inputs["sampled_trajectories"] = torch.ones(
        1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32
    )
    inputs["ego_agent_past"] = torch.randn(1, INPUT_T + 1, POSE_DIM, dtype=torch.float32)
    inputs["ego_current_state"] = torch.randn(1, 10, dtype=torch.float32)
    inputs["neighbor_agents_past"] = torch.randn(
        1, MAX_NUM_NEIGHBORS, INPUT_T + 1, 11, dtype=torch.float32
    )
    inputs["static_objects"] = torch.randn(1, NUM_STATIC_OBJECTS, 10, dtype=torch.float32)
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
    inputs["polygons"] = torch.randn(
        1, NUM_POLYGONS, POINTS_PER_POLYGON, 2 + POLYGON_TYPE_NUM, dtype=torch.float32
    )
    inputs["line_strings"] = torch.randn(
        1, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM, dtype=torch.float32
    )
    inputs["goal_pose"] = torch.randn(1, POSE_DIM, dtype=torch.float32)
    inputs["ego_shape"] = torch.tensor([[2.75, 4.34, 1.70]], dtype=torch.float32)
    inputs["turn_indicators"] = torch.randint(0, 3, (1, INPUT_T + 1), dtype=torch.float32)
    inputs["delay"] = torch.zeros(1, 1, dtype=torch.float32)
    return inputs


def load_validation_inputs(eval_npz_path: Path | None) -> TensorDict:
    if eval_npz_path:
        return build_inputs_from_npz(eval_npz_path)
    return build_dummy_inputs()


def build_decoder_inputs(inputs: TensorDict, encoding: torch.Tensor) -> TensorDict:
    return {
        "encoding": encoding,
        "sampled_trajectories": inputs["sampled_trajectories"],
        "diffusion_time": torch.ones(1, MAX_NUM_AGENTS, OUTPUT_T + 1, 1, dtype=torch.float32),
        "neighbor_agents_past": inputs["neighbor_agents_past"],
    }


def build_turn_indicator_inputs(encoding: torch.Tensor, final_x0: torch.Tensor) -> TensorDict:
    return {
        "encoding": encoding,
        "final_x0": final_x0,
    }


def load_model(config_json_path: str, ckpt_path: str, use_ema: bool) -> Diffusion_Planner:
    config_obj = Config(config_json_path)
    model = Diffusion_Planner(config_obj)
    model.eval()
    model.encoder.eval()
    model.decoder.eval()
    model.decoder.training = False

    ckpt = torch.load(ckpt_path)
    if use_ema:
        if "ema_state_dict" not in ckpt:
            raise ValueError(f"EMA state dict not found in checkpoint: {ckpt_path}")
        state_dict = ckpt["ema_state_dict"]
        print("Loading EMA model weights")
    else:
        state_dict = ckpt["model"]
        print("Loading regular model weights")
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    return model


def build_wrappers(model: Diffusion_Planner) -> ModelWrappers:
    return ModelWrappers(
        full=FullONNXWrapper(model).eval(),
        encoder=EncoderONNXWrapper(model).eval(),
        decoder=DecoderONNXWrapper(model).eval(),
        turn_indicator=TurnIndicatorONNXWrapper(model).eval(),
    )


def build_export_specs(
    wrappers: ModelWrappers,
    inputs: TensorDict,
    decoder_inputs: TensorDict,
    turn_indicator_inputs: TensorDict,
    full_onnx_path: Path,
    encoder_onnx_path: Path,
    decoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
) -> list[ExportSpec]:
    return [
        ExportSpec(
            wrapper=wrappers.full,
            inputs=inputs,
            input_names=FULL_INPUT_NAMES,
            output_names=FULL_OUTPUT_NAMES,
            output_path=full_onnx_path,
        ),
        ExportSpec(
            wrapper=wrappers.encoder,
            inputs=inputs,
            input_names=ENCODER_INPUT_NAMES,
            output_names=ENCODER_OUTPUT_NAMES,
            output_path=encoder_onnx_path,
        ),
        ExportSpec(
            wrapper=wrappers.decoder,
            inputs=decoder_inputs,
            input_names=DECODER_INPUT_NAMES,
            output_names=DECODER_OUTPUT_NAMES,
            output_path=decoder_onnx_path,
        ),
        ExportSpec(
            wrapper=wrappers.turn_indicator,
            inputs=turn_indicator_inputs,
            input_names=TURN_INDICATOR_INPUT_NAMES,
            output_names=TURN_INDICATOR_OUTPUT_NAMES,
            output_path=turn_indicator_onnx_path,
        ),
    ]


def build_dynamic_axes(
    input_names: list[str], output_names: list[str]
) -> dict[str, dict[int, str]]:
    return {name: {0: "batch"} for name in [*input_names, *output_names]}


def export_onnx(
    wrapper: nn.Module,
    inputs: TensorDict,
    input_names: list[str],
    output_names: list[str],
    output_path: Path,
    use_simplify: bool,
    opset_version: int,
    external_data: bool = False,
) -> None:
    torch_input_tuple = tuple(inputs[name] for name in input_names)
    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "dynamo": False,
        "dynamic_axes": build_dynamic_axes(input_names, output_names),
        "external_data": external_data,
    }

    print(f"Creating ONNX model with legacy exporter: {output_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            torch_input_tuple,
            str(output_path),
            **export_kwargs,
        )

    if use_simplify:
        try:
            import onnx
            from onnxsim import simplify

            model_proto = onnx.load(str(output_path))
            model_simp, check = simplify(model_proto)
            if check:
                onnx.save(model_simp, str(output_path))
                print(f"Simplified ONNX saved: {output_path}")
            else:
                print("WARNING: onnxsim validation failed, keeping unsimplified model")
        except ImportError:
            print("WARNING: onnxsim not installed, skipping (pip install onnxsim)")
        except Exception as exc:
            print(f"WARNING: onnxsim failed ({exc}), keeping unsimplified model")


def export_spec(
    spec: ExportSpec,
    use_simplify: bool,
    opset_version: int,
    external_data: bool = False,
) -> None:
    export_onnx(
        spec.wrapper,
        spec.inputs,
        spec.input_names,
        spec.output_names,
        spec.output_path,
        use_simplify,
        opset_version,
        external_data,
    )


def run_ort_in_subprocess(model_path: Path, np_inputs: NumpyDict) -> list[np.ndarray]:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/inputs.npz"
        output_path = f"{tmpdir}/outputs.npz"
        np.savez(input_path, **np_inputs)

        script = f"""
import numpy as np
import onnxruntime as ort
data = np.load("{input_path}", allow_pickle=True)
inputs = {{k: data[k] for k in data.files}}
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
sess = ort.InferenceSession(
    "{model_path}", sess_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ORT providers:", sess.get_providers())
outputs = sess.run(None, inputs)
np.savez("{output_path}", **{{f"out_{{i}}": o for i, o in enumerate(outputs)}})
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ORT subprocess failed:\n{result.stderr[-1000:]}")
        print(result.stdout.strip())
        data = np.load(output_path, allow_pickle=True)
        return [data[f"out_{i}"] for i in range(len(data.files))]


def compare(name: str, torch_output: np.ndarray, onnx_output: np.ndarray) -> None:
    abs_diff = np.abs(torch_output - onnx_output)
    print(f"{name}: torch={torch_output.shape}, onnx={onnx_output.shape}")
    print(f"  Max diff: {abs_diff.max()}")
    print(f"  Mean diff: {abs_diff.mean()}")
    print(f"  Close? {np.allclose(torch_output, onnx_output, rtol=1e-03, atol=1e-05)}")


def validate_full_model(
    wrappers: ModelWrappers,
    inputs: TensorDict,
    full_onnx_path: Path,
) -> None:
    with torch.no_grad():
        torch_prediction, torch_turn_indicator = wrappers.full(
            *(inputs[name] for name in FULL_INPUT_NAMES)
        )

    full_onnx_inputs = {name: inputs[name].cpu().numpy() for name in FULL_INPUT_NAMES}
    onnx_prediction, onnx_turn_indicator = run_ort_in_subprocess(full_onnx_path, full_onnx_inputs)
    compare("prediction", torch_prediction.cpu().numpy(), onnx_prediction)
    compare(
        "full_turn_indicator_logit",
        torch_turn_indicator.cpu().numpy(),
        onnx_turn_indicator,
    )


def validate_split_models(
    wrappers: ModelWrappers,
    inputs: TensorDict,
    decoder_inputs: TensorDict,
    encoder_onnx_path: Path,
    decoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
) -> None:
    with torch.no_grad():
        torch_encoding = wrappers.encoder(*(inputs[name] for name in ENCODER_INPUT_NAMES))
        torch_model_output = wrappers.decoder(
            torch_encoding,
            decoder_inputs["sampled_trajectories"],
            decoder_inputs["diffusion_time"],
            decoder_inputs["neighbor_agents_past"],
        )
        torch_turn_indicator = wrappers.turn_indicator(torch_encoding, torch_model_output)

    encoder_onnx_inputs = {name: inputs[name].cpu().numpy() for name in ENCODER_INPUT_NAMES}
    onnx_encoding = run_ort_in_subprocess(encoder_onnx_path, encoder_onnx_inputs)[0]
    compare("encoding", torch_encoding.cpu().numpy(), onnx_encoding)

    decoder_onnx_inputs = {
        "encoding": onnx_encoding,
        "sampled_trajectories": decoder_inputs["sampled_trajectories"].cpu().numpy(),
        "diffusion_time": decoder_inputs["diffusion_time"].cpu().numpy(),
        "neighbor_agents_past": decoder_inputs["neighbor_agents_past"].cpu().numpy(),
    }
    onnx_model_output = run_ort_in_subprocess(decoder_onnx_path, decoder_onnx_inputs)[0]
    compare("model_output", torch_model_output.cpu().numpy(), onnx_model_output)

    turn_indicator_onnx_inputs = {
        "encoding": onnx_encoding,
        "final_x0": onnx_model_output,
    }
    onnx_turn_indicator = run_ort_in_subprocess(
        turn_indicator_onnx_path, turn_indicator_onnx_inputs
    )[0]
    compare("turn_indicator_logit", torch_turn_indicator.cpu().numpy(), onnx_turn_indicator)


def convert_model(
    config_json_path: str,
    ckpt_path: str,
    full_onnx_path: Path,
    encoder_onnx_path: Path,
    decoder_onnx_path: Path,
    turn_indicator_onnx_path: Path,
    eval_npz_path: Path | None,
    use_ema: bool = False,
    use_simplify: bool = False,
    opset_version: int = 20,
    external_data: bool = False,
    ego_from_control: bool = True,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"Converting: {ckpt_path}")
    print(f"Config: {config_json_path}")
    print(f"Full output: {full_onnx_path}")
    print(f"Encoder output: {encoder_onnx_path}")
    print(f"Decoder output: {decoder_onnx_path}")
    print(f"Turn indicator output: {turn_indicator_onnx_path}")
    print(f"Using EMA: {use_ema}")
    print("ONNX exporter: legacy")
    print(f"ONNX opset version: {opset_version}")
    print(f"{'=' * 80}\n")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model(config_json_path, ckpt_path, use_ema)

    if ego_from_control:
        if model.decoder._output_mode != OUTPUT_MODE_TRAJECTORY_AND_CONTROL:
            raise ValueError(
                f"--ego-from-control requires output_mode='trajectory_and_control', "
                f"but got '{model.decoder._output_mode}'"
            )
        model.decoder._ego_prediction_from_control = True
        print("Ego prediction will use control->trajectory conversion")

    wrappers = build_wrappers(model)

    export_inputs = build_dummy_inputs()

    with torch.no_grad():
        encoding = wrappers.encoder(*(export_inputs[name] for name in ENCODER_INPUT_NAMES))

    decoder_inputs = build_decoder_inputs(export_inputs, encoding)
    with torch.no_grad():
        final_x0 = wrappers.decoder(
            decoder_inputs["encoding"],
            decoder_inputs["sampled_trajectories"],
            decoder_inputs["diffusion_time"],
            decoder_inputs["neighbor_agents_past"],
        )
    turn_indicator_inputs = build_turn_indicator_inputs(encoding, final_x0)

    export_specs = build_export_specs(
        wrappers,
        export_inputs,
        decoder_inputs,
        turn_indicator_inputs,
        full_onnx_path,
        encoder_onnx_path,
        decoder_onnx_path,
        turn_indicator_onnx_path,
    )
    for spec in export_specs:
        export_spec(spec, use_simplify, opset_version, external_data)

    print("\nORT validation")
    validation_inputs = load_validation_inputs(eval_npz_path)
    with torch.no_grad():
        validation_encoding = wrappers.encoder(
            *(validation_inputs[name] for name in ENCODER_INPUT_NAMES)
        )
    validation_decoder_inputs = build_decoder_inputs(validation_inputs, validation_encoding)

    validate_full_model(wrappers, validation_inputs, full_onnx_path)
    validate_split_models(
        wrappers,
        validation_inputs,
        validation_decoder_inputs,
        encoder_onnx_path,
        decoder_onnx_path,
        turn_indicator_onnx_path,
    )

    print(
        "\nSuccessfully converted to ONNX:"
        f"\n  {full_onnx_path}"
        f"\n  {encoder_onnx_path}"
        f"\n  {decoder_onnx_path}"
        f"\n  {turn_indicator_onnx_path}\n"
    )


if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root_dir)

    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' does not exist")
        exit(1)
    if not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a directory")
        exit(1)

    pth_files = list(root_dir.rglob("*.pth"))
    print(f"Found {len(pth_files)} .pth file(s) in '{root_dir}'")

    skipped_count = 0
    for pth_file in pth_files:
        pth_dir = pth_file.parent
        config_file = pth_dir / "args.json"
        full_onnx_file = pth_dir / f"{args.output_prefix}.onnx"
        encoder_onnx_file = pth_dir / f"{args.output_prefix}_encoder.onnx"
        decoder_onnx_file = pth_dir / f"{args.output_prefix}_decoder.onnx"
        turn_indicator_onnx_file = pth_dir / f"{args.output_prefix}_turn_indicator.onnx"

        print(f"\n{'#' * 80}")
        print(f"Processing: {pth_file.relative_to(root_dir)}")

        if not config_file.exists():
            print(f"Skipping: args.json not found in {pth_dir}")
            skipped_count += 1
            continue

        convert_model(
            config_json_path=str(config_file),
            ckpt_path=str(pth_file),
            full_onnx_path=full_onnx_file,
            encoder_onnx_path=encoder_onnx_file,
            decoder_onnx_path=decoder_onnx_file,
            turn_indicator_onnx_path=turn_indicator_onnx_file,
            eval_npz_path=args.eval_npz,
            use_ema=args.use_ema,
            use_simplify=args.use_simplify,
            opset_version=args.opset_version,
            external_data=args.external_data,
            ego_from_control=not args.no_ego_from_control,
        )

    print(f"\n{'=' * 80}")
    print("Conversion Summary:")
    print(f"  Total found: {len(pth_files)}")
    print(f"  Skipped (no args.json): {skipped_count}")
    print(f"{'=' * 80}")
