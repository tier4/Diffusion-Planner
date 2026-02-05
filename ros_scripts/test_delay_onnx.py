"""
Test script to verify that the delay mechanism works correctly in the ONNX model.

This script tests the same behavior as test_delay.py but using ONNX runtime instead of PyTorch.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from diffusion_planner.dimensions import *
from diffusion_planner.utils.config import Config

torch.backends.mha.set_fastpath_enabled(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Directory containing model.onnx and args.json")
    parser.add_argument("--delay", type=int, default=10, help="Delay steps to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    return args


def create_test_inputs(seed: int):
    """Create test inputs with random noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    inputs = {}
    # Create sampled trajectories with distinctive values
    sampled_traj = torch.randn(1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32)
    sampled_traj = sampled_traj * 10.0
    inputs["sampled_trajectories"] = sampled_traj

    inputs["ego_agent_past"] = torch.randn(1, INPUT_T + 1, EGO_HISTORY_DIM, dtype=torch.float32)
    inputs["ego_current_state"] = torch.randn(1, 10, dtype=torch.float32)
    inputs["neighbor_agents_past"] = torch.randn(
        1, MAX_NUM_NEIGHBORS, INPUT_T + 1, 12, dtype=torch.float32
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
    inputs["delay"] = torch.zeros(1, 1, dtype=torch.int64)

    return inputs


def test_delay_mechanism_onnx(model_dir: Path, delay_steps: int, seed: int):
    """Test that delay mechanism works correctly in ONNX model."""
    config_file = model_dir / "args.json"
    onnx_file = model_dir / "model.onnx"

    if not config_file.exists():
        print(f"Error: {config_file} not found")
        return False

    if not onnx_file.exists():
        print(f"Error: {onnx_file} not found")
        return False

    print(f"Loading config from {config_file}")
    config_obj = Config(str(config_file))

    print(f"Loading ONNX model from {onnx_file}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_file), sess_options, providers=["CPUExecutionProvider"])

    # Get input names from ONNX model
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"ONNX input names: {input_names}")

    print(f"\n{'='*80}")
    print(f"Testing delay mechanism with multiple delay values (ONNX)")
    print(f"{'='*80}\n")

    # Create test inputs ONCE (unnormalized, like C++ does)
    inputs = create_test_inputs(seed)

    print("Input sampled_trajectories[0, 0, 0:10, 0] (raw, before normalization):")
    print(f"  {inputs['sampled_trajectories'][0, 0, 0:10, 0].numpy()}")

    # Prepare ONNX inputs
    onnx_inputs = {
        "sampled_trajectories": inputs["sampled_trajectories"].numpy(),
        "ego_agent_past": inputs["ego_agent_past"].numpy(),
        "ego_current_state": inputs["ego_current_state"].numpy(),
        "neighbor_agents_past": inputs["neighbor_agents_past"].numpy(),
        "static_objects": inputs["static_objects"].numpy(),
        "lanes": inputs["lanes"].numpy(),
        "lanes_speed_limit": inputs["lanes_speed_limit"].numpy(),
        "lanes_has_speed_limit": inputs["lanes_has_speed_limit"].numpy(),
        "route_lanes": inputs["route_lanes"].numpy(),
        "route_lanes_speed_limit": inputs["route_lanes_speed_limit"].numpy(),
        "route_lanes_has_speed_limit": inputs["route_lanes_has_speed_limit"].numpy(),
        "polygons": inputs["polygons"].numpy(),
        "line_strings": inputs["line_strings"].numpy(),
        "goal_pose": inputs["goal_pose"].numpy(),
        "ego_shape": inputs["ego_shape"].numpy(),
        "turn_indicators": inputs["turn_indicators"].numpy(),
        "delay": np.array([[0]], dtype=np.float32),
    }

    # Get normalized version and apply transformations for comparison
    normalized_inputs = config_obj.observation_normalizer(inputs)
    norm_st = normalized_inputs["sampled_trajectories"].reshape(1, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM)

    # Apply state_normalizer.inverse to get what should be in the output if copied
    expected_if_copied = config_obj.state_normalizer.inverse(norm_st)

    # Test with multiple delay values
    delay_values = [0, 1, 10, 40]
    results = []

    for delay_val in delay_values:
        print(f"\n{'='*80}")
        print(f"Testing with delay={delay_val}:")
        print(f"{'='*80}")

        onnx_inputs["delay"] = np.array([[delay_val]], dtype=np.float32)

        outputs = session.run(None, onnx_inputs)
        prediction = outputs[0]  # [B, P, T, 4]

        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Prediction[0, 0, 0:10, 0] (x values): {prediction[0, 0, 0:10, 0]}")

        # Count how many positions are actually copied
        copied_count = 0
        for t in range(OUTPUT_T):
            input_val = expected_if_copied[0, 0, t+1, 0].item()  # +1 because output is [:,:,1:]
            pred_val = prediction[0, 0, t, 0]
            diff = abs(pred_val - input_val)

            if diff < 0.1:
                copied_count += 1

        predicted_count = OUTPUT_T - copied_count

        print(f"  Steps COPIED from input: {copied_count}/{OUTPUT_T}")
        print(f"  Steps PREDICTED (not copied): {predicted_count}/{OUTPUT_T}")
        print(f"  Expected copied: {delay_val}")

        if copied_count == OUTPUT_T:
            status = f"❌ BUG: ALL steps copied!"
        elif copied_count == delay_val or copied_count == delay_val + 1:
            status = f"✓ Correct"
        else:
            status = f"⚠ Unexpected"

        print(f"  Status: {status}")

        results.append({
            "delay": delay_val,
            "copied": copied_count,
            "predicted": predicted_count,
            "expected": delay_val,
        })

    # Summary
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}\n")

    print(f"{'Delay':<10} {'Copied':<10} {'Predicted':<10} {'Expected':<10} {'Status':<15}")
    print("-" * 60)

    success = True
    for result in results:
        delay_val = result["delay"]
        copied = result["copied"]
        predicted = result["predicted"]
        expected = result["expected"]

        if copied == OUTPUT_T:
            status = "❌ ALL COPIED"
            success = False
        elif copied == expected or copied == expected + 1:
            status = "✓ Correct"
        else:
            status = "⚠ Unexpected"
            success = False

        print(f"{delay_val:<10} {copied:<10} {predicted:<10} {expected:<10} {status:<15}")

    # Check for success
    print(f"\n{'='*80}")
    print("Verification:")
    print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    if success:
        print("✓ TEST PASSED: Delay mechanism works correctly in ONNX!")
    else:
        print("✗ TEST FAILED: Delay mechanism is not working as expected in ONNX")
    print(f"{'='*80}\n")

    return success


if __name__ == "__main__":
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Error: {model_dir} is not a valid directory")
        exit(1)

    success = test_delay_mechanism_onnx(model_dir, args.delay, args.seed)
    exit(0 if success else 1)
