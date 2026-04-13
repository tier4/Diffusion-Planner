"""Unit tests for rlvr.grpo_sampler -- diverse trajectory generation.

Requires a loaded model. Skip if model not available.
Run: python3 rlvr/test_grpo_sampler.py --model_path <path.pth> --npz_path <path.npz>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from rlvr.grpo_sampler import SampledTrajectory, SamplerConfig, generate_diverse_group


def test_sampler_config_defaults():
    cfg = SamplerConfig()
    assert cfg.n_trajectories == 8
    assert cfg.noise_scale_range == (0.5, 2.0)
    assert cfg.enable_guidance is True
    assert cfg.enable_centerline is True
    assert cfg.enable_anchor is True
    assert cfg.enable_collision is False
    assert cfg.enable_route_following is False
    assert cfg.enable_lane_keeping is False
    assert cfg.guidance_prob == 0.5
    assert cfg.prototypes_path is None
    print("  PASS  sampler_config_defaults")


def test_first_trajectory_is_deterministic(model, model_args, data, device):
    config = SamplerConfig(n_trajectories=4)
    results = generate_diverse_group(model, model_args, data, config, device)
    assert results[0].is_deterministic is True
    assert results[0].noise_scale == 0.0
    assert results[0].label == "det"
    print("  PASS  first_trajectory_is_deterministic")


def test_correct_count(model, model_args, data, device):
    config = SamplerConfig(n_trajectories=4)
    results = generate_diverse_group(model, model_args, data, config, device)
    assert len(results) == 4, f"Expected 4, got {len(results)}"
    print("  PASS  correct_count")


def test_trajectory_shapes(model, model_args, data, device):
    config = SamplerConfig(n_trajectories=4)
    results = generate_diverse_group(model, model_args, data, config, device)
    for i, st in enumerate(results):
        assert st.trajectory.shape == (80, 4), f"Traj {i} shape: {st.trajectory.shape}"
    print("  PASS  trajectory_shapes")


def test_diverse_configs(model, model_args, data, device):
    config = SamplerConfig(n_trajectories=8, enable_centerline=True, guidance_prob=1.0)
    results = generate_diverse_group(model, model_args, data, config, device)
    noise_scales = {st.noise_scale for st in results[1:]}
    assert len(noise_scales) > 1, f"Expected diverse noise scales, got {noise_scales}"
    print("  PASS  diverse_configs")


def test_labels_descriptive(model, model_args, data, device):
    config = SamplerConfig(n_trajectories=4)
    results = generate_diverse_group(model, model_args, data, config, device)
    for st in results:
        assert len(st.label) > 0
        assert isinstance(st.label, str)
    print("  PASS  labels_descriptive")


def test_no_prototypes_disables_anchor(model, model_args, data, device):
    config = SamplerConfig(n_trajectories=8, enable_anchor=True, guidance_prob=1.0, prototypes_path=None)
    results = generate_diverse_group(model, model_args, data, config, device)
    for st in results[1:]:
        if st.guidance_config is not None:
            for fn in st.guidance_config.functions:
                assert fn.name != "anchor_following", f"anchor_following found without prototypes"
    print("  PASS  no_prototypes_disables_anchor")


if __name__ == "__main__":
    print("=" * 60)
    print("GRPO Sampler Test Suite")
    print("=" * 60 + "\n")

    # Config-only tests (no model needed)
    failed = 0
    try:
        test_sampler_config_defaults()
    except Exception as e:
        print(f"  ERROR test_sampler_config_defaults: {e}")
        failed += 1

    # Model-dependent tests
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--npz_path", type=Path, default=None)
    args, _ = parser.parse_known_args()

    if args.model_path is None or args.npz_path is None:
        print("\n  SKIP  model-dependent tests (provide --model_path and --npz_path)")
    else:
        from preference_optimization.model_utils import load_model
        from preference_optimization.utils import load_npz_data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n  Loading model from {args.model_path} on {device}...")
        model, model_args = load_model(args.model_path, device)
        model.eval()

        print(f"  Loading data from {args.npz_path}...")
        data = load_npz_data(str(args.npz_path), device)

        model_tests = [
            test_first_trajectory_is_deterministic,
            test_correct_count,
            test_trajectory_shapes,
            test_diverse_configs,
            test_labels_descriptive,
            test_no_prototypes_disables_anchor,
        ]

        for t in model_tests:
            try:
                t(model, model_args, data, device)
            except AssertionError as e:
                print(f"  FAIL  {t.__name__}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR {t.__name__}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"{failed} TEST(S) FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
