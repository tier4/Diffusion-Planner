"""Tests for CPU/GPU pipelining in batch_generate.py."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scenario_generation.batch_generate import (
    _load_snippet,
    _save_scene,
    run_batch,
)
from scenario_generation.scene_context import SceneContext


class TestLoadSnippet:
    def test_loads_basic_snippet(self, tmp_path):
        data = {"lanelet_ids": [1, 2, 3]}
        snip_path = tmp_path / "test.pkl"
        with open(snip_path, "wb") as f:
            pickle.dump(data, f)

        name, ids, ego_pose = _load_snippet(snip_path)
        assert name == "test"
        assert ids == [1, 2, 3]
        assert ego_pose is None

    def test_loads_with_ego_pose(self, tmp_path):
        data = {"lanelet_ids": [10], "ego_pose": [1.0, 2.0, 0.5]}
        snip_path = tmp_path / "posed.pkl"
        with open(snip_path, "wb") as f:
            pickle.dump(data, f)

        name, ids, ego_pose = _load_snippet(snip_path)
        assert ego_pose == (1.0, 2.0, 0.5)


class TestSaveScene:
    def test_creates_expected_files(self, synthetic_scene, tmp_path):
        scene_dir = tmp_path / "scene_000"
        _save_scene(synthetic_scene, scene_dir, "test_snippet")

        assert (scene_dir / "scene.pkl").exists()
        assert (scene_dir / "info.json").exists()
        assert (scene_dir / "initial.png").exists()

        with open(scene_dir / "info.json") as f:
            info = json.load(f)
        assert info["snippet"] == "test_snippet"
        assert info["n_agents"] == 4

    def test_pickle_roundtrips(self, synthetic_scene, tmp_path):
        scene_dir = tmp_path / "scene_001"
        _save_scene(synthetic_scene, scene_dir, "rt")

        with open(scene_dir / "scene.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, SceneContext)
        assert len(loaded.agents) == len(synthetic_scene.agents)


class TestRunBatch:
    def test_no_sim_produces_outputs(self, synthetic_scene, tmp_path):
        """Run batch with mock builder, no simulation, verify output structure."""
        snippets_dir = tmp_path / "snippets"
        snippets_dir.mkdir()

        for i in range(2):
            data = {"lanelet_ids": [1, 2]}
            with open(snippets_dir / f"snip_{i}.pkl", "wb") as f:
                pickle.dump(data, f)

        builder = MagicMock()
        builder.build_scene_context.return_value = synthetic_scene

        config = {
            "snippets_dir": str(snippets_dir),
            "generation": {"n_scenes_per_snippet": 2},
            "simulation": {"enabled": False},
        }

        output_dir = tmp_path / "output"
        run_batch(config, builder, output_dir, model_path=None, device="cpu")

        # 2 snippets x 2 scenes = 4 scene dirs
        scene_dirs = list(output_dir.rglob("scene_*"))
        assert len(scene_dirs) == 4

        for sd in scene_dirs:
            assert (sd / "scene.pkl").exists()
            assert (sd / "info.json").exists()
            assert (sd / "initial.png").exists()

    def test_scene_count_matches(self, synthetic_scene, tmp_path):
        """Verify pipeline produces same count as sequential would."""
        snippets_dir = tmp_path / "snippets"
        snippets_dir.mkdir()

        for i in range(3):
            data = {"lanelet_ids": [i + 1]}
            with open(snippets_dir / f"snip_{i}.pkl", "wb") as f:
                pickle.dump(data, f)

        builder = MagicMock()
        builder.build_scene_context.return_value = synthetic_scene

        config = {
            "snippets_dir": str(snippets_dir),
            "generation": {"n_scenes_per_snippet": 1},
            "simulation": {"enabled": False},
        }

        output_dir = tmp_path / "output"
        run_batch(config, builder, output_dir, model_path=None, device="cpu")

        scene_dirs = list(output_dir.rglob("scene_*"))
        assert len(scene_dirs) == 3
