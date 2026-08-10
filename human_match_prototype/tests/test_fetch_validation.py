import json

import numpy as np

from human_match_prototype.fetch_validation import create_subsample, verify_route_format


class TestCreateSubsample:
    def test_deterministic(self, tmp_path):
        """Same seed produces same subsample."""
        full = [f"/path/to/frame_{i:05d}.npz" for i in range(1000)]
        full_json = tmp_path / "full.json"
        full_json.write_text(json.dumps(full))
        out1 = tmp_path / "sub1.json"
        out2 = tmp_path / "sub2.json"
        create_subsample(str(full_json), 50, seed=42, output_path=str(out1))
        create_subsample(str(full_json), 50, seed=42, output_path=str(out2))
        assert json.loads(out1.read_text()) == json.loads(out2.read_text())

    def test_correct_count(self, tmp_path):
        full = [f"/path/frame_{i}.npz" for i in range(200)]
        full_json = tmp_path / "full.json"
        full_json.write_text(json.dumps(full))
        out = tmp_path / "sub.json"
        create_subsample(str(full_json), 50, seed=0, output_path=str(out))
        result = json.loads(out.read_text())
        assert len(result) == 50


class TestVerifyRouteFormat:
    def test_standard_shape(self, tmp_path):
        npz_path = tmp_path / "test.npz"
        route_lanes = np.zeros((25, 20, 33), dtype=np.float32)
        route_lanes[0, :, 0] = np.linspace(0, 19, 20)  # x values
        np.savez(npz_path, route_lanes=route_lanes, ego_agent_future=np.zeros((80, 3)))
        info = verify_route_format(str(npz_path))
        assert info["shape"] == (25, 20, 33)
        assert info["n_nonempty_segments"] == 1

    def test_batched_shape(self, tmp_path):
        npz_path = tmp_path / "test.npz"
        route_lanes = np.zeros((1, 25, 20, 33), dtype=np.float32)
        route_lanes[0, 0, :, 0] = np.linspace(0, 19, 20)
        np.savez(npz_path, route_lanes=route_lanes, ego_agent_future=np.zeros((80, 3)))
        info = verify_route_format(str(npz_path))
        assert info["shape_raw"] == (1, 25, 20, 33)
        assert info["shape"] == (25, 20, 33)
