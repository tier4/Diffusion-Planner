import json

import pytest
from diffusion_planner.data_pipeline import sidecar
from diffusion_planner.data_pipeline.errors import SidecarError
from tests.dp_fixtures import make_sidecar


@pytest.mark.parametrize("variant", ["psim", "skip", "neighbor", "full"])
def test_parse_variants_yield_full_column_set(variant):
    raw = json.dumps(make_sidecar(variant, 3)).encode()
    f = sidecar.parse_sidecar(raw)
    assert set(f) == {name for name, _ in sidecar.SIDECAR_FIELDS}
    assert f["timestamp"] == 1_700_000_000_000_000_000 + 3 * 300_000_000
    if variant == "psim":
        assert f["is_skipped"] is None and f["project_id"] is None and f["neighbor_count"] is None
    if variant in ("neighbor", "full"):
        assert f["neighbor_count"] == 2
    if variant == "full":
        assert f["project_id"] == "projA" and f["skip_label"] == 0


def test_missing_sidecar_is_all_null_and_kept():
    f = sidecar.parse_sidecar(None)
    assert all(v is None for v in f.values())
    assert sidecar.is_rejected(f) is False


def test_rejected_only_when_true():
    assert sidecar.is_rejected({"is_skipped": True}) is True
    assert sidecar.is_rejected({"is_skipped": False}) is False
    assert sidecar.is_rejected({"is_skipped": None}) is False


def test_malformed_raises():
    with pytest.raises(SidecarError):
        sidecar.parse_sidecar(b"{not json")
    with pytest.raises(SidecarError):
        sidecar.parse_sidecar(b"[1,2]")
    with pytest.raises(SidecarError):
        sidecar.parse_sidecar(json.dumps({"timestamp": "abc"}).encode())


def test_sidecar_path_and_neighbor_ids(tmp_path):
    assert sidecar.sidecar_path_for(tmp_path / "a/b_00000001.npz") == tmp_path / "a/b_00000001.json"
    assert sidecar.neighbor_ids_of(json.dumps(make_sidecar("neighbor", 0)).encode()) == [
        "id0a",
        "id0b",
    ]
    assert sidecar.neighbor_ids_of(json.dumps(make_sidecar("psim", 0)).encode()) is None
