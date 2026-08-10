# human_match_prototype/tests/test_build_lanelet_index.py
import warnings
from unittest.mock import MagicMock, patch

from human_match_prototype.build_lanelet_index import (
    add_lanelet_ids,
    load_lanelet_index,
    save_lanelet_index,
)


def _make_index(n=5):
    """Synthetic index entries with world positions along a line."""
    return [
        {
            "npz_path": f"/data/frame_{i}.npz",
            "x": 89100.0 + i * 10.0,
            "y": 42400.0,
            "heading_deg": 0.0,
            "timestamp": 1000 + i,
        }
        for i in range(n)
    ]


def test_add_lanelet_ids_adds_key():
    """add_lanelet_ids adds lanelet_id to each entry."""
    index = _make_index(3)

    mock_ll_layer = MagicMock()

    def fake_find_nearest(layer, point, count):
        return [(0.5, MagicMock(id=42))]

    with (
        patch("human_match_prototype.build_lanelet_index._find_nearest_lanelet", fake_find_nearest),
        patch("human_match_prototype.build_lanelet_index._make_point", lambda x, y: (x, y)),
    ):
        result = add_lanelet_ids(index, mock_ll_layer)

    assert len(result) == 3
    for entry in result:
        assert "lanelet_id" in entry
        assert entry["lanelet_id"] == 42


def test_save_load_roundtrip(tmp_path):
    """Save and load preserves all columns and map hash."""
    index = _make_index(3)
    for entry in index:
        entry["lanelet_id"] = 42

    path = str(tmp_path / "test_index.parquet")
    save_lanelet_index(index, path, map_hash="abc123")
    loaded, loaded_hash = load_lanelet_index(path)

    assert loaded_hash == "abc123"
    assert len(loaded) == 3
    for orig, loaded_entry in zip(index, loaded):
        assert loaded_entry["npz_path"] == orig["npz_path"]
        assert loaded_entry["lanelet_id"] == 42
        assert abs(loaded_entry["x"] - orig["x"]) < 1e-6
        assert abs(loaded_entry["heading_deg"] - orig["heading_deg"]) < 1e-6


def test_add_lanelet_ids_drops_no_match_with_warning():
    """When _find_nearest_lanelet returns [], the entry is dropped and a warning is emitted."""
    index = _make_index(3)

    call_count = 0

    def fake_find_nearest(layer, point, count):
        nonlocal call_count
        call_count += 1
        # Return empty list for the second entry (index 1)
        if call_count == 2:
            return []
        return [(0.5, MagicMock(id=99))]

    mock_ll_layer = MagicMock()

    with (
        patch("human_match_prototype.build_lanelet_index._find_nearest_lanelet", fake_find_nearest),
        patch("human_match_prototype.build_lanelet_index._make_point", lambda x, y: (x, y)),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = add_lanelet_ids(index, mock_ll_layer)

    # One entry dropped
    assert len(result) == 2
    # Remaining entries have correct lanelet_id
    for entry in result:
        assert entry["lanelet_id"] == 99

    # A warning was emitted mentioning the dropped npz_path
    lanelet_warnings = [w for w in caught if "No lanelet match" in str(w.message)]
    assert len(lanelet_warnings) == 1
    assert "frame_1" in str(lanelet_warnings[0].message)
