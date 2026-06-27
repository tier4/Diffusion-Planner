"""scene_skip: drop only the sidecar-flagged frames, keep everything else, and be a
no-op (backward compatible) when sidecars/flags are absent."""

import json

from diffusion_planner.utils.scene_skip import filter_scene_list, is_skipped


def _frame(tmp, stem, skipped=None):
    """Create <stem>.npz and (optionally) a sibling <stem>.json with is_skipped."""
    npz = tmp / f"{stem}.npz"
    npz.write_bytes(b"")  # content irrelevant; we only resolve the sidecar
    if skipped is not None:
        (tmp / f"{stem}.json").write_text(json.dumps({"x": 0.0, "is_skipped": skipped}))
    return str(npz)


def test_is_skipped_reads_sibling_sidecar(tmp_path):
    flagged = _frame(tmp_path, "a", skipped=True)
    keep = _frame(tmp_path, "b", skipped=False)
    nosc = _frame(tmp_path, "c", skipped=None)  # no sidecar
    assert is_skipped(flagged) is True
    assert is_skipped(keep) is False
    assert is_skipped(nosc) is False  # missing sidecar => not skipped


def test_filter_drops_only_flagged(tmp_path):
    scenes = [
        _frame(tmp_path, "a", skipped=True),
        _frame(tmp_path, "b", skipped=False),
        _frame(tmp_path, "c", skipped=True),
        _frame(tmp_path, "d", skipped=False),
    ]
    kept = filter_scene_list(scenes, label="unit")
    assert kept == [scenes[1], scenes[3]]


def test_missing_sidecar_passthrough(tmp_path):
    scenes = [_frame(tmp_path, f"n{i}", skipped=None) for i in range(3)]
    kept = filter_scene_list(scenes)  # no sidecars => all kept (backward compatible)
    assert kept == scenes


def test_disabled_is_noop(tmp_path):
    scenes = [_frame(tmp_path, "a", skipped=True), _frame(tmp_path, "b", skipped=False)]
    assert filter_scene_list(scenes, enabled=False) == scenes


def test_dict_entries_preserved(tmp_path):
    a, b = _frame(tmp_path, "a", skipped=True), _frame(tmp_path, "b", skipped=False)
    scenes = [{"path": a, "weight": 1}, {"npz": b, "weight": 2}]
    kept = filter_scene_list(scenes)
    assert kept == [{"npz": b, "weight": 2}]  # the flagged dict dropped, entry type kept


def test_sidecar_root_index(tmp_path):
    # NPZs in one dir, sidecars nested under a separate root (padded-corpus style).
    npz_dir, sc_root = tmp_path / "npz", tmp_path / "sc" / "date" / "bag"
    npz_dir.mkdir(parents=True)
    sc_root.mkdir(parents=True)
    (npz_dir / "x.npz").write_bytes(b"")
    (sc_root / "x.json").write_text(json.dumps({"is_skipped": True}))
    # sidecar resolved by stem under sidecar_root => the flagged frame is dropped
    assert is_skipped(npz_dir / "x.npz", sidecar_root=tmp_path / "sc") is True
    assert filter_scene_list([str(npz_dir / "x.npz")], sidecar_root=tmp_path / "sc") == []
