import hashlib
import json

import pytest
from diffusion_planner.data_pipeline.validation import prepare_path_list as PP


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_rewrites_prefix_and_reports_digest(tmp_path):
    src = _write(tmp_path / "in.json", ["/old/root/a/b/1.npz", "/old/root/a/b/2.npz"])
    out = tmp_path / "out.json"
    rep = PP.prepare(src, out, "/old/root", "/new/root")
    assert rep["entries"] == 2
    assert rep["rewritten"] == 2
    assert json.loads(out.read_text()) == ["/new/root/a/b/1.npz", "/new/root/a/b/2.npz"]
    assert rep["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_prefix_must_match_on_a_component_boundary(tmp_path):
    src = _write(tmp_path / "in.json", ["/old/rootstuff/a/1.npz"])
    with pytest.raises(ValueError, match="does not start with"):
        PP.prepare(src, tmp_path / "out.json", "/old/root", "/new/root")


def test_rejects_non_npz_and_duplicates_and_bad_shape(tmp_path):
    with pytest.raises(ValueError, match="not a .npz path"):
        PP.prepare(
            _write(tmp_path / "a.json", ["/old/root/a/1.txt"]),
            tmp_path / "o1.json",
            "/old/root",
            "/new/root",
        )
    with pytest.raises(ValueError, match="duplicate"):
        PP.prepare(
            _write(tmp_path / "b.json", ["/old/root/a/1.npz", "/old/root/a/1.npz"]),
            tmp_path / "o2.json",
            "/old/root",
            "/new/root",
        )
    with pytest.raises(ValueError, match="list of strings"):
        PP.prepare(
            _write(tmp_path / "c.json", {"paths": []}),
            tmp_path / "o3.json",
            "/old/root",
            "/new/root",
        )
