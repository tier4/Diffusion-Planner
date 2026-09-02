import json

import numpy as np
import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline.encoding import arrays_bitexact, load_npz_bytes
from diffusion_planner.data_pipeline.export import export
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.reader import ShardReader
from tests.dp_fixtures import make_tree

LAYOUT = [("pA/mX/manual/2026-01-01/t1/r", 30, "full"), ("pB/mY/manual/2026-01-02/t1/r", 8, "psim")]


@pytest.fixture
def packed(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    keys = make_tree(src, LAYOUT)
    PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=4),
            shard_size_bytes=48 * 1024,
        )
    )
    return src, dst, keys


def test_query_get_iter_agree(packed):
    src, dst, keys = packed
    rd = ShardReader(dst, "latest")
    assert rd.version.tag == "v1"
    t = rd.query("project_id = 'projA'", columns=["key", "timestamp"])
    assert t.num_rows == 30 and t.column("key").to_pylist() == sorted(
        k for k in keys if k.startswith("pA")
    )
    k0 = keys[0]
    assert arrays_bitexact(rd.get(k0), load_npz_bytes((src / f"{k0}.npz").read_bytes()))
    with pytest.raises(KeyError):
        rd.get("nope/x")
    seen = dict(
        rd.iter("project_id IS NULL")
    )  # psim rows have no project_id → seek/skim path both exercised
    assert set(seen) == {k for k in keys if k.startswith("pB")}
    for k, arrays in seen.items():
        assert arrays_bitexact(arrays, load_npz_bytes((src / f"{k}.npz").read_bytes()))
    k, arrays = next(rd.iter("project_id = 'projA'", training_view=True))
    assert len(arrays) == 17 and "version" not in arrays


def test_export_roundtrip_for_legacy_tools(packed, tmp_path):
    src, dst, keys = packed
    out = tmp_path / "exported"
    n = export(
        dst, "v1", "project_id = 'projA' AND timestamp < 1700000000000000000 + 5 * 300000000", out
    )
    assert n == 5
    man = json.loads((out / "export_manifest.json").read_text())
    assert man["n"] == 5 and man["version"] == "v1" and len(man["keys"]) == 5
    for k in man["keys"]:
        exported = dict(np.load(out / f"{k}.npz", allow_pickle=True))  # legacy np.load works
        assert arrays_bitexact(exported, load_npz_bytes((src / f"{k}.npz").read_bytes()))
        js = json.loads((out / f"{k}.json").read_text())
        orig = json.loads((src / f"{k}.json").read_text())
        assert js["is_skipped"] == orig["is_skipped"] and js["timestamp"] == orig["timestamp"]
        assert (
            js["project_id"] == orig["project_id"]
            and js["skipping_info"]["label"] == orig["skipping_info"]["label"]
        )
        assert "neighbor_count" not in js
