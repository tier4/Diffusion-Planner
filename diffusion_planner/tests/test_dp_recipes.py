import subprocess
import sys
from pathlib import Path

import pytest
from diffusion_planner.data_pipeline import packer as PK
from diffusion_planner.data_pipeline import recipes as R
from diffusion_planner.data_pipeline.errors import PlanError
from diffusion_planner.data_pipeline.partition import PartitionRule
from diffusion_planner.data_pipeline.reader import ShardReader
from tests.dp_fixtures import make_tree

UTIL = Path(__file__).resolve().parents[1] / "util_scripts"
LAYOUT = [
    ("pA/mX/manual/2026-01-01/t1/r", 40, "full"),
    ("pA/mX/manual/2026-01-02/t1/r", 25, "skip"),
    ("psim/b_mobility_seed_200_poses_100/psim_training_bag_0_0", 12, "psim"),
    ("psim/kashiwa_seed_201_poses_100/psim_training_bag_1_0", 9, "psim"),
]


def _run(*argv):
    subprocess.run([sys.executable, *map(str, argv)], check=True, cwd=UTIL)


@pytest.fixture
def world(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src, LAYOUT, seed=0, skipped_every=5)
    PK.pack(
        PK.PackOptions(
            source=src,
            dest=dst,
            base="none",
            tag="v1",
            rule=PartitionRule(depth=2),
            shard_size_bytes=64 * 1024,
        )
    )
    return src, dst


def test_create_train_set_path_equivalence(world, tmp_path):
    src, dst = world
    out = tmp_path / "legacy.json"
    _run(UTIL / "create_train_set_path.py", src / "pA", "--save_path", out)
    legacy = R.legacy_keys(out, src)
    new = R.keys_for(ShardReader(dst, "v1"), R.root_filter("pA"))
    assert new == sorted(legacy) and len(new) == len(legacy)


def test_every_n_and_head_equivalence(world, tmp_path):
    src, dst = world
    base = tmp_path / "base.json"
    _run(UTIL / "create_train_set_path.py", src / "pA", "--save_path", base)
    _run(UTIL / "filter_json.py", base, "--num_filter", "3", "--num_filter_mode", "interval")
    legacy = R.legacy_keys(base.with_name("base_every_3.json"), src)
    rd = ShardReader(dst, "v1")
    assert R.keys_for(rd, R.every_n(R.root_filter("pA"), 3)) == legacy  # ordered multiset equality

    _run(UTIL / "filter_json.py", base, "--num_filter", "4", "--num_filter_mode", "head")
    legacy_head = R.legacy_keys(base.with_name("base_head_4.json"), src)
    assert R.keys_for(rd, R.head_n(R.root_filter("pA"), 4)) == legacy_head


def test_psim_per_location_and_concat_keep_duplicates(world, tmp_path):
    src, dst = world
    psim = tmp_path / "psim.json"
    _run(UTIL / "create_train_set_path.py", src / "psim", "--save_path", psim)
    _run(UTIL / "filter_json_special.py", psim, "--num_filter", "2")
    legacy = R.legacy_keys(next(tmp_path.glob("psim_*.json")), src)
    rd = ShardReader(dst, "v1")
    assert R.keys_for(rd, R.psim_per_location(R.root_filter("psim"), 2, component_k=2)) == legacy

    both = tmp_path / "both.json"
    _run(UTIL / "concat_data_list_jsons.py", psim, psim, "--save_path", both)
    assert R.legacy_keys(both, src) == R.keys_for_all(
        rd, [R.root_filter("psim"), R.root_filter("psim")]
    )  # duplicates preserved


def test_ranked_handle_composition_is_guarded():
    """every_n/head_n/psim_per_location take a plain WHERE clause, not a ranked handle: composing
    them (`every_n(head_n(...), n)`) or colliding with the internal `:::` separator must raise
    ValueError rather than silently misparsing into wrong SQL.
    """
    with pytest.raises(ValueError):
        R.every_n(R.head_n(R.root_filter("pA"), 4), 3)
    with pytest.raises(ValueError):
        R.every_n("a:::b", 2)


def test_keys_for_routes_through_reader_guard(world):
    """keys_for must go through ShardReader's public, guarded execute()/query() — proven here by
    the `;` guard now firing for a plain WHERE passed straight through.
    """
    src, dst = world
    rd = ShardReader(dst, "v1")
    with pytest.raises(PlanError):
        R.keys_for(rd, "1=1; DROP TABLE x")
