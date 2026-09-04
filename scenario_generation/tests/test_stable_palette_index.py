"""The id -> palette-slot mapping must be seed-independent: reverting ``stable_palette_index``
to ``abs(hash(agent_id))`` fails ``test_index_is_identical_under_different_hash_seeds``."""

import os
import subprocess
import sys

import pytest

from scenario_generation.replay import stable_palette_index

# Hex track UUIDs as ``_draw_step`` builds them: no all-digit suffix, so these take the
# hashed branch -- the one that used to be seed-dependent.
UUID_IDS = ["nb_a3f91c2e", "nb_7bd4e10a", "nb_ffc23b90", "nb_00d1f4bc"]

_PROBE = (
    "import json,sys;"
    "from scenario_generation.replay import stable_palette_index as f;"
    "print(json.dumps([f(s) for s in sys.argv[1:]]))"
)


def _indices_with_hash_seed(seed: str) -> list[int]:
    """Palette slots computed in a fresh interpreter with the given PYTHONHASHSEED."""
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, *UUID_IDS],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return __import__("json").loads(out.stdout.strip().splitlines()[-1])


def test_numeric_suffix_is_used_directly():
    assert stable_palette_index("npc_5") == 5
    assert stable_palette_index("npc_0") == 0


def test_index_is_deterministic_in_process():
    assert [stable_palette_index(s) for s in UUID_IDS] == [
        stable_palette_index(s) for s in UUID_IDS
    ]


@pytest.mark.parametrize("seed", ["1", "12345"])
def test_index_is_identical_under_different_hash_seeds(seed):
    """Pinning PYTHONHASHSEED to two different values reproduces "parent vs spawned worker"
    without needing a pool."""
    baseline = _indices_with_hash_seed("0")
    assert _indices_with_hash_seed(seed) == baseline
    # the in-process value agrees too, whatever seed the test runner happens to have
    assert [stable_palette_index(s) for s in UUID_IDS] == baseline
