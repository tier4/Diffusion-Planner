import json

import numpy as np
from tests.dp_fixtures import SIDECAR_VARIANTS, make_arrays, make_tree


def test_make_arrays_has_18_real_dtypes():
    arrays = make_arrays(np.random.default_rng(0))
    assert len(arrays) == 18 and "version" in arrays
    assert arrays["version"].dtype == np.uint32
    assert arrays["lanes_has_speed_limit"].dtype == np.bool_
    assert arrays["turn_indicators"].dtype == np.int32
    assert arrays["lanes"].dtype == np.float32


def test_make_tree_writes_pairs_per_variant(tmp_path):
    keys = make_tree(
        tmp_path,
        [
            ("projA/mapX/manual/2026-01-01/t1/route_0", 3, "full"),
            ("psim/loc_seed_1/bag_0", 2, "psim"),
            ("projB/mapY/manual/2026-01-02/t2/route_0", 1, "none"),
        ],
    )
    assert len(keys) == 6
    npz = sorted(p for p in tmp_path.rglob("*.npz"))
    assert len(npz) == 6
    full = json.loads(
        (tmp_path / "projA/mapX/manual/2026-01-01/t1/route_0").glob("*.json").__next__().read_text()
    )
    assert set(SIDECAR_VARIANTS["full"]) <= set(full)
    assert not list((tmp_path / "projB/mapY/manual/2026-01-02/t2/route_0").glob("*.json"))
