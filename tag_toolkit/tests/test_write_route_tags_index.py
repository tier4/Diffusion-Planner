"""``TagStore.from_source`` opens ``<source stem>.tags.db`` without checking whether the
sidecars moved on since it was built, so a tag writer that skips the index hands every
later reader the previous run's answers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tag_toolkit" / "scripts" / "tag_management" / "write_route_tags_from_csv.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tag_toolkit import TagStore  # noqa: E402

_spec = importlib.util.spec_from_file_location("write_route_tags_from_csv", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


@pytest.fixture
def dataset(tmp_path: Path):
    """One route whose sidecars carry a t4_dataset_id, plus a path list naming it."""
    route = tmp_path / "proj" / "1234_site" / "auto" / "2026-01-01" / "00-00-00"
    route.mkdir(parents=True)
    for i in (1, 2):
        npz = route / f"00-00-00_00000000_0000000{i}.npz"
        npz.write_bytes(b"")
        npz.with_suffix(".json").write_text(json.dumps({"t4_dataset_id": "abc", "tags": []}))
    source = tmp_path / "path_list.json"
    source.write_text(json.dumps([str(route)]))
    csv_path = tmp_path / "map.csv"
    csv_path.write_text("t4_dataset_id,devops_site\nabc,odaiba\n")
    return source, csv_path


def _run(source: Path, csv_path: Path, *extra: str) -> int:
    argv = [
        "write_route_tags_from_csv.py",
        str(source),
        str(csv_path),
        "--match-col",
        "t4_dataset_id",
        "--tag-dimensions",
        "devops_site",
        *extra,
    ]
    old, sys.argv = sys.argv, argv
    try:
        return script.main()
    finally:
        sys.argv = old


def test_a_second_run_leaves_the_index_agreeing_with_the_sidecars(dataset):
    """The failure this guards: tags change, the index does not, readers see the old ones."""
    source, csv_path = dataset
    _run(source, csv_path)

    csv_path.write_text("t4_dataset_id,devops_site\nabc,shiojiri\n")
    _run(source, csv_path)

    indexed = TagStore(source.parent / "path_list.tags.db").tags_of(
        scope=source, granularity="route"
    )
    assert "devops_site:shiojiri" in indexed
