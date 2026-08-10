"""Test shard splitting and CSV merge logic."""

import csv
import tempfile
from pathlib import Path

from human_match_prototype.run_full_eval import merge_shard_csvs, split_into_shards


def test_split_into_shards_distributes_evenly():
    paths = list(range(10))
    shards = split_into_shards(paths, 3)
    assert len(shards) == 3
    assert sum(len(s) for s in shards) == 10
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_split_into_shards_more_shards_than_paths():
    paths = [1, 2]
    shards = split_into_shards(paths, 5)
    non_empty = [s for s in shards if s]
    assert sum(len(s) for s in shards) == 2
    assert len(non_empty) == 2


def test_merge_shard_csvs_concatenates():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fields = ["npz_path", "es_2s", "es_4s"]

        for i, rows in enumerate(
            [
                [{"npz_path": "/a.npz", "es_2s": "1.0", "es_4s": "2.0"}],
                [
                    {"npz_path": "/b.npz", "es_2s": "3.0", "es_4s": "4.0"},
                    {"npz_path": "/c.npz", "es_2s": "5.0", "es_4s": "6.0"},
                ],
            ]
        ):
            shard = tmp_path / f"shard_{i}.csv"
            with open(shard, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

        out = tmp_path / "merged.csv"
        n = merge_shard_csvs(
            [tmp_path / "shard_0.csv", tmp_path / "shard_1.csv"],
            out,
        )
        assert n == 3
        with open(out) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 3
        assert reader[0]["npz_path"] == "/a.npz"
        assert reader[2]["npz_path"] == "/c.npz"


def test_merge_skips_empty_shards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fields = ["npz_path", "es_2s"]

        shard0 = tmp_path / "shard_0.csv"
        with open(shard0, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{"npz_path": "/a.npz", "es_2s": "1.0"}])

        shard1 = tmp_path / "shard_1.csv"  # does not exist

        out = tmp_path / "merged.csv"
        n = merge_shard_csvs([shard0, shard1], out)
        assert n == 1
