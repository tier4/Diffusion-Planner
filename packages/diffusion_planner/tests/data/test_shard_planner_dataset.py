"""ShardPlannerDataset: rank coverage, bit-exact frames, transforms, loader wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffusion_planner.data import ShardPlannerDataset, build_shard_dataloader
from planner_shards.data_pipeline.pack_shards import main as pack_main

KEYS = {"a": (3, 2), "b": (4,), "c": (2, 2, 2)}
PARTITION_REGEX = r"^(?P<partition>.+)/[0-9]{6}$"


def _write_frames(
    root: Path, *, bags: int, frames: int
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    expected: dict[str, dict[str, np.ndarray]] = {}
    for bag in range(bags):
        directory = root / f"bag{bag}"
        directory.mkdir(parents=True)
        for index in range(frames):
            arrays = {
                key: rng.standard_normal(shape).astype(np.float32)
                for key, shape in KEYS.items()
            }
            np.savez(directory / f"{index:06d}.npz", **arrays)
            expected[f"bag{bag}/{index:06d}"] = arrays
    return expected


@pytest.fixture
def packed(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, np.ndarray]]]:
    source = tmp_path / "src"
    dest = tmp_path / "dest"
    expected = _write_frames(source, bags=2, frames=40)
    assert (
        pack_main(
            [
                "pack",
                "--source",
                str(source),
                "--dest",
                str(dest),
                "--base",
                "none",
                "--tag",
                "v1",
                "--partition-regex",
                PARTITION_REGEX,
                "--quiet",
            ]
        )
        == 0
    )
    keyset = tmp_path / "keys.parquet"
    assert (
        pack_main(
            [
                "keyset",
                "--dest",
                str(dest),
                "--tag",
                "v1",
                "--where",
                "TRUE",
                "--out",
                str(keyset),
            ]
        )
        == 0
    )
    return dest, keyset, expected


def _matches(
    sample: dict[str, np.ndarray], arrays: dict[str, np.ndarray], scale: float = 1.0
) -> bool:
    return all(np.array_equal(sample[key], scale * arrays[key]) for key in KEYS)


def test_two_ranks_cover_every_frame_exactly_once_bit_exact(packed) -> None:
    dest, keyset, expected = packed
    seen: list[dict[str, np.ndarray]] = []
    for rank in range(2):
        dataset = ShardPlannerDataset(
            dest,
            "latest",
            keyset,
            batch_size=4,
            num_workers=0,
            world_size=2,
            rank=rank,
            seed=1,
        )
        dataset.set_epoch(0)
        seen.extend(
            {key: value.numpy() for key, value in sample.items()} for sample in dataset
        )
    # 80 frames over 2 slots of 40 = whole batches of 4: the plan pads nothing.
    assert len(seen) == len(expected) == 80
    remaining = dict(expected)
    for sample in seen:
        key = next(
            (k for k, arrays in remaining.items() if _matches(sample, arrays)), None
        )
        assert key is not None, "decoded frame matches no source frame bit-exact"
        del remaining[key]
    assert not remaining


def test_transforms_apply_in_order_and_frames_are_writable(packed) -> None:
    dest, keyset, expected = packed

    def double(frame):
        return {key: value * 2.0 for key, value in frame.items()}

    def plus_one(frame):
        return {key: value + 1.0 for key, value in frame.items()}

    dataset = ShardPlannerDataset(
        dest,
        "latest",
        keyset,
        batch_size=4,
        num_workers=0,
        world_size=1,
        rank=0,
        shuffle=False,
        transforms=[double, plus_one],
    )
    first = {key: value.numpy() for key, value in next(iter(dataset)).items()}
    assert set(first) == set(KEYS)
    assert any(
        all(np.array_equal(first[key], arrays[key] * 2.0 + 1.0) for key in KEYS)
        for arrays in expected.values()
    )
    first["a"] += 1.0  # arrays handed to transforms and callers are writable copies


def test_dataloader_yields_whole_batches_and_matching_len(packed) -> None:
    dest, keyset, _ = packed
    dataset = ShardPlannerDataset(
        dest, "latest", keyset, batch_size=8, num_workers=1, world_size=1, rank=0
    )
    loader = build_shard_dataloader(
        dataset, batch_size=8, num_workers=1, pin_memory=False, prefetch_factor=2
    )
    batches = list(loader)
    assert len(batches) == len(loader) == dataset.steps_per_epoch == 10
    assert all(batch["a"].shape == (8, 3, 2) for batch in batches)


def test_rank_and_world_size_default_to_the_torchrun_environment(
    packed, monkeypatch
) -> None:
    dest, keyset, _ = packed
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    dataset = ShardPlannerDataset(dest, "latest", keyset, batch_size=4, num_workers=0)
    assert (dataset.config.world_size, dataset.config.rank) == (2, 1)


def test_loader_settings_must_match_the_plan(packed) -> None:
    dest, keyset, _ = packed
    dataset = ShardPlannerDataset(
        dest, "latest", keyset, batch_size=4, num_workers=0, world_size=1, rank=0
    )
    with pytest.raises(ValueError, match="batch_size"):
        build_shard_dataloader(dataset, batch_size=8, num_workers=0)
    with pytest.raises(ValueError, match="num_workers"):
        build_shard_dataloader(dataset, batch_size=4, num_workers=2)
    with pytest.raises(ValueError, match="drop_last"):
        build_shard_dataloader(dataset, batch_size=4, num_workers=0, drop_last=False)


def test_run_record_names_the_version_and_keyset(packed) -> None:
    dest, keyset, _ = packed
    dataset = ShardPlannerDataset(
        dest, "latest", keyset, batch_size=4, num_workers=0, world_size=1, rank=0
    )
    record = dataset.run_record()
    assert (
        record["version"] == "latest"
        and record["keyset_digest"]
        and record["version_hash"]
    )
