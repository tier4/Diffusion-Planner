import hashlib
import tarfile

import pytest
from diffusion_planner.data_pipeline import tar_shards as T
from diffusion_planner.data_pipeline.errors import IntegrityError


def _payloads(n, size=1000):
    return [bytes([i % 256]) * size + i.to_bytes(4, "little") for i in range(n)]


def test_writer_rolls_shards_and_records_offsets(tmp_path):
    payloads = _payloads(10)
    w = T.ShardWriter(tmp_path, shard_size_bytes=4 * 1024)  # ~3 members per shard incl. headers
    recs = [w.add(p) for p in payloads]
    names = w.close()
    assert names[0] == "shard-0000.tar" and len(names) >= 3
    assert not list(tmp_path.glob("*.tmp"))
    assert [r.sample_index for r in recs if r.shard_id == 0] == list(
        range(sum(1 for r in recs if r.shard_id == 0))
    )
    for r, p in zip(recs, payloads):
        with open(tmp_path / names[r.shard_id], "rb") as f:
            assert T.read_member(f, r.offset, r.size) == p
        assert r.payload_sha256 == hashlib.sha256(p).digest()


def test_ustar_format_and_member_names(tmp_path):
    w = T.ShardWriter(tmp_path, shard_size_bytes=1 << 30)
    [w.add(p) for p in _payloads(3)]
    (name,) = w.close()
    with tarfile.open(tmp_path / name) as t:
        infos = t.getmembers()
        assert [i.name for i in infos] == [
            "000000.safetensors.zst",
            "000001.safetensors.zst",
            "000002.safetensors.zst",
        ]
        assert all(i.type == tarfile.REGTYPE and i.mtime == 0 for i in infos)
        assert infos[0].offset_data == 512  # ustar header only, no PAX blocks


def test_skim_and_list_agree(tmp_path):
    payloads = _payloads(5)
    w = T.ShardWriter(tmp_path, shard_size_bytes=1 << 30)
    [w.add(p) for p in payloads]
    (name,) = w.close()
    assert [(i, p) for i, p in T.iter_members(tmp_path / name)] == list(enumerate(payloads))
    listed = T.list_members(tmp_path / name)
    assert [i for i, _, _ in listed] == list(range(5))
    with open(tmp_path / name, "rb") as f:
        assert [T.read_member(f, o, s) for _, o, s in listed] == payloads


def test_short_read_is_integrity_error(tmp_path):
    w = T.ShardWriter(tmp_path, shard_size_bytes=1 << 30)
    r = w.add(b"x" * 100)
    (name,) = w.close()
    with open(tmp_path / name, "rb") as f, pytest.raises(IntegrityError):
        T.read_member(f, r.offset, r.size + 10_000_000)


def test_truncated_mid_payload_raises_integrity_error(tmp_path):
    payloads = _payloads(3)
    w = T.ShardWriter(tmp_path, shard_size_bytes=1 << 30)
    recs = [w.add(p) for p in payloads]
    (name,) = w.close()
    tar_path = tmp_path / name

    member1_offset = recs[1].offset
    truncate_pos = member1_offset + 10

    with open(tar_path, "r+b") as f:
        f.truncate(truncate_pos)

    with pytest.raises(IntegrityError):
        list(T.iter_members(tar_path))


def test_truncated_mid_header_raises_integrity_error(tmp_path):
    payloads = _payloads(3)
    w = T.ShardWriter(tmp_path, shard_size_bytes=1 << 30)
    recs = [w.add(p) for p in payloads]
    (name,) = w.close()
    tar_path = tmp_path / name

    member1_offset = recs[1].offset
    truncate_pos = member1_offset - 512 + 100

    with open(tar_path, "r+b") as f:
        f.truncate(truncate_pos)

    with pytest.raises(IntegrityError):
        list(T.iter_members(tar_path))

    member0_padded_end = recs[0].offset + recs[0].size
    padded_size = recs[0].size + (-recs[0].size % 512)
    clean_cut = recs[0].offset - 512 + padded_size

    with open(tar_path, "r+b") as f:
        f.truncate(clean_cut)

    with pytest.raises(IntegrityError):
        list(T.iter_members(tar_path))


def test_expected_count_mismatch_raises(tmp_path):
    payloads = _payloads(5)
    w = T.ShardWriter(tmp_path, shard_size_bytes=1 << 30)
    [w.add(p) for p in payloads]
    (name,) = w.close()
    tar_path = tmp_path / name

    with pytest.raises(IntegrityError, match="member count mismatch"):
        list(T.iter_members(tar_path, expected_count=99))

    list(T.iter_members(tar_path, expected_count=5))
