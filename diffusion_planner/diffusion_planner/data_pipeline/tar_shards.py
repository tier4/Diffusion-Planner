"""USTAR tar shards with one safetensors.zst member per sample (spec §3)."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from diffusion_planner.data_pipeline.encoding import MEMBER_EXT
from diffusion_planner.data_pipeline.errors import IntegrityError

SHARD_NAME_FMT = "shard-{:04d}.tar"
_MEMBER_RE = re.compile(r"^(\d{6})" + re.escape(MEMBER_EXT) + r"$")


def member_name(index: int) -> str:
    return f"{index:06d}{MEMBER_EXT}"


@dataclass(frozen=True)
class MemberRecord:
    shard_id: int
    sample_index: int
    offset: int
    size: int
    payload_sha256: bytes


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size_bytes: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = int(shard_size_bytes)
        self._names: list[str] = []
        self._tar: tarfile.TarFile | None = None
        self._fh: BinaryIO | None = None
        self._tmp: Path | None = None
        self._index = 0
        self._bytes = 0

    def _open(self) -> None:
        name = SHARD_NAME_FMT.format(len(self._names))
        self._tmp = self.out_dir / (name + ".tmp")
        self._fh = open(self._tmp, "wb")
        self._tar = tarfile.open(fileobj=self._fh, mode="w", format=tarfile.USTAR_FORMAT)
        self._names.append(name)
        self._index = 0
        self._bytes = 0

    def _roll(self) -> None:
        assert self._tar is not None and self._fh is not None and self._tmp is not None
        self._tar.close()
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        final = self._tmp.with_name(self._tmp.name[: -len(".tmp")])
        os.replace(self._tmp, final)
        _fsync_dir(self.out_dir)
        self._tar = self._fh = self._tmp = None

    def add(self, payload: bytes) -> MemberRecord:
        need = 512 + len(payload) + (-len(payload) % 512)
        if self._tar is None:
            self._open()
        elif self._bytes + need > self.shard_size_bytes and self._index > 0:
            self._roll()
            self._open()
        assert self._tar is not None and self._fh is not None
        offset_before = self._fh.tell()
        info = tarfile.TarInfo(member_name(self._index))
        info.size, info.mtime, info.mode, info.uid, info.gid = len(payload), 0, 0o444, 0, 0
        info.uname = info.gname = ""
        self._tar.addfile(info, io.BytesIO(payload))
        offset_data = offset_before + 512
        rec = MemberRecord(
            len(self._names) - 1,
            self._index,
            offset_data,
            len(payload),
            hashlib.sha256(payload).digest(),
        )
        self._index += 1
        self._bytes += need
        return rec

    def close(self) -> list[str]:
        if self._tar is not None:
            self._roll()
        return list(self._names)


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _index_of(name: str) -> int:
    m = _MEMBER_RE.match(name)
    if not m:
        raise IntegrityError(f"unexpected tar member name {name!r}")
    return int(m.group(1))


def iter_members(
    tar_path: Path, expected_count: int | None = None
) -> Iterator[tuple[int, int, int, bytes]]:
    """Yield ``(index, offset_data, size, payload)`` for each member in streaming order."""
    count = 0
    try:
        with tarfile.open(tar_path, mode="r|") as t:
            for info in t:
                f = t.extractfile(info)
                if f is None:
                    raise IntegrityError(f"non-regular member {info.name!r}")
                payload = f.read()
                if len(payload) != info.size:
                    raise IntegrityError(
                        f"short member {info.name!r}: {len(payload)} of {info.size} bytes"
                    )
                yield _index_of(info.name), info.offset_data, info.size, payload
                count += 1
    except tarfile.TarError as e:
        raise IntegrityError(f"invalid tar archive: {e}") from e
    except EOFError as e:
        raise IntegrityError(f"truncated tar archive: {e}") from e
    except OSError as e:
        raise IntegrityError(f"i/o error reading tar: {e}") from e

    size = os.path.getsize(tar_path)
    if size % 512 != 0:
        raise IntegrityError(f"truncated tar: size {size} not multiple of 512")

    if size >= 1024:
        with open(tar_path, "rb") as f:
            f.seek(size - 1024)
            marker = f.read(1024)
            if marker != b"\x00" * 1024:
                raise IntegrityError("missing end-of-archive marker")
    elif size > 0:
        raise IntegrityError("tar too small for end-of-archive marker")

    if expected_count is not None and count != expected_count:
        raise IntegrityError(f"member count mismatch: got {count}, expected {expected_count}")


def list_members(tar_path: Path) -> list[tuple[int, int, int]]:
    with tarfile.open(tar_path, mode="r") as t:
        return [(_index_of(i.name), i.offset_data, i.size) for i in t.getmembers()]


def read_member(fileobj: BinaryIO, offset: int, size: int) -> bytes:
    fileobj.seek(offset)
    data = fileobj.read(size)
    if len(data) != size:
        raise IntegrityError(f"short read at offset {offset}: {len(data)} of {size} bytes")
    return data
