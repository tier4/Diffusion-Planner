"""DDP sharding helpers for closed-loop evaluation."""

from __future__ import annotations


def shard_items(items: list, rank: int, world_size: int) -> list:
    """Round-robin assignment: rank ``r`` gets indices r, r+world_size, ..."""
    if world_size <= 1:
        return items
    return [items[i] for i in range(rank, len(items), world_size)]
