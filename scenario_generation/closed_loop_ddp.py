"""DDP sharding helpers for closed-loop evaluation."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence


def shard_items(items: list, rank: int, world_size: int) -> list:
    """Round-robin assignment: rank ``r`` gets indices r, r+world_size, ..."""
    if world_size <= 1:
        return items
    return [items[i] for i in range(rank, len(items), world_size)]


def balance_items(
    items: Sequence[Any],
    world_size: int,
    *,
    cost: Callable[[Any], float],
    key: Callable[[Any], str],
) -> list[list[Any]]:
    """Partition ``items`` into ``world_size`` buckets with LPT greedy bin packing.

    Closed-loop jobs differ in cost by an order of magnitude (a route costs roughly what it is
    long), so the round-robin ``shard_items`` above balances *counts* and leaves the wall clock
    set by whichever rank happened to draw the long routes. LPT -- sort descending by cost, hand
    each item to the currently-cheapest bucket -- is within 4/3 of optimal and is what makes a
    single-barrier run actually approach ``world_size``x.

    Determinism is load-bearing: every rank computes this independently and they must agree
    exactly, or jobs get run twice / not at all while the merged result still looks healthy.
    Ties therefore break on ``key(item)`` (not input order), and the bucket choice breaks ties on
    the lowest bucket index.

    Returns one list per rank, each in ``key`` order so a rank's own execution order is stable.
    """
    if world_size <= 1:
        return [list(items)]
    ordered = sorted(items, key=lambda it: (-float(cost(it)), key(it)))
    buckets: list[list[Any]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for item in ordered:
        target = min(range(world_size), key=lambda r: (loads[r], r))
        buckets[target].append(item)
        loads[target] += float(cost(item))
    return [sorted(b, key=key) for b in buckets]


def plan_fingerprint(keys: Sequence[str]) -> int:
    """Order-sensitive 60-bit fingerprint of a job-key list, for cross-rank agreement checks.

    Ranks build their assignment plan independently from the filesystem; anything that makes two
    ranks enumerate routes differently (glob order feeding ``enumerate_multi_root_routes``'s
    collision disambiguation, a partially-written dataset, a stale NFS listing) silently turns
    the plan into a non-partition -- some routes run twice, others never, and the merged summary
    still looks perfectly healthy. Comparing this number across ranks makes that loud instead.
    """
    digest = hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)
