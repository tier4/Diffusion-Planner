"""Migration adapters: today's path-list recipes as manifest queries (spec §5b).

These are **migration adapters, not format contracts**. Each function here expresses
one of the shell-chained legacy scripts in `util_scripts/` (`create_train_set_path.py`,
`filter_json.py`, `filter_json_special.py`, `concat_data_list_jsons.py`) as a manifest
WHERE-clause (or a small internal handle carrying a window-function query), so that
today's training-subset recipes can be reproduced against a packed dataset version
without re-running the legacy scripts. `tests/test_dp_recipes.py` proves each adapter
equivalent to its legacy script as an ordered multiset of keys by actually invoking the
legacy scripts via subprocess on a synthetic fixture tree.
"""

from __future__ import annotations

import json
from pathlib import Path

from diffusion_planner.data_pipeline.reader import ShardReader

# Internal string protocol for `where` handles that need a window function (row_number/count)
# rather than a plain WHERE clause. Kept internal to this module; `keys_for` is the only
# place that interprets it.
_RANKED = "__RANKED__:"
_RANKED_PART = "__RANKED_PART__:"
_SEP = ":::"


def root_filter(rel_root: str) -> str:
    """Mirror `create_train_set_path.py <root_dir> --save_path out.json` (no `--no_exclude_skipped`).

    The legacy script globs `*.npz` under `root_dir` and, by default, drops frames whose
    sibling json has `is_skipped == True` (keeping `False` and unverifiable/missing sidecars).
    The packer already drops `is_skipped == True` frames at pack time (`PackOptions.drop_skipped`,
    default True) and never stores rejected frames, so `is_skipped IS NOT TRUE` here is a no-op
    on rows that reach the manifest — but it documents the same exclusion policy and keeps this
    filter correct if ever applied to an un-filtered relation.
    """
    return f"(rel_dir LIKE '{rel_root}/%' OR rel_dir = '{rel_root}') AND is_skipped IS NOT TRUE"


def every_n(where: str, n: int) -> str:
    """Mirror `filter_json.py <json> --num_filter N --num_filter_mode interval`.

    The legacy script keeps `files[::n]` of the (already sorted-by-glob) list — i.e. 0-based
    indices `0, n, 2n, …`. With a 1-based `row_number() OVER (ORDER BY key)` as `rn`, index `0`
    is `rn == 1`, so the kept set is `rn IN {1, n+1, 2n+1, …}`, i.e. `(rn - 1) % n == 0`.
    """
    return f"{_RANKED}{where}{_SEP}(rn - 1) % {n} = 0"


def head_n(where: str, n: int) -> str:
    """Mirror `filter_json.py <json> --num_filter N --num_filter_mode head`.

    The legacy script keeps `files[: len(files) // n]` — the first `count // n` elements using
    *floor* (integer) division, not ceiling. `count_all // n` below is DuckDB integer division,
    matching Python's `//` for the non-negative counts involved here.
    """
    return f"{_RANKED}{where}{_SEP}rn <= count_all // {n}"


def psim_per_location(where: str, n: int, component_k: int) -> str:
    """Mirror `filter_json_special.py <json> --num_filter N`.

    The legacy script buckets each path by "location" — parsed from the directory two levels
    above the npz file (`parts[-3]`), split on `_`, and joined back up to (excluding) the first
    literal `seed` token (e.g. `b_mobility_seed_200_poses_100` -> `b_mobility`) — then, per
    location bucket, sorts the paths and keeps the first `count // n` (floor division, like
    `head_n`). `component_k` is the 1-based `string_split(rel_dir, '/')` index of that directory
    component (e.g. `psim/<location>_seed_.../<bag>` -> `component_k=2`).
    """
    loc = f"regexp_extract(string_split(rel_dir, '/')[{component_k}], '^(.*)_seed_', 1)"
    return f"{_RANKED_PART}{loc}:{where}{_SEP}rn <= count_part // {n}"


def concat(*wheres: str) -> list[str]:
    """Mirror `concat_data_list_jsons.py <json>... --save_path out.json`: the list of WHEREs to
    concatenate with `keys_for_all` (UNION ALL semantics — duplicates preserved, order preserved).
    """
    return list(wheres)


def keys_for(reader: ShardReader, where: str) -> list[str]:
    """Resolve one `where` (or adapter handle) to its ordered list of keys (`ORDER BY key`)."""
    files = reader._files
    if where.startswith(_RANKED):
        inner, cond = where[len(_RANKED) :].split(_SEP, 1)
        sql = (
            "SELECT key FROM (SELECT key, row_number() OVER (ORDER BY key) AS rn, "
            "count(*) OVER () AS count_all FROM read_parquet(?) WHERE " + inner + ") "
            "WHERE " + cond + " ORDER BY key"
        )
    elif where.startswith(_RANKED_PART):
        rest = where[len(_RANKED_PART) :]
        loc, rest = rest.split(":", 1)
        inner, cond = rest.split(_SEP, 1)
        sql = (
            "SELECT key FROM (SELECT key, row_number() OVER (PARTITION BY "
            + loc
            + " ORDER BY key) AS rn, count(*) OVER (PARTITION BY "
            + loc
            + ") AS count_part FROM read_parquet(?) WHERE "
            + inner
            + ") WHERE "
            + cond
            + " ORDER BY key"
        )
    else:
        sql = f"SELECT key FROM read_parquet(?) WHERE {where} ORDER BY key"
    return [k for (k,) in reader._con.execute(sql, [files]).fetchall()]


def keys_for_all(reader: ShardReader, wheres: list[str]) -> list[str]:
    """UNION ALL semantics, like `concat_data_list_jsons.py`: concatenate, preserving duplicates."""
    out: list[str] = []
    for w in wheres:
        out += keys_for(reader, w)
    return out


def legacy_keys(path_list_json: Path, source_root: Path) -> list[str]:
    """Convert a legacy path-list json (absolute npz paths) to manifest keys, in list order."""
    root = Path(source_root).resolve()
    return [
        Path(p).resolve().relative_to(root).as_posix()[: -len(".npz")]
        for p in json.loads(Path(path_list_json).read_text())
    ]
