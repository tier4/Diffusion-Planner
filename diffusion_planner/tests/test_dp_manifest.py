import duckdb
import pyarrow as pa
from diffusion_planner.data_pipeline import manifest as M
from diffusion_planner.data_pipeline.sidecar import SIDECAR_FIELDS, parse_sidecar


def _rows():
    sc_full = parse_sidecar(
        b'{"is_skipped": false, "timestamp": 5, "project_id": "p", "neighbor_ids": ["a"]}'
    )
    sc_none = parse_sidecar(None)
    sc_skip = parse_sidecar(b'{"is_skipped": true, "timestamp": 6}')
    mk = lambda k, i, sc: M.ManifestRow(
        k, "part/a", "part/a/d", 0, i, 512 + i * 1024, 100, b"\x01" * 32, b"\x02" * 32, sc
    )
    return [
        mk("part/a/d/k2", 1, sc_none),
        mk("part/a/d/k1", 0, sc_full),
        mk("part/a/d/k3", 2, sc_skip),
    ]


def test_schema_is_index_plus_sidecar_only():
    names = M.manifest_schema().names
    assert names[:9] == [n for n, _ in M.INDEX_FIELDS]
    assert names[9:] == [n for n, _ in SIDECAR_FIELDS]


def test_table_sorted_by_key_and_nulls(tmp_path):
    t = M.rows_to_table(_rows())
    assert t.column("key").to_pylist() == ["part/a/d/k1", "part/a/d/k2", "part/a/d/k3"]
    assert t.column("project_id").to_pylist() == ["p", None, None]
    assert t.column("neighbor_count").to_pylist() == [1, None, None]
    p = tmp_path / "m.parquet"
    M.write_manifest(p, t, {"format_version": "1", "data_rev": "abc"})
    assert M.read_metadata(p) == {"format_version": "1", "data_rev": "abc"}
    assert M.read_manifest(p, columns=M.PLANNING_COLUMNS).num_columns == 5
    assert not list(tmp_path.glob("*.tmp"))


def test_skip_predicate_keeps_null_rows(tmp_path):
    p = tmp_path / "m.parquet"
    M.write_manifest(p, M.rows_to_table(_rows()), {})
    kept = duckdb.sql(
        f"SELECT key FROM read_parquet('{p}') WHERE is_skipped IS NOT TRUE ORDER BY key"
    ).fetchall()
    assert [k for (k,) in kept] == ["part/a/d/k1", "part/a/d/k2"]


def test_meta_rev_depends_only_on_sidecar_values():
    rows = _rows()
    a = M.meta_rev(M.rows_to_table(rows))
    rows[0] = M.ManifestRow(*rows[0].__dict__.values())  # identical copy
    assert M.meta_rev(M.rows_to_table(rows)) == a and len(a) == 16
    changed = [
        M.ManifestRow(**{**r.__dict__, "sidecar": {**r.sidecar, "is_skipped": True}})
        if r.key.endswith("k1")
        else r
        for r in rows
    ]
    assert M.meta_rev(M.rows_to_table(changed)) != a
    moved = [M.ManifestRow(**{**r.__dict__, "offset": r.offset + 4096}) for r in rows]
    assert M.meta_rev(M.rows_to_table(moved)) == a  # index columns do not affect meta_rev
