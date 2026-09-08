"""Shared DuckDB query helper — wraps raw DuckDB calls with `;` guard and
parser/binder/catalog error → PlanError conversion (spec §5b)."""

from __future__ import annotations

import duckdb
import pyarrow as pa

from diffusion_planner.data_pipeline.errors import PlanError


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pa.Table:
    """Execute a single read-only SQL statement and return its result as an Arrow table.

    Raises:
        PlanError: if `sql` contains `;` or if DuckDB raises a parser/binder/catalog error
    """
    if ";" in sql:
        raise PlanError("SQL must be a single statement (no ';')")
    try:
        return con.execute(sql, params or []).arrow().read_all()
    except (duckdb.ParserException, duckdb.BinderException, duckdb.CatalogException) as e:
        raise PlanError(
            f'invalid SQL ({e}); double-quote reserved column names, e.g. "offset"'
        ) from e
