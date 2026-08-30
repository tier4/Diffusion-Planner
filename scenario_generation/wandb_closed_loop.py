"""Build ``wandb.Table``s and charts for closed-loop evaluation results.

For each json_label (e.g. ``sites_sample``, ``sites_sample__noobj``,
``close_loop_devops_override_label``) this module produces:

- one **abs** ``wandb.Table`` (raw counts / fraction over the entire run),
  ready to be logged to W&B directly.
- one **per_1000steps** ``wandb.Table`` with a Run column, enabling cross-run
  comparison via W&B Custom Chart (shared axes, dynamic run filtering).
- one stacked-bar HTML panel per json_label (:func:`build_per_1000steps_stacked_panels`)
  showing the 5 count-metric columns normalized per 1000 steps.  The panel uses
  ECharts inlined via ``wandb.Html`` so it can be uploaded without any
  pre-registered Vega spec on the W&B backend.
- one **cross-run Custom Chart** per json_label (:func:`log_cross_run_charts`)
  that automatically appears in W&B Workspace with shared axes.

Cross-run Comparison (Fully Automatic):
--------------------------------------
1. Each run calls ``build_closed_loop_tables(by_json, run_name=run.name)``
2. Call ``log_cross_run_charts(run, by_json, ...)`` to log charts
3. Open W&B link → charts appear automatically with shared axes
4. Toggle runs on/off in the left panel to update the comparison

Prerequisites:
- Run ``python wandb_closed_loop_workspace.py`` once to create the Vega preset
- The preset is reusable across all runs
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import wandb
from scenario_generation.closed_loop_score_keys import extract_score

# (display column name, source key in a per-group summary dict)
_ABS_COLUMNS = [
    ("Group", "group"),
    ("Segments", "n_segments"),
    ("Steps", "total_steps"),
    ("Route completion (%)", "mean_route_completion"),
    ("Pass rate (%)", "pass_rate"),
    ("Fails", "fail_count"),
    ("Curb hits", "total_curb_hits"),
    ("Snaps", "total_snaps"),
    ("Red light", "total_red_light_violations"),
    ("Strong brakes", "total_strong_brakes"),
    ("Segs diverged", "n_segments_diverged"),
    ("Collisions", "total_collision_events"),
]

_PER_1000STEPS_COLUMNS = [
    ("Group", "group"),
    ("Segments", "n_segments"),
    ("Steps", "total_steps"),
    ("Route completion (%)", "mean_route_completion"),
    ("Pass rate (%)", "pass_rate"),
    ("Curb hits / 1k steps", "total_curb_hits"),
    ("Snaps / 1k steps", "total_snaps"),
    ("Red light / 1k steps", "total_red_light_violations"),
    ("Strong brakes / 1k steps", "total_strong_brakes"),
    ("Segs diverged / 1k steps", "n_segments_diverged"),
    ("Collisions / 1k steps", "total_collision_events"),
]


def _short_label(group_key: str) -> str:
    """``sites_sample/group_a`` → ``group_a``. ``group_a`` → ``group_a``."""
    return group_key.split("/", 1)[1] if "/" in group_key else group_key


def _add_run_column(table: wandb.Table, run_name: str) -> wandb.Table:
    """Prepend a 'Run' column to a table for cross-run comparison."""
    new_data = [[run_name] + list(row) for row in table.data]
    return wandb.Table(columns=["Run", *table.columns], data=new_data)


def _segment_weighted_mean(values: list[dict], key: str) -> float:
    """Segment-weighted mean of ``key`` across groups, e.g. mean_route_completion or pass_rate."""
    n_segments = sum(int(s.get("n_segments", 0) or 0) for s in values)
    if n_segments == 0:
        return 0.0
    total = sum(float(s.get(key, 0.0) or 0.0) * int(s.get("n_segments", 0) or 0) for s in values)
    return total / n_segments


def _abs_value(source_key: str, summary: dict):
    """Numeric value for an abs column.

    Looks up the raw key first so the cross-group aggregate row (which is a
    flat dict like ``{"total_curb_hits": 39, ...}``) is read correctly.
    Falls back to ``extract_score`` for raw per-group summaries whose
    headline numbers live in nested categories like ``road_border``. A key the
    summary simply doesn't carry -- ``pass_rate`` / ``fail_count`` on a run with no
    pass condition -- resolves to None there and reads as 0, so an old
    ``summary.json`` still logs instead of blowing up the upload.
    """
    # Float fields: direct lookup or extract_score fallback.
    if source_key in ("mean_route_completion", "pass_rate"):
        if source_key in summary:
            val = summary[source_key]
            return float(val if isinstance(val, (int, float)) else 0.0)
        return float(extract_score(summary, source_key) or 0.0)

    # Int fields.
    if source_key in ("n_segments", "total_steps"):
        return int(summary.get(source_key, 0) or 0)
    if source_key in summary:
        return int(summary[source_key] or 0)
    return int(extract_score(summary, source_key) or 0)


def _per_1000steps_value(source_key: str, summary: dict) -> float:
    """Counts normalized per 1000 steps (or per 1000 segments for ``n_segments_diverged``)."""
    # Pass-through fields that are already a fraction or count, not a rate.
    if source_key in ("n_segments", "total_steps", "mean_route_completion", "pass_rate"):
        return _abs_value(source_key, summary)
    denom_key = "n_segments" if source_key == "n_segments_diverged" else "total_steps"
    denom = int(summary.get(denom_key, 0) or 0)
    if denom <= 0:
        return 0.0
    raw = summary.get(source_key)
    if raw is None:
        raw = extract_score(summary, source_key)
    return int(raw or 0) / denom * 1000.0


def _aggregate(group_summaries: dict[str, dict]) -> dict:
    """Cross-group aggregate, same shape as a per-group summary dict.

    ``__noobj`` groups are excluded from collision sums (they're 0 by
    construction in the no-object ablation). ``route_completion`` and
    ``pass_rate`` are segment-weighted means, not plain averages.
    """
    if not group_summaries:
        return {}

    values = list(group_summaries.values())
    objects_only_values = [s for k, s in group_summaries.items() if "__noobj/" not in k]

    n_segments = sum(int(s.get("n_segments", 0) or 0) for s in values)

    agg: dict = {
        "n_groups": len(values),
        "n_segments": n_segments,
        "total_steps": sum(int(s.get("total_steps", 0) or 0) for s in values),
        "mean_route_completion": _segment_weighted_mean(values, "mean_route_completion"),
        "pass_rate": _segment_weighted_mean(values, "pass_rate"),
        "fail_count": sum(int(s.get("fail_count", 0) or 0) for s in values),
    }
    for k in (
        "total_curb_hits",
        "total_snaps",
        "total_red_light_violations",
        "total_strong_brakes",
        "n_segments_diverged",
    ):
        agg[k] = sum(int(extract_score(s, k) or 0) for s in values)
    agg["total_collision_events"] = sum(
        int(extract_score(s, "total_collision_events") or 0) for s in objects_only_values
    )
    return agg


def _build_table(
    json_label: str,
    kind: str,  # "abs" or "per_1000steps"
    group_summaries: dict[str, dict],
) -> wandb.Table:
    cols = _ABS_COLUMNS if kind == "abs" else _PER_1000STEPS_COLUMNS
    value_fn = _abs_value if kind == "abs" else _per_1000steps_value

    rows: list[list] = []
    for group_key in sorted(group_summaries.keys()):
        summary = group_summaries[group_key]
        rows.append(
            [
                _short_label(group_key) if src == "group" else value_fn(src, summary)
                for _, src in cols
            ]
        )

    all_agg = _aggregate(group_summaries)
    rows.append(["All" if src == "group" else value_fn(src, all_agg) for _, src in cols])

    return wandb.Table(columns=[c[0] for c in cols], data=rows)


def build_closed_loop_tables(
    by_json: dict[str, dict[str, dict]],
    *,
    run_name: str | None = None,
) -> dict[str, wandb.Table]:
    """Build abs table for each json_label.

    ``by_json`` maps ``json_label`` → ``group_key`` → per-group summary dict.

    Returns:
        ``Closed-Loop-{json_label}/metrics``: abs table with raw counts and route completion
    """
    if run_name is None:
        run_name = wandb.run.name if wandb.run is not None else "unknown"

    out: dict[str, wandb.Table] = {}

    for json_label, group_summaries in sorted(by_json.items()):
        abs_table = _build_table(json_label, "abs", group_summaries)
        out[f"Closed-Loop-{json_label}/metrics"] = abs_table

    return out


# Metrics stacked into the per-1000-steps bar chart. Order is preserved so
# colors stay stable across runs in the rendered ECharts panel.
_STACKED_METRICS = (
    ("Curb hits / 1k steps", "#4C78A8"),
    ("Snaps / 1k steps", "#F58518"),
    ("Red light / 1k steps", "#E45756"),
    ("Strong brakes / 1k steps", "#72B7B2"),
    ("Collisions / 1k steps", "#EECA3B"),
)


# Metrics for cross-run stacked bar chart
_CROSS_RUN_METRICS = (
    ("Curb hits / 1k steps", "curb_hits"),
    ("Snaps / 1k steps", "snaps"),
    ("Red light / 1k steps", "red_light"),
    ("Strong brakes / 1k steps", "strong_brakes"),
    ("Collisions / 1k steps", "collisions"),
)


def _build_cross_run_vega_spec() -> dict:
    """Build the Vega-Lite spec for cross-run grouped stacked bar chart.

    Uses W&B template variables (${field:...}) for dynamic data binding.
    Supports clicking legend items to show/hide individual event types.
    """
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Closed-loop events per 1k steps, grouped by run and stacked by event type.",
        "data": {"name": "wandb"},
        "params": [
            {
                "name": "selected_event_types",
                "select": {"type": "point", "fields": ["event_type"]},
                "bind": "legend",
            }
        ],
        "transform": [
            {"filter": "datum['${field:group}'] !== 'All'"},
            {
                "joinaggregate": [
                    {
                        "op": "distinct",
                        "field": "${field:run}",
                        "as": "visible_run_count",
                    }
                ],
            },
            {
                "calculate": "max(14, min(20, 30 / datum.visible_run_count))",
                "as": "bar_thickness",
            },
            {
                "fold": [
                    "${field:curb_hits}",
                    "${field:snaps}",
                    "${field:red_light}",
                    "${field:strong_brakes}",
                    "${field:collisions}",
                ],
                "as": ["event_key", "event_value"],
            },
            {
                "calculate": "datum.event_key === '${field:curb_hits}' ? 'Curb hits' : datum.event_key === '${field:snaps}' ? 'Snaps' : datum.event_key === '${field:red_light}' ? 'Red light' : datum.event_key === '${field:strong_brakes}' ? 'Strong brakes' : 'Collisions'",
                "as": "event_type",
            },
            {"filter": {"param": "selected_event_types"}},
        ],
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "y": {
                "field": "${field:group}",
                "type": "nominal",
                "title": "Group",
                "axis": {"labelLimit": 320},
            },
            "yOffset": {"field": "${field:run}", "type": "nominal"},
            "x": {
                "aggregate": "sum",
                "field": "event_value",
                "type": "quantitative",
                "stack": "zero",
                "title": "Events / 1k steps",
                "scale": {"zero": True},
            },
            "size": {
                "field": "bar_thickness",
                "type": "quantitative",
                "scale": None,
                "legend": None,
            },
            "color": {
                "field": "event_type",
                "type": "nominal",
                "title": "Event type",
                "scale": {
                    "domain": [
                        "Curb hits",
                        "Snaps",
                        "Red light",
                        "Strong brakes",
                        "Collisions",
                    ],
                    "range": ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#EECA3B"],
                },
            },
            "order": {
                "field": "event_type",
                "sort": [
                    "Curb hits",
                    "Snaps",
                    "Red light",
                    "Strong brakes",
                    "Collisions",
                ],
            },
            "tooltip": [
                {"field": "${field:group}", "type": "nominal", "title": "Group"},
                {"field": "${field:run}", "type": "nominal", "title": "Run"},
                {"field": "event_type", "type": "nominal", "title": "Event"},
                {
                    "aggregate": "sum",
                    "field": "event_value",
                    "type": "quantitative",
                    "title": "Events / 1k steps",
                    "format": ".3f",
                },
            ],
        },
        "height": {"step": 42},
    }


def _build_table_vega_spec() -> dict:
    """Build a table-like Vega-Lite spec for cross-run metrics comparison.

    Each Group × Run pair occupies one row. The visible row label contains only
    the Group name, while the row background color identifies the Run. The full
    Run name remains available in the legend and tooltip.

    W&B replaces ``${field:...}`` expressions using the mapping passed to
    ``wandb.plot_table(fields=...)``.
    """
    metric_field_ids = [source_key for _, source_key in _ABS_COLUMNS[1:]]
    metric_titles = [display_name for display_name, _ in _ABS_COLUMNS[1:]]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": ("Closed-loop metrics table with colored rows for cross-run comparison."),
        "data": {"name": "wandb"},
        "transform": [
            # Build an internal unique row key. The axis label later strips the
            # Run suffix so long Run names are not rendered as row labels.
            {
                "calculate": ("datum['${field:group}'] + '|||' + datum['${field:run}']"),
                "as": "row_key",
            },
            # Keep normal groups alphabetically ordered and put All last.
            {
                "calculate": (
                    "(datum['${field:group}'] === 'All' "
                    "? '~~~~All' : datum['${field:group}']) "
                    "+ '|' + datum['${field:run}']"
                ),
                "as": "row_sort",
            },
            # Convert the fixed metric columns into table cells.
            # Group and Run are dimensions, so they must not be folded.
            {
                "fold": [f"${{field:{field_id}}}" for field_id in metric_field_ids],
                "as": ["column_name", "column_value"],
            },
            # Format integer counts without decimals and other numeric values
            # to three decimal places.
            {
                "calculate": (
                    "!isValid(datum.column_value) "
                    "? '—' "
                    ": isNumber(datum.column_value) "
                    "? (datum.column_value % 1 === 0 "
                    "   ? format(datum.column_value, ',.0f') "
                    "   : format(datum.column_value, ',.3f')) "
                    ": datum.column_value"
                ),
                "as": "display_value",
            },
        ],
        # x/y/tooltip are shared by both the background and text layers.
        "encoding": {
            "x": {
                "field": "column_name",
                "type": "nominal",
                "title": None,
                "sort": metric_titles,
                "axis": {
                    "orient": "top",
                    "labelAngle": 0,
                    "labelLimit": 180,
                    "labelPadding": 8,
                    "title": None,
                },
            },
            "y": {
                "field": "row_key",
                "type": "nominal",
                "title": "Group",
                "sort": {
                    "field": "row_sort",
                    "op": "min",
                    "order": "ascending",
                },
                "axis": {
                    # "group|||very-long-run-name" -> "group"
                    "labelExpr": "split(datum.label, '|||')[0]",
                    "labelLimit": 280,
                    "labelPadding": 8,
                    "titlePadding": 12,
                },
            },
            "tooltip": [
                {
                    "field": "${field:group}",
                    "type": "nominal",
                    "title": "Group",
                },
                {
                    "field": "${field:run}",
                    "type": "nominal",
                    "title": "Run",
                },
                {
                    "field": "column_name",
                    "type": "nominal",
                    "title": "Metric",
                },
                {
                    "field": "display_value",
                    "type": "nominal",
                    "title": "Value",
                },
            ],
        },
        "layer": [
            {
                # One lightly colored cell background per Run.
                "mark": {
                    "type": "rect",
                    "opacity": 0.16,
                },
                "encoding": {
                    "color": {
                        "field": "${field:run}",
                        "type": "nominal",
                        "title": "Run",
                        "scale": {
                            "scheme": "tableau10",
                        },
                        "legend": {
                            "labelLimit": 180,
                            "symbolType": "square",
                            "symbolOpacity": 0.7,
                            "titleLimit": 180,
                        },
                    },
                },
            },
            {
                # Text is deliberately dark rather than colored, preserving
                # readability over the lightly colored row background.
                "mark": {
                    "type": "text",
                    "align": "center",
                    "baseline": "middle",
                    "fontSize": 12,
                    "color": "#222222",
                },
                "encoding": {
                    "text": {
                        "field": "display_value",
                        "type": "nominal",
                    },
                },
            },
        ],
        "height": {
            "step": 26,
        },
        "width": {
            "step": 120,
        },
        "config": {
            "view": {
                "stroke": None,
            },
            "axis": {
                "grid": False,
                "domain": False,
                "tickSize": 0,
            },
        },
    }


def log_metrics_tables(
    run: wandb.sdk.wandb_run.Run,
    by_json: dict[str, dict[str, dict]],
) -> None:
    entity = getattr(run, "entity", None)
    if not entity:
        raise ValueError("W&B run.entity is unavailable")

    # create_custom_chart does not update an existing preset.
    # Increment this suffix whenever the Vega spec changes.
    preset_name = "closed_loop_metrics_table"
    vega_spec_name = f"{entity}/{preset_name}"

    try:
        wandb.Api().create_custom_chart(
            entity=entity,
            name=preset_name,
            display_name="Closed-Loop Metrics Table",
            spec_type="vega2",
            access="private",
            spec=_build_table_vega_spec(),
        )
        print(f"wandb: created custom chart preset '{vega_spec_name}'")
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" in message or "duplicate" in message:
            print(f"wandb: using existing preset '{vega_spec_name}'")
        else:
            raise RuntimeError(f"Failed to create W&B preset '{vega_spec_name}'") from exc

    fields = {
        "run": "Run",
        "group": "Group",
        **{source: display for display, source in _ABS_COLUMNS[1:]},
    }

    for json_label in sorted(by_json):
        abs_table = _build_table(
            json_label,
            "abs",
            by_json[json_label],
        )
        table_with_run = _add_run_column(abs_table, run.name)

        chart = wandb.plot_table(
            vega_spec_name=vega_spec_name,
            data_table=table_with_run,
            fields=fields,
            split_table=True,
        )
        run.log(
            {
                f"Closed-Loop-{json_label}/metrics_table": chart,
            }
        )
        print(f"wandb: logged Closed-Loop-{json_label}/metrics_table")


def log_cross_run_charts(
    run: wandb.sdk.wandb_run.Run,
    by_json: dict[str, dict[str, dict]],
) -> None:
    """Log cross-run Custom Charts for grouped stacked bar comparison.

    This function:
    1. Creates a Vega preset (if not exists) for cross-run stacked bar
    2. Logs a Custom Chart for each json_label

    Args:
        run: W&B run instance (from wandb.init() or passed in)
        by_json: Mapping of json_label -> group_key -> summary dict
    """
    entity = getattr(run, "entity", None) or "unknown"

    # Create the Vega preset once
    vega_spec_name = f"{entity}/closed_loop_cross_run_stacked_bar"
    try:
        api = wandb.Api()
        api.create_custom_chart(
            entity=entity,
            name="closed_loop_cross_run_stacked_bar",
            display_name="Closed-Loop Cross-Run Stacked Bar",
            spec_type="vega2",
            access="private",
            spec=_build_cross_run_vega_spec(),
        )
        print(f"wandb: created custom chart preset '{vega_spec_name}'")
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print(f"wandb: custom chart preset '{vega_spec_name}' already exists")
        else:
            print(f"wandb: warning - failed to create custom chart preset: {e}")

    # Build fields mapping: Vega field name -> table column name
    chart_fields = {
        "run": "Run",
        "group": "Group",
    }
    for display_name, field_key in _CROSS_RUN_METRICS:
        chart_fields[field_key] = display_name

    # Log a Custom Chart for each json_label
    for json_label in sorted(by_json.keys()):
        table = _build_table(json_label, "per_1000steps", by_json[json_label])
        table_with_run = _add_run_column(table, run.name)

        chart = wandb.plot_table(
            vega_spec_name=vega_spec_name,
            data_table=table_with_run,
            fields=chart_fields,
            split_table=True,
        )
        run.log({f"Closed-Loop-{json_label}/cross_run_chart": chart})
        print(f"wandb: logged Closed-Loop-{json_label}/cross_run_chart")


def _cfg_field(cfg: object | None, field: str) -> str:
    """Read ``field`` off a config that may be a dataclass, a mapping, or ``None``.

    ``run_all_groups_closed_loop`` and ``train.py`` both hand us a
    ``ClosedLoopConfig`` dataclass, while older callers passed a plain dict, so
    read through both rather than assuming ``.get``.
    """
    if cfg is None:
        return ""
    if isinstance(cfg, Mapping):
        return cfg.get(field) or ""
    return getattr(cfg, field, "") or ""


def log_closed_loop_to_wandb(
    cfg: "object | dict | None",
    group_names: list[str],
    group_summaries: dict[str, dict],
    run: "wandb.sdk.wandb_run.Run | None" = None,
) -> None:
    """Push per-group closed-loop scalar metrics + Custom Charts to W&B.

    Reuses ``run`` if given, else starts its own.
    Sets up W&B Custom Chart presets for cross-run comparison.

    Args:
        cfg: ``ClosedLoopConfig`` (or a dict) carrying ``wandb_project_name`` and
             ``exp_name``. If None, or if ``wandb_project_name`` is empty and no
             ``run`` was supplied, the upload is skipped.
        group_names: List of group keys.
        group_summaries: Dict mapping group key -> summary dict.
        run: W&B run instance. If None, starts a new one.
    """
    if not group_summaries:
        return

    if run is None:
        project = _cfg_field(cfg, "wandb_project_name")
        if not project:
            # ``wandb_project_name`` is documented as "empty = disabled". Without this
            # guard we would wandb.init(project=None) and either create a run in the
            # default project or block on a missing API key -- after the evaluation
            # has already finished, which is the worst possible place to fail.
            print("wandb: wandb_project_name is empty, skipping closed-loop upload")
            return
        run = wandb.init(project=project, name=_cfg_field(cfg, "exp_name") or None)
        own_run = True
    else:
        own_run = False

    try:
        by_json: dict[str, dict[str, dict]] = {}
        for key in group_names:
            summary = group_summaries[key]
            if "__noobj/" in key:
                json_label = key.split("__noobj/", 1)[0] + "__noobj"
            else:
                json_label = key.split("/", 1)[0]
            by_json.setdefault(json_label, {})[key] = summary

        log_metrics_tables(run, by_json)
        log_cross_run_charts(run, by_json)
    finally:
        if own_run:
            wandb.finish()
