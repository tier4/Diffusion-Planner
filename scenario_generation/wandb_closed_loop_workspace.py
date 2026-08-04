"""Build (or refresh) a curated W&B workspace view for closed-loop eval.

The scalars/media/table this project logs under ``closed_loop_scores/``,
``closed_loop_overview/``, ``closed_loop_media/``, and ``closed_loop_episodes/`` (see
:mod:`scenario_generation.wandb_closed_loop`) are already namespaced into sections, but
W&B's DEFAULT auto-generated workspace still renders one panel per distinct key — for
N sites x ~8 score metrics that's still dozens of small single-line panels, one per
(metric, site) pair, with no overlay.

This module uses the ``wandb-workspaces`` SDK to define an explicit, curated workspace
instead: one LinePlot PER METRIC with every site overlaid as its own colored line (via
``metric_regex``, so it doesn't need to know site names), one MediaBrowser per site (its
colormap gallery + video), and one table panel for the combined episode table. Built with
``auto_generate_panels=True`` (default) so every other logged key (train/valid loss, lr,
epdms, replan consistency, ...) still gets its usual auto panel alongside these curated
sections — nothing from the default workspace is lost. Pass ``include_auto_panels=False``
for a closed-loop-only view.

This is a project-level view, not a per-run artifact — run it ONCE (or whenever the set of
score metrics changes) via the CLI below, not from inside the training loop. The resulting
``.url`` is a saved, shareable, LIVE view: it re-renders with each new ``wandb.log()`` call
under the same project, exactly like the default workspace does.

Example::

    python -m scenario_generation.wandb_closed_loop_workspace \\
        --entity advanced-technology-department \\
        --project Diffusion-Planner-smoke-test \\
        --site_names odaiba takanawa
"""

from __future__ import annotations

import argparse

from scenario_generation.closed_loop_score_keys import (
    COMPARISON_OVERVIEW_SUM_KEYS,
    COMPARISON_SCORE_KEYS,
    OBJECTS_ONLY_OVERVIEW_SUM_KEYS,
    OBJECTS_ONLY_SCORE_KEYS,
)

# Panel groups for the objects-ablation dashboard, derived from the single source of truth in
# closed_loop_score_keys (dependency-free, so importing it here doesn't pull in the heavier
# torch/matplotlib deps scenario_generation.wandb_closed_loop needs):
# - COMPARISON: meaningful in both objects and empty-world ("__noobj") mode — one panel per
#   metric, objects and noobj overlaid as separate lines via metric_regex (no site enumeration
#   needed, so a newly added site is picked up automatically).
# - OBJECTS-ONLY: structurally 0 with no traffic to react to (collision counts) — plotted with
#   an explicit y=[...] list of ONLY the objects labels, so the noobj line (always 0) doesn't
#   show up as a flat, meaningless addition to the panel.
#
# The overview panel list additionally includes "route_completion" -- build_sites_aggregate_log
# computes it as its own segment-weighted-average key, not part of the generic sum loop
# COMPARISON_OVERVIEW_SUM_KEYS drives.
_COMPARISON_OVERVIEW_KEYS = ("route_completion", *COMPARISON_OVERVIEW_SUM_KEYS)
_OBJECTS_ONLY_OVERVIEW_KEYS = OBJECTS_ONLY_OVERVIEW_SUM_KEYS


def build_closed_loop_workspace(
    entity: str,
    project: str,
    *,
    site_names: list[str],
    name: str = "Closed-Loop Dashboard",
    include_noobj: bool = True,
    include_auto_panels: bool = True,
) -> str:
    """Create/save a curated closed-loop workspace view and return its URL.

    ``site_names`` are BASE names (no ``__noobj`` suffix, e.g. ``["odaiba", "takanawa"]``) —
    this builder expands each into its objects label (unchanged) and, when ``include_noobj``,
    its empty-world-ablation label (``{site}__noobj``). Only the per-site media galleries need
    those exact labels (W&B media panels take explicit
    keys, no regex); the comparison score line plots use ``metric_regex`` and pick up any site
    automatically, present or added later. Set ``include_noobj=False`` for an objects-only
    dashboard (e.g. before any ablation run exists for this project).
    """
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    objects_labels = list(site_names)
    noobj_labels = [f"{n}__noobj" for n in site_names] if include_noobj else []
    all_labels = objects_labels + noobj_labels

    overview_panels = (
        [
            wr.LinePlot(title=key, y=[f"closed_loop_overview/{key}"])
            for key in _COMPARISON_OVERVIEW_KEYS
        ]
        + [
            wr.LinePlot(title=key, y=[f"closed_loop_overview/{key}"])
            for key in _OBJECTS_ONLY_OVERVIEW_KEYS
        ]
        + [
            wr.LinePlot(
                title="n_sites / n_segments",
                y=["closed_loop_overview/n_sites", "closed_loop_overview/n_segments"],
            ),
        ]
    )

    # Comparison group: one panel PER METRIC, every site+mode overlaid as its own line
    # (metric_regex matches "closed_loop_scores/{metric}/{any_label}" without needing to
    # enumerate labels) — this is the key fix over the default workspace's
    # one-tiny-panel-per-(metric,site) layout, AND it's what makes objects vs noobj directly
    # comparable in the same chart.
    comparison_score_panels = [
        wr.LinePlot(title=metric, metric_regex=rf"^closed_loop_scores/{metric}/.*$")
        for metric in COMPARISON_SCORE_KEYS
    ]
    # Objects-only group: explicit y=[...] over objects_labels ONLY (no regex) — a noobj line
    # here would just be a flat, meaningless 0 (no traffic to collide with by construction).
    objects_only_score_panels = [
        wr.LinePlot(
            title=metric,
            y=[f"closed_loop_scores/{metric}/{label}" for label in objects_labels],
        )
        for metric in OBJECTS_ONLY_SCORE_KEYS
    ]

    # Two SEPARATE sections (not just two panel types in one section): the colormap gallery
    # (an indexed LIST of 5 images) and the video (a single file) are different W&B media
    # shapes — combining both key types in one MediaBrowser with gallery_axis="index" silently
    # drops the video (the index axis only applies to the list-shaped key). Grouping all
    # labels' overlays together and all labels' videos together (rather than interleaving)
    # also keeps same-purpose panels next to each other.
    overlay_panels = [
        wr.MediaBrowser(
            title=label,
            media_keys=[f"closed_loop_media/{label}"],
            mode="gallery",
            gallery_axis="index",
        )
        for label in all_labels
    ]
    video_panels = [
        wr.MediaBrowser(title=label, media_keys=[f"closed_loop_media/{label}__video"])
        for label in all_labels
    ]

    episodes_panels = [
        wr.WeavePanelSummaryTable(
            table_name="closed_loop_episodes/all", layout=wr.Layout(w=24, h=16)
        ),
    ]

    # Plain training-loop metrics (every epoch, not just closed-loop-firing epochs) — placed
    # FIRST so loss/lr trends are visible above the closed-loop sections, in one unified view
    # instead of needing the separate default auto-generated workspace for these.
    training_panels = [
        wr.LinePlot(title="train_loss", metric_regex=r"^train_loss/.*$"),
        wr.LinePlot(title="valid_loss", metric_regex=r"^valid_loss/.*$"),
        wr.LinePlot(title="learning_rate", metric_regex=r"^lr/.*$"),
    ]

    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        auto_generate_panels=include_auto_panels,
        sections=[
            ws.Section(name="Training", panels=training_panels, is_open=True, pinned=True),
            ws.Section(name="Overview", panels=overview_panels, is_open=True, pinned=True),
            ws.Section(
                name="Scores: comparison (objects vs no-objects)"
                if include_noobj
                else "Scores: comparison",
                panels=comparison_score_panels,
                is_open=True,
            ),
            ws.Section(
                name="Scores: objects-only (collision)",
                panels=objects_only_score_panels,
                is_open=True,
            ),
            ws.Section(name="Trajectory Overlay (per site)", panels=overlay_panels, is_open=False),
            ws.Section(name="Video (per site)", panels=video_panels, is_open=False),
            ws.Section(
                name="Episodes (all sites, one table)", panels=episodes_panels, is_open=False
            ),
        ],
    )
    workspace.save()
    return workspace.url


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--site_names",
        nargs="+",
        required=True,
        help="BASE site names, no '__noobj' suffix (e.g. odaiba takanawa) — the builder expands "
        "each into its objects/no-objects labels itself (see --include_noobj)",
    )
    parser.add_argument("--name", default="Closed-Loop Dashboard")
    parser.add_argument(
        "--no_noobj",
        action="store_true",
        help="objects-only dashboard: skip the empty-world-ablation ('__noobj') labels entirely",
    )
    parser.add_argument(
        "--no_auto_panels",
        action="store_true",
        help="closed-loop-only view: skip the auto-generated panel for every other logged key "
        "(train/valid loss, lr, epdms, replan consistency, ...)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    url = build_closed_loop_workspace(
        args.entity,
        args.project,
        site_names=args.site_names,
        name=args.name,
        include_noobj=not args.no_noobj,
        include_auto_panels=not args.no_auto_panels,
    )
    print(f"Saved workspace view: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
