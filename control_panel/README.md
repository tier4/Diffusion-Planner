# Autoresearch Control Panel

A single Gradio app that wraps the existing autoresearch CLI tools in **forms + Run/Stop +
live log**, driven by a **structured workspace folder** so you select assets from dropdowns
instead of typing paths. It adds no domain logic — every Run shells out to the existing
`python -m <tool>` and streams its output.

## Launch

```bash
source .venv/bin/activate
python -m control_panel            # http://localhost:7888
# options: --port 7888  --host 0.0.0.0  --editor_port 7899  --share
```

## Workspace — the one thing you set

Everything is organized under a single **workspace root** with a fixed layout:

```
<workspace>/
  models/<name>/        best_model.pth + args.json   (🧠)
  loras/<name>/         adapter_config.json + weights (🧩)
  policies/<name>/      exploration_policy_config.json + .pth (🛰 guidance)
  configs/<name>.json   reward configs (⚙️)
  maps/<name>.osm       lanelet2 maps (🗺)
  datasets/scenes/<name>.json   individual-scene lists (🎬)
  datasets/routes/<name>/       contiguous per-frame NPZ dirs (📁)
  runs/<tool>/<name>/   eval / merge / mining outputs
```

In the **Workspace** tab:
- Set the **Workspace root** (📁 Browse), then **🆕 Create folders** (scaffold empty structure)
  or **🔄 Scan workspace** (auto-detect every asset by its on-disk signature; follows symlinks).
- Assets you can't move can be **symlinked** into the layout — Scan follows symlinks.
- A one-off asset outside the workspace can still be added with the Browse + Add buttons.

## Picking assets — one dropdown each

Every asset field across the tabs is a **single dropdown** with a type icon, listing the
scanned assets, plus:
- **(none)** for optional fields, and
- **➕ Browse to add…** — opens your OS file picker, registers the picked asset (named from its
  stem; a model auto-attaches its `args.json`), and selects it. The new asset instantly appears
  in that asset's dropdown in **every** tab.

Model + LoRA sit on one row. Scene datasets (individual) and route datasets (contiguous frames)
are **separate types**, so each tool only offers the right kind:
- eval / train / PRiSM / ghost → **scene** datasets
- mine collisions / render / scene editor → **route** datasets

When a tool **creates** a dataset (e.g. `disturb_and_replay`) it lands in
`datasets/scenes|routes/<name>/` and the dropdowns **auto-refresh** when the run finishes — no
manual scan needed.

## Tabs

| Tab | Tools |
|---|---|
| **Workspace** | set root, Create folders, Scan, register one-offs, load a training run → register a checkpoint |
| **Train** | `run_experiment` (ranked-SFT): just **Training scenes** + **Validation scenes** + live per-epoch metric table |
| **Evaluate → Metrics** | `eval_full_metrics` (all metrics on the det trajectory; optional 2nd model for A/B; **Render** dumps per-scene PNGs) — or tick **Use guidance policy** for `eval_policy_avoidance` |
| **Evaluate → L2 loss** | `valid_predictor` (DDP) |
| **Merge + Export** | `merge_lora` → `torch2onnx` |
| **Data generation → PRiSM** | one tab, three ordered steps: perturb → rank K candidates by reward → filter (keep top percentile, drop scenes no candidate improved) |
| **Render** | one tab, a mode dropdown: closed-loop A/B · open-loop A/B · generated candidates · render route/scenes. A/B sides each take model + optional LoRA + optional guidance policy (any combination) |
| **Scene Editor** | the Scene Branch Editor, launched as a subprocess (lanelet env) + embedded via iframe; export/save dirs pre-pointed into the workspace |

## Training config presets (Train → RSFT)

The training **mode** lives inside the GRPO/experiment config JSON you pick (the `🎛` dropdown),
not as a separate GUI toggle. Ready-made presets ship in `rlvr/configs/`:

| Preset | `ranked_sft_mode` | What it does |
|---|---|---|
| `rsft_curated_sft_ego_gt` | `curated` | **Plain SFT on the ego GT** — no generation, no ranking; target = each NPZ's `ego_agent_future`. lr 1e-4. |
| `rsft_curated_sft_ego_gt_lowlr` | `curated` | Same, gentler lr 5e-5 / 25 epochs (tighter L2 control). |
| `rsft_ranked_gt_neighbor` | `gt_neighbor` | Ranked-SFT: generate K, rank by reward, train on the winner; GT neighbor targets. |
| `rsft_ranked_baseline_neighbor` | `baseline_neighbor` | Ranked-SFT with base-model neighbor targets + neighbor reg (anti-drift). |

All set `sft_velocity_weight=true` (required for curated, else longitudinal L2 drifts) and
`lora_target=all` (train all blocks, ablate post-hoc). `ranked_sft_mode` also accepts the alias
`"sft"`/`"gt"` for the curated path. Use the optional **Epochs** field to override per run.

## Reward config & the SC gotcha

Avoidance (`sc`) metrics are only measured when the reward config enables static-collision
scoring. Use the **SC-enabled** config (`static_collision_enabled=true`) — otherwise
`sc_min_dist` reads 99 everywhere and the eval prints a NOTE telling you so.

## Jobs

Each Run launches a **detached** subprocess (survives closing the panel) writing to
`~/.diffusion_planner_jobs/<ts>_<key>/run.log`. **Attach to latest** re-attaches; **■ Stop**
ends the job (SIGTERM→SIGKILL). **Preview command** shows the exact argv.

## Architecture

- `workflows.py` — declarative registry (`Workflow` + `ArgSpec`). `shared` = which library type a
  field draws from; `auto` = output path under the workspace; `creates` = scenes/routes/None;
  `derive_from` pulls args.json/lora from a chosen model; `hidden` = fixed flag.
- `presets.py` — the asset library + `scan_workspace` / `create_workspace`.
- `runner.py` — detached launch, log tail/stream, Stop, per-epoch metric parse, `lanelet` env.
- `app.py` / `__main__.py` — the Gradio UI (form builder, inline-add dropdowns, tab extras).
- `_dev_presets.py` (gitignored) — points at a local dev workspace of symlinks for testing.
