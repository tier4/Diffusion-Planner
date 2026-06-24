# Autoresearch Control Panel

A single Gradio app that wraps the existing autoresearch CLI tools in **forms + Run/Stop +
live log**, with a central **asset library** so you register a model/dataset once and pick it
from dropdowns everywhere. It adds no domain logic — every Run shells out to the existing
`python -m <tool>` and streams its output.

## Launch

```bash
source .venv/bin/activate
python -m control_panel            # http://localhost:7888
# options: --port 7888  --host 0.0.0.0  --editor_port 7899  --share
```

## Workspace (register assets once)

The first tab is the asset library. Browse with the file explorer, name a path, and **Add as
→ Model / Guidance policy / Dataset / Reward config / Map / Run dir**. Set the global
`ego_shape` and default `output_dir`. Every workflow tab then offers these as **dropdowns**
(plus a `custom…` option). A registered **Model** entry can carry its `args_json` and a
`lora_dir`, which downstream tools pick up automatically.

**Load a training run:** point "Run dir" at a `run_experiment` output, **Scan run** to list its
`lora_epoch_*` / `merged.pth` / `best_model.pth`, then **Register epoch as Model** — it appears
in every model dropdown immediately, so you can train → register → Evaluate/Viz it in seconds.

The library persists to `~/.diffusion_planner_presets.json` (in `$HOME`, not the repo). On
first run it is seeded from the gitignored `control_panel/_dev_presets.py` (`DEV_LIBRARY`) if
present — temporary testing defaults; delete that file for a clean/shared setup.

## Tabs

| Tab | Wraps |
|---|---|
| **Workspace** | asset library + file explorer + training-run loader |
| **Train** | `run_experiment` (ranked-SFT) + live per-epoch metric table |
| **Evaluate** | `eval_det_avoidance` (+ summary viewer), guided `eval_policy_avoidance`, `valid_predictor` L2, `eval_border_distance`, `eval_detailed_metrics`; frozen baseline column |
| **Merge + Export** | `merge_lora` → `torch2onnx` |
| **PRiSM** | `disturb_and_replay` → `viz_p4_recovery` → `percentile_filter_perturbed` |
| **Reproducer / Viz** | `mine_collisions_reproducer` + `ghost_replay_openloop` / `compare_models_ghost` / `render_npz_dir` with a WebM/PNG viewer |
| **Scene Editor** | the Scene Branch Editor, run in-process on its own port and shown via iframe |

Two-model comparisons (ghost A/B) take two model dropdowns; guidance/exploration policies are a
first-class asset type usable in Eval and Viz.

## Jobs & Stop

Each Run launches a **detached** subprocess writing to
`~/.diffusion_planner_jobs/<ts>_<key>/run.log` with a `job.json`. Closing/restarting the panel
does not kill it — **Attach to latest** re-attaches and resumes tailing. **■ Stop** ends the
job (SIGTERM → SIGKILL) — a user-initiated stop, distinct from the never-auto-kill policy.

**Preview command** shows the exact argv so you can copy it to a terminal.

## Scene Editor (in-process)

The Scene Editor is the existing `scenario_generation.tools.scene_branch_editor`. Set its NPZ
dir (+ optional model/reward/map from the library) and click **Open / Reload editor**: it is
built and launched as a Gradio sub-server in this same process (`prevent_thread_lock=True`) on
`--editor_port`, and embedded in the tab via an iframe. Reload rebuilds it for a new dataset.
Reuses `scene_branch_editor.build_demo_from_paths`.

## Architecture

- `workflows.py` — declarative registry (`Workflow` + `ArgSpec`); `ArgSpec.shared` marks a field
  as library-sourced; `derive_from`/`derive_field` pull args.json/lora from a chosen model.
- `presets.py` — load/save the asset library (`~/.diffusion_planner_presets.json`).
- `runner.py` — detached launch, log tail/stream, `stop`, per-epoch metric parsing, job listing.
- `app.py` / `__main__.py` — the Gradio UI (form builder + Workspace + tab extras).
- `_dev_presets.py` — gitignored temporary testing seed.
