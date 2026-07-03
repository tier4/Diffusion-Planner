# R2LPL Lifelong Workflow

Canonical entrypoint: `python -m rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds`

This workflow is the repo's intended single-command R2LPL-style loop for route-based replay mining, repair-target generation, replay-memory update, and training.

## Where To Look

When the autoresearch skill needs to recover this workflow quickly, start here:

- orchestration entrypoint:
  `rlvr/autoresearch/tools/run_lifelong_r2lpl_rounds.py`
- open-loop classification and saved-prediction reuse:
  `rlvr/autoresearch/tools/classify_scene_failures.py`
- event mining and closed-loop reproduction:
  `rlvr/autoresearch/tools/mine_credit_window_scenes.py`
- reproducer mechanics:
  `scenario_generation/reproducer_rollout.py`

For manual event inspection renders, `python -m rlvr.autoresearch.tools.render_reproducer_segment`
now supports both `--tracker_mode perfect` and `--tracker_mode mpc`. The R2LPL mining workflow
also accepts `perception_reproducer.tracker_mode`, and now defaults to MPC tracking unless
explicitly changed.

## Single-Entry Contract

Normal usage should go through the single orchestrator and provide:

- `model_path`
- one route source: `scene_list` or `route_root`
- optional `saved_predictions_dir`
- `workflow_config`
- `training_config`
- `output_dir`

The orchestrator owns classify -> event mining -> reproduction -> repair -> replay -> train.

## Intent

The workflow is deliberately split into four distinct semantics:

1. Open-loop event discovery on deterministic predicted trajectories.
2. Closed-loop perception reproduction judged on realized ego-state violations.
3. Repair-window export ending at the first realized violation.
4. One-best-valid repair target selection for training.

These are separate on purpose. The open-loop planner prediction is used only to find candidate source events. The closed-loop reproducer is used to verify whether an issue is actually realized when the model is driving.

## Event Semantics

- A classified route can contain many violating timestamps for the same event.
- Those timestamps are collapsed into distinct events before reproduction.
- One event triggers exactly one reproducer rollout.
- A scene is one NPZ. Generated scene count is the number of exported NPZs, so it should be a multiple of reproduced event count when every event exports the same repair-window length.
- Reporting should distinguish:
  - violating timestamps
  - selected open-loop events
  - reproduced events
  - generated scene count

## Anchor And Rollout

There are three different horizons and they must not be coupled:

- `anchor_horizon_steps`
  - Default `40` steps (`4.0s` at 10 Hz).
  - Used only to choose the source scene inside each open-loop event cluster.
  - The chosen source scene is the one whose predicted violation ETA is closest to this horizon.

- `max_rollout_steps`
  - Default `80` steps (`8.0s`).
  - Used only for the closed-loop perception reproducer.
  - Reproduction starts from the chosen source scene and stops at the first realized violation or the rollout cap.

- `credit_window_config`
  - Defines how many scenes to save before the realized violation.
  - The common case is `15` scenes (`1.5s`) for the main dangerous labels.

The key design choice is: do not anchor at the repair window itself. Anchor earlier, reproduce forward, then cut the repair window back from the realized offense point.

## Saved Prediction Reuse

Two inference modes are supported:

- `inference.mode="det"`
  - runs deterministic inference during classification
  - if `saved_predictions_dir` is provided, predictions are saved in the `valid_predictor.py` NPZ format
- `inference.mode="saved_predictions"`
  - reuses an existing `saved_predictions_dir`
  - the prediction directory is the only prediction input the workflow needs
  - the workflow still needs the route source (`scene_list` or `route_root`) for event mining, reproduction, and source-scene accounting

Saved prediction NPZs are written in a source-path-mirrored layout, so later reuse can resolve the original route NPZs directly from the prediction path. For standalone `classify_scene_failures.py` runs without a scene list, pass `--source_scene_root` so the classifier can map prediction NPZs back to route NPZs.
That root must sit above the mirrored relative path stored under `saved_predictions_dir`. For example, if predictions are saved under `saved_predictions/<dataset>/<session>/...`, then `--source_scene_root` should be the parent dataset root such as `.../validation`, not the deeper `.../validation/<dataset>/<session>` subfolder.

## Realized-Event Judgement

The reproducer verification stage is judged on realized rollout state, not by re-running the open-loop future-trajectory classifier at each step.

Current intended realized labels:

- `moving_collision`
  - Triggered by realized ego-vs-neighbor contact in the closed-loop rollout.
  - For this workflow, rear-end contact counts as a collision. We do not silently
    suppress a following vehicle hitting the ego from behind during mining or repair.
  - The moving-collision decision uses the same `moving_collision_thresh` from
    `scene_failure_thresholds.json` as the open-loop classifier and repair validity
    checks, so open-loop and realized closed-loop judgement use one shared rule.

- `road_border_crossing`
  - Triggered from the realized current-pose road-border distance at that rollout step.

This means the realized violation label is allowed to differ from the source open-loop label.

For this workflow, the intended default reproducer progression is fixed clock time:

- `timeline_progress_mode="clock"` for R2LPL runs
- `neighbor_history_mode="sim"`
- `tracker_mode="mpc"`

That keeps neighbor motion advancing even when the ego does not make enough spatial progress to trigger the pose-based cursor.

## Saved Scene Semantics

Saved event-window scenes are all-live scenes from the closed-loop rollout.

- `ego_agent_past`
  - Built from the realized ego history.

- `turn_indicators`
  - In `neighbor_history_mode="sim"`, this is the closed-loop turn-signal history:
    seeded from the anchor scene, then rolled forward with the model's own turn-indicator predictions.

- `neighbor_agents_past`
  - In `neighbor_history_mode="sim"`, this is the simulated shown neighbor history, not the raw logged history.

- `neighbor_agents_future`
  - Uses the simulated shown future first, UUID-matched into the current slot order.
  - If the remaining simulated horizon is shorter than the model future horizon, only the unsimulated tail is filled from the recorded route future, still UUID-matched into the live slot order.
  - This keeps the near-term target consistent with the realized rollout while avoiding short future targets late in the reproducer.

- `ego_agent_future`
  - For mined scenes this is only a temporary realized rollout future.
  - Repair generation replaces it with the accepted repaired trajectory before training.

## Repair Generation

For each reproduced scene:

1. Generate `K` candidate ego trajectories once.
2. Score all candidates against the configured gates.
3. Keep at most one winner.
4. Drop the scene if no candidate passes.

Before candidate generation, scenes already overlapped with a moving neighbor at
the current frame are discarded as unrecoverable. Those are not valid repair scenes.

Default v1 winner rule:

- safest valid candidate first
- then lower deviation penalty
- then stable first index

This avoids ambiguous multi-target training rows.

## Training Semantics

Training reuses the existing base SFT path by default.

- Default real workflow: full-model SFT.
- LoRA remains optional through training config.
- Replay memory mixes current repaired scenes with prior accepted scenes.
- DER, if enabled, applies only on replay-role scenes and anchors replay outputs to the prior-round model, while the supervised target remains the accepted repaired trajectory.

## Outputs

Per round, the workflow should report at least:

- route scene count
- deterministic predictions loaded vs computed
- violating timestamps by label
- selected open-loop events by label
- reproduced event count
- generated scene count
- accepted repaired scene count
- discarded unrepaired scene count
- replay memory size
- final training scene count

Artifacts belong under the SSD `auto_research` area, not inside the git repo.
