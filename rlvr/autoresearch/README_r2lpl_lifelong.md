# R2LPL Lifelong Workflow

Canonical entrypoint: `python -m rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds`

This workflow is the repo's intended single-command R2LPL-style loop for route-based replay mining, repair-target generation, replay-memory update, and training.

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
