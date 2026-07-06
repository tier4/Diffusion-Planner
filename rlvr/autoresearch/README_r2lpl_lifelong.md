# R2LPL Lifelong Workflow

Canonical entrypoint:

```bash
python -m rlvr.autoresearch.tools.run_lifelong_r2lpl_rounds
```

This is the single-command loop for direct reproducer scene mining, repair-target
generation, replay-memory update, and training.

## Where To Look

- Orchestration:
  `rlvr/autoresearch/tools/run_lifelong_r2lpl_rounds.py`
- Direct chunk mining:
  `rlvr/autoresearch/tools/mine_direct_reproducer_chunks.py`
- Reproducer mechanics:
  `scenario_generation/reproducer_rollout.py`
- Repair target generation:
  `rlvr/autoresearch/tools/build_avoiding_target.py`
- Replay memory:
  `rlvr/autoresearch/tools/lifelong_replay_memory.py`

## Inputs

The orchestrator requires:

- `model_path`
- one mining source: `scene_list` or `chunk_manifest`
- `workflow_config`
- `training_config`
- `output_dir`

`scene_list` is a contiguous NPZ path list. The miner samples chunk starts at
fixed stride, defaulting to every 80 scenes. `chunk_manifest` is the compact
JSONL output from a planning pass and is preferred for sharded production runs
because every worker avoids reparsing the full scene list.

## Round Flow

Each round runs:

1. `mine_direct_reproducer_chunks`
2. `build_avoiding_target`
3. `lifelong_replay_memory`
4. training through either base SFT or `rlvr.autoresearch.run_experiment`

The miner writes `credit_windows.jsonl` directly. Repair generation consumes that
file and writes accepted repaired scenes. Replay memory merges current accepted
scenes with prior replay scenes. Training uses the repaired current scenes plus
the replay list.

## Chunk Semantics

- Default chunk length: 80 frames.
- Default start stride: 80 frames.
- The miner stops a chunk at scene-list discontinuities, frame-id jumps, pose
  jumps, excessive implied speed, or excessive unwrapped yaw change.
- Default guards:
  - `max_pose_step_m=10`
  - `max_pose_speed_mps=20`
  - `max_yaw_step_rad=1.57`
- By default, chunks shorter than `chunk_len` are discarded.
- `sample_fraction` and `sample_seed` provide deterministic random subsampling.
- `num_shards` and `shard_index` split chunk starts without overlap.

The scene list is treated as already mostly contiguous. The workflow does not
build route groups before mining; it detects discontinuities locally while
building each chunk.

## Reproducer Defaults

For this workflow, the intended reproducer settings are:

- `tracker_mode="mpc"`
- `timeline_progress_mode="clock"`
- `neighbor_history_mode="sim"`
- `gpu_transform=true`

Clock progression keeps neighbor motion advancing with simulation time. Sim
neighbor history ensures saved scenes reflect the realized rollout state.

## Saved Scene Semantics

Saved event-window scenes are all-live scenes from the reproducer rollout.

- `ego_agent_past` is built from realized ego history.
- `turn_indicators` are rolled forward from the simulated run.
- `neighbor_agents_past` uses simulated shown neighbor history.
- `neighbor_agents_future` uses simulated shown future first, then UUID-matched
  recorded tail if the simulated horizon is short.
- `ego_agent_future` is temporary until repair generation replaces it with the
  accepted repaired trajectory.

## Repair Generation

For each mined scene:

1. Generate `K` candidate ego trajectories.
2. Score all candidates against the configured gates.
3. Keep at most one winner.
4. Drop the scene if no candidate passes.

Scenes already overlapped with a moving neighbor at the current frame are
discarded as unrecoverable repair rows.

For mixed-platform corpora, set repair `ego_shape` to `from_npz`. Repair scoring
then uses each scene's own required `ego_shape` field instead of enforcing one
global vehicle shape across the whole run.

Default winner rule:

- safest valid candidate first
- then lower deviation penalty
- then stable first index

## Outputs

Per round, the workflow reports:

- source scene count when a scene list is provided
- planned chunks
- simulated chunks
- skipped chunks
- mined event count by label
- generated scene count
- accepted repaired scene count
- discarded unrepaired scene count
- replay memory size
- final training scene count

Artifacts belong under the SSD `auto_research` area, not inside the git repo.
