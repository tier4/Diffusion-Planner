# Autoresearch Tools

Standalone evaluation, diagnostic, and data preparation tools for GRPO autoresearch.
All tools use `python -m` and require the project root in PYTHONPATH.

## Evaluation Tools

### eval_driving_metrics.py
Evaluates driving quality: speed, lateral acceleration (curvature-based), path length, stopped scenes.
```bash
python -m rlvr.autoresearch.tools.eval_driving_metrics \
  --model_path <base_model.pth> --lora_path <lora_dir> \
  --scenes <driving_scenes.json> --tag <name>
```

### eval_lane_border_distance.py
Combined lane departure + road border distance metrics. Reports lane_dep count, rb_crossings, min/avg border distance.
```bash
python -m rlvr.autoresearch.tools.eval_lane_border_distance \
  --model_path <model.pth> --scenes <scenes.json> --tag <name>
```

### eval_psim_centerline.py
Closed-loop psim-rosbag centerline-tracking comparison between two runs (e.g. baseline
vs a PRiSM-trained checkpoint). Reuses the official lateral metric:
`compute_centerline_score_batch` from `rlvr.reward` via the helper
`lat_offset_and_naive_score` (same code the GRPO training reward sees), so reported
numbers are directly comparable to training metrics.

Why it's separate from `eval_centerline_metrics.py`: that tool generates trajectories
from a model and scores them on each npz's own `route_lanes` field. This tool consumes
*already-rolled-out* psim ego poses across an entire run and projects them onto a
reference centerline built from the rosbag's `LaneletRoute` + the lanelet2 map. The npz
`route_lanes` field is unreliable in psim teleport bags (40–75% of frames have empty
route segments — the route is published once before teleport and the converter cannot
re-anchor it), so the reference must be reconstructed from `route.json` and the map.

The signed lateral metric is exposed as `signed_lat_offset_m` in
`lat_offset_and_naive_score`'s output dict — `+` = ego left of route direction, `−` =
right (same sign convention `compute_centerline_score_batch` uses internally to pick
between `left_hw` and `−right_hw`).

```bash
# 1. Convert rosbags to npz. psim bags lack /perception/traffic_light_recognition/
#    traffic_signals; inject empty TL msgs first (see the helper script paired with
#    your dataset), then run:
python ros_scripts/parse_rosbag_by_cpp.py \
    cpp_tools/build/autoware_diffusion_planner_tools/data_converter \
    <bag_with_tl> <lanelet2_map.osm> <out>/npz/<run_name> \
    --step 2 --min_frames 100 --interpolation 1

# 2. Export the route message as JSON (one-time, from the route rosbag):
#    python extract_route.py <route_bag> --output route.json

# 3. Heatmap comparison:
python -m rlvr.autoresearch.tools.eval_psim_centerline \
    --baseline_dir <out>/npz/baseline \
    --prism_dir <out>/npz/<prism_run> \
    --osm <lanelet2_map.osm> \
    --route_json <route.json> \
    --out_dir <out>/analysis
```
Outputs: `summary.txt`, `{baseline,prism}_offsets.npz`, `route_polyline.npy`,
`heatmap_combined.png` (3-panel: baseline | prism | Δ), `heatmap_centerline_xy.png`
(2-panel scatter), `heatmap_centerline_diff.png` (Δ binned along route arc-length and
re-projected onto the polyline so it remains visible at any zoom),
`histogram_offsets.png`, `timeseries.png`, `progress_vs_offset.png`.

### eval_psim_centerline_nway.py
N-way generalisation of `eval_psim_centerline`. Compare any number of runs (≥2)
side-by-side. Each run is a `--run name=<label>,kind=<npz|trajlog>,path=<...>` triple,
mixing rosbag-derived npz directories and `scenario_generation.replay`
`trajectory_log.json` outputs in the same comparison. Reuses the same lateral metric
helper, so all per-run numbers stay comparable across rosbag-converted and
closed-loop replay runs.
```bash
python -m rlvr.autoresearch.tools.eval_psim_centerline_nway \
    --run name=baseline,kind=npz,path=<dir>/npz \
    --run name=run_a,kind=trajlog,path=<dir>/trajectory_log.json \
    --run name=run_b,kind=trajlog,path=<dir>/trajectory_log.json \
    --osm <map>.osm --route_json <route>.json --out_dir <out>
```
Outputs: `summary.txt`, `route_polyline.npy`, per-run `<name>_offsets.npz`,
`heatmap_nway_xy.png` (N-panel scatter), `heatmap_nway_diff.png` (Δ panels for each
non-baseline run, projected onto the polyline), `histogram_offsets.png`,
`progress_vs_offset.png`.

### scenario_generation.render_npz_dir
Render a directory of training-style npz files as `step_NNNN.png` PNGs that look like
`scenario_generation.replay` output. Reuses the replay's per-step renderer
(`_save_step_figure`), so the viewport, ego box, route overlay, road borders, agent
predictions and HUD line all match closed-loop sim renders exactly. Useful for
qualitative side-by-side video comparisons of a recorded rosbag and a closed-loop
replay.
```bash
python -m scenario_generation.render_npz_dir \
    --npz_dir <path>/npz --output_dir <path>/render \
    [--route_pkl <route>.pkl] \
    [--ego_length 7.24 --ego_width 2.29 --ego_wheelbase 4.76] \
    [--workers 8] [--limit 100] [--stride 1]
```
The route pickle is optional; without it the route polyline isn't drawn but everything
else (lane network, road borders, agents, predicted trajectory, HUD) still appears.
Ego dimensions can be overridden when the npz was converted with the data_converter's
default ego_length / ego_width but the rosbag was recorded on a vehicle with different
dimensions.

### eval_reward_vs_gt.py
Per-scene reward breakdown comparing model output vs ground truth. Useful for diagnosing which reward components are driving training.
```bash
python -m rlvr.autoresearch.tools.eval_reward_vs_gt \
  --model_path <model.pth> --scenes <scenes.json> --tag <name>
```

### eval_plan_comfort.py
Fast OPEN-LOOP **plan** gentleness metric (deterministic inference only — no psim/ROS), cheap enough to run every epoch alongside the avoidance/L2 evals. Computes planned lateral-accel = `|yaw_rate·speed|` and lateral jerk = `|d(lat_accel)/dt|` from the model's `[80,4]` plan at the KNOWN plan timestep (`dt`=0.1s, `RewardConfig.dt`), and reports the across-scene distribution of per-plan p95 values plus a `curve_speed` column (mean planned speed at steps where lat-accel exceeds `--curve_lat`, default 1.0 m/s² — the signal a slow-in-curves re-timing is meant to drop). Catches a comfort regression at training time — the blind spot the geometric (cl/RB) metrics miss. Point it at a curve scene-list to score curve behavior specifically. NOTE: open-loop is an optimistic proxy (the realized closed-loop drive is rougher), so use it as a *relative-improvement* signal on route/cruise scenes; the closed-loop `psim_comfort_heatmap` (post-hoc, ≥3 runs) is the ground-truth gate. Reuses `load_model`/`det_inference_batched` from `eval_det_avoidance.py`.
```bash
python -m rlvr.autoresearch.tools.eval_plan_comfort \
  --model_path <merged.pth> --scenes <route_scenes.json> --output_dir <dir>
```

### grpo_viz.py
Visualizes all K GRPO trajectories per scene with reward breakdown. Each scene gets one figure:
left panel shows trajectories colored by rank (green=best, red=worst) with road borders and lane
boundaries; right panel shows per-trajectory reward table with progress, smoothness, lane status, path length.
```bash
python -m rlvr.autoresearch.tools.grpo_viz \
  --model_path <model.pth> --scenes <scenes.json> --output_dir <dir> \
  --indices 0 4 8 --K 16 --enable_lane --survival \
  --w_progress 1.0 --lane_near_scale 50.0
```

## Diagnostic Tools

### diagnose_grpo_signal.py
Diagnoses per-scene GRPO reward signal. Generates K trajectories per scene (batched), scores them, and reports advantage distribution. Useful for understanding why GRPO isn't learning on specific scenes.
```bash
python -m rlvr.autoresearch.tools.diagnose_grpo_signal \
  --model_path <model.pth> --scenes <scenes.json> --n_scenes 10
```

### viz_guidance_actual.py
Visualizes actual DiT inference with and without guidance. Shows the effect of different guidance configs on trajectory generation.
```bash
python -m rlvr.autoresearch.tools.viz_guidance_actual \
  --model_path <model.pth> --scenes <scenes.json> --output_dir <dir>
```

### viz_lane_departure.py
Visualizes lane departure and road border distance checks. Supports three modes:
- `--mode lane`: lane departure polygon containment + outer boundary distance
- `--mode rb`: road border point-to-segment distance (auto-picks closest-to-border timestep)
- `--mode both`: side-by-side comparison
```bash
python -m rlvr.autoresearch.tools.viz_lane_departure \
  --scenes <scenes.json> --indices 0 5 10 --output_dir <dir> \
  --mode rb --zoom 12
```

### viz_lane_gate_rank_flip.py
Visualizes the `_classify_outer_boundaries` classifier per lane-boundary segment:
outer (road-edge, black) vs shared-between-lanes (blue) vs junction-gap (orange dashed,
reclassified as shared to suppress false road-edge detections). Useful for debugging LD-gate
decisions on new scenes.
```bash
python -m rlvr.autoresearch.tools.viz_lane_gate_rank_flip \
  --scenes <scenes.json> --n_scenes 10 --output_dir <dir>
```

### viz_intersection_polygons.py
Overlays `POLYGON_TYPE_INTERSECTION_AREA` polygons from the NPZ `polygons` field on lane
segments. Use when tuning the intersection-polygon filter in the signed-distance lane gate.
```bash
python -m rlvr.autoresearch.tools.viz_intersection_polygons \
  --scenes <scenes.json> --output_dir <dir>
```

### viz_kinematic_floored.py
Visualizes trajectories flagged as kinematically infeasible by `compute_kinematic_gate`
(absolute yaw rate + bicycle-model curvature). Colors perimeter points by which gate fired
and at which timestep.
```bash
python -m rlvr.autoresearch.tools.viz_kinematic_floored \
  --model_path <model.pth> --scenes <scenes.json> --output_dir <dir>
```

### viz_t1_crossing_debug.py
Per-timestep signed-distance debug for a single scene: plots ego perimeter points colored by
signed distance to the nearest outer boundary. Useful when the gate flags a scene but the
trajectory looks fine visually (or vice versa).
```bash
python -m rlvr.autoresearch.tools.viz_t1_crossing_debug \
  --model_path <model.pth> --scenes <scenes.json> \
  --reference_config <config.json> \
  --scene_substr <scene_name_substring> --traj_idx <0..K-1> \
  --target_t <timestep> --lane_cross_thresh <threshold> \
  --output <fig.png>
```

### calibrate_rb_vs_lane.py
Measures lane vs RB reward-component contribution on a set of generated trajectories, and
sweeps RB scale vs top-1-agreement with the reference config. Use when picking
`rb_near_scale` / `rb_wide_scale` / `rb_cont_scale` for a new dataset.
```bash
python -m rlvr.autoresearch.tools.calibrate_rb_vs_lane \
  --model_path <model.pth> --scenes <scenes.json> \
  --reference_config <config.json> --output <calibration.json>
```

### det_path_drift.py
Tracks how the deterministic (no-noise) path length drifts across epochs vs a frozen
LoRA-less baseline. Useful for diagnosing progress collapse during ranked-SFT training.
```bash
python -m rlvr.autoresearch.tools.det_path_drift \
  --run_dir <experiment_run_dir> --model_path <base.pth> \
  --scenes <scenes.json> --epochs 1 3 5 7 9
```

### compute_baseline_cache.py
Precomputes baseline model and GT path lengths per scene. Used by `run_experiment.py --baseline_cache` to report progress ratios (model_path/gt_path and model_path/baseline_path) during training.
```bash
python -m rlvr.autoresearch.tools.compute_baseline_cache \
  --model_path <base_model.pth> --scenes <scenes.json> --output <cache.json>
```

### rb_campaign_launcher.py
Auto-queues experiment batches (max 2 in parallel). Requires environment variables:
```bash
export RB_CAMPAIGN_EXP_DIR=/path/to/experiments
export RB_CAMPAIGN_MODEL=/path/to/base_model.pth
python -m rlvr.autoresearch.tools.rb_campaign_launcher
```

## Data Preparation

### cleanse_lane_scenes.py
Filters scene lists by t=0 lane/border clearance. Removes scenes where ego starts out of lane or too close to road border. Used to clean training/eval scene lists.
```bash
python -m rlvr.autoresearch.tools.cleanse_lane_scenes \
  --scenes <input.json> --output <cleaned.json> --min_clearance 0.2
```

### build_comfort_speed_targets.py
Re-times curated cl-guided **curve** targets to cap curve lateral-accel by *slowing where curvature is high* (`lat = v²·κ`), re-sampling along the SAME path polyline (the geometry/trace is preserved — waypoints move *along* the unchanged line, so centering / road-border are kept) and interpolating the original headings — only the curve speed drops, straights are unchanged. Produces curated SFT targets for a slow-in-curves leg (curated `ranked_sft_mode`). Keep `--a_long_max ≥ 1.5`: lower over-slows the ego and blows up ego-L2. Usage: `python -m rlvr.autoresearch.tools.build_comfort_speed_targets --scenes <json> --out_dir <dir> --out_list <json> [--a_lat_max 1.5 --a_long_max 1.5 --v_min 2.0]`.

### build_clguided_comfort_target.py
Comfort-aware variant of `build_clguided_target` (target *selection*, not re-timing): among the K cl-guided candidates per scene, keeps those that are kinematically feasible (`compute_kinematic_gate==1`) AND centered-enough (`cl ≥ max_cl − cl_margin`), then picks the one with the lowest comfort cost (`mean|lat_accel| + jerk_weight·mean|jerk|`) over `ego_agent_past ⊕ candidate_future` (continuity-aware). Drops scenes with no feasible+centered candidate (no poison). An *alternative* comfort lever to `build_comfort_speed_targets` (which keeps the path and re-times speed); the re-timing approach is the one in the shipped curve-comfort recipe — this selection approach is kept for experiments. Usage: `python -m rlvr.autoresearch.tools.build_clguided_comfort_target --model <base.pth> --scenes <json> --ego_shape WB,L,W --out_dir <dir> --out_list <json> [--cl_margin 0.08 --jerk_weight 0.1 --peak_weight 0 --mean_weight 1.0]`.

### sweep_epoch_comfort.py
Per-epoch sweep that merges each LoRA epoch and scores it on BOTH avoidance (via `eval_det_avoidance.score_det_scenes`) and open-loop comfort (lat-accel/jerk percentiles via `eval_driving_metrics.lat_accel_smoothed`), so you can pick the epoch on the comfort↔avoidance↔L2 frontier. Pair with `valid_predictor` for L2. Usage: `python -m rlvr.autoresearch.tools.sweep_epoch_comfort --run_dir <lora_run_dir> --base_model <warmstart.pth> --scenes <json> --config <reward.json> --ego_shape WB,L,W --output_dir <dir> [--epochs all]`.

## PRiSM — Perturbation-Recovery iterative Self-Mining

Tools for the self-improvement loop described in `rlvr/README.md` (see "PRiSM" section). Per round: sim NPZ pool → mine warm scenes → perturb → K=N + reward.py rank → filter → ranked-SFT warmstarted → iterate.

### disturb_and_replay.py
Generates perturbed NPZ training inputs from warm source scenes. Applies parallel offsets / yaw / velocity / jitter / combined perturbations to `ego_current_state` and `ego_agent_past`; lanes / route_lanes / line_strings stay in the original world frame. Emits `manifest.json` with per-NPZ `dx, dy, dtheta_deg, dv, lateral_offset_m, longitudinal_offset_m, source_scene, kind, variant_name`.
```bash
python -m rlvr.autoresearch.tools.disturb_and_replay \
  --scenes <warm_scenes.json> \
  --output_dir <out_dir> \
  --output_scene_list <out_list.json> \
  --kind combined --offsets 0.3,0.5,0.8 --yaw_degs 5,10 \
  --n_per_scene 9 --reject_threshold 0.15
```
Pass `--kind parallel_only` for offset-only perturbations. The legacy `--base_model` flag (overwrites `ego_agent_future` with a baseline forward pass) is optional and **not recommended**: ranked-SFT ignores `ego_agent_future`, so the rewrite is wasted compute.

### viz_p4_recovery.py
Per-scene K=N + reward.py rank-1 visualization on perturbed scenes. Loads a model (with optional `--lora_path`), runs K=N under the configured `generation_variant`, ranks trajectories by `compute_reward_batch` total reward (ties on cl), and renders one PNG per scene with the lanelet/road-border base, all K trajectories in faint grey, the deterministic prediction in blue, the rank-1 winner in red. Yellow translucent ego footprint anchored at the perturbed pose `(dx, dy)` in the world frame; trajectories translated by `(dx, dy)` so footprints visually start at that anchor. Records rank-1 safety flags per scene for downstream filtering.
```bash
python -m rlvr.autoresearch.tools.viz_p4_recovery \
  --model_path <base.pth> --lora_path <lora_dir> \
  --scenes <perturbed_list.json> \
  --manifest <perturbed_dir>/manifest.json \
  --config <reward_config.json> \
  --output_dir <out_dir> --K 8
```
Outputs per-scene PNGs into `improve/` and `no_improve/` subdirs based on whether rank-1's CL beats the perturbed t=0, plus `summary.json` and `improve_scenes.json`.

### viz_prism_compare.py
Multi-model overlay on the same perturbed scene. Up to three trajectories on one panel (baseline / warmstart / PRiSM) — model output translated to the lanelet frame so the perturbation magnitude is visible. Optional summary-JSON-driven scene ranking by `Δ_PRiSM-vs-reference` for "show me the biggest gain scenes" workflows. Per-deployment labels passed via `--baseline_label`, `--warmstart_label`, `--prism_label`.
```bash
python -m rlvr.autoresearch.tools.viz_prism_compare \
  --baseline_model <pth> \
  --warmstart_model <pth> \
  --prism_model <pth> --prism_lora <lora_dir> \
  --scenes <perturbed_list.json> \
  --manifest <manifest.json> --config <reward_config.json> \
  --output_dir <out_dir> \
  [--top_delta --rank_by baseline] [--hide_warmstart]
```

### recovery_sim_ghost.py
Ghost-overlay 8-second closed-loop rollout. Runs the rollout under two models on the same perturbed scene and writes per-step PNGs with both ego footprints + plans overlaid (blue / red), plus an optional WebM via ffmpeg. Reuses the scene-rendering primitives from `recovery_sim.py`.
```bash
python -m rlvr.autoresearch.tools.recovery_sim_ghost \
  --scene <source_npz> \
  --kind parallel --magnitude 1.0 --side - \
  --baseline_model <pth> --prism_model <pth> [--prism_lora <dir>] \
  --output_dir <out_dir> --steps 80 --make_webm
```

### recovery_test.py
One-shot K=N + closed-loop rollout recovery diagnostic. Applies parallel / yaw / velocity / combined perturbations in-memory (no NPZ dump) and reports recovery rates per kind / magnitude. Quick diagnostic before committing to a full disturb_and_replay → viz_p4_recovery pipeline.

### recovery_sim.py
Sim-style closed-loop rollout PNG renderer for a single perturbed scene. Used both standalone and as the rollout primitive for `recovery_sim_ghost`.

## Road-border (HEAL) & realized-ego analysis

Tools for measuring and fixing **road-border** behaviour against the *real map* borders. The parsed psim NPZ `line_strings` conflate lane lines with curbs, so reward RB scoring / `road_border` guidance on them is invalid — these tools rebuild and score against true map borders. All reuse `reward.py` OBB geometry (`_obb_corners`, `_point_to_segments_min_dist`, `compute_road_border_penalty`) and `LaneletSceneBuilder` — no hand-rolled geometry. Lanelet-dependent ones must run under a ROS env.

### rebuild_realborder_npz.py
Rewrite a scene NPZ's `line_strings` with the REAL map borders in the scene's ego frame. Recovers ego world pose (`recover_ego_world_pose_from_goal`), rebuilds the canonical stop_line+road_border tensor (`LaneletSceneBuilder.build_line_strings_tensor`, ch2=stop_line/ch3=road_border), transforms point xy into ego frame, injects `ego_shape`, copies other fields. After this, `viz_p4_recovery` / `compute_reward_batch` RB scoring AND `road_border` guidance act on the true curb. Usage: `--scenes <json> --route <pkl> --ego_shape WB,L,W --out_dir <dir> --out_list <json>`.

### verify_target_realborder.py
Check whether a scene's target `ego_agent_future` crosses the road border, using `compute_road_border_penalty` with real map borders (not the parsed NPZ line_strings). The tool reports the crossing gate (t≥1, the gate excludes t=0 by construction) and the per-timestep min border distance over t≥1. Pipeline rule: before training toward a target, confirm the target itself stays off the curb; discard scenes whose target crosses. **Related rule (enforce separately):** because the crossing gate ignores t=0, also require the ego OBB to be off the border at t=0 — build the ego current pose and read `compute_road_border_penalty` `per_ts_min[:,0] ≥ rb_cross_thresh` — you cannot train recovery from an already-violating start state.

### psim_realized_rb.py
Road-border crossings of the REALIZED closed-loop ego (from a psim `.db3` bag's `/localization/kinematic_state`), scored vs real map borders. `--stride 10` subsamples ~100Hz localization to ~10Hz; `--front_cut`/`--tail_cut` skip route ends; `--localize` bins crossings by arc. Reports full distribution (min/p5/p50/mean) + crossings. Usage: `--route <pkl> --bag <bag_dir> --ego_shape WB,L,W --stride 10 --front_cut 50 --tail_cut 50 --localize`.

### psim_comfort_heatmap.py
COMFORT/dynamics of the REALIZED closed-loop ego from a psim `.db3` bag, arc-binned (the RB-heatmap analog for dynamics). Per-arc lateral-accel, lateral jerk, curvature, speed + centerline-offset, all from MEASURED signals — NO position derivatives — with **dt derived from the message timestamps** (psim localization is ~40Hz, NOT 100Hz; assuming a fixed dt inflates speed ~2.5×/lat-accel ~6×/jerk ~16× — a real bug this avoids). The PRIMARY lateral signal is measured `|accel.y|` from `/localization/acceleration` (base_link EKF, the felt accel; first msg dropped, interpolated onto pose timestamps), with kinematic `|yaw_rate·speed|` reported as a SECONDARY sanity column — in psim there's no IMU so the EKF derives `accel.y` as `v·ω` and the two are identical (the tool prints their `corr`/`max|Δ|`; on a REAL-vehicle bag the gap is diagnostic of slip/noise). Jerk = `|d(accel.y)/dt|` (one derivative, never 3rd-deriv of position). Pass `--baseline_bag` for a side-by-side table + heatmaps + a v-vs-κ decomposition. Reports p95 as the primary statistic (not mean/max). Run under a ROS env (reads `/localization/{kinematic_state,acceleration}`; reuses `_heatmap_common` arc-binning). Usage: `python -m rlvr.autoresearch.tools.psim_comfort_heatmap --route <pkl> --bag <bag_dir> [--baseline_bag <bag_dir>] --ego_shape WB,L,W --out_dir <dir> [--stride 10 --bin_m 100 --front_cut 50 --tail_cut 50]`.

### psim_per_arc_metrics.py
Per-arc CL + road-border table for one-or-more psim bags side by side. Per pose: arc + |lateral| from route centerline (CL) via `project_to_polyline`, and border distance via the reward OBB (RB). Bins by arc and prints per-bin clμ (mean |lat|), clmx (max |lat|), rb min-dist, X (crossings) + an in-bounds total. Usage: `python -m rlvr.autoresearch.tools.psim_per_arc_metrics --route <pkl> --ego_shape WB,L,W [--bin_m 250 --front_cut 50 --tail_cut 50 --stride 10] --models LABEL1 BAG1 LABEL2 BAG2 ...`.

### psim_rb_crossing_viz.py
**Video of where the RB crossings happen.** Renders REALIZED RB-crossing regions as dual-ego WebM clips (baseline vs model, arc-synced over the actual map lanes + road borders + route centerline; the ego footprint gets a red outline on frames where its perimeter is within `rb_cross_thresh` of a border). Auto-detects the crossing arc windows from EITHER bag (so you see exactly the places the realized ego grazes the curb) and emits one WebM per window. Same footprint/border style as the perfect-track ghost sims; fails loudly if ffmpeg errors. Run under a ROS env. Usage: `python -m rlvr.autoresearch.tools.psim_rb_crossing_viz --route <pkl> --baseline_bag <bag> --model_bag <bag> --model_label <NAME> --ego_shape WB,L,W --output_dir <dir> [--stride 5 --front_cut 50 --tail_cut 50 --view_half 22 --pad_m 60 --fps 10]`.

### ghost_replay_openloop.py
Dual-model OPEN-LOOP perfect-tracking replay video per scene (baseline vs model): one deterministic inference at t0 per model, each ego perfect-tracks its 80-step plan, preceded by the recorded ego history; per-neighbor visibility windowing. WebM.

### mine_arc_from_goalpose.py
Filter already-converted (ego-centric) psim NPZs to a route arc band: recover ego world pose from each NPZ's `goal_pose` (`recover_ego_world_pose_from_goal`) → `project_to_polyline` → keep frames in `[arc_lo, arc_hi]`. Closes the frame→arc gap for arc-targeted MEND mining. Usage: `--npz_dir <dir> --route_pkl <pkl> --arc_lo <m> --arc_hi <m> --out_list <json>`.

### track_cl_heal_learning.py
Per-epoch L2-to-target distribution (+ per-arc) tracker for a HEAL/curated run, on a held-out arc scene set — distinguishes genuine healing (held-out improves) from training-fit overfit.

### build_baseline_det_target.py
Build curated GRAFT-CL targets = a competent model's deterministic trajectory. Runs `--model`'s det inference (`eval_det_avoidance.det_inference_batched`) on a scene list, unit-normalizes the (cos,sin) heading columns, and writes the result into each NPZ's `ego_agent_future` (the curated SFT target) — for HEAL Mechanism B (train the wounded model toward a known-good line instead of ranking its own samples). `--model` is the TARGET source (e.g. the baseline that keeps the line where the grafted model drifted), NOT the model being trained; train curated (lr 5e-5) warm-started from the wounded model. Usage: `--model <competent.pth> --scenes <json> --ego_shape WB,L,W --out_dir <dir> --out_list <json>`.

## Data preparation (NPZ format / mining)

### convert_3col_to_4col.py / convert_4col_to_3col.py
Convert ego/neighbor futures between canonical 3-col `(x,y,heading)` and 4-col `(x,y,cos,sin)`. Curated ranked-SFT cats prob+normal in one batch, so all scenes must share a column format. No silent fallbacks: missing fields raise.

### pad_neighbors_320.py
Pad/truncate `neighbor_agents_*` to a fixed neighbor-slot count (e.g. 320) so heterogeneous NPZs stack into one batch.

### cpp_bin_to_npz.py
Convert the C++ converter's binary tensor dumps to training-format NPZs (heading → cos/sin where needed).

### filter_avoidance_fittable.py
Keep only scenes where a competent target source can produce a candidate that clears all safety/feasibility gates with real margin — i.e. genuinely fittable avoidance scenes; drops scenes even an expert can't satisfy.

### cata_select_or_window.py
Select the time-window NPZs around an override moment from a per-session parsed-rosbag NPZ dir (sibling `.json` carries the ego timestamp).

### ghost_render.py
Standalone dual-ego ghost-overlay frame renderer (footprints over map lanes/borders/route); rollout primitive shared by the ghost-sim tools.

### Related tools in other dirs
- `scenario_generation/tools/select_npz_by_arc_range.py` — select scenes by route arc-range (filters on arc-range + `--speed_thresh`; records signed/abs lateral per scene but does not filter on it), optionally inject `ego_shape`; world pose via `recover_ego_world_pose_from_goal`.
- `ros_scripts/extract_route_from_chunk0.py` — extract/pickle the latched route message from a session's chunk-0 rosbag (run under ROS env).

## Frame-transform note

`disturb_and_replay` shifts `ego_current_state[0:4]` and `ego_agent_past` by `(dx, dy, dtheta)` but leaves `lanes` / `route_lanes` / `line_strings` in the original world frame. The model output is in an "ego-current-pose-relative" frame: `pred[0]` is always near `(1, 0)` for a 10 m/s ego, regardless of perturbation magnitude. To draw the trajectory in the world/lanelet frame, **add `(dx, dy)` to every step** and place the perturbed-ego footprint at `(dx, dy)`. Skip this and the perturbation will look invisible. `viz_p4_recovery` and `viz_prism_compare` both apply the translation; the canonical helper is `viz_prism_compare._to_lanelet_frame`.

## Tests

Tests for closed-loop components are in `rlvr/autoresearch/tests/`:

```bash
# GAE computation
python -m pytest rlvr/autoresearch/tests/test_gae.py -x -q

# State update coordinate transforms
python -m pytest rlvr/autoresearch/tests/test_state_update.py -x -q

# Integration test with real scene data
python -m rlvr.autoresearch.tests.test_real_scene
```
