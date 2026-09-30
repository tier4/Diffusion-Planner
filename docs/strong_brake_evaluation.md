# Evaluation: controller and strong-brake metric

The following ML Planner rows use the same 2231 H5 route, seed 0, and replan interval shown. The last column is an **offline rescore of the saved PerfectTracker plans/trajectory**, not a rerun of the latest branch. Values are event counts.

| Model | Replan | Perfect + old actual metric | Replan-only MPC + old actual metric | Every-tick MPC + old actual metric | Perfect + new C metric |
|---|---:|---:|---:|---:|---:|
| frenet_recovery_aug_66m_epoch45 | 1 | 280 | 1 | 1 | 4 |
| frenet_recovery_aug_66m_epoch45 | 2 | 8 | 146 | 2 | 0 |
| frenet_recovery_aug_66m_epoch45 | 4 | 425 | 20 | 4 | 1 |
| frenet_recovery_aug_66m_epoch45 | 8 | 212 | 5 | 0 | 10 |
| 0914-dev33m-sft50 | 1 | 347 | 0 | 0 | 0 |
| 0914-dev33m-sft50 | 2 | 55 | 8 | 0 | 0 |
| 0914-dev33m-sft50 | 4 | 0 | 1 | 0 | 0 |
| 0914-dev33m-sft50 | 8 | 2 | 3 | 0 | 2 |

Additional **legacy DP / NPZ diagnostic**, shown separately because its input schema and model are different:

| Model | Replan | Perfect + old actual metric | Replan-only MPC + old actual metric | Every-tick MPC + old actual metric | Perfect + new C metric |
|---|---:|---:|---:|---:|---:|
| r2lpl_rewindow2_ep85 (`diffusion_planner.onnx`) | 8 | 0 | — | — | 0 |

Old actual metric: 0.1-second realized speed difference, `≤−2.5 m/s²` on two consecutive frames, original 3-clear event count. New C metric: from the **current active raw plan suffix**, `C=(||p2−p1||−||p1−p0||)/0.1²`; two consecutive threshold crossings enter an event, five consecutive **raw-score** clear frames end it. The C counts were independently recomputed from saved raw local `plans.jsonl` with both float32 and float64 chord arithmetic; both agree with world-point reconstruction in all eight ML runs. Changing the metric on the saved PerfectTracker trajectories does not change their motion or completion.

Interpretation limits: PerfectTracker recovery r1 and sft50 r1/r2 reached `max_steps` with little route progress. Replan-only MPC recovery r1 (reused from every-tick MPC, since every tick is a replan at r1) diverged; all other MPC rows reached goal. Replan-only MPC recovery r2 has one snap; its other rows have none. Every-tick MPC is a separate controller experiment, not the selected C metric. These outcomes and the different event definitions mean the four columns should not be treated as a controlled model-quality ranking.

Full ML model paths:

- `frenet_recovery_aug_66m_epoch45`: `/mnt/storage_rdma/diffusion_planner/validation_model/odaiba_exp_20260915/frenet_recovery_aug_66m_epoch45/diffusion_planner_for_x2/ml_planner.onnx`
- `0914-dev33m-sft50`: `/mnt/storage_rdma/diffusion_planner/validation_model/odaiba_exp_20260915/0914-dev33m-sft50/diffusion_planner_for_x2/ml_planner.onnx`
- ML route: `/mnt/storage_rdma/diffusion_planner/dataset/validation_dataset/closed_loop_lap/h5_versions/v2.0.0/data/h5/x2_dev/2231_odaiba_shinagawa_copied_from_xx1/train/2026-01-15/13-24-21/route_00000000/frames.h5` (`planning_setting=ml_planner`, `backend=new_dp_h5`).

Legacy DP reference: `/mnt/storage_rdma/diffusion_planner/validation_model/odaiba_exp_20260915/r2lpl_rewindow2_ep85/diffusion_planner_for_x2/diffusion_planner.onnx`; input `/mnt/storage_rdma/diffusion_planner/dataset/20260909_basic_dataset/x2_dev/2231_odaiba_shinagawa_copied_from_xx1/manual/2026-01-15/13-24-21/route_00000000` (`planning_setting=diffusion_planner`, NPZ loader). It represents the same logged drive at a different dataset schema/length, not a one-for-one H5 input swap; this fresh diagnostic is not the historical 9/15 Confluence run. It reached goal in 4,973 steps with snap=0. Raw count/steps=0/0; new C count/steps=0/0, with four isolated single-frame crossings and no adjacent pair.

Artifacts are saved under `/mnt/storage_rdma/workspaces/kem/Diffusion-Planner-Meta-Repository/outputs/`:

- `controller_comparison_20260929/matrix_runs.json`: Perfect/every-tick MPC runs.
- `legacy_hybrid_20260930/matrix_runs.json`: replan-only MPC runs.
- `three_brake_state_machine_20260930/three_algorithm_counts.json`: offline C counts.
- `three_brake_state_machine_20260930/c_local_precision_audit.json`: local-point precision audit.
- `old_dp_2231_diagnostic_20260930/old_r2lpl_2231_r8_perfect/`: complete legacy DP diagnostic.
