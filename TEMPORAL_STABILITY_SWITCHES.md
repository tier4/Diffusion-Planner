# Temporal-stability attempts — switch guide

All 7 temporal-stability attempts (Jira Epic **T4DEV-55768**) live on the single branch
`feat/temporal-stability`, gated by flags. They were developed incrementally in one working tree,
so the code overlaps and is hard to split into per-attempt branches cleanly (the per-run
`git diff` snapshots don't even capture the new files like `model/jepa/`). Instead, **every
attempt is a flag you turn on/off** — documented here.

## Golden rule
**With every flag at its default, the model is byte-identical to the base** (single-frame, no
temporal code runs). Each flag is `coeff_*=0.0` or `*_train=False` by default. To run an attempt,
set ITS flag(s); to turn it off, restore the default (just omit the CLI arg).

The four paired-training modes are **mutually exclusive** — set exactly ONE. Dispatch order in
`train_epoch._temporal_consistency_term`: `history_cond_train` → `prior_cond_train` → `sdedit_train`
→ (else) plain consistency. All paired modes (attempts 2–7) **require `coeff_temporal_consistency>0`**
(that's what switches `train.py` to the paired `DiffusionPlannerPairData`). Attempt 1 (JEPA) is
independent: single-frame + an auxiliary energy loss.

## The switches (training)

| # | Attempt | Jira | Flags to ENABLE (all else default) | Launcher | Verdict |
|---|---------|------|--------------------------------------|----------|---------|
| — | **base** (off) | — | *(nothing — all defaults)* | any base run | byte-identical base |
| 1 | SAGE-JEPA energy | T4DEV-55836 | `--coeff_jepa_consistency_loss 0.x` + `--jepa_encoder_ckpt … --jepa_predictor_ckpt …` (single-frame; loads frozen JEPA) | scratchpad smoke/AB | ❌ negative |
| 2 | consistency loss | T4DEV-55837 | `--coeff_temporal_consistency 0.5` (and sdedit/prior/history all False) | `run_temporal_ab.sh`, `train_temporal_run.sh` | ❌ wash |
| 3 | SDEdit-init | T4DEV-55838 | `--coeff_temporal_consistency 0.5 --sdedit_train True` | `run_sdedit_train.sh` | ❌ copies |
| 4 | prior-cond (no incentive) | T4DEV-55839 | **superseded** — see note | `run_prior_cond_train.sh` | ❌ ignored |
| 5 | synthesis (prior + consistency) | T4DEV-55840 | `--coeff_temporal_consistency 0.5 --prior_cond_train True` | `run_prior_loss_train.sh` | ❌ copies |
| 6 | scene-aware gate | T4DEV-55872 | `--coeff_temporal_consistency 0.5 --prior_cond_train True --tc_scene_gate True --tc_gate_q 0.5 --tc_gate_tau 6.0` | `run_prior_loss_gate_train.sh` | ❌ wash |
| 7 | history-context | T4DEV-55946 | `--coeff_temporal_consistency 0.5 --history_cond_train True` | `run_history_cond_train.sh` | 🟡 in progress |

**Note on attempt 4:** "prior input, GT-only, no consistency loss" was an *earlier* state of the
`prior_cond_train` branch. In the current code that branch always adds the consistency loss when
`coeff>0` (= attempt 5), and the prior input needs the paired dataset (which needs `coeff>0`). So
attempt 4's exact behavior is no longer reproducible by a flag — it's documented in the Jira task only.

## Shared knobs (apply to the paired modes)
- `--coeff_temporal_consistency` (float, def 0.0): master switch for paired training + the consistency-loss weight.
- `--tc_step_g 3` frame gap; `--tc_fixed_t 0.5` near-clean diffusion-t for the consistency forward; `--tc_cons_scale 10.0` normalises cons(m)→loss units; `--tc_w_heading 1.0`.
- `--tc_gate_q` / `--tc_gate_tau`: only used when `tc_scene_gate True` (attempt 6).
- `--init_weights_path <ep60>`: warm-start (weights-only) used by all attempts.

## Inference-side hooks (for deploying a winning method — fed via the `inputs` dict, not CLI)
- SDEdit (attempt 3): `inputs["sdedit_prior"]`, `inputs["sdedit_t_start"]`.
- prior-cond (attempts 5/6): `inputs["prior_traj"]` (propagated prev plan), `inputs["prior_keep"]`.
- history-context (attempt 7): `inputs["hist_ctx"]` (prev frame's pooled `encoder(...).mean(1)`), `inputs["hist_keep"]`.
- Absent these keys ⇒ the decoder runs exactly like the base model.

## Where the code lives
- Flags: `diffusion_planner/diffusion_planner/train_config.py` + `train_predictor.py` (argparse).
- Dispatch + losses: `train_epoch.py::_temporal_consistency_term`.
- Modules: `model/module/decoder.py` (`PriorEncoder`, `HistoryEncoder`, SDEdit hooks); `model/jepa/` (attempt 1).
- Metric: `planner_metrics/replan_consistency.py` (flicker/replan-consistency + the differentiable `temporal_consistency_loss`).
