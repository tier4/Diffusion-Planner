# Plan — SAGE-style JEPA temporal-stability for the single-mode Diffusion Planner

**Branch:** `feat/sage-jepa-temporal-stability` (off `tier4-main` @ `0ae5b25`).
**Status:** PLAN ONLY — no code written yet. Everything below is default-OFF and additive.
**Companion analysis:** `OnePlanner/refer/SAGE_analysis_for_diffusion_planner.md`.

Goal (user): use SAGE's idea (JEPA latent-consistency energy) to improve **temporal stability**
of *this* Diffusion Planner, **keeping it single-mode**. SAGE's native mechanism (re-rank C
candidates) does not apply to a single-mode planner; we keep only its **signal** — a frozen JEPA +
action-conditioned predictor whose per-trajectory latent-consistency **energy**
`E(τ)=1/K·Σ‖f_η(z_k,a_k)−z_{k+1}‖₁` scores how dynamically self-consistent one trajectory is — and
apply it as a **training-time regulariser** (primary) and/or a **test-time guidance term**
(secondary). Both leave the architecture single-mode.

---

## 0. Repo facts this plan is built on (verified, file:line)

- Single-mode: `P = 1 + predicted_neighbor_num` is the **agent** axis, not a mode axis
  (`model/module/decoder.py:81,371,460`). No mode head. ✔ keep as-is.
- Horizon/units: `OUTPUT_T=80`, `POSE_DIM=4` (x,y,cos,sin) (`dimensions.py:43-44`); 10 Hz
  (turn-indicator keyframes `x[:,0,1::10]`, `decoder.py:379`) → dt=0.1 s → **K=10 = 1 s prefix**.
- Velocity rep: `model_output [B,P,T,4]` IS per-step displacement (dx,dy,cos,sin) when
  `use_velocity_representation` (`loss.py:15`, `decoder.py:134,512`). → the per-step **action**
  `a_k` SAGE needs is already the model output; no inverse-dynamics model needed (unlike SAGE's
  state-only D4RL planners).
- Training loss assembled additively, coefficient-gated (`train_epoch.py:70-76`); per-term losses
  produced by `compute_training_loss` (`decoder.py:65-270`).
- **The predicted ego world-trajectory is already reconstructed** for the geometric penalties:
  `ego_pred_world [B,T,4]` at `decoder.py:214-222`, fed to `compute_ego_edge_points`. This is the
  exact spot to also compute the JEPA energy. `encoding [B,N,D]` + `encoding_pooled [B,D]` (scene
  context) are in scope (`decoder.py:602`).
- Penalty wiring precedent to mirror: `coeff_road_border_loss` / `coeff_neighbor_collision_loss`
  (config `train_config.py:90-95`; compute `decoder.py:229-250`; sum `train_epoch.py:74-75`).
- Frozen-module-via-args precedent: `state_normalizer` / `observation_normalizer` are
  `Optional[...]` placeholders set during execution (`train_config.py:134-135`).
- Guidance system exists and is rich: `model/guidance/{base,registry,composer}.py` + many terms
  (collision, road_border, route_following, …); classifier guidance wired in inference at
  `decoder.py:476-503` via `self._guidance_fn`. `hidden_dim=256` (`train_config.py:116`) == SAGE `d_z`.

---

## 1. New code — the JEPA energy module (the only genuinely new ML)

New subpackage `diffusion_planner/diffusion_planner/model/jepa/` (mirrors `refer/sage/jepa/`,
adapted to our 4-d ego pose + velocity action + optional scene context):

| New file | Contents | SAGE analogue |
|---|---|---|
| `model/jepa/encoder.py` | `TrajStateEncoder`: MLP `pose(4) [⊕ ctx] → 512 → 512 → 256`, GELU+LayerNorm. EMA-teacher copy. | `Encoder` (`jepa/utils.py:279`) + EMA |
| `model/jepa/predictor.py` | `ACLatentPredictor`: block-causal Transformer over `[z_k, a_k]` bundles, latent-delta head `ẑ_{k+1}=z_k+Δz`. 2 layers/4 heads/256. | `ACTinyTransformer.forward_teacher` (`jepa/utils.py:584`) |
| `model/jepa/energy.py` | `compute_traj_energy(ego_traj, velocity, ctx, K) -> [B]` = `1/K·Σ‖f_η(z_k,a_k)−z_{k+1}‖₁`. Both encoder+predictor frozen, `requires_grad_(False)`; grad flows only through `ego_traj`/`velocity`. | `SAGEEnergyScorer.compute_energy_from_traj` (`energy.py:177`) |
| `model/jepa/losses.py` | Stage-I JEPA loss (L2 latent align to stop-grad EMA target + VICReg var/cov) and Stage-II AC losses (teacher-forced L1 `L_tf` + rollout `L_ro` + action-usage hinge `L_neg`). | `main.tex` §3.1/§3.2, `pre_train_enc.py`/`posttrain_ac.py` |

Two offline trainer scripts (self-supervised, **annotation-free**, reuse `utils/dataset.py` ego
trajectories — past+future give the state/action windows):
- `diffusion_planner/train_jepa_encoder.py` — Stage I → `jepa_encoder_ema.pt`.
- `diffusion_planner/train_jepa_predictor.py` — Stage II (frozen encoder) → `jepa_predictor.pt`.

**State choice (the one real design call — see §6):** start with **ego-pose ⊕ pooled scene
context** (`encoding_pooled`), not ego-only. Ego-only just relearns kinematics the velocity rep +
`unicycle_accel_curvature.py` + geometric penalties already enforce; the scene-conditioned version
is what makes the energy penalise *scene-inconsistent* (off-route / into-agent) motion = real
temporal stability.

---

## 2. Use A — training-time auxiliary loss (PRIMARY)

1. `model/module/decoder.py` `compute_training_loss`, inside/after the `need_ego_edge` block
   (`:213-225`) where `ego_pred_world [B,T,4]` already exists:
   - build the prefix: `s_0..s_K` = `[ego_current ; ego_pred_world[:,:K]]`, `a_k` =
     `model_output[:,0,:K]` (velocity), `ctx` = `encoding_pooled`;
   - `loss["jepa_consistency_loss"] = compute_traj_energy(...)` (guarded `coeff>0 and x_start`,
     else `tensor(0.)` — exact pattern of `road_border_loss` at `:229-237`).
2. `train_epoch.py:70-76`: add `+ args.coeff_jepa_consistency_loss * loss["jepa_consistency_loss"]`.
3. `train_config.py` (after `:95`): `coeff_jepa_consistency_loss: float = 0.0`,
   `jepa_prefix_K: int = 10`, `jepa_encoder_ckpt: Optional[str] = None`,
   `jepa_predictor_ckpt: Optional[str] = None`; frozen modules attached at init (see 4).
4. `train.py` (model/optoptimizer setup): if `coeff_jepa_consistency_loss>0`, load the two ckpts,
   `.eval()`, `requires_grad_(False)`, move to device, attach to `args` (normalizer pattern,
   `train_config.py:134-135`). DDP: frozen no-grad modules need no wrapping.
5. Gradient hygiene: reuse the detach-window idea (`loss.py:_detached_integral`) so the energy
   shapes the **early prefix** without fighting the velocity term over all 80 steps (matches SAGE
   `K≤10`; `K≥20` degraded in the paper).

Default-off ⇒ bit-identical to current training when `coeff=0`.

## 3. Use B — inference guidance term (SECONDARY, cheap, parallel)

`model/guidance/jepa_consistency.py`: `JEPAConsistencyGuidance(BaseGuidance)`, `@register`,
`_compute(x, inputs) -> -E(τ)` (higher=better contract, `base.py`). Drops into the existing composer
+ classifier-guidance path (`decoder.py:476-503`); guides the **single** sample, no re-ranking.
Carries the paper's own caveat (guidance can distort the learnt distribution — their reason to prefer
selection); the `t∈(0.005,0.1)` gate (`base.py`) + small scale mitigate. One file, no training.

## 4. Metric-first — temporal-stability probe (DO THIS FIRST)

`planner_metrics/` + `validate_model.py`: add, as pure eval (no model change):
- **smoothness**: jerk / curvature-rate of the predicted ego trajectory;
- **replan-consistency**: overlap-jump between predictions at consecutive frames (the closest
  open-loop proxy for SAGE's closed-loop "prefix-cascade" failure);
- reuse existing comfort terms.
This is the ruler. Without it we cannot see a temporal-stability win (see §7).

---

## 5. Build order

1. **Metric probe** (§4) — eval-only, immediately useful, de-risks everything.
2. JEPA module + Stage-I/II trainers (§1); validate the energy localises injected
   action-corruptions (SAGE's AUROC diagnostic, `main.tex` §4.2) on our data **before** wiring it in.
3. Use A wiring (§2), `coeff=0` default; cold A/B (coeff 0 vs >0) read on the §4 metrics + PDMS.
4. Use B (§3) optional parallel probe.

## 6. Open decisions for the user

- **JEPA state**: ego-pose⊕scene-context (recommended) vs ego-only (cheaper, likely marginal).
- **A vs B first** after the metric probe (recommend A — only one that improves the model itself).
- Whether to commit this plan doc to the branch now.

## 7. The caveat that decides whether this is worth it ⚠

SAGE's gains are **all closed-loop** (500-seed receding-horizon); its target failure (infeasible
prefix cascading under replanning) **only exists in closed loop**. Our eval is open-loop single-shot
PDMS (ρ≈−0.36 with closed-loop). This is the same wall that banked the RL-hybrid and decoupled-xattn
negatives — *the bottleneck has repeatedly been the eval signal, not the lever.* So: the re-rank use
is off the table (single-mode + open-loop); Use A is the only variant that can help **intrinsically**
and is **partly** visible via §4. **Hence metric-first.** If §4 shows no movement and we have no
closed-loop sim, stop before this becomes a third "runs but can't be shown to help" result.
