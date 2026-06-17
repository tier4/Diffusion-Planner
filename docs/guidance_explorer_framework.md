# Training a Guidance-Explorer Policy: A General Framework

This document describes a case-agnostic methodology for training a small
learned policy that steers a FROZEN diffusion planner via classifier
guidance — "guidance explorer" — for a target behavior (e.g. obstacle
avoidance) while staying inert everywhere else. It distills the logic and
the order of operations; it deliberately contains no experiment-specific
numbers.

## 1. Architecture and contract

- The planner is frozen. A policy (~small MLP over the frozen encoder's
  scene encoding + the planner's own deterministic trajectory as a
  reference input) outputs per-head guidance commands ("etas") in [-1, 1]
  through Beta distributions; deterministic inference uses the means.
- Each head maps to a guidance energy applied during diffusion sampling
  through a composer. The contract per head: eta = 0 must be EXACTLY inert
  (zero energy and zero gradient), magnitude must mean roughly the same
  thing in every scene, and commands must be O(1).
- **Optional learned strength gate (`use_strength_head`).** Instead of (or
  alongside) relying on every eta head reaching exactly 0 to be inert, the
  policy can emit ONE extra scalar `g in (0,1)` that the composer multiplies
  into the *total* guidance energy (`g -> 0` = exactly the unguided plan).
  Supervise it as a per-scene gate (target 1 when guidance is wanted, 0 when
  not — including scenes that contain obstacles the ego does NOT need to
  avoid). This decouples "engage? / how hard" (the scalar) from "which way"
  (the etas) and is a strong false-positive suppressor: a single dedicated
  gate is easier to drive to 0 than two noisy heads, and an inference-time
  multiplier on `g` (a deploy knob) then amplifies guidance ONLY where the
  gate is already open, so it cannot wake guidance on non-avoidance scenes.
- **Reachable eta range (`head_max_conc`).** The deterministic Beta mean
  saturates at the concentration cap (default 10 -> |eta| <~ 0.82), so
  scenes whose best guidance sits at the grid edge (|eta| = 1) are
  unreachable and stay marginal. Note this: a residual contact whose swept
  solution needs |eta| ~ 1 is an architecture ceiling, NOT a data problem,
  and raising `g` (energy) does NOT fix it — energy scales how hard the plan
  tracks a target *offset* that eta itself sets, so more energy just
  oscillates. Raise the concentration cap to let the mean reach near +-1.

## 2. Calibrate the guidances FIRST (most failures start here)

Before any policy training, measure each guidance's response curve:
sweep eta over [-1, 1] on a handful of diverse scenes, generate, and plot
realized displacement / task metric vs eta, plus rendered fans.
Acceptance: monotonic, side-symmetric, and scene-consistent gain.
Findings to expect and fix:
- Linear "push" energies without a bounded target have scene-dependent
  endpoint gain (the same eta can move the plan 50x more in one scene
  than another) — prefer bounded-target quadratic energies.
- Per-step-normalized energies concentrate curvature and can destabilize
  the sampler; full-horizon (mean-over-all-steps) energies with held or
  ramped targets are the stable family.
- Demanding a target offset at the first future step distorts the plan
  head (closed-loop poison); ramp the target over the first ~2 s.
- Larger guidance magnitude is NOT more capability: past the corridor
  width it trades the target metric for boundary violations. Calibrate the
  envelope (lambda/scales) so eta = +-1 spans what the road allows.

## 3. Labels by sweep, selected for ROBUSTNESS

- For each training scene run a dense grid sweep over the joint eta space,
  score every combo with the canonical task reward, and keep the metadata
  for ALL combos.
- Proactive margins: label a scene as needing action when the unguided
  plan's clearance is below a chosen margin (not only when it violates);
  the target is the best CLEAN combo achieving the margin, with a
  best-improvement fallback when nothing reaches it.
- CRITICAL — robust selection: a regression policy cannot hit exact eta
  values, and a Beta mean can never reach +-1. The diffusion prior is
  multi-modal, so the response surface has cliffs where a small eta error
  flips the outcome. Therefore: pick combos at the center of a clean
  PLATEAU (maximize the worst outcome over the eta-neighbourhood), and
  EXCLUDE scenes whose only solutions are cliff-edge — they are
  unlearnable by regression and belong to other mechanisms. Expect
  behavioral metrics to improve even when label-fit metrics get worse.

## 4. Data coverage is the real lever

- Perturbation diversity of a few base scenes generalizes far better than
  adding more unique scenes.
- Mid-maneuver states: re-anchor scenes along the policy's own OPEN-LOOP
  guided trajectories (executed prefix spliced into history) so the policy
  learns to continue and to decay an action in progress. (Labels derived
  from CLOSED-LOOP visited states do not transfer — feedback compounding
  makes open-loop-solved labels wrong there.)
- When evaluation reveals a failing geometry class, MINT it: select seed
  scenes containing that geometry, perturbation-generate variants, sweep,
  and label. Teaching the true geometry tends to improve precision
  everywhere (imitation-deviation cost can drop at the same time).
- Hygiene screens that must run on every pool: (a) scenes already
  violating at t=0 are dropped; (b) zero-target "normal" scenes must be
  verified conflict-free with the canonical clearance (a mislabeled normal
  teaches driving into obstacles); (c) any neighbour that is stopped but
  has an empty future track must be repaired (future = pose repeated) or
  the future-based reward machinery is blind to it; (d) never let
  training pools touch the metric rulers (check by path AND basename).

## 5. Splits and rulers

- Hold out by BASE-SCENE GROUP across all label files (perturbation and
  rolled siblings leave together), or overfit detection is meaningless.
- Judge candidates behaviorally, in this order: (1) closed-loop
  per-step-replanning rollouts on held-out sets, scored by the canonical
  collision machinery — batch all scenes in lockstep for speed; apply the
  same plan smoothing the production stack uses; (2) open-loop counts and
  clearance distributions; (3) imitation deviation on a dedicated
  validation list the policy never touched; (4) inertness on no-action
  scenes (mean and max trajectory deviation, raw eta magnitudes).
  Label-fit metrics (MSE, sign accuracy) are diagnostics only — they can
  anti-correlate with behavior.
- Evaluate beyond the nominal horizon occasionally: a pass that is clean
  at the protocol cutoff can still graze later. Report minimum clearances,
  not just counts — "no contact" at a few cm is an operational near-miss.

## 6. Iteration discipline

- Change ONE variable per retrain; with cached encoder features a retrain
  is minutes, so prefer many small single-variable runs.
- Diagnose per scene before choosing the next lever: is the policy BLIND
  (eta ~ 0 in a real conflict -> coverage problem), MISCALIBRATED (right
  sign, wrong magnitude -> robustness/labels), or is the scene INFEASIBLE
  under the envelope (sweep ceiling says no clean combo -> different
  action type, or out of scope)?
- Loss-balance knobs (avoid-vs-zero weighting) have a narrow useful range
  and a noise floor; if two settings straddle the target without meeting
  it, the fix is data or labels, not finer knob steps.
- RL on top of a converged supervised policy: treat as polish only.
  Value-head warmup must not touch shared trunks (freeze everything but
  the value head, or the critic's gradients drift the actor); evaluate
  EVERY epoch checkpoint and select, because policy-gradient epochs drift.
  Expect it to defend, not advance, the supervised result.

## 7. Inference engineering

- Cache anything constant across denoise steps (observation
  inverse-normalization), short-circuit the whole guidance path when all
  etas are ~0 (most frames), and avoid recomputing model forwards the
  solver already made. Gate every optimization behind an equivalence
  battery: response curves unchanged, closed-loop tables within noise,
  bit-identical trajectories where exactness is claimed.
- The end state for a latency-critical deployment is distillation: use the
  frozen-planner+policy combo as a data engine (guided trajectories as
  curated targets) and bake the behavior into the planner via standard
  fine-tuning, removing inference-time guidance entirely.
- A trained explorer doubles as a scene CLASSIFIER: run it deterministically
  over any scene list and threshold the per-head |eta| — a strong lateral or
  collision request means the policy sees something to avoid, near-zero means
  a normal scene (`rlvr/autoresearch/tools/classify_avoidance_scenes.py`,
  `--lat_thresh`/`--col_thresh`, rule any/both). Useful for mining avoidance
  scenes from large unlabeled pools and for auditing dataset composition; the
  inertness contract makes the signal bimodal, so the threshold is forgiving.

## 8b. False-detection–robust recipe (discrimination data + learned gate)

The most common deployment complaint is the explorer firing when nothing needs
avoiding (false positives on normal driving). A policy trained only to react to
the PRESENCE of a stopped agent learns the wrong invariant; the fix is to make
the data and the architecture both encode PROXIMITY-TO-PATH instead.

Data — three zero-target classes plus the positives, deliberately inert-heavy:
- positives: a real in-path obstacle, PLUS far "distractor" stopped agents the
  ego must ignore (place them beside the planned path, kept only if the whole
  plan clears them by a margin via the canonical OBB clearance — so they are
  provably non-candidates: `add_distractor_neighbors_npz`);
- anchors: the same scenes with the in-path obstacle removed
  (`strip_neighbors_npz`) and/or distractor-only — stopped agents present, none
  to avoid, target 0;
- real normals: genuine driving, target 0 (the real-domain anchor — sim-built
  zeros ALONE teach domain blindness; real zeros keep it honest).
Balance is normal-heavy (engage ≈ 1 : inert ≈ 5), because restraint is the hard
skill; counter the imbalance with the avoid-weight so positives still drive
aggressive swerves. Mix distractor / no-distractor within each class so the
learned invariant is proximity, not distractor-presence.

Architecture — a learned strength GATE (`use_strength_head`): the policy emits
one extra scalar g in (0,1) that the composer multiplies into the TOTAL guidance
energy (g->0 = exactly the unguided plan). Supervise it as the engage signal
(1 on positives, 0 on every zero-target class). This is a stronger FP suppressor
than relying on the eta heads alone (one dedicated gate drives to 0 far more
reliably than two heads both landing on 0), and it replaces the envelope's
hand-tuned "how aggressive" constant with a learned per-scene decision. A
deploy-time multiplier on g is FP-safe (g~0 on normals) but pushing it above 1
is closed-loop NEGATIVE (over-strong per-step guidance oscillates).

Closing the residual avoidance gap WITHOUT regressing FPs: do NOT add energy
gain or mid-maneuver "roll" data (both reawaken FPs / oscillate here). The
limiter is usually eta REACHABILITY — the deterministic Beta mean saturates
below |eta|=1 at the concentration cap, so scenes whose swept solution sits at
the grid edge stay marginal; raise the cap (`head_max_conc`) so the mean can
reach near +-1 while the gate keeps it FP-safe.

## 8. Failure modes checklist

blind scenes (coverage) | cliff labels (robust selection) | scene-gain
variance (guidance design) | plan-head distortion (target ramps / closed
loop) | inertness leak on unfamiliar normals (zero-target coverage +
balance) | poisoned normals / blind rewards (hygiene screens) | holdout
leakage via siblings (grouped split) | proxy-metric chasing (behavioral
rulers) | RL drift (warmup + per-epoch selection) | "clean at cutoff"
passes (extended-horizon spot checks).
