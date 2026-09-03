# Ego-dimension diagnostics

Pre-deployment checks for a failure that open-loop metrics do not detect: the model
plans a **standstill** on geometry it should drive through, at some values of the
`ego_shape` (vehicle-dimension) input.

`ego_shape` is supposed to describe geometry — it should change the shape of a
manoeuvre, never whether the vehicle moves at all. But if a training corpus mixes
vehicle platforms and those platforms also differ in how fast they drive, the token
becomes the cheapest predictor of speed available to the model, and it gets read as
"which speed distribution do I imitate". The resulting model can score normally on
displacement error while being undeployable, because the collapse is confined to the
geometry and dimensions where it plans to stop.

These tools measure that directly, in seconds, without a closed-loop simulation. Run
them from the repo root, and point `--model` at a **deployable** checkpoint: a training
milestone carries both raw optimizer iterates and the EMA copy, and loading the raw ones
gives a plausible but wrong verdict. Every script here refuses such a checkpoint and
prints the conversion.

## 1. `check_tl_gate.py` — the acceptance gate, traffic-light stratified

The one to run on every checkpoint when the scenes carry traffic-light state.

```
python -m rlvr.autoresearch.ego_shape_diag.check_tl_gate \
    --model <best_model.pth> --scenes <scene dir> \
    --shape <wheelbase,length,width> [--goal X,Y]
```

A stop is only a failure if the light says go. Grading scenes that span a light cycle
with a single threshold gets this backwards twice over: it marks the red frames as
stalls, and it scores a model that *runs* red lights as healthy. So scenes are grouped
by the traffic-light state their own route lanes carry, and the gate asks for
`mean span >= --min_green_m` on green and `<= --max_red_m` on red. Amber is reported
but not gated — easing and proceeding are both defensible, and the choice is a
behaviour difference between checkpoints rather than a pass or a fail.

If no scene carries a signalled route lane the tool fails rather than reporting a
vacuous pass.

## 2. `check_ego_shape_gate.py` — the acceptance gate, dimension sweep

For scenes without traffic-light state, or to locate the cliff.

```
python -m rlvr.autoresearch.ego_shape_diag.check_ego_shape_gate \
    --model <best_model.pth> --scenes <dir-or-glob> [--wheelbases 2.75,3.5,4.0,4.5,4.76]
```

Sweeps the wheelbase with the other dimensions fixed and reports the plan span at each,
plus the spread. A healthy model is flat across the sweep. A cliff — normal spans at one
end, near-zero at the other — means the token is acting as a speed prior, and the spread
is the size of the effect.

## 3. `check_training_data.py` — find the confound in the corpus

```
python -m rlvr.autoresearch.ego_shape_diag.check_training_data \
    --scenes <dir-or-json-list> [--wheelbase_split 4.0] [--sample 800]
```

Splits a scene set by wheelbase and compares travel over the prediction horizon between
the two classes. Two red flags: only a couple of distinct `ego_shape` values (a binary
switch is the easiest thing for a model to key on), and a large gap in median travel
between the classes. Healthy corpora have overlapping travel distributions. This runs on
data alone, so it can be checked before committing to a training run.

## 4. `check_route_ab.py` — pathological stops versus real ones

```
python -m rlvr.autoresearch.ego_shape_diag.check_route_ab \
    --model <best_model.pth> --scenes <route dir> --shape_a W,L,W --shape_b W,L,W
```

Scores every scene along a route at two dimension settings. A stop where one setting
stands still and the other drives is attributable to the token; a stop where both agree
is a real one. Report the fractions, not individual frames — and cross-check the
pathological timestamps against traffic-light state before concluding anything, since
stops on red are legitimate however the settings differ.

## 5. `check_token_ablation.py` — which input is holding the plan

```
python -m rlvr.autoresearch.ego_shape_diag.check_token_ablation \
    --model <best_model.pth> --scene <npz> --shape W,L,W [--split_line_strings]
```

Zeroes each input group in turn; a group whose removal releases the plan is what the
model is reacting to. `--split_line_strings` separates stop lines from road borders and
also pushes the borders outward, which distinguishes "cannot fit" from "will not cross":
if dropping borders releases the plan but moving them away does not, it is the latter.

## Scenes

These tools need scenes at the geometry where the model misbehaves — typically converted
from a recorded run, with the same converter the runtime uses so the tensors match.

Two conversion details worth knowing, both handled here:

* `ego_agent_past` may arrive as float64; the encoder needs float32.
* a converter may rewrite `goal_pose` onto the ego when the recorded run ends stopped,
  which makes the scene meaningless. `--goal X,Y` restores the true ego-frame goal.
