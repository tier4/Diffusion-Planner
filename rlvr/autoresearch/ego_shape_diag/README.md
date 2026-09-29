# Traffic-light acceptance gate

A pre-deployment check on a planner checkpoint: **does it plan to go on green and hold on
red?** Seconds of compute, no closed-loop simulation.

```
python -m rlvr.autoresearch.ego_shape_diag.check_tl_gate \
    --model <best_model.pth> --scenes <scene dir> \
    [--shape wheelbase,length,width] [--goal X,Y]
```

## Why a stratified gate

Open-loop trajectory error does not detect either failure this gate is for. A model can
hold a standstill plan at signalised geometry and still score normally on displacement
metrics; a model that drives through stop lines scores normally too, because the recorded
future it is compared against is a car that stopped for perfectly ordinary reasons.

Grading such scenes with a single plan-span threshold is worse than useless. Scenes taken
across a light cycle contain frames where standing still is the *correct* answer, so a
one-threshold gate marks those as stalls — and, in the same breath, scores a model that
runs reds as healthy, since running a red produces exactly the long confident plan the
threshold rewards.

So scenes are grouped by the traffic-light state their own route lanes carry, and each
group is held to its own standard:

| group | requirement | catches |
|---|---|---|
| green | mean plan span ≥ `--min_green_m` | a checkpoint that will not commit to go |
| red | mean plan span ≤ `--max_red_m` | a checkpoint that drives through the stop |
| amber | reported, never graded | — |

Amber is deliberately not a criterion: easing and proceeding are both defensible, so it
is a behaviour difference between checkpoints rather than a pass or a fail. It is printed
because that difference is worth seeing.

## What makes a result trustworthy

* **Both halves must be exercised.** They catch opposite failures, so a set carrying only
  one leaves the other undetectable while still printing PASS. A green-only set cannot
  notice a red-runner. The gate refuses such a set; `--allow_one_sided` overrides that
  and names the untested half in the result, which is then not a full acceptance.
* **Point `--model` at a deployable checkpoint.** A training milestone holds both the raw
  optimizer iterates and the EMA copy, and the loader takes the raw ones — a plausible
  but wrong verdict. The gate refuses a milestone and prints the conversion.
* **Scenes need a signalled route.** If no scene carries a traffic-light state the gate
  fails rather than reporting a vacuous pass.

## Scenes

Use scenes at the geometry you care about, converted with the same converter the runtime
uses so the tensors match. Two conversion details, both handled here:

* `ego_agent_past` may arrive as float64; the encoder needs float32.
* a converter may rewrite `goal_pose` onto the ego when the recorded run ends stopped,
  which makes the scene meaningless — `--goal X,Y` restores the true ego-frame goal.

`--shape` overrides the recorded vehicle dimensions, which is how to check a checkpoint at
the geometry it will actually be deployed on rather than the one it was recorded with.
