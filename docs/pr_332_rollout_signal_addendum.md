# PR #332 addendum: Decoder rollout and traffic-light visualization

## Summary

This addendum documents two diagnostic extensions to the token analysis tools:

1. Decoder-to-input Attention rollout, which combines Decoder cross-attention
   with residual-aware Fusion Attention rollout.
2. Traffic-light-aware lane/route visualization, which filters map tokens by
   their explicit traffic-light attributes and overlays attention on the
   corresponding lane geometry.

These tools are offline diagnostics only. They do not modify the Diffusion
Planner model, training procedure, or inference outputs.

## Decoder rollout

The rollout captures the ego Decoder query's cross-attention at every Decoder
layer and diffusion step. Fusion matrices are augmented with the identity,
row-normalized, and multiplied in sequence. Decoder weights are averaged over
captured events and multiplied by the Fusion rollout to obtain token-level
Decoder-to-input scores.

The implementation provides:

- a Fusion attention image for comparison;
- an all-token Decoder rollout image and video;
- a neighbor-only view filtered from the same rollout scores;
- JSON records containing token indices, classes, positions, percentages, and
  the number of captured Decoder cross-attention events.

Attention rollout indicates connectivity to the Decoder query. It must not be
interpreted as causal feature importance; ablation and closed-loop evaluation
remain necessary for that claim.

## Traffic-light visualization

Traffic-light state is embedded in lane and route tokens rather than exposed as
a separate token. The five attributes are green, yellow, red, white, and
no-signal. The signal view excludes explicit no-signal and all-zero/unknown
attributes, draws lanes as solid lines and routes as dashed lines, and uses
attention for line width, opacity, and marker size.

The display defaults to the six highest-attention signal-bearing tokens. Use
`TOP_K=10` for the PR validation scenes. The JSON still records all selected
signal-bearing tokens and their total attention share.

## Usage

```bash
CUDA_VISIBLE_DEVICES=3 \
MODEL_DIR=/path/to/best_model \
DATADIR=/path/to/dataset \
VALID_LIST=/path/to/path_list_valid.json \
CENTER_INDEX=464 \
FRAMES_BEFORE=20 \
FRAMES_AFTER=40 \
TOP_K=10 \
VIDEO_WIDTH=640 \
VIDEO_HEIGHT=360 \
DEVICE=cuda \
./run_signal_attention_video.sh
```

For rollout videos, use `run_attention_rollout_video.sh`. For long signal
sequences, `run_long_signal_attention_video.sh` saves each rendered frame and
progress JSON immediately, so an interrupted run does not lose completed
frames.

## Validation

- Python syntax compilation passed for all rollout and signal scripts.
- pre-commit lint, format, and file checks passed.
- GPU 3 execution succeeded on the mini dataset sample.
- Generated MP4s were verified as 1280×720 or 640×360, `yuv420p`, BT.709,
  with correct frame counts and aspect ratio.
- The long-video runner was verified to save per-frame PNGs and progress JSON
  incrementally.

## Limitations

- Long sequences can be compute-intensive because every frame executes the
  model and attention capture.
- The current signal sequence may end before a later maneuver; a left-turn
  video requires a path list containing those frames.
- White and unknown attributes are dataset labels and should not be assumed to
  have the same semantics as standard vehicle traffic lights.
