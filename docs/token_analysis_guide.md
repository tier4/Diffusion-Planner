# Token Importance and Attention Analysis Tool Guide

## Overview

This guide describes an analysis workflow that evaluates whether the Diffusion Planner
Encoder's token configuration (up to 564 tokens) is appropriate. The workflow
runs and visualizes:

- feature importance through input-class ablation;
- nearest-first Top-K evaluation for neighbors, lanes, and line strings;
- distance-cutoff evaluation for neighbors and lanes;
- Attention analysis inside the Fusion Encoder;
- valid-token counts and slot utilization by class;
- data-parallel sample sharding and result aggregation across multiple GPUs;
- per-scene neighbor-only and all-token Attention overlays, plus continuous
  neighbor-Attention MP4 visualization;
- residual-aware Decoder-to-input Attention rollout overlays and videos;
- traffic-light-aware lane/route overlays, rankings, and videos;
- numerical tables and graphs for FDE, ADE, min FDE, and min ADE in an HTML report.

The model architecture and training procedure are unchanged. These are offline
analysis tools that operate on a saved checkpoint and an evaluation dataset.

## Tool Components

| File | Description |
|---|---|
| `scripts/token_importance.py` | Evaluates input ablation, nearest-first Top-K, and distance cutoffs, then writes FDE / ADE / min FDE / min ADE and deltas from baseline to TSV |
| `scripts/attention_analysis.py` | Writes class Attention, selectivity, value-weighted share, per-layer share, distance bins, turning/straight comparison, and valid-token statistics to JSON |
| `scripts/token_occupancy_scan.py` | Independently scans an NPZ dataset to summarize token-slot occupancy |
| `scripts/visualize_neighbor_attention.py` | Draws a large bird's-eye-view overlay in which neighbor-token color and marker area represent Attention |
| `scripts/visualize_neighbor_attention_video.py` | Renders consecutive overlay frames with one global Attention scale and encodes an MP4 |
| `scripts/visualize_all_token_attention.py` | Draws Attention for every valid Fusion token, including non-spatial tokens in the ranking and JSON |
| `scripts/visualize_all_token_attention_video.py` | Renders consecutive all-token overlays with one global Attention scale and encodes an MP4 |
| `run_token_analysis.sh` | Runs both PyTorch analyses, executes the Japanese and English Notebooks, and builds portable HTML reports |
| `run_neighbor_attention_visualization.sh` | Selects or fixes one scene and generates the neighbor-Attention PNG and JSON |
| `run_neighbor_attention_video.sh` | Generates a neighbor-Attention MP4 and frame-level JSON around one dataset index |
| `run_all_token_attention_visualization.sh` | Selects or fixes one scene and generates the all-token Attention PNG and JSON |
| `run_all_token_attention_video.sh` | Generates an all-token Attention MP4 and frame-level JSON around one dataset index |
| `scripts/visualize_attention_rollout.py` | Captures Decoder cross-attention and propagates it through residual Fusion attention to produce all-token and neighbor-only rollout reports |
| `scripts/visualize_attention_rollout_video.py` | Generates all-token and neighbor-only Decoder rollout videos |
| `scripts/visualize_signal_attention.py` | Visualizes only lane/route tokens carrying explicit traffic-light attributes |
| `scripts/visualize_signal_attention_video.py` | Generates Fusion and Decoder-rollout videos for traffic-light-bearing lane/route tokens |
| `run_attention_rollout_visualization.sh` | Generates Fusion, all-token rollout, and neighbor-only rollout PNG/JSON files |
| `run_attention_rollout_video.sh` | Generates all-token and neighbor-only Decoder-rollout MP4 files |
| `run_signal_attention_video.sh` | Generates traffic-light Fusion and Decoder-rollout MP4 files |
| `run_long_signal_attention_video.sh` | Runs a reduced-resolution long signal video with persistent per-frame PNG saving |
| `scripts/visualize_prediction_overlay.py` | Overlays recorded and model-predicted ego/neighbor trajectories with selectable sources |
| `scripts/visualize_prediction_overlay_video.py` | Creates a sequentially saved trajectory-overlay MP4 |
| `run_prediction_overlay.sh` | Renders one selectable trajectory overlay scene |
| `run_prediction_overlay_video.sh` | Renders a selectable trajectory-overlay video |
| `notebook/token_analysis.ipynb` | Japanese explanation of the experiment, metrics, complete numerical results, graphs, and interpretation |
| `notebook/token_analysis_en.ipynb` | English version of the same analysis |
| `notebook/generate_english_notebook.py` | Generates the English Notebook while keeping its code cells synchronized with the Japanese source |
| `notebook/portable_html.j2` | Template for a self-contained HTML file with no external CSS or JavaScript |
| `test_scripts/test_token_analysis_helpers.py` | Regression tests for neighbor validity, ONNX output device handling, valid-token statistics, and merging distributed accumulators |

## Analysis

### 1. Feature Importance Through Input Ablation

For the same evaluation scenes, the workflow compares the unchanged baseline
with runs in which one input group is replaced by zeros.

```text
importance = error after ablation - baseline error
```

- Positive: hiding the input degraded performance, suggesting that the model
  uses information from that input.
- Near zero: no clear contribution was observed for this dataset and metric.
- Negative: hiding the input improved performance. This does not immediately
  prove that the input is unnecessary; sampling error and out-of-distribution
  ablation inputs must also be considered.

For variable-length inputs such as neighbors and maps, zeroed rows become
padding and are excluded by the Attention mask.

Goal pose and turn indicators are fixed-length tokens. Their experiments
replace input information with a constant value rather than completely
removing the token.

### 2. Nearest-First Top-K

Only the K inputs nearest to the ego vehicle are retained.

```text
nbr_top:16   # retain only the 16 nearest neighbors
lane_top:40  # retain only the 40 nearest lanes
ls_top:10    # retain only the 10 nearest line strings
```

The smallest K at which FDE and ADE stabilize near the baseline is a candidate
for reducing the token limit.

### 3. Distance Cutoffs

Only inputs within a specified physical distance are retained.

```text
nbr_within:50
nbr_within:100
lane_within:50
lane_within:100
```

Top-K evaluates a count limit, while a distance cutoff evaluates the physical
input range. They answer different questions.

### 4. Attention Analysis

Fusion Encoder Attention weights are aggregated by token class.

| Metric | Meaning |
|---|---|
| count share | Fraction of all valid tokens belonging to the class |
| ego-query share | Total Attention sent from the ego token to the class |
| all-query share | Mean Attention received by the class from all valid queries |
| selectivity | `Attention share / count share` |
| value-weighted share | Approximate contribution obtained by multiplying Attention by the value-vector magnitude |

Selectivity can be interpreted as follows:

- `1.0`: Attention is approximately proportional to token count.
- `> 1.0`: The class is preferred beyond its token-count share.
- `< 1.0`: The class receives less Attention than its token-count share.

Attention indicates where the model looked; it is not itself a causal measure
of importance to the final prediction. It should be interpreted together with
feature importance.

### 5. Valid-Token Counts and Utilization

The Attention padding mask is used to count valid tokens in the same evaluation
samples. The following values are saved to the Attention JSON:

- maximum slot count;
- mean;
- p50;
- p95;
- p99;
- maximum observed count;
- mean utilization;
- integer capacities that cover p95 and p99.

p95 and p99 are initial candidates for reducing slot limits. They do not imply
that discarding the remaining high-density scenes is safe. Top-K and
closed-loop evaluations are also required.

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| FDE | Distance between the final predicted point and the final ground-truth point |
| ADE | Mean positional error over all predicted time steps |
| min FDE / min ADE | Error of the candidate trajectory closest to ground truth |
| Δ | Difference between an ablation result and the baseline |

For a model with a single trajectory output, the top and min metrics are
identical.

FDE evaluates the final destination, while ADE evaluates the complete path.
The report therefore includes both.

## Prerequisites

The model directory must contain at least:

```text
best_model/
├── args.json
└── best_model.pth
```

The Notebook and HTML dependencies are declared in `pyproject.toml` and pinned
in `uv.lock`. Set up the environment with:

```bash
uv sync
```

## Recommended Usage

Run `run_token_analysis.sh` from the repository root:

```bash
MODEL_DIR=best_models/20260730/best_model \
DATADIR=/path/to/dataset \
N_SAMPLES=1024 \
BATCH_SIZE=64 \
NUM_GPUS=4 \
NUM_WORKERS=2 \
DEVICE=cuda \
./run_token_analysis.sh
```

If `DATADIR/path_list_valid.json` exists, it is used automatically.

To explicitly select an evaluation list:

```bash
MODEL_DIR=best_models/20260730/best_model \
DATADIR=/path/to/dataset \
VALID_LIST=/path/to/path_list_valid.json \
N_SAMPLES=1024 \
BATCH_SIZE=64 \
DEVICE=cuda \
./run_token_analysis.sh
```

If neither `path_list_valid.json` nor `VALID_LIST` is available, the Shell
recursively collects every NPZ below `DATADIR`. Depending on the directory
layout, that may mix training and validation data. An explicit `VALID_LIST` is
recommended for formal evaluation.

## Parameters

| Environment variable | Default | Description |
|---|---:|---|
| `MODEL_DIR` | `best_models/20260730/best_model` | Directory containing `args.json` and `best_model.pth` |
| `DATADIR` | mini dataset | Evaluation dataset root |
| `VALID_LIST` | `$DATADIR/path_list_valid.json` | List of NPZ files to evaluate |
| `N_SAMPLES` | `128` | Maximum number of moving scenes to evaluate |
| `BATCH_SIZE` | `32` | Inference batch size per GPU |
| `NUM_GPUS` | `1` | Number of GPUs used for data parallelism |
| `NUM_WORKERS` | `0` | Number of DataLoader workers in each GPU process |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `MOVE_MIN_M` | `5.0` | Minimum final displacement of an evaluated scene |
| `TURN_DEG` | `15.0` | Final bearing threshold used to classify a turning scene |
| `OUT_DIR` | derived from dataset name | Output directory for results and reports |
| `PYTHON_BIN` | `.venv/bin/python` | Python interpreter |

Start with a small `N_SAMPLES` to verify the setup, then use at least 512–1024
samples for formal evaluation.

`BATCH_SIZE` is the value for one GPU. The effective batch size is
approximately `BATCH_SIZE × NUM_GPUS`; adjust it to the available GPU memory.

## Multi-GPU Data Parallel Execution

When `NUM_GPUS` is at least 2, the integrated Shell launches `torchrun`. The
selected evaluation samples are partitioned across ranks without duplication,
and each GPU processes only its own shard using the same checkpoint.
Moving-scene selection first checks candidates distributed uniformly across
the dataset, then expands to unvisited offsets if necessary. It therefore
selects `N_SAMPLES` scenes when enough matching data is available.

```bash
MODEL_DIR=/path/to/best_model \
VALID_LIST=/path/to/path_list_valid.json \
N_SAMPLES=4096 \
BATCH_SIZE=16 \
NUM_GPUS=4 \
NUM_WORKERS=2 \
DEVICE=cuda \
./run_token_analysis.sh
```

- Feature importance uses `all_reduce` on the per-rank error sums and sample
  counts, producing the mean over all selected samples.
- Attention analysis gathers per-sample statistics from every rank on rank 0,
  then computes means, percentiles, and turning/straight statistics.
- Only rank 0 writes TSV, JSON, Notebook, and HTML outputs.
- Attention analysis requires only Fusion Attention, so it runs the Encoder
  without executing the Decoder.
- Distributed multi-GPU execution is not supported for the ONNX backend.

## Per-Scene Neighbor-Attention Visualization

Dataset-level Attention statistics describe average behavior but do not identify
which agent the model reads in one scene. The separate visualization tools
overlay ego-query Fusion Attention directly on neighbor-agent positions.

- Marker shape identifies vehicle, pedestrian, or bicycle tokens.
- Marker color and area increase with Attention share within valid neighbor
  tokens.
- The bird's-eye view is substantially larger than the Top-K bar chart.
- The bar chart reports each token's Attention share among all valid tokens.
- `LAYER=mean` averages all Fusion layers. `last` or a zero-based layer index
  can also be selected.
- A JSON file preserves the displayed token class, slot, position, distance,
  aggregate Attention, and per-layer Attention.

This is an additional diagnostic workflow. It is intentionally separate from
`run_token_analysis.sh` and does not change the dataset-level Notebook report.

### Render One Scene

To automatically select a turning scene using the most-attended neighbor token,
regardless of object class:

```bash
MODEL_DIR=/path/to/best_model \
DATADIR=/path/to/dataset \
SELECT_CLASS=any \
TURN_ONLY=1 \
DEVICE=cuda \
./run_neighbor_attention_visualization.sh
```

To select a scene based only on pedestrian tokens:

```bash
SELECT_CLASS=pedestrian \
TURN_ONLY=1 \
CANDIDATE_COUNT=128 \
./run_neighbor_attention_visualization.sh
```

`SELECT_CLASS` accepts `any`, `vehicle`, `pedestrian`, or `bicycle`. It affects
automatic scene selection only; the resulting image always draws every valid
neighbor class. To render a known entry from the evaluation list:

```bash
SAMPLE_INDEX=996 \
OUTPUT_NAME=intersection_right_turn_index996 \
./run_neighbor_attention_visualization.sh
```

`SAMPLE_INDEX` is the zero-based position in `VALID_LIST`, not an identifier
stored inside the NPZ filename.

### Appearance

The sequential Matplotlib colormap and marker-area range are configurable:

```bash
COLORMAP=plasma \
MARKER_SIZE_MIN=45 \
MARKER_SIZE_MAX=950 \
TOP_K=12 \
VIEW_RANGE=80 \
./run_neighbor_attention_visualization.sh
```

| Environment variable | Default | Description |
|---|---:|---|
| `LAYER` | `mean` | `mean`, `last`, or a zero-based Fusion layer index |
| `TOP_K` | `12` | Number of ranked tokens shown and annotated |
| `VIEW_RANGE` | `80` | Bird's-eye-view range in meters |
| `COLORMAP` | `viridis` | Matplotlib sequential colormap |
| `MARKER_SIZE_MIN` | `45` | Marker area at zero Attention |
| `MARKER_SIZE_MAX` | `950` | Marker area at the maximum Attention scale |
| `OUTPUT_NAME` | `neighbor_attention` | PNG/JSON basename |

The static runner writes:

```text
<OUT_DIR>/<OUTPUT_NAME>.png
<OUT_DIR>/<OUTPUT_NAME>.json
```

### Create a Continuous MP4

The video runner uses consecutive entries around a center index:

```bash
CENTER_INDEX=996 \
FRAMES_BEFORE=20 \
FRAMES_AFTER=40 \
FPS=10 \
VIDEO_WIDTH=1920 \
VIDEO_HEIGHT=1080 \
DEVICE=cuda \
./run_neighbor_attention_video.sh
```

`ffmpeg` with the H.264 encoder must be available in `PATH`. All frames use one
global Attention maximum so that colors and marker areas remain comparable over
time. Encoding preserves the figure aspect ratio by scaling and padding to the
requested even-sized frame. The output uses BT.709 limited-range `yuv420p`,
avoiding the gray colors and half-width distortion caused by ambiguous color
metadata or an incorrect sample aspect ratio.

The sequence stops before crossing into a different parent directory in the
data list. A meaningful temporal video therefore requires `VALID_LIST` entries
to be in chronological order within each log directory. `STEP` can skip entries,
and `KEEP_FRAMES=1` retains the intermediate PNG files.

```text
<OUT_DIR>/<OUTPUT_NAME>.mp4
<OUT_DIR>/<OUTPUT_NAME>.json
<OUT_DIR>/<OUTPUT_NAME>_frames/   # only with KEEP_FRAMES=1
```

## Per-Scene All-Token Attention Visualization

The all-token visualization uses the same ego-query Fusion Attention as the
neighbor-only tool but includes every valid encoder token:

- ego;
- neighbors;
- static objects;
- lanes;
- route lanes;
- polygons;
- line strings;
- goal pose;
- ego shape;
- turn indicator.

Spatial tokens are drawn on the bird's-eye view using a different marker shape
for each class. Color and marker area represent the token's percentage of
ego-query Attention over all valid tokens. The Top-K chart ranks spatial and
non-spatial tokens together. Ego shape and turn indicator have no scene
coordinate, so they appear in the chart and JSON but not as map markers.

The representative positions follow the Encoder definition: current position
for neighbors and static objects, the middle point for lane/route/polygon/line
tokens, the goal coordinate for goal pose, and the origin for ego.

### Run with a Fixed Scene

The Shell runner is the simplest entry point:

```bash
MODEL_DIR=/path/to/best_model \
DATADIR=/path/to/dataset \
VALID_LIST=/path/to/path_list_valid.json \
SAMPLE_INDEX=996 \
DEVICE=cuda \
./run_all_token_attention_visualization.sh
```

`SAMPLE_INDEX` is the zero-based position in `VALID_LIST`. The command writes:

```text
<OUT_DIR>/<OUTPUT_NAME>.png
<OUT_DIR>/<OUTPUT_NAME>.json
```

The same operation can be run directly:

```bash
.venv/bin/python scripts/visualize_all_token_attention.py \
  --run_dir /path/to/best_model \
  --valid_set_list /path/to/path_list_valid.json \
  --sample_index 996 \
  --layer mean \
  --top_k 20 \
  --view_range 80 \
  --colormap plasma \
  --marker_size_min 25 \
  --marker_size_max 700 \
  --device cuda \
  --out_png all_token_attention.png \
  --out_json all_token_attention.json
```

### Automatically Select a Scene

Without `SAMPLE_INDEX`, the runner searches moving scenes and selects the scene
containing the strongest eligible token:

```bash
SELECT_CLASS=any \
TURN_ONLY=1 \
CANDIDATE_COUNT=128 \
./run_all_token_attention_visualization.sh
```

To search by one token class:

```bash
SELECT_CLASS=route ./run_all_token_attention_visualization.sh
SELECT_CLASS=goal_pose ./run_all_token_attention_visualization.sh
SELECT_CLASS=polygons ./run_all_token_attention_visualization.sh
```

`SELECT_CLASS` accepts `any`, `ego`, `neighbors`, `static`, `lanes`, `route`,
`polygons`, `line_strings`, `goal_pose`, `ego_shape`, or `turn_indicator`.
Class selection affects scene search only; the generated report always contains
all valid classes.

### Display Parameters

| Environment variable | Default | Description |
|---|---:|---|
| `LAYER` | `mean` | `mean`, `last`, or a zero-based Fusion layer index |
| `TOP_K` | `20` | Number of tokens ranked and map annotations attempted |
| `VIEW_RANGE` | `80` | Bird's-eye-view range in meters |
| `COLORMAP` | `plasma` | Sequential Matplotlib colormap |
| `MARKER_SIZE_MIN` | `25` | Marker area at zero Attention |
| `MARKER_SIZE_MAX` | `700` | Marker area at the maximum Attention value |
| `SELECT_CLASS` | `any` | Token class used for automatic scene selection |
| `TURN_ONLY` | `1` | Restrict automatic selection to turning scenes |
| `CANDIDATE_COUNT` | `128` | Number of candidate moving scenes |
| `OUTPUT_NAME` | `all_token_attention` | PNG/JSON basename |

The JSON contains:

- total valid-token count and Attention sum;
- valid-token count and total Attention by class;
- global token index and class-local index;
- Attention value and percentage;
- representative position and distance for spatial tokens;
- neighbor subclass for neighbor tokens.

The percentages over all valid tokens should sum to approximately 100%, apart
from floating-point rounding. A high-ranked token indicates where the ego query
read strongly in the selected Fusion layer or layer average. It does not prove
that the token caused the final trajectory; compare the result with ablation
metrics and scene context.

### Create an All-Token MP4

The video runner uses the same all-token records as the static visualization and
keeps one Attention color/size scale across every frame:

```bash
CENTER_INDEX=996 \
FRAMES_BEFORE=20 \
FRAMES_AFTER=40 \
FPS=10 \
VIDEO_WIDTH=1920 \
VIDEO_HEIGHT=1080 \
DEVICE=cuda \
./run_all_token_attention_video.sh
```

The script writes `all_token_attention_index996.mp4` and its JSON metadata under
`OUT_DIR`. It requires `ffmpeg`, accepts the same `LAYER`, `TOP_K`, `VIEW_RANGE`,
`COLORMAP`, `MARKER_SIZE_MIN`, and `MARKER_SIZE_MAX` settings as the static
runner, and supports `STEP` and `KEEP_FRAMES=1`. Frames are scaled and padded
to preserve the figure aspect ratio and encoded as BT.709 limited-range
`yuv420p` with a 1:1 sample aspect ratio. As with the neighbor video, the
selected list must be chronological within one log directory for the sequence
to represent time.

## Decoder-to-Input Attention Rollout

Fusion Attention describes how the ego query reads encoder tokens. The rollout
tool adds Decoder cross-attention and propagates it backward through the
residual Fusion self-attention matrices:

```text
Decoder ego-query attention × Fusion residual rollout = input-token relevance
```

The Fusion rollout uses `(A + I) / 2` at each layer, followed by row
normalization. Decoder weights are averaged over decoder layers and diffusion
steps before multiplication. This makes the result useful for asking which
input tokens are connected to the final Decoder query, but it is still an
attention-based diagnostic, not a causal attribution or an ablation result.

### Static rollout visualization

```bash
CUDA_VISIBLE_DEVICES=3 \
MODEL_DIR=/path/to/best_model \
DATADIR=/path/to/dataset \
VALID_LIST=/path/to/path_list_valid.json \
SAMPLE_INDEX=464 \
DEVICE=cuda \
OUT_DIR=/tmp/rollout-index464 \
./run_attention_rollout_visualization.sh
```

The runner writes Fusion attention, all-token Decoder rollout, neighbor-only
rollout, and a JSON report. The neighbor-only view is a filtered presentation
of the same rollout scores; it does not recompute attention among neighbors.

### Rollout videos

```bash
CUDA_VISIBLE_DEVICES=3 \
MODEL_DIR=/path/to/best_model \
DATADIR=/path/to/dataset \
VALID_LIST=/path/to/path_list_valid.json \
CENTER_INDEX=464 \
FRAMES_BEFORE=20 \
FRAMES_AFTER=40 \
FPS=10 \
DEVICE=cuda \
./run_attention_rollout_video.sh
```

This produces separate all-token and neighbor-only MP4 files. Interpret a
large rollout value as strong Decoder-to-input attention connectivity, not as
proof that removing the token would change ADE/FDE.

## Traffic-Light-Aware Lane/Route Visualization

Traffic lights are not independent encoder tokens. They are attributes of each
`lanes` and `route_lanes` token (`x[..., 8:13]`). The signal tool selects lane
and route tokens with an explicit green, yellow, red, or white signal state;
`no_signal` and all-zero/unknown attributes are excluded from the signal view.

- solid lines represent `lanes` and dashed lines represent `route_lanes`;
- line color identifies the signal state;
- line width, opacity, and marker size represent attention;
- the ranking uses attention as a percentage of all valid encoder tokens;
- only the top six signal-bearing tokens are shown by default in static/video
  reports; set `TOP_K=10` to show ten.

The signal-bearing total is useful for comparing how much of the model's
attention reaches traffic-regulated road context. It does not establish that
the model understood the semantic meaning of a red or white signal. Unknown
attributes should be reported separately as missing/unset data rather than as
a real signal state.

### Signal video

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

For long sequences, use the persistent-frame runner:

```bash
CUDA_VISIBLE_DEVICES=3 ./run_long_signal_attention_video.sh
```

It saves each PNG immediately under `<out_json>.frames/fusion/` and
`<out_json>.frames/rollout/`, and updates the JSON progress file after every
frame. If execution is interrupted, completed frames remain available for
inspection or later encoding. Existing frame directories are protected by
default; set `OVERWRITE_FRAMES=1` to explicitly replace their PNG files.

## Prediction and Agent Trajectory Overlay

The trajectory overlay separates recorded trajectories from the direct model
prediction returned by the Decoder. Ego and neighbor display modes are
independent and accept `none`, `ground_truth`, `prediction`, or `both`:

```bash
CUDA_VISIBLE_DEVICES=3 \
SAMPLE_INDEX=464 \
EGO_MODE=both \
NEIGHBOR_MODE=prediction \
DEVICE=cuda \
./run_prediction_overlay.sh
```

For a sequence, use the video runner with the same mode variables:

```bash
CUDA_VISIBLE_DEVICES=3 \
CENTER_INDEX=464 \
FRAMES_BEFORE=20 \
FRAMES_AFTER=40 \
VIDEO_WIDTH=1920 \
VIDEO_HEIGHT=1080 \
EGO_MODE=both \
NEIGHBOR_MODE=both \
./run_prediction_overlay_video.sh
```

Ego GT is cyan, ego prediction is orange, neighbor GT is blue, and neighbor
prediction is teal. Frames are saved incrementally. The direct model
prediction uses the deterministic zero-initialized sampled trajectory input,
so it can differ from prediction NPZ files generated by a validation run with
another sampling setup.

## Outputs

By default, `eval_<dataset name>/` is created next to `MODEL_DIR`.

```text
eval_<dataset name>/
├── token_importance_n1024.tsv
├── token_importance_n1024.log
├── attention_n1024.json
├── attention_analysis_n1024.log
├── token_analysis_n1024.executed.ipynb
├── token_analysis_n1024.html
├── token_analysis_en_n1024.executed.ipynb
└── token_analysis_en_n1024.html
```

Both HTML reports embed their images and styles. They can be opened as
standalone files without the TSV, JSON, or repository.

## Latest Validation Run

The integrated Shell was run end to end on July 31, 2026, using two A100 80GB
GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_DIR=best_models/20260730/best_model \
DATADIR=../sample/il_odaiba_shinagawa_j6_npz \
N_SAMPLES=128 \
BATCH_SIZE=2 \
NUM_GPUS=2 \
NUM_WORKERS=2 \
DEVICE=cuda \
./run_token_analysis.sh
```

The target directory contained 33,275 NPZ files. Because it did not contain
`path_list_valid.json`, the workflow generated `path_list_all.json`. It selected
128 moving scenes and assigned 64 scenes to each rank. Attention analysis
classified 27 scenes as turning and 101 as straight.

This run validates the complete workflow over all NPZ files, not a fixed-split
model benchmark. A formal evaluation should provide `VALID_LIST` explicitly to
prevent unintended mixing of training and validation data.

### Feature-Importance Results

Values are in meters. Each delta is the ablation value minus the baseline.

| Configuration | FDE | ΔFDE | ADE | ΔADE |
|---|---:|---:|---:|---:|
| baseline | 3.8648 | 0.0000 | 1.4088 | 0.0000 |
| drop:neighbors | 4.0700 | +0.2051 | 1.7803 | +0.3715 |
| drop:lanes | 6.9302 | +3.0654 | 2.7215 | +1.3127 |
| drop:route | 4.4235 | +0.5587 | 1.6063 | +0.1976 |
| drop:line_strings | 3.5936 | -0.2712 | 1.4316 | +0.0228 |
| drop:polygons | 4.0002 | +0.1354 | 1.4719 | +0.0632 |
| drop:goal_pose | 6.9039 | +3.0391 | 2.7630 | +1.3543 |
| drop:turn_indicators | 4.2464 | +0.3816 | 1.5467 | +0.1380 |

Lane and goal-pose ablation produced the largest degradation in this 128-scene
run. Line-string ablation produced a negative FDE delta. These are observations,
not conclusions that an input can be removed. They require reproduction with
more samples, scene-level inspection, retraining, and closed-loop evaluation.

### Valid-Token Counts

| Class | Slots | Mean | p95 | p99 | Maximum | Mean utilization |
|---|---:|---:|---:|---:|---:|---:|
| Total | 564 | 133.21 | 192.30 | 212.30 | 218 | 23.62% |
| neighbors | 320 | 26.03 | 57.95 | 87.60 | 161 | 8.13% |
| lanes | 140 | 58.93 | 91.30 | 99.46 | 108 | 42.09% |
| route | 25 | 2.98 | 6.00 | 7.00 | 8 | 11.94% |
| polygons | 10 | 1.52 | 3.00 | 5.00 | 5 | 15.16% |
| line strings | 60 | 39.75 | 60.00 | 60.00 | 60 | 66.25% |

Neighbors used only 8.13% of their slots on average, but the run included a
dense scene with 161 valid neighbors. Lane p99 was approximately 100, while
line strings reached the 60-slot limit at p95. Capacity decisions must therefore
consider p95, p99, maximum counts, and Top-K results rather than mean utilization
alone.

### Attention Results

- Fusion Attention was aggregated for all six layers.
- For ego-query Attention, selectivity was `1.12x` for neighbors, `1.71x` for
  route, `1.29x` for line strings, and `1.79x` for goal pose.
- Route share was 3.69% in turning scenes and 4.15% in straight scenes.
- Within-lane Attention was 27.65% at 0–25 m, 18.57% at 25–50 m, 40.69% at
  50–100 m, and 13.09% beyond 100 m.
- Within-neighbor Attention was 14.17% at 0–25 m, 22.58% at 25–50 m, 40.82% at
  50–100 m, and 22.42% beyond 100 m.

Distance-bin values are not corrected for the number of tokens in each bin.
They must not be treated as causal evidence that the model prefers distant
inputs. Interpret them together with feature importance, token counts, and
scene-level visualization.

### Generated and Validated Artifacts

The workflow generated the 24-configuration TSV, Attention JSON, executed
Japanese and English Notebooks, and portable HTML reports under
`best_models/20260730/eval_il_odaiba_shinagawa_j6_npz/`.

- Both Notebooks pass nbformat schema validation.
- All code cells complete with zero execution errors.
- Each Notebook contains six generated graph images.
- Both HTML files contain embedded `data:image/png` images.
- Invalid `execution_count` fields were removed from Markdown cells in the
  English Notebook.

## Notebook and HTML Contents

- model, dataset list, sample count, batch size, device, and thresholds;
- FDE / ADE / min FDE / min ADE and baseline deltas for all 24 configurations;
- class-level ΔFDE and ΔADE;
- separate FDE and ADE Top-K curves;
- FDE and ADE for distance cutoffs;
- count, ego-query, all-query, value-weighted share, and selectivity for every class;
- ego-query Attention for every Fusion layer;
- route share for turning and straight scenes;
- all lane and neighbor distance bins;
- valid-token counts and slot utilization by class and in total;
- interpretation guidance and a review checklist.

## ONNX

`scripts/token_importance.py` can also be used directly with an ONNX model:

```bash
.venv/bin/python scripts/token_importance.py \
  --onnx /path/to/diffusion_planner.onnx \
  --args_json /path/to/args.json \
  --valid_set_list /path/to/path_list_valid.json \
  --n_samples 128 \
  --batch_size 8 \
  --device cpu \
  --out_tsv token_importance_onnx.tsv
```

The ONNX graph does not expose Fusion Encoder Attention weights, so
`attention_analysis.py` requires a PyTorch checkpoint. The integrated Shell
also uses the PyTorch checkpoint.

## Interpretation Caveats

- The analysis measures what the current checkpoint uses on the selected
  evaluation dataset.
- A small mean feature-importance value can hide an effect on a small number of
  safety-critical scenes.
- Top-K and zero filling may create inputs outside the training distribution.
- High Attention and high causal contribution to the final prediction are not
  equivalent.
- Utilization is the fraction of occupied slots, not a measure of token
  usefulness.
- Actual input reduction requires retraining, multiple splits, scene-level
  inspection, and closed-loop safety evaluation.

## Validation

- Python syntax checks
- Ruff lint and format
- Six regression tests
- NCCL distributed execution on two A100 80GB GPUs, with 64 of 128 samples per rank
- `all_reduce` aggregation for all 24 feature-importance configurations
- Rank-0 aggregation of Attention statistics
- Static neighbor-token Attention PNG/JSON generation
- Continuous H.264 MP4/JSON generation with a global Attention scale
- Static all-token Attention PNG/JSON generation with class-level totals
- Continuous all-token H.264 MP4/JSON generation with a global Attention scale
- Notebook JSON validation
- Sequential execution of every Notebook code cell
- Conversion to HTML with embedded images
- Verification that the HTML has no external CSS, JavaScript, or CDN references
