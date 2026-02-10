# Annotation UI (Lichtblick)

This extension recreates the annotation GUI as multiple Lichtblick panels. It connects to the Python WebSocket server in `preference_optimization/annotation_ws_server.py`.

## Quick Start

1. Start the WebSocket server:

```
python preference_optimization/annotation_ws_server.py \
  --model-path /path/to/model.pth \
  --npz-list /path/to/train_npz_list.json \
  --device cuda:0
```

2. Install the extension (from this folder):

```
npm install
npm run local-install
```

This uses `lichtblick-extension install` via the locally installed CLI from `create-lichtblick-extension`.

3. In Lichtblick, add panels:
   - Annotation Sidebar
   - Annotation Navigation
   - Annotation Controls
   - Annotation Selection
   - Annotation Trajectory Plot
   - Annotation Velocity Plot
   - Annotation Lateral Plot
   - Annotation Metrics Table
   - Annotation Metrics Compact

## WebSocket API Schema

### Client → Server

- `get_state`: request current state.
- `set_params`: update parameters. Payload keys: `noise_scale`, `fde_threshold`, `ade_threshold`, `max_retries`, `zoom_level`, `time_step`, `gt_similarity_mode`.
- `load_sample`: load current sample and generate trajectories.
- `regenerate`: regenerate stochastic trajectory.
- `select_winner`: payload `{ "winner": "trajectory_1" | "trajectory_2" | "green" | "orange" }`.
- `jump`: payload `{ "delta": int }`.
- `jump_to_index`: payload `{ "target_index": int }` (1-indexed).
- `jump_to_next_unlabeled`: no payload.
- `toggle_filter`: payload `{ "filter_mode": "All" | "Finished" | "Unfinished" }`.
- `set_auto_skip`: payload `{ "enabled": boolean }`.
- `update_time`: payload `{ "time_step": int }`.
- `update_zoom`: payload `{ "zoom_level": int }`.
- `launch_training`: no payload.

### Server → Client

- `state_update`:
  - `texts`: progress/metric/metrics/sidebar/history strings
  - `plots`: base64 PNGs for trajectory/velocity/lateral
  - `params`: current parameters
  - `status`: annotation status counters
  - `server`: protocol metadata and uptime
- `hello_ack`: protocol handshake response
- `pong`: health check response
- `error`: payload `{ "message": string }`

All messages may include `request_id` for request/response correlation.

## ROS Image Panels

Use native Lichtblick Image panels for camera topics. These panels do not require the WebSocket server.
