# ml_planner_data

Python bindings that build ML planner model inputs and training labels directly
from rosbags. Input preprocessing is shared with the ROS 2 inference node through
`autoware::ml_planner::preprocess::create_input_data_map`.

## Whole-bag dataset generation

`create_bag_frame_data` selects every usable frame in one bag, generates its model inputs
and future labels, and returns arrays stacked on the leading frame dimension.

```python
import ml_planner_data as mpd

spec = mpd.VehicleSpec(
    base_link_to_front=3.55,
    vehicle_length=4.65,
    vehicle_width=1.85,
)

param = mpd.DatasetBuilderParam()
param.frame_interval_s = 0.5
param.min_travel_distance = 1000.0
param.traffic_light_timeout_s = 0.2
param.neighbor_observation_timeout_s = 0.3

result = mpd.create_bag_frame_data(
    bag_path="path/to/bag",
    map_path="path/to/lanelet2_map.osm",
    vehicle_spec=spec,
    param=param,
)

ego_past = result["frames"]["ego_agent_past"]
frame_times = result["metadata"]["frame_time_ns"]
print(ego_past.shape[0], frame_times.shape[0])
```

The result contains:

- `frames`: every input and label tensor, stacked as `(num_frames, ...)`.
- `metadata`: frame timestamps and curation values.
- `warnings`: topic dropout and individual frame generation diagnostics.
- `stats`: candidate, usable, created, failed, and skipped counts.

Only frames with complete history, future ego coverage, an available route, and acceptable
topic publication gaps are returned. Bags below `min_travel_distance` are skipped.

The standard dataset command writes one H5 file per bag and a split-level Parquet index:

```bash
uv run python scripts/dataset/create_h5_dataset.py \
  root=/path/to/rosbags \
  output_root=/path/to/h5-dataset \
  split=train
```

## Single-frame access

`FrameDataCache` remains available for interactive rosbag inspection in the dashboard.
Training does not use this API after the H5 dataset has been generated.

```python
cache = mpd.FrameDataCache(reader_capacity=16, map_capacity=4)
frame = cache.create_frame_data(
    bag_path="path/to/bag",
    map_path="path/to/lanelet2_map.osm",
    frame_time_ns=frame_time_ns,
    vehicle_spec=spec,
)
```

The result is one unbatched `dict[str, np.ndarray]`, or `None` when the requested frame is
not usable.
