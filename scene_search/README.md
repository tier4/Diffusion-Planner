# Scene Search GUI

Visual tool for searching and curating NPZ driving scenes by clicking on a lanelet2 road map.

## Setup

Requires ROS 2 Humble + Autoware lanelet2 extension (for map loading).

```bash
source /opt/ros/humble/setup.bash
source ~/autoware/install/setup.bash  # or set AUTOWARE_INSTALL env var
source .venv/bin/activate
```

## Launch

```bash
python -m scene_search.app \
  --map_path /path/to/lanelet2_map.osm \
  --npz_list /path/to/path_list.json_or_npz_directory \
  [--port 7860]
```

Example with shinagawa_odaiba map:
```bash
python -m scene_search.app \
  --map_path ~/autoware_map/shinagawa_odaiba_stable/lanelet2_map.osm \
  --npz_list /media/danielsanchez/2fb4af16-188c-4b7d-8ebb-4a7d0c90d207/xx1_grpo_v4_data/npz/
```

## Usage

### Map interaction
- **Drag** — pan the map
- **Scroll** — zoom in/out (centered on cursor)
- **Shift+drag** — draw arrow (sets search position + heading)

### Search workflow
1. Navigate to area of interest (drag/scroll)
2. **Shift+drag** to place an arrow (or type X, Y, Heading manually)
3. Adjust **radius**, **heading tolerance**, **frames before/after** in sidebar
4. Click **Search** — finds matching scenes, expands to contiguous batches
5. Review thumbnail previews (every 10th scene per batch)
6. Click **Keep Batch N** or **Keep All Batches** to add to kept collection
7. Draw a new arrow elsewhere, search again, keep more batches
8. Click **Save All Kept → JSON** to export

### Constraints
Collapsible panels in the sidebar for filtering scenes:
- **Neighbor Count** — min/max active neighbors within a radius
- **Ego Speed** — min/max speed at t=0 (km/h)
- **Travel Distance** — min/max GT trajectory distance (meters)

Enable a constraint checkbox, set parameters, then Search.

### Saving
- Set the **base name** (default: `<cwd>/kept_scenes`)
- Files auto-increment: `kept_scenes_0.json`, `kept_scenes_1.json`, ...
- **Downsample to N** — randomly sample N scenes from all kept batches
- Output format: JSON list of NPZ paths (compatible with `path_list.json`)

## Adding constraints

Create a new file in `scene_search/constraints/` following the plugin pattern:

```python
from scene_search.constraints.base import BaseConstraint
from scene_search.constraints.registry import register

@register("my_constraint")
class MyConstraint(BaseConstraint):
    name = "My Constraint"
    description = "Filter by something"

    def get_params_spec(self):
        return {
            "threshold": {"type": "float", "default": 5.0, "label": "Threshold",
                          "min": 0.0, "max": 100.0, "step": 1.0},
        }

    def filter(self, npz_path, npz_data, params):
        # Return True if scene passes
        return some_value <= params["threshold"]
```

Then add the import to `scene_search/constraints/__init__.py`.

## CLI backend

The search backend can also be used standalone:

```bash
python diffusion_planner/util_scripts/search_scenes.py \
  /path/to/path_list.json \
  --center 89130,42440 --radius 50 \
  --heading 76,136 \
  --stats --group-sequences
```
