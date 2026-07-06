# Greenstop Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-on C++ greenstop skip filter to the Diffusion Planner data converter, using Sakayori reference thresholds and neighbor-only lead blockers.

**Architecture:** Follow the existing redrun integration: keep detection in pure header-only helpers under `frame_filters.hpp`, route thresholds through `ConverterOptions` and `FrameFilterParams`, and choose the skip label in `decide_frame_skip()`. Append a new `GreenStop` label to preserve existing serialized enum values.

**Tech Stack:** C++17, ROS 2/Autoware colcon package, GoogleTest, CLI11, existing `cpp_tools/build.sh --test` workflow, optional sakurab validation with `../clip-review-tool`.

---

## File Structure

- Modify `cpp_tools/src/autoware_diffusion_planner_tools/include/cli/converter_options.hpp`
  - Add greenstop threshold fields to `ConverterOptions`.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/src/cli/converter_options.cpp`
  - Register CLI options, set reference defaults, and validate non-negative thresholds.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/src/data_converter.cpp`
  - Print greenstop options beside redrun options.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/README.md`
  - Document the new converter options and greenstop skip behavior.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_filters.hpp`
  - Add the pure `detect_green_stop()` helper and small corridor helpers.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_skip_decision.hpp`
  - Add greenstop fields to `FrameFilterParams`.
  - Add `ego_current` to `decide_frame_skip()` inputs.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_skip_decision.cpp`
  - Run the detector before generic no-future-progress.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_processor.cpp`
  - Pass `ego_current` and greenstop thresholds into the skip decision.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/include/types/skipping_info.hpp`
  - Append `SkippingLabel::GreenStop` and add `SkippingInfo::green_stop()`.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/test/test_converter_options.cpp`
  - Add defaults and validation coverage.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_filters.cpp`
  - Add greenstop detector unit tests.
- Modify `cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_skip_decision.cpp`
  - Add decision priority and label tests.

---

### Task 1: Add Converter Options And Documentation

**Files:**
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/test/test_converter_options.cpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/include/cli/converter_options.hpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/src/cli/converter_options.cpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/src/data_converter.cpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/README.md`

- [ ] **Step 1: Write failing option tests**

In `cpp_tools/src/autoware_diffusion_planner_tools/test/test_converter_options.cpp`, add the greenstop fields to `make_default_opts()` after the redrun fields:

```cpp
  o.green_stop_heading_tol_deg = 45.0f;
  o.green_stop_stay_radius_m = 2.0f;
  o.green_stop_speed_max_mps = 1.0f;
  o.green_stop_ahead_m = 40.0f;
  o.green_stop_lead_fwd_m = 30.0f;
  o.green_stop_lead_lat_m = 2.0f;
```

In `DefaultConverterOptionsTest.UsesSharedDefaults`, add:

```cpp
  EXPECT_FLOAT_EQ(opts.green_stop_heading_tol_deg, 45.0f);
  EXPECT_FLOAT_EQ(opts.green_stop_stay_radius_m, 2.0f);
  EXPECT_FLOAT_EQ(opts.green_stop_speed_max_mps, 1.0f);
  EXPECT_FLOAT_EQ(opts.green_stop_ahead_m, 40.0f);
  EXPECT_FLOAT_EQ(opts.green_stop_lead_fwd_m, 30.0f);
  EXPECT_FLOAT_EQ(opts.green_stop_lead_lat_m, 2.0f);
```

At the end of the validation tests, add:

```cpp
TEST(ValidateOptionsTest, NegativeGreenStopHeadingToleranceReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.green_stop_heading_tol_deg = -1.0f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeGreenStopStayRadiusReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.green_stop_stay_radius_m = -0.1f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeGreenStopSpeedMaxReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.green_stop_speed_max_mps = -0.1f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeGreenStopAheadReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.green_stop_ahead_m = -0.1f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeGreenStopLeadForwardReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.green_stop_lead_fwd_m = -0.1f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeGreenStopLeadLateralReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.green_stop_lead_lat_m = -0.1f;
  EXPECT_TRUE(validate_options(opts).has_value());
}
```

- [ ] **Step 2: Run option tests and verify they fail**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected: build fails with errors like `ConverterOptions has no member named green_stop_heading_tol_deg`.

- [ ] **Step 3: Add greenstop fields to converter options**

In `cpp_tools/src/autoware_diffusion_planner_tools/include/cli/converter_options.hpp`, add after the redrun fields:

```cpp
  // Green-stop filter. A frame is skipped when ego stays put at a green
  // heading-aligned route lane and no neighbor is ahead to justify stopping.
  // Static-object blockers are not checked because this converter currently
  // writes zero static_objects.
  float green_stop_heading_tol_deg;
  float green_stop_stay_radius_m;
  float green_stop_speed_max_mps;
  float green_stop_ahead_m;
  float green_stop_lead_fwd_m;
  float green_stop_lead_lat_m;
```

In `cpp_tools/src/autoware_diffusion_planner_tools/src/cli/converter_options.cpp`, register CLI options after the redrun options:

```cpp
  app.add_option(
    "--green_stop_heading_tol_deg", green_stop_heading_tol_deg,
    "Maximum heading difference in degrees between ego heading and a green route lane.");
  app.add_option(
    "--green_stop_stay_radius_m", green_stop_stay_radius_m,
    "Maximum future spatial extent in meters for a green-stop stationary window.");
  app.add_option(
    "--green_stop_speed_max_mps", green_stop_speed_max_mps,
    "Maximum current ego speed in m/s for green-stop stationary detection.");
  app.add_option(
    "--green_stop_ahead_m", green_stop_ahead_m,
    "Maximum forward distance in meters to a heading-aligned green route-lane entry.");
  app.add_option(
    "--green_stop_lead_fwd_m", green_stop_lead_fwd_m,
    "Forward extent in meters of the green-stop lead-neighbor corridor.");
  app.add_option(
    "--green_stop_lead_lat_m", green_stop_lead_lat_m,
    "Half-width in meters of the green-stop lead-neighbor corridor.");
```

In `ConverterOptions::default_converter_options()`, add after redrun defaults:

```cpp
  // Green-stop detector defaults match Sakayori's npz_cleansing reference.
  options.green_stop_heading_tol_deg = 45.0f;
  options.green_stop_stay_radius_m = 2.0f;
  options.green_stop_speed_max_mps = 1.0f;
  options.green_stop_ahead_m = 40.0f;
  options.green_stop_lead_fwd_m = 30.0f;
  options.green_stop_lead_lat_m = 2.0f;
```

In `validate_options()`, add after redrun validation:

```cpp
  if (opts.green_stop_heading_tol_deg < 0.0f) {
    return "green_stop_heading_tol_deg must be non-negative.";
  }
  if (opts.green_stop_stay_radius_m < 0.0f) {
    return "green_stop_stay_radius_m must be non-negative.";
  }
  if (opts.green_stop_speed_max_mps < 0.0f) {
    return "green_stop_speed_max_mps must be non-negative.";
  }
  if (opts.green_stop_ahead_m < 0.0f) {
    return "green_stop_ahead_m must be non-negative.";
  }
  if (opts.green_stop_lead_fwd_m < 0.0f) {
    return "green_stop_lead_fwd_m must be non-negative.";
  }
  if (opts.green_stop_lead_lat_m < 0.0f) {
    return "green_stop_lead_lat_m must be non-negative.";
  }
```

- [ ] **Step 4: Print and document the new options**

In `cpp_tools/src/autoware_diffusion_planner_tools/src/data_converter.cpp`, extend the `fmt::print()` format string after the redrun line:

```cpp
    "Red-light-run filter radius_m: {}, heading_tol_deg: {}\n"
    "Green-stop filter heading_tol_deg: {}, stay_radius_m: {}, speed_max_mps: {}, "
    "ahead_m: {}, lead_fwd_m: {}, lead_lat_m: {}\n",
```

Append the six greenstop values after `converter.red_light_run_heading_tol_deg`:

```cpp
    converter.red_light_run_heading_tol_deg, converter.green_stop_heading_tol_deg,
    converter.green_stop_stay_radius_m, converter.green_stop_speed_max_mps,
    converter.green_stop_ahead_m, converter.green_stop_lead_fwd_m,
    converter.green_stop_lead_lat_m);
```

In `cpp_tools/src/autoware_diffusion_planner_tools/README.md`, add these rows to the converter options table after the redrun rows:

```markdown
| `--green_stop_heading_tol_deg D` | Maximum heading difference used to match ego's green route lane | `45.0` |
| `--green_stop_stay_radius_m M` | Maximum future spatial extent for stopped-on-green detection | `2.0` |
| `--green_stop_speed_max_mps MPS` | Maximum current speed for stopped-on-green detection | `1.0` |
| `--green_stop_ahead_m M` | Maximum forward distance to a heading-aligned green route-lane entry | `40.0` |
| `--green_stop_lead_fwd_m M` | Forward extent of the lead-neighbor exclusion corridor | `30.0` |
| `--green_stop_lead_lat_m M` | Half-width of the lead-neighbor exclusion corridor | `2.0` |
```

In the per-frame JSON section, replace the `is_skipped` description with:

```markdown
| `is_skipped` (bool) | `true` if the production filter would have dropped this frame (stale data, invalid covariance, red/yellow-light run, stopped at red/yellow, greenstop, no future progress, GT collision, or off-lane). See also `skipping_info.label`. |
```

- [ ] **Step 5: Run option tests and verify they pass**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected: all `autoware_diffusion_planner_tools` tests pass.

- [ ] **Step 6: Commit option changes**

Run:

```bash
git add \
  cpp_tools/src/autoware_diffusion_planner_tools/test/test_converter_options.cpp \
  cpp_tools/src/autoware_diffusion_planner_tools/include/cli/converter_options.hpp \
  cpp_tools/src/autoware_diffusion_planner_tools/src/cli/converter_options.cpp \
  cpp_tools/src/autoware_diffusion_planner_tools/src/data_converter.cpp \
  cpp_tools/src/autoware_diffusion_planner_tools/README.md
git commit -m "feat: add greenstop converter options"
```

Expected: one commit containing option, print, and README changes.

---

### Task 2: Add Pure Greenstop Detector

**Files:**
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_filters.cpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_filters.hpp`

- [ ] **Step 1: Write failing detector tests**

Append this complete block to `cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_filters.cpp`:

```cpp
namespace
{

std::vector<float> make_stationary_future(float x = 0.2f, float y = 0.0f)
{
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POSE_DIM;
  std::vector<float> ego_future(OUTPUT_T * POSE_DIM, 0.0f);
  for (int64_t t = 0; t < OUTPUT_T; ++t) {
    ego_future[t * POSE_DIM + 0] = x;
    ego_future[t * POSE_DIM + 1] = y;
    ego_future[t * POSE_DIM + 2] = 1.0f;
    ego_future[t * POSE_DIM + 3] = 0.0f;
  }
  return ego_future;
}

std::vector<float> make_ego_current(float cos_h = 1.0f, float sin_h = 0.0f, float vx = 0.0f)
{
  std::vector<float> ego_current(10, 0.0f);
  ego_current[2] = cos_h;
  ego_current[3] = sin_h;
  ego_current[4] = vx;
  ego_current[5] = 0.0f;
  return ego_current;
}

std::vector<float> make_route_lanes()
{
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;
  return std::vector<float>(NUM_SEGMENTS_IN_ROUTE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM, 0.0f);
}

std::vector<float> make_neighbor_past()
{
  using autoware::diffusion_planner::INPUT_T;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  constexpr int64_t np_dim = 11;
  return std::vector<float>(MAX_NUM_NEIGHBORS * (INPUT_T + 1) * np_dim, 0.0f);
}

void set_route_lane_light(
  std::vector<float> & route_lanes, int64_t segment_idx, int64_t light_index,
  float entry_x = 8.0f, float entry_y = 0.0f, float end_x = 20.0f, float end_y = 0.0f)
{
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;
  using autoware::diffusion_planner::TRAFFIC_LIGHT;
  using autoware::diffusion_planner::TRAFFIC_LIGHT_ONE_HOT_DIM;
  const int64_t first = (segment_idx * POINTS_PER_SEGMENT + 0) * SEGMENT_POINT_DIM;
  const int64_t second = (segment_idx * POINTS_PER_SEGMENT + 1) * SEGMENT_POINT_DIM;
  route_lanes[first + 0] = entry_x;
  route_lanes[first + 1] = entry_y;
  route_lanes[second + 0] = end_x;
  route_lanes[second + 1] = end_y;
  for (int64_t k = 0; k < TRAFFIC_LIGHT_ONE_HOT_DIM; ++k) {
    route_lanes[first + TRAFFIC_LIGHT + k] = 0.0f;
  }
  route_lanes[first + light_index] = 1.0f;
}

void set_neighbor_current(std::vector<float> & neighbor_past, float x, float y)
{
  using autoware::diffusion_planner::INPUT_T;
  constexpr int64_t past = INPUT_T + 1;
  constexpr int64_t np_dim = 11;
  const int64_t base = (0 * past + INPUT_T) * np_dim;
  neighbor_past[base + 0] = x;
  neighbor_past[base + 1] = y;
  neighbor_past[base + 2] = 1.0f;
  neighbor_past[base + 3] = 0.0f;
}

bool call_green_stop(
  const std::vector<float> & ego_future, const std::vector<float> & ego_current,
  const std::vector<float> & route_lanes, const std::vector<float> & neighbor_past)
{
  return detect_green_stop(
    ego_future, ego_current, route_lanes, neighbor_past,
    2.0f,   // stay_radius_m
    1.0f,   // speed_max_mps
    40.0f,  // green_ahead_m
    30.0f,  // lead_fwd_m
    2.0f,   // lead_lat_m
    45.0f   // heading_tol_deg
  );
}

}  // namespace

TEST(GreenStopTest, StoppedAtGreenNoNeighborAheadReturnsTrue)
{
  auto ego_future = make_stationary_future();
  const auto ego_current = make_ego_current();
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN);

  EXPECT_TRUE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}

TEST(GreenStopTest, MovingEgoReturnsFalse)
{
  auto ego_future = make_stationary_future();
  const auto ego_current = make_ego_current(1.0f, 0.0f, 1.5f);
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN);

  EXPECT_FALSE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}

TEST(GreenStopTest, RedRouteLaneReturnsFalse)
{
  auto ego_future = make_stationary_future();
  const auto ego_current = make_ego_current();
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_RED);

  EXPECT_FALSE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}

TEST(GreenStopTest, PerpendicularGreenLaneReturnsFalse)
{
  auto ego_future = make_stationary_future();
  const auto ego_current = make_ego_current();
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(
    route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN,
    8.0f, 0.0f, 8.0f, 20.0f);

  EXPECT_FALSE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}

TEST(GreenStopTest, NeighborAheadReturnsFalse)
{
  auto ego_future = make_stationary_future();
  const auto ego_current = make_ego_current();
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN);
  set_neighbor_current(neighbor_past, 12.0f, 0.5f);

  EXPECT_FALSE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}

TEST(GreenStopTest, NeighborOutsideCorridorReturnsTrue)
{
  auto ego_future = make_stationary_future();
  const auto ego_current = make_ego_current();
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN);
  set_neighbor_current(neighbor_past, 12.0f, 4.0f);

  EXPECT_TRUE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}

TEST(GreenStopTest, PaddedFutureReturnsFalse)
{
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POSE_DIM;
  std::vector<float> ego_future(OUTPUT_T * POSE_DIM, 0.0f);
  const auto ego_current = make_ego_current();
  auto route_lanes = make_route_lanes();
  auto neighbor_past = make_neighbor_past();
  set_route_lane_light(route_lanes, 0, autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN);

  EXPECT_FALSE(call_green_stop(ego_future, ego_current, route_lanes, neighbor_past));
}
```

- [ ] **Step 2: Run detector tests and verify they fail**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected: build fails with `detect_green_stop` not declared.

- [ ] **Step 3: Add the detector implementation**

In `cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_filters.hpp`, add this helper before `detect_red_light_run()`:

```cpp
inline bool object_in_forward_corridor(
  float x, float y, float ch, float sh, float lead_fwd_m, float lead_lat_m)
{
  const float fwd = x * ch + y * sh;
  const float lat = -x * sh + y * ch;
  return (fwd > 0.5f) && (fwd < lead_fwd_m) && (std::fabs(lat) < lead_lat_m);
}
```

Add this detector after the helper:

```cpp
inline bool detect_green_stop(
  const std::vector<float> & ego_future, const std::vector<float> & ego_current,
  const std::vector<float> & route_lanes, const std::vector<float> & neighbor_past,
  float stay_radius_m, float speed_max_mps, float green_ahead_m, float lead_fwd_m,
  float lead_lat_m, float heading_tol_deg)
{
  using autoware::diffusion_planner::INPUT_T;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_ROUTE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::POSE_DIM;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;
  using autoware::diffusion_planner::TRAFFIC_LIGHT;
  using autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN;
  using autoware::diffusion_planner::TRAFFIC_LIGHT_ONE_HOT_DIM;

  if (ego_future.size() < static_cast<size_t>(OUTPUT_T * POSE_DIM) || ego_current.size() < 6) {
    return false;
  }

  int64_t exact_zero_rows = 0;
  const float first_x = ego_future[0];
  const float first_y = ego_future[1];
  float max_excursion = 0.0f;
  for (int64_t t = 0; t < OUTPUT_T; ++t) {
    const float x = ego_future[t * POSE_DIM + 0];
    const float y = ego_future[t * POSE_DIM + 1];
    if (x == 0.0f && y == 0.0f) {
      ++exact_zero_rows;
    }
    const float dx = x - first_x;
    const float dy = y - first_y;
    max_excursion = std::max(max_excursion, std::sqrt(dx * dx + dy * dy));
  }
  if (exact_zero_rows > 5 || max_excursion >= stay_radius_m) {
    return false;
  }

  const float ch = ego_current[2];
  const float sh = ego_current[3];
  const float speed = std::hypot(ego_current[4], ego_current[5]);
  if (speed >= speed_max_mps) {
    return false;
  }
  const float ego_heading_deg = std::atan2(sh, ch) * 180.0f / static_cast<float>(M_PI);

  bool has_green_ahead = false;
  for (int64_t seg = 0; seg < NUM_SEGMENTS_IN_ROUTE; ++seg) {
    const float * segment = &route_lanes[seg * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM];
    const float * tl = &segment[TRAFFIC_LIGHT];
    float tl_max = tl[0];
    int tl_argmax = 0;
    for (int k = 1; k < TRAFFIC_LIGHT_ONE_HOT_DIM; ++k) {
      if (tl[k] > tl_max) {
        tl_max = tl[k];
        tl_argmax = k;
      }
    }
    if (tl_max <= 0.5f || (TRAFFIC_LIGHT + tl_argmax) != TRAFFIC_LIGHT_GREEN) {
      continue;
    }

    int64_t first_valid = -1;
    int64_t last_valid = -1;
    for (int64_t p = 0; p < POINTS_PER_SEGMENT; ++p) {
      const float * pt = &segment[p * SEGMENT_POINT_DIM];
      if (std::fabs(pt[0]) <= 1e-6f && std::fabs(pt[1]) <= 1e-6f) {
        continue;
      }
      if (first_valid < 0) {
        first_valid = p;
      }
      last_valid = p;
    }
    if (first_valid < 0 || last_valid <= first_valid) {
      continue;
    }

    const float * entry = &segment[first_valid * SEGMENT_POINT_DIM];
    const float * end = &segment[last_valid * SEGMENT_POINT_DIM];
    const float lane_heading_deg =
      std::atan2(end[1] - entry[1], end[0] - entry[0]) * 180.0f / static_cast<float>(M_PI);
    if (heading_diff_deg(lane_heading_deg, ego_heading_deg) >= heading_tol_deg) {
      continue;
    }
    const float forward = entry[0] * ch + entry[1] * sh;
    if (forward > 0.0f && forward < green_ahead_m) {
      has_green_ahead = true;
      break;
    }
  }
  if (!has_green_ahead) {
    return false;
  }

  constexpr int64_t past = INPUT_T + 1;
  constexpr int64_t np_dim = 11;
  constexpr int64_t last = INPUT_T;
  if (neighbor_past.size() < static_cast<size_t>(MAX_NUM_NEIGHBORS * past * np_dim)) {
    return false;
  }
  for (int64_t n = 0; n < MAX_NUM_NEIGHBORS; ++n) {
    const float * cur = &neighbor_past[(n * past + last) * np_dim];
    const bool active =
      std::fabs(cur[0]) + std::fabs(cur[1]) + std::fabs(cur[2]) + std::fabs(cur[3]) > 1e-6f;
    if (!active) {
      continue;
    }
    if (object_in_forward_corridor(cur[0], cur[1], ch, sh, lead_fwd_m, lead_lat_m)) {
      return false;
    }
  }

  return true;
}
```

- [ ] **Step 4: Run detector tests and verify they pass**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected: all `autoware_diffusion_planner_tools` tests pass.

- [ ] **Step 5: Commit detector changes**

Run:

```bash
git add \
  cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_filters.cpp \
  cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_filters.hpp
git commit -m "feat: add greenstop detector"
```

Expected: one commit containing only the pure detector and its tests.

---

### Task 3: Wire Greenstop Into Skip Decisions

**Files:**
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_skip_decision.cpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/include/types/skipping_info.hpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_skip_decision.hpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_skip_decision.cpp`
- Modify: `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_processor.cpp`

- [ ] **Step 1: Write failing skip-decision tests**

In `cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_skip_decision.cpp`, update `make_default_filter_params()` to include greenstop values:

```cpp
  return FrameFilterParams{
    0.0f, 0.0f, 0.0f, 5, 6.0f, 1, 5.0f, 30.0f,
    45.0f, 2.0f, 1.0f, 40.0f, 30.0f, 2.0f};
```

Add `ego_current` to `ZeroVectors`:

```cpp
  std::vector<float> ego_current;
```

Initialize it in `ZeroVectors()` after `ego_future`:

```cpp
    ego_current.assign(10, 0.0f);
    ego_current[2] = 1.0f;
    ego_current[3] = 0.0f;
```

Update `call_decide()` to pass `vecs.ego_current` after `vecs.ego_future`:

```cpp
  return decide_frame_skip(
    inputs, vecs.ego_future, vecs.ego_current, vecs.ego_shape, vecs.static_objects,
    vecs.neighbor_future, vecs.neighbor_past, vecs.line_strings, vecs.lanes,
    vecs.route_lanes, make_default_filter_params());
```

Add these helpers inside the existing anonymous namespace:

```cpp
void set_stationary_future(std::vector<float> & ego_future, float x = 0.2f, float y = 0.0f)
{
  for (int64_t t = 0; t < OUTPUT_T; ++t) {
    ego_future[t * POSE_DIM + 0] = x;
    ego_future[t * POSE_DIM + 1] = y;
    ego_future[t * POSE_DIM + 2] = 1.0f;
    ego_future[t * POSE_DIM + 3] = 0.0f;
  }
}

void set_route_lane_green(std::vector<float> & route_lanes, int64_t segment_idx)
{
  using autoware::diffusion_planner::TRAFFIC_LIGHT_GREEN;
  route_lanes[(segment_idx * POINTS_PER_SEGMENT + 0) * SEGMENT_POINT_DIM + TRAFFIC_LIGHT_GREEN] =
    1.0f;
}

void set_neighbor_current(std::vector<float> & neighbor_past, float x, float y)
{
  const int64_t past = INPUT_T + 1;
  const int64_t np_dim = 11;
  const int64_t base = (0 * past + INPUT_T) * np_dim;
  neighbor_past[base + 0] = x;
  neighbor_past[base + 1] = y;
  neighbor_past[base + 2] = 1.0f;
  neighbor_past[base + 3] = 0.0f;
}
```

Add these tests near the existing `NoFutureProgress` tests:

```cpp
TEST(DecideFrameSkipTest, GreenStopSkip)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;
  set_stationary_future(vecs.ego_future);
  set_route_lane_green(vecs.route_lanes, 0);
  set_route_lane_point(vecs.route_lanes, 0, 0, 8.0f, 0.0f);
  set_route_lane_point(vecs.route_lanes, 0, 1, 20.0f, 0.0f);

  const FrameSkipInputs inputs = make_clear_inputs();

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::GreenStop);
}

TEST(DecideFrameSkipTest, GreenStopBeforeNoFutureProgress)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;
  set_stationary_future(vecs.ego_future);
  set_route_lane_green(vecs.route_lanes, 0);
  set_route_lane_point(vecs.route_lanes, 0, 0, 8.0f, 0.0f);
  set_route_lane_point(vecs.route_lanes, 0, 1, 20.0f, 0.0f);

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.no_future_progress_x_step = 31;

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::GreenStop);
}

TEST(DecideFrameSkipTest, GreenStopKeptWhenNeighborAhead)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;
  set_stationary_future(vecs.ego_future);
  set_route_lane_green(vecs.route_lanes, 0);
  set_route_lane_point(vecs.route_lanes, 0, 0, 8.0f, 0.0f);
  set_route_lane_point(vecs.route_lanes, 0, 1, 20.0f, 0.0f);
  set_neighbor_current(vecs.neighbor_past, 12.0f, 0.5f);

  const FrameSkipInputs inputs = make_clear_inputs();

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::NotSkipped);
}

TEST(DecideFrameSkipTest, StaleDataWinsOverGreenStop)
{
  ZeroVectors vecs;
  vecs.lanes[0] = 1.0f;
  set_stationary_future(vecs.ego_future);
  set_route_lane_green(vecs.route_lanes, 0);
  set_route_lane_point(vecs.route_lanes, 0, 0, 8.0f, 0.0f);
  set_route_lane_point(vecs.route_lanes, 0, 1, 20.0f, 0.0f);

  FrameSkipInputs inputs = make_clear_inputs();
  inputs.max_msg_age_ns = 600'000'000LL;

  const SkippingInfo info = call_decide(inputs, vecs);
  EXPECT_EQ(info.label, SkippingLabel::IncompleteData);
}
```

- [ ] **Step 2: Run skip-decision tests and verify they fail**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected: build fails with errors for missing `SkippingLabel::GreenStop`, the old `FrameFilterParams` arity, and the old `decide_frame_skip()` signature.

- [ ] **Step 3: Add the GreenStop skip label**

In `cpp_tools/src/autoware_diffusion_planner_tools/include/types/skipping_info.hpp`, append to `enum class SkippingLabel` after `AcceleratingAtTrafficLight`:

```cpp
  GreenStop,  // Ego stays put at a heading-aligned green light with no lead neighbor ahead.
```

Add this constructor after `accelerating_at_traffic_light()`:

```cpp
  static SkippingInfo green_stop()
  {
    return {
      SkippingLabel::GreenStop,
      "Stopped at green light with no lead neighbor ahead",
      {},
      {}};
  }
```

- [ ] **Step 4: Extend skip-decision types and signature**

In `cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_skip_decision.hpp`, add greenstop fields to `FrameFilterParams` after the redrun fields:

```cpp
  float green_stop_heading_tol_deg;
  float green_stop_stay_radius_m;
  float green_stop_speed_max_mps;
  float green_stop_ahead_m;
  float green_stop_lead_fwd_m;
  float green_stop_lead_lat_m;
```

Change the `decide_frame_skip()` declaration so `ego_current` comes after `ego_future`:

```cpp
SkippingInfo decide_frame_skip(
  const FrameSkipInputs & inputs, const std::vector<float> & ego_future,
  const std::vector<float> & ego_current, const std::vector<float> & ego_shape,
  const std::vector<float> & static_objects, const std::vector<float> & neighbor_future,
  const std::vector<float> & neighbor_past, const std::vector<float> & line_strings,
  const std::vector<float> & lanes, const std::vector<float> & route_lanes,
  const FrameFilterParams & filter_params);
```

Make the same signature change in `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_skip_decision.cpp`.

- [ ] **Step 5: Run greenstop in the skip decision**

In `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_skip_decision.cpp`, add this block after the existing `StoppedAtTrafficLight` block and before `NoFutureProgress`:

```cpp
  if (frame_filters::detect_green_stop(
        ego_future, ego_current, route_lanes, neighbor_past,
        filter_params.green_stop_stay_radius_m, filter_params.green_stop_speed_max_mps,
        filter_params.green_stop_ahead_m, filter_params.green_stop_lead_fwd_m,
        filter_params.green_stop_lead_lat_m, filter_params.green_stop_heading_tol_deg)) {
    return SkippingInfo::green_stop();
  }
```

This preserves current red/yellow stop labeling while still ensuring greenstop is more specific than generic `NoFutureProgress`.

- [ ] **Step 6: Pass ego_current and greenstop params from frame processing**

In `cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_processor.cpp`, expand `filter_params`:

```cpp
    const frame_processor::FrameFilterParams filter_params{
      options.static_object_margin,
      options.neighbor_margin,
      options.road_border_margin,
      options.collision_time_stride,
      options.offlane_max_score,
      options.offlane_time_stride,
      options.red_light_run_radius_m,
      options.red_light_run_heading_tol_deg,
      options.green_stop_heading_tol_deg,
      options.green_stop_stay_radius_m,
      options.green_stop_speed_max_mps,
      options.green_stop_ahead_m,
      options.green_stop_lead_fwd_m,
      options.green_stop_lead_lat_m};
```

Update the `decide_frame_skip()` call:

```cpp
    const SkippingInfo skipping_info = frame_processor::decide_frame_skip(
      skip_inputs, ego_future, ego_current, ego_shape, static_objects, neighbor_future,
      neighbor_past, line_strings, lanes, route_lanes, filter_params);
```

- [ ] **Step 7: Run skip-decision tests and verify they pass**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected: all `autoware_diffusion_planner_tools` tests pass.

- [ ] **Step 8: Commit skip wiring**

Run:

```bash
git add \
  cpp_tools/src/autoware_diffusion_planner_tools/test/test_frame_skip_decision.cpp \
  cpp_tools/src/autoware_diffusion_planner_tools/include/types/skipping_info.hpp \
  cpp_tools/src/autoware_diffusion_planner_tools/include/processing/frame_skip_decision.hpp \
  cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_skip_decision.cpp \
  cpp_tools/src/autoware_diffusion_planner_tools/src/processing/frame_processor.cpp
git commit -m "feat: skip greenstop frames during conversion"
```

Expected: one commit containing the skip label, decision routing, and frame processor call-site changes.

---

### Task 4: Full Verification And Low-Footprint Sampling

**Files:**
- No required source edits.
- Read: `../clip-review-tool/README.md`
- Runtime output on sakurab: `/mnt/nvme/chenglin/greenstop_review_2026-07-06/`

- [ ] **Step 1: Run full local package tests**

Run:

```bash
cd cpp_tools
./build.sh --test
```

Expected:

```text
[INFO] Build and run unit tests with colcon test
Summary: 1 package finished
```

and `colcon test-result --all` reports no failures.

- [ ] **Step 2: Inspect git diff and status**

Run:

```bash
git status --short
git log --oneline -4
```

Expected:

- only unrelated pre-existing untracked `reference/` remains uncommitted
- the latest commits are the spec commit and the three implementation commits

- [ ] **Step 3: Preflight sakurab disk and create a bounded review directory**

Run on sakurab:

```bash
df -h /mnt/nvme
REVIEW_DIR=/mnt/nvme/chenglin/greenstop_review_2026-07-06
mkdir -p "$REVIEW_DIR/lists" "$REVIEW_DIR/videos/skipped" "$REVIEW_DIR/videos/kept" "$REVIEW_DIR/reviews"
du -sh "$REVIEW_DIR"
```

Expected:

- `/mnt/nvme` has enough free space for 40-80 short rendered clips
- `du -sh "$REVIEW_DIR"` is near zero before rendering

- [ ] **Step 4: Discover the two target cohorts without scanning the whole disk repeatedly**

Run on sakurab:

```bash
REVIEW_DIR=/mnt/nvme/chenglin/greenstop_review_2026-07-06
find /mnt/nvme -type d -path '*erga*hiratsuka*' -print > "$REVIEW_DIR/lists/erga_hiratsuka_dirs.txt"
find /mnt/nvme -type d -path '*x2_dev*Fujiyoshida_diffusion_planner*' -print > "$REVIEW_DIR/lists/fujiyoshida_dirs.txt"
head -20 "$REVIEW_DIR/lists/erga_hiratsuka_dirs.txt"
head -20 "$REVIEW_DIR/lists/fujiyoshida_dirs.txt"
```

Expected:

- each file contains at least one dataset directory
- if either file is empty, stop sampling and ask for the correct sakurab dataset root

- [ ] **Step 5: Build a capped NPZ pool from those cohorts**

Run on sakurab:

```bash
REVIEW_DIR=/mnt/nvme/chenglin/greenstop_review_2026-07-06
while read -r d; do find "$d" -type f -name '*.npz'; done < "$REVIEW_DIR/lists/erga_hiratsuka_dirs.txt" \
  | shuf -n 50000 > "$REVIEW_DIR/lists/erga_hiratsuka_pool.txt"
while read -r d; do find "$d" -type f -name '*.npz'; done < "$REVIEW_DIR/lists/fujiyoshida_dirs.txt" \
  | shuf -n 50000 > "$REVIEW_DIR/lists/fujiyoshida_pool.txt"
cat "$REVIEW_DIR/lists/erga_hiratsuka_pool.txt" "$REVIEW_DIR/lists/fujiyoshida_pool.txt" \
  | shuf > "$REVIEW_DIR/lists/combined_pool.txt"
wc -l "$REVIEW_DIR/lists/"*_pool.txt
```

Expected:

- `combined_pool.txt` contains no more than 100000 paths
- this creates text lists only, not converted datasets or videos

- [ ] **Step 6: Generate small skipped and kept review lists**

Use the reference Python detector as the sampling oracle because this implementation intentionally matches its thresholds except for static-object blockers. Limit rendering separately in the next step.

Run on sakurab from the repository root:

```bash
REVIEW_DIR=/mnt/nvme/chenglin/greenstop_review_2026-07-06
PYTHONPATH=reference/at-team-tools/sakayori python3 - <<'PY'
import json
import random
from pathlib import Path

import numpy as np
from npz_cleansing.filters.greenstop import detect_greenstop

review_dir = Path("/mnt/nvme/chenglin/greenstop_review_2026-07-06")
pool = [Path(line.strip()) for line in (review_dir / "lists/combined_pool.txt").read_text().splitlines() if line.strip()]
random.seed(20260706)
random.shuffle(pool)

cfg = {
    "head_tol": 45.0,
    "stay_radius": 2.0,
    "speed_max": 1.0,
    "green_ahead": 40.0,
    "lead_fwd": 30.0,
    "lead_lat": 2.0,
}

skipped = []
kept = []
errors = []
for path in pool:
    if len(skipped) >= 60 and len(kept) >= 60:
        break
    try:
        with np.load(path, allow_pickle=False) as z:
            ego_future = np.asarray(z["ego_agent_future"], dtype=np.float64)
            route_lanes = np.asarray(z["route_lanes"], dtype=np.float64)
            hit = detect_greenstop(z, ego_future, route_lanes, cfg)
            if hit and len(skipped) < 60:
                skipped.append(str(path))
            elif not hit and len(kept) < 60:
                kept.append(str(path))
    except Exception as exc:
        if len(errors) < 20:
            errors.append({"path": str(path), "error": repr(exc)})

(review_dir / "lists/skipped_greenstop_candidates.txt").write_text("\n".join(skipped) + "\n")
(review_dir / "lists/kept_candidates.txt").write_text("\n".join(kept) + "\n")
(review_dir / "lists/sample_errors.json").write_text(json.dumps(errors, indent=2))
print({"skipped": len(skipped), "kept": len(kept), "errors": len(errors)})
PY
wc -l "$REVIEW_DIR/lists/skipped_greenstop_candidates.txt" "$REVIEW_DIR/lists/kept_candidates.txt"
```

Expected:

- `skipped_greenstop_candidates.txt` contains up to 60 paths
- `kept_candidates.txt` contains up to 60 paths
- if skipped count is below 20, stop and inspect whether the cohort paths point at the intended Erga/Fujiyoshida NPZ pool

- [ ] **Step 7: Render a small clip set with clip-review-tool**

Run on sakurab:

```bash
REVIEW_DIR=/mnt/nvme/chenglin/greenstop_review_2026-07-06
cd ../clip-review-tool
uv run render-video-txt "$REVIEW_DIR/lists/skipped_greenstop_candidates.txt" "$REVIEW_DIR/videos/skipped" --workers 4
uv run render-video-txt "$REVIEW_DIR/lists/kept_candidates.txt" "$REVIEW_DIR/videos/kept" --workers 4
du -sh "$REVIEW_DIR"
```

Expected:

- rendered videos exist under `$REVIEW_DIR/videos/skipped` and `$REVIEW_DIR/videos/kept`
- total review directory size remains small enough for the current `/mnt/nvme` free space

- [ ] **Step 8: Review clips and record the outcome**

Run on sakurab:

```bash
REVIEW_DIR=/mnt/nvme/chenglin/greenstop_review_2026-07-06
cd ../clip-review-tool
uv run --with streamlit --with pyyaml streamlit run src/app.py -- \
  --npz_list "$REVIEW_DIR/lists/skipped_greenstop_candidates.txt" \
  --db_path manifest.sqlite
```

Expected:

- skipped samples visually look like ego stopped on green without a lead blocker
- kept samples visually look legitimate, not greenstop, or outside the stationary-future threshold
- review notes are saved under `../clip-review-tool/reviews/`; copy the relevant JSONL into `$REVIEW_DIR/reviews/` after review

- [ ] **Step 9: Commit any sampling note added after review**

If a concise markdown note is added under `docs/`, commit it:

```bash
git add docs
git commit -m "docs: record greenstop sampling results"
```

Expected: commit exists only if review notes are summarized in the repository. If no repository note is added, leave git unchanged.

---

## Self-Review Checklist

- Spec coverage:
  - Default-on greenstop: Task 3 routes it through `decide_frame_skip()`.
  - New skip label: Task 3 appends `GreenStop`.
  - Reference thresholds: Task 1 sets defaults and Task 3 passes them through.
  - Neighbor-only blockers: Task 2 detector checks `neighbor_past` only.
  - Static-object extraction out of scope: documented in Task 1 field comment and README context.
  - Low-footprint sakurab validation: Task 4 caps pools and rendered clips under one review directory.
- Placeholder scan:
  - No unresolved markers.
  - No step asks for unspecified tests or unspecified error handling.
- Type consistency:
  - `ConverterOptions` greenstop field names match `FrameFilterParams`.
  - `detect_green_stop()` argument order matches the call in `decide_frame_skip()`.
  - `decide_frame_skip()` signature changes are reflected in tests and `frame_processor.cpp`.
