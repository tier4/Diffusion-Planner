// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PROCESSING__FRAME_FILTERS_HPP_
#define PROCESSING__FRAME_FILTERS_HPP_

// Frame-level "collision free" filter, ported from
// ros_scripts/filter_collision_free_npz.py.
//
// A frame is dropped when its GT ego trajectory (ego_agent_future) collides with
// a static object, a neighbor's future, or a road-border line string. All inputs
// are the flat per-frame float vectors assembled in process_sequence, already in
// the ego-centric frame at t=0, so no transform is required. The data layout
// mirrors what convert_cpp_bin_to_python_npz.py reads back, i.e. the same arrays
// the python filter operated on (ego_future/neighbor_future store [x,y,cos,sin]
// here instead of [x,y,heading], so cos/sin are used directly).

#include <autoware/diffusion_planner/constants.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace frame_filters
{

struct CollisionResult
{
  std::vector<std::string> reasons;  // any of "static_object", "neighbor", "road_border"
  bool collided() const { return !reasons.empty(); }
};

using Corner = std::array<float, 2>;
using Corners = std::array<Corner, 4>;  // 4 corners in CCW order: FR, FL, RL, RR

// Oriented rectangle corners centred at (cx, cy) with heading (cos_h, sin_h).
// Matches compute_rect_corners() in the python filter.
inline Corners make_rect(float cx, float cy, float cos_h, float sin_h, float length, float width)
{
  const float hl = length * 0.5f;
  const float hw = width * 0.5f;
  const std::array<Corner, 4> local = {{{hl, hw}, {hl, -hw}, {-hl, -hw}, {-hl, hw}}};
  Corners out;
  for (int k = 0; k < 4; ++k) {
    const float lx = local[k][0];
    const float ly = local[k][1];
    out[k][0] = cx + cos_h * lx - sin_h * ly;
    out[k][1] = cy + sin_h * lx + cos_h * ly;
  }
  return out;
}

// Separating-axis-theorem overlap test for two oriented rectangles.
inline bool rect_overlap_sat(const Corners & a, const Corners & b)
{
  std::array<Corner, 4> axes;
  auto edge_normals = [&axes](const Corners & c, int base) {
    const float e0x = c[1][0] - c[0][0];
    const float e0y = c[1][1] - c[0][1];
    const float e1x = c[2][0] - c[1][0];
    const float e1y = c[2][1] - c[1][1];
    axes[base] = {-e0y, e0x};
    axes[base + 1] = {-e1y, e1x};
  };
  edge_normals(a, 0);
  edge_normals(b, 2);

  for (int ax = 0; ax < 4; ++ax) {
    const float axx = axes[ax][0];
    const float axy = axes[ax][1];
    float min_a = 1e30f, max_a = -1e30f, min_b = 1e30f, max_b = -1e30f;
    for (int k = 0; k < 4; ++k) {
      const float pa = a[k][0] * axx + a[k][1] * axy;
      const float pb = b[k][0] * axx + b[k][1] * axy;
      min_a = std::min(min_a, pa);
      max_a = std::max(max_a, pa);
      min_b = std::min(min_b, pb);
      max_b = std::max(max_b, pb);
    }
    if (max_a < min_b || max_b < min_a) return false;  // found a separating axis
  }
  return true;
}

inline float point_segment_dist(float px, float py, float ax, float ay, float bx, float by)
{
  const float abx = bx - ax;
  const float aby = by - ay;
  float denom = abx * abx + aby * aby;
  if (denom < 1e-8f) denom = 1e-8f;
  float t = ((px - ax) * abx + (py - ay) * aby) / denom;
  t = std::max(0.0f, std::min(1.0f, t));
  const float cx = ax + t * abx;
  const float cy = ay + t * aby;
  const float dx = px - cx;
  const float dy = py - cy;
  return std::sqrt(dx * dx + dy * dy + 1e-12f);
}

inline float cross2(float ux, float uy, float vx, float vy)
{
  return ux * vy - uy * vx;
}

// Proper (general-position) segment intersection test, matching _segments_intersect_any().
inline bool segments_intersect(
  float p1x, float p1y, float p2x, float p2y, float p3x, float p3y, float p4x, float p4y)
{
  const float d1 = cross2(p4x - p3x, p4y - p3y, p1x - p3x, p1y - p3y);
  const float d2 = cross2(p4x - p3x, p4y - p3y, p2x - p3x, p2y - p3y);
  const float d3 = cross2(p2x - p1x, p2y - p1y, p3x - p1x, p3y - p1y);
  const float d4 = cross2(p2x - p1x, p2y - p1y, p4x - p1x, p4y - p1y);
  return (d1 * d2 < 0.0f) && (d3 * d4 < 0.0f);
}

// Ego bounding-box corners for each evaluated future step (every `stride`-th step).
// ego_future is laid out as OUTPUT_T * POSE_DIM with channels [x, y, cos, sin].
// ego_shape is [wheel_base, length, width]; the box centre is offset forward by
// wheel_base / 2 (ego future xy is the rear axle), matching compute_ego_corners().
inline std::vector<Corners> compute_ego_corners(
  const std::vector<float> & ego_future, const std::vector<float> & ego_shape, int64_t stride)
{
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POSE_DIM;
  const float wheel_base = ego_shape[0];
  const float length = ego_shape[1];
  const float width = ego_shape[2];
  const float cog_offset = 0.5f * wheel_base;

  std::vector<Corners> corners;
  for (int64_t t = 0; t < OUTPUT_T; t += stride) {
    const float * f = &ego_future[t * POSE_DIM];
    const float x = f[0];
    const float y = f[1];
    const float c = f[2];
    const float s = f[3];
    corners.push_back(make_rect(x + c * cog_offset, y + s * cog_offset, c, s, length, width));
  }
  return corners;
}

inline bool check_static_object_collision(
  const std::vector<Corners> & ego, const std::vector<float> & static_objects, float margin)
{
  const int64_t num = autoware::diffusion_planner::STATIC_OBJECTS_SHAPE[1];  // 5
  const int64_t dim = autoware::diffusion_planner::STATIC_OBJECTS_SHAPE[2];  // 10
  for (int64_t n = 0; n < num; ++n) {
    const float * o = &static_objects[n * dim];  // [x, y, cos, sin, w, l, type x4]
    if (std::fabs(o[0]) + std::fabs(o[1]) + std::fabs(o[2]) + std::fabs(o[3]) <= 1e-6f) continue;
    const Corners oc =
      make_rect(o[0], o[1], o[2], o[3], o[5] + 2.0f * margin, o[4] + 2.0f * margin);
    for (const auto & ec : ego) {
      if (rect_overlap_sat(ec, oc)) return true;
    }
  }
  return false;
}

inline bool check_neighbor_collision(
  const std::vector<Corners> & ego, const std::vector<float> & neighbor_future,
  const std::vector<float> & neighbor_past, float margin, int64_t stride)
{
  using autoware::diffusion_planner::INPUT_T;
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POSE_DIM;
  const int64_t past = INPUT_T + 1;  // 31
  const int64_t np_dim = 11;
  const int64_t last = past - 1;

  for (int64_t n = 0; n < MAX_NUM_NEIGHBORS; ++n) {
    const float * lp = &neighbor_past[(n * past + last) * np_dim];
    // neighbor valid iff its last past frame is non-zero
    if (std::fabs(lp[0]) + std::fabs(lp[1]) + std::fabs(lp[2]) + std::fabs(lp[3]) <= 1e-6f)
      continue;
    const float nw = std::max(lp[6] + 2.0f * margin, 1e-3f);  // width
    const float nl = std::max(lp[7] + 2.0f * margin, 1e-3f);  // length

    int64_t k = 0;
    for (int64_t t = 0; t < OUTPUT_T && k < static_cast<int64_t>(ego.size()); t += stride, ++k) {
      const float * f = &neighbor_future[(n * OUTPUT_T + t) * POSE_DIM];
      const float nx = f[0];
      const float ny = f[1];
      if (std::fabs(nx) + std::fabs(ny) <= 1e-6f) continue;  // padded step
      const Corners nc = make_rect(nx, ny, f[2], f[3], nl, nw);
      if (rect_overlap_sat(ego[k], nc)) return true;
    }
  }
  return false;
}

inline bool check_road_border_collision(
  const std::vector<Corners> & ego, const std::vector<float> & line_strings, float margin)
{
  using autoware::diffusion_planner::LINE_STRING_TYPE_NUM;
  using autoware::diffusion_planner::NUM_LINE_STRINGS;
  using autoware::diffusion_planner::POINTS_PER_LINE_STRING;
  const int64_t dim = 2 + LINE_STRING_TYPE_NUM;  // per-point channels: [x, y, type0, type1]
  const int64_t pts = POINTS_PER_LINE_STRING;
  constexpr int64_t border_channel = 3;  // channel 3 > 0.5 marks a road border

  std::vector<std::array<float, 4>> segments;  // {ax, ay, bx, by}
  for (int64_t n = 0; n < NUM_LINE_STRINGS; ++n) {
    const float * base = &line_strings[n * pts * dim];
    bool is_border = false;
    for (int64_t p = 0; p < pts; ++p) {
      if (base[p * dim + border_channel] > 0.5f) {
        is_border = true;
        break;
      }
    }
    if (!is_border) continue;
    for (int64_t p = 0; p + 1 < pts; ++p) {
      const float ax = base[p * dim + 0];
      const float ay = base[p * dim + 1];
      const float bx = base[(p + 1) * dim + 0];
      const float by = base[(p + 1) * dim + 1];
      const bool va = std::fabs(ax) + std::fabs(ay) > 1e-6f;
      const bool vb = std::fabs(bx) + std::fabs(by) > 1e-6f;
      if (va && vb) segments.push_back({ax, ay, bx, by});
    }
  }
  if (segments.empty()) return false;

  // 1) any ego corner within `margin` of a border segment
  if (margin > 0.0f) {
    for (const auto & ec : ego) {
      for (int c = 0; c < 4; ++c) {
        for (const auto & s : segments) {
          if (point_segment_dist(ec[c][0], ec[c][1], s[0], s[1], s[2], s[3]) < margin) return true;
        }
      }
    }
  }

  // 2) any ego box edge crossing a border segment (dominant case when margin == 0)
  for (const auto & ec : ego) {
    for (int c = 0; c < 4; ++c) {
      const float p1x = ec[c][0];
      const float p1y = ec[c][1];
      const float p2x = ec[(c + 1) % 4][0];
      const float p2y = ec[(c + 1) % 4][1];
      for (const auto & s : segments) {
        if (segments_intersect(p1x, p1y, p2x, p2y, s[0], s[1], s[2], s[3])) return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// In-lanelet ("off-lane") filter, ported from score_offroad_npz.py /
// filter_in_lanelet_npz.py: score = mean over the (strided) future steps of the
// minimum distance from ego_agent_future xy to any valid lane centerline point.
// ---------------------------------------------------------------------------

struct OffLaneResult
{
  float mean_distance;  // the score; mean over evaluated steps of min centerline distance
  float max_distance;
  bool has_centerline;  // false == no valid centerline points (treated as +inf score)
};

inline OffLaneResult compute_offlane_score(
  const std::vector<float> & ego_future, const std::vector<float> & lanes, int64_t time_stride)
{
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::POSE_DIM;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;
  if (time_stride < 1) time_stride = 1;

  // Valid lane centerline points (xy), matching collect_centerline_points().
  std::vector<std::array<float, 2>> pts;
  pts.reserve(NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT);
  for (int64_t s = 0; s < NUM_SEGMENTS_IN_LANE; ++s) {
    for (int64_t p = 0; p < POINTS_PER_SEGMENT; ++p) {
      const float * c = &lanes[(s * POINTS_PER_SEGMENT + p) * SEGMENT_POINT_DIM];
      if (std::fabs(c[0]) + std::fabs(c[1]) > 1e-6f) pts.push_back({c[0], c[1]});
    }
  }

  OffLaneResult result{0.0f, 0.0f, !pts.empty()};
  if (pts.empty()) return result;  // score == +inf

  double sum = 0.0;
  int64_t count = 0;
  for (int64_t t = 0; t < OUTPUT_T; t += time_stride) {
    const float ex = ego_future[t * POSE_DIM + 0];
    const float ey = ego_future[t * POSE_DIM + 1];
    float best_sq = 1e30f;
    for (const auto & pt : pts) {
      const float dx = ex - pt[0];
      const float dy = ey - pt[1];
      const float d_sq = dx * dx + dy * dy;
      if (d_sq < best_sq) best_sq = d_sq;
    }
    const float d = std::sqrt(best_sq);
    sum += d;
    result.max_distance = std::max(result.max_distance, d);
    ++count;
  }
  result.mean_distance = static_cast<float>(sum / std::max<int64_t>(count, 1));
  return result;
}

// A frame is off-lane (dropped) when there is no centerline or the mean distance
// reaches max_score, matching filter_in_lanelet_npz.py's `score >= max_score` drop.
inline bool is_off_lane(const OffLaneResult & r, float max_score)
{
  return !r.has_centerline || r.mean_distance >= max_score;
}

// ---------------------------------------------------------------------------
// "Accelerating into a red/yellow signal" detector.
//
// The RedOrYellowLight skip originally fired only when the ego was fully stopped
// (linear.x < 0.1) with a forward GT future. That gate leaked every low-speed /
// creeping start where the ego is already rolling (>= 0.1 m/s) but the GT future
// still drives forward through a red or yellow light. Rather than loosen the speed
// gate (which would also drop legitimate decelerate-to-stop trajectories), we look
// at the *shape* of the GT future speed profile: a frame is flagged only when the
// ego clearly accelerates into the signal — the peak future speed ramps well above
// the initial speed. Pure decel-to-stop and minor creeps (peak < ~3 m/s) are kept.
// This is meant to be OR-ed with the original stopped-at-red condition so coverage
// only ever grows.
// ---------------------------------------------------------------------------

// Leading future steps averaged into the initial speed (0.5 s at 10 Hz).
constexpr int64_t kAccelInitWindow = 5;
constexpr float kAccelDeltaVThreshold = 2.0f;  // m/s gained from initial to peak
constexpr float kAccelPeakThreshold = 3.0f;    // m/s minimum peak future speed
constexpr float kAccelRatioThreshold = 1.3f;   // peak must exceed v_initial * this

struct FutureAccelResult
{
  float v_initial;  // mean speed over the first kAccelInitWindow future steps (m/s)
  float v_peak;     // max per-step speed over the future horizon (m/s)
};

// Per-step speed profile of the GT future, in the ego frame at t=0. ego_future is
// laid out as OUTPUT_T * POSE_DIM with channels [x, y, cos, sin]; speed at step j is
// |p_{j+1} - p_j| / dt. Matches the python analysis used to size the thresholds.
inline FutureAccelResult compute_future_accel(const std::vector<float> & ego_future)
{
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POSE_DIM;
  constexpr float dt =
    static_cast<float>(autoware::diffusion_planner::constants::PREDICTION_TIME_STEP_S);

  float v_peak = 0.0f;
  double init_sum = 0.0;
  int64_t init_count = 0;
  for (int64_t j = 0; j + 1 < OUTPUT_T; ++j) {
    const float dx = ego_future[(j + 1) * POSE_DIM + 0] - ego_future[j * POSE_DIM + 0];
    const float dy = ego_future[(j + 1) * POSE_DIM + 1] - ego_future[j * POSE_DIM + 1];
    const float speed = std::sqrt(dx * dx + dy * dy) / dt;
    v_peak = std::max(v_peak, speed);
    if (j < kAccelInitWindow) {
      init_sum += speed;
      ++init_count;
    }
  }
  const float v_initial = init_count > 0 ? static_cast<float>(init_sum / init_count) : 0.0f;
  return {v_initial, v_peak};
}

// True when the GT future clearly ramps up speed (accelerates) rather than holding
// or decelerating — the signature of driving *into* a red/yellow light.
inline bool is_accelerating(const FutureAccelResult & a)
{
  return (a.v_peak - a.v_initial > kAccelDeltaVThreshold) && (a.v_peak > kAccelPeakThreshold) &&
         (a.v_peak > a.v_initial * kAccelRatioThreshold);
}

// Top-level: returns the list of collision reasons for one frame (empty == keep).
inline CollisionResult check_collision(
  const std::vector<float> & ego_future, const std::vector<float> & ego_shape,
  const std::vector<float> & static_objects, const std::vector<float> & neighbor_future,
  const std::vector<float> & neighbor_past, const std::vector<float> & line_strings,
  float static_object_margin, float neighbor_margin, float road_border_margin, int64_t time_stride)
{
  CollisionResult result;
  if (time_stride < 1) time_stride = 1;
  const std::vector<Corners> ego = compute_ego_corners(ego_future, ego_shape, time_stride);
  if (ego.empty()) return result;

  if (check_static_object_collision(ego, static_objects, static_object_margin)) {
    result.reasons.emplace_back("static_object");
  }
  if (check_neighbor_collision(ego, neighbor_future, neighbor_past, neighbor_margin, time_stride)) {
    result.reasons.emplace_back("neighbor");
  }
  if (check_road_border_collision(ego, line_strings, road_border_margin)) {
    result.reasons.emplace_back("road_border");
  }
  return result;
}

}  // namespace frame_filters

#endif  // PROCESSING__FRAME_FILTERS_HPP_
