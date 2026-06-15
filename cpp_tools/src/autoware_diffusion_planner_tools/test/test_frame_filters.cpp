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

#include "processing/frame_filters.hpp"

#include <autoware/diffusion_planner/dimensions.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace frame_filters;

// ---------------------------------------------------------------------------
// make_rect
// ---------------------------------------------------------------------------

TEST(MakeRectTest, AxisAlignedBox)
{
  // heading = east (cos=1, sin=0), length=4, width=2, centred at origin
  const Corners c = make_rect(0.0f, 0.0f, 1.0f, 0.0f, 4.0f, 2.0f);
  // Expected corners (FR, FL, RL, RR): (2,1),(2,-1),(-2,-1),(-2,1)
  EXPECT_NEAR(c[0][0], 2.0f, 1e-5f);
  EXPECT_NEAR(c[0][1], 1.0f, 1e-5f);
  EXPECT_NEAR(c[1][0], 2.0f, 1e-5f);
  EXPECT_NEAR(c[1][1], -1.0f, 1e-5f);
  EXPECT_NEAR(c[2][0], -2.0f, 1e-5f);
  EXPECT_NEAR(c[2][1], -1.0f, 1e-5f);
  EXPECT_NEAR(c[3][0], -2.0f, 1e-5f);
  EXPECT_NEAR(c[3][1], 1.0f, 1e-5f);
}

TEST(MakeRectTest, RotatedBox)
{
  // heading = north (cos=0, sin=1), length=2, width=2, centred at (3,4)
  const float angle = static_cast<float>(M_PI_2);
  const Corners c = make_rect(3.0f, 4.0f, std::cos(angle), std::sin(angle), 2.0f, 2.0f);
  // Half-extents in local: hl=1, hw=1
  // FR local=(1,1) → world: (3 + 0*1 - 1*1, 4 + 1*1 + 0*1) = (2,5)
  EXPECT_NEAR(c[0][0], 2.0f, 1e-5f);
  EXPECT_NEAR(c[0][1], 5.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// rect_overlap_sat
// ---------------------------------------------------------------------------

TEST(RectOverlapSatTest, OverlappingAxisAligned)
{
  const Corners a = make_rect(0.0f, 0.0f, 1.0f, 0.0f, 4.0f, 2.0f);
  const Corners b = make_rect(1.0f, 0.0f, 1.0f, 0.0f, 4.0f, 2.0f);  // overlapping
  EXPECT_TRUE(rect_overlap_sat(a, b));
}

TEST(RectOverlapSatTest, SeparatedAxisAligned)
{
  const Corners a = make_rect(0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.0f);
  const Corners b = make_rect(5.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.0f);  // gap > 0
  EXPECT_FALSE(rect_overlap_sat(a, b));
}

TEST(RectOverlapSatTest, TouchingEdgesAreOverlapping)
{
  // Two 2x2 boxes touching at x=1
  const Corners a = make_rect(0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.0f);
  const Corners b = make_rect(2.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.0f);
  // Extents: a right=1, b left=1 → max_a == min_b so NOT a strict gap: no separation
  EXPECT_TRUE(rect_overlap_sat(a, b));
}

// ---------------------------------------------------------------------------
// segments_intersect
// ---------------------------------------------------------------------------

TEST(SegmentsIntersectTest, CrossingSegments)
{
  // (+) and (|) cross at origin
  EXPECT_TRUE(segments_intersect(-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f));
}

TEST(SegmentsIntersectTest, ParallelSegments)
{
  EXPECT_FALSE(segments_intersect(0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f));
}

TEST(SegmentsIntersectTest, TShapeNoIntersect)
{
  // T-shape: horizontal from (-1,0)→(1,0), vertical from (0,0)→(0,1) — shares endpoint
  EXPECT_FALSE(segments_intersect(-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f));
}

// ---------------------------------------------------------------------------
// compute_offlane_score / is_off_lane
// ---------------------------------------------------------------------------

TEST(OfflaneScoreTest, EgoOnLaneReturnsLowScore)
{
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::POSE_DIM;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;

  // Minimal lanes: one valid point at (0,0), rest zero
  const int64_t lanes_size = NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM;
  std::vector<float> lanes(lanes_size, 0.0f);
  // Place a centerline point at (0, 0) — first segment, first point, x=0, y=0 (non-zero check
  // requires fabs(x)+fabs(y) > 1e-6; use (1,0) instead)
  lanes[0] = 1.0f;  // x = 1, y = 0 for first point

  // Ego future: all steps at (1, 0)
  const int64_t future_size = OUTPUT_T * POSE_DIM;
  std::vector<float> ego_future(future_size, 0.0f);
  for (int64_t t = 0; t < OUTPUT_T; ++t) {
    ego_future[t * POSE_DIM + 0] = 1.0f;  // x = 1
    ego_future[t * POSE_DIM + 1] = 0.0f;  // y = 0
  }

  const OffLaneResult result = compute_offlane_score(ego_future, lanes, 1);
  EXPECT_TRUE(result.has_centerline);
  EXPECT_NEAR(result.mean_distance, 0.0f, 1e-3f);
  EXPECT_FALSE(is_off_lane(result, 6.0f));
}

TEST(OfflaneScoreTest, EgoFarFromLaneReturnsHighScore)
{
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::POSE_DIM;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;

  const int64_t lanes_size = NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM;
  std::vector<float> lanes(lanes_size, 0.0f);
  // Single centerline point at (1, 0)
  lanes[0] = 1.0f;

  // Ego future: all steps at (100, 0)
  const int64_t future_size = OUTPUT_T * POSE_DIM;
  std::vector<float> ego_future(future_size, 0.0f);
  for (int64_t t = 0; t < OUTPUT_T; ++t) {
    ego_future[t * POSE_DIM + 0] = 100.0f;
    ego_future[t * POSE_DIM + 1] = 0.0f;
  }

  const OffLaneResult result = compute_offlane_score(ego_future, lanes, 1);
  EXPECT_TRUE(result.has_centerline);
  EXPECT_GT(result.mean_distance, 6.0f);
  EXPECT_TRUE(is_off_lane(result, 6.0f));
}

TEST(OfflaneScoreTest, EmptyLanesNoCenterline)
{
  using autoware::diffusion_planner::NUM_SEGMENTS_IN_LANE;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POINTS_PER_SEGMENT;
  using autoware::diffusion_planner::POSE_DIM;
  using autoware::diffusion_planner::SEGMENT_POINT_DIM;

  const int64_t lanes_size = NUM_SEGMENTS_IN_LANE * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM;
  std::vector<float> lanes(lanes_size, 0.0f);  // all zero → no valid centerline

  const int64_t future_size = OUTPUT_T * POSE_DIM;
  std::vector<float> ego_future(future_size, 0.0f);

  const OffLaneResult result = compute_offlane_score(ego_future, lanes, 1);
  EXPECT_FALSE(result.has_centerline);
  EXPECT_TRUE(is_off_lane(result, 6.0f));
}

// ---------------------------------------------------------------------------
// check_collision (top-level) — basic smoke tests
// ---------------------------------------------------------------------------

TEST(CheckCollisionTest, NoObjectsNoCollision)
{
  using autoware::diffusion_planner::MAX_NUM_NEIGHBORS;
  using autoware::diffusion_planner::OUTPUT_T;
  using autoware::diffusion_planner::POSE_DIM;
  using autoware::diffusion_planner::STATIC_OBJECTS_SHAPE;
  using autoware::diffusion_planner::INPUT_T;

  const std::vector<float> ego_future(OUTPUT_T * POSE_DIM, 0.0f);
  const std::vector<float> ego_shape = {2.75f, 4.34f, 1.70f};
  const std::vector<float> static_objects(STATIC_OBJECTS_SHAPE[1] * STATIC_OBJECTS_SHAPE[2], 0.0f);
  const int64_t past = INPUT_T + 1;
  const int64_t np_dim = 11;
  const std::vector<float> neighbor_past(MAX_NUM_NEIGHBORS * past * np_dim, 0.0f);
  const std::vector<float> neighbor_future(MAX_NUM_NEIGHBORS * OUTPUT_T * POSE_DIM, 0.0f);
  // No line strings
  using autoware::diffusion_planner::LINE_STRING_TYPE_NUM;
  using autoware::diffusion_planner::NUM_LINE_STRINGS;
  using autoware::diffusion_planner::POINTS_PER_LINE_STRING;
  const int64_t ls_dim = 2 + LINE_STRING_TYPE_NUM;
  const std::vector<float> line_strings(NUM_LINE_STRINGS * POINTS_PER_LINE_STRING * ls_dim, 0.0f);

  const CollisionResult r = check_collision(
    ego_future, ego_shape, static_objects, neighbor_future, neighbor_past, line_strings,
    0.0f, 0.0f, 0.0f, 5);
  EXPECT_FALSE(r.collided());
}
