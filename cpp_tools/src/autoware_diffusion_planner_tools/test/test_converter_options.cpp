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

#include "cli/converter_options.hpp"

#include <gtest/gtest.h>

// Helper: build a default-ish ConverterOptions for testing apply_named_arg.
static ConverterOptions make_default_opts()
{
  ConverterOptions o{};
  o.step = 1;
  o.limit = -1;
  o.min_frames = 1700;
  o.search_nearest_route = 1;
  o.convert_yellow = 0;
  o.convert_red = 0;
  o.interpolation = 1;
  o.min_distance = 50.0;
  o.ego_wheel_base = 2.75f;
  o.ego_length = 4.34f;
  o.ego_width = 1.70f;
  o.static_object_margin = 0.0f;
  o.neighbor_margin = 0.0f;
  o.road_border_margin = 0.0f;
  o.collision_time_stride = 5;
  o.offlane_max_score = 6.0f;
  o.offlane_time_stride = 1;
  o.write_skipped_npz = false;
  return o;
}

// ---------------------------------------------------------------------------
// apply_named_arg tests
// ---------------------------------------------------------------------------

TEST(ApplyNamedArgTest, RecognisedStep)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--step=3"));
  EXPECT_EQ(opts.step, 3);
}

TEST(ApplyNamedArgTest, RecognisedLimit)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--limit=100"));
  EXPECT_EQ(opts.limit, 100);
}

TEST(ApplyNamedArgTest, RecognisedMinFrames)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--min_frames=500"));
  EXPECT_EQ(opts.min_frames, 500);
}

TEST(ApplyNamedArgTest, RecognisedMinDistance)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--min_distance=20.5"));
  EXPECT_DOUBLE_EQ(opts.min_distance, 20.5);
}

TEST(ApplyNamedArgTest, RecognisedConvertYellow)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--convert_yellow=1"));
  EXPECT_EQ(opts.convert_yellow, 1);
}

TEST(ApplyNamedArgTest, RecognisedConvertRed)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--convert_red=1"));
  EXPECT_EQ(opts.convert_red, 1);
}

TEST(ApplyNamedArgTest, RecognisedInterpolation)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--interpolation=0"));
  EXPECT_EQ(opts.interpolation, 0);
}

TEST(ApplyNamedArgTest, RecognisedEgoDimensions)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--ego_wheel_base=3.0"));
  EXPECT_FLOAT_EQ(opts.ego_wheel_base, 3.0f);

  EXPECT_TRUE(apply_named_arg(opts, "--ego_length=5.0"));
  EXPECT_FLOAT_EQ(opts.ego_length, 5.0f);

  EXPECT_TRUE(apply_named_arg(opts, "--ego_width=2.0"));
  EXPECT_FLOAT_EQ(opts.ego_width, 2.0f);
}

TEST(ApplyNamedArgTest, RecognisedCollisionFilterParams)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--static_object_margin=0.5"));
  EXPECT_FLOAT_EQ(opts.static_object_margin, 0.5f);

  EXPECT_TRUE(apply_named_arg(opts, "--neighbor_margin=0.3"));
  EXPECT_FLOAT_EQ(opts.neighbor_margin, 0.3f);

  EXPECT_TRUE(apply_named_arg(opts, "--road_border_margin=0.1"));
  EXPECT_FLOAT_EQ(opts.road_border_margin, 0.1f);

  EXPECT_TRUE(apply_named_arg(opts, "--collision_time_stride=2"));
  EXPECT_EQ(opts.collision_time_stride, 2);
}

TEST(ApplyNamedArgTest, RecognisedOfflaneFilterParams)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--offlane_max_score=4.0"));
  EXPECT_FLOAT_EQ(opts.offlane_max_score, 4.0f);

  EXPECT_TRUE(apply_named_arg(opts, "--offlane_time_stride=2"));
  EXPECT_EQ(opts.offlane_time_stride, 2);
}

TEST(ApplyNamedArgTest, RecognisedWriteSkippedNpz)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_TRUE(apply_named_arg(opts, "--write_skipped_npz=1"));
  EXPECT_TRUE(opts.write_skipped_npz);
}

TEST(ApplyNamedArgTest, UnrecognisedArgReturnsFalse)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_FALSE(apply_named_arg(opts, "--unknown_arg=42"));
  EXPECT_FALSE(apply_named_arg(opts, "not_an_option"));
  EXPECT_FALSE(apply_named_arg(opts, ""));
}

TEST(ApplyNamedArgTest, InvalidValueReturnsFalse)
{
  ConverterOptions opts = make_default_opts();
  // Malformed values should not throw — they return false.
  EXPECT_FALSE(apply_named_arg(opts, "--step=abc"));
  EXPECT_FALSE(apply_named_arg(opts, "--ego_wheel_base=not_a_number"));
  EXPECT_FALSE(apply_named_arg(opts, "--min_distance=x"));
}

TEST(ApplyNamedArgTest, DoesNotModifyOtherFieldsOnSingleArg)
{
  ConverterOptions before = make_default_opts();
  ConverterOptions after = make_default_opts();
  apply_named_arg(after, "--step=7");
  // Only step should differ
  EXPECT_EQ(after.step, 7);
  EXPECT_EQ(after.limit, before.limit);
  EXPECT_EQ(after.min_frames, before.min_frames);
}

// ---------------------------------------------------------------------------
// validate_options tests
// ---------------------------------------------------------------------------

TEST(ValidateOptionsTest, ValidDimensionsReturnsNullopt)
{
  ConverterOptions opts = make_default_opts();
  EXPECT_FALSE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeWheelBaseReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.ego_wheel_base = -1.0f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeLengthReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.ego_length = -0.1f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, NegativeWidthReturnsError)
{
  ConverterOptions opts = make_default_opts();
  opts.ego_width = -1.0f;
  EXPECT_TRUE(validate_options(opts).has_value());
}

TEST(ValidateOptionsTest, ZeroWheelBasePasses)
{
  ConverterOptions opts = make_default_opts();
  opts.ego_wheel_base = 0.0f;
  // Validation requires < 0.0; zero itself is accepted.
  EXPECT_FALSE(validate_options(opts).has_value());
}
