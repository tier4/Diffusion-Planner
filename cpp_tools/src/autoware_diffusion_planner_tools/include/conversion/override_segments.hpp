// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0.

#ifndef CONVERSION__OVERRIDE_SEGMENTS_HPP_
#define CONVERSION__OVERRIDE_SEGMENTS_HPP_

#include "types/override_segment.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ControlModeSample;

std::vector<OverrideSegment> build_override_segments(
  const std::vector<ControlModeSample> & control_modes);

void save_override_segments_json(
  const std::string & output_dir, const std::vector<OverrideSegment> & segments,
  std::size_t control_mode_sample_count);

#endif  // CONVERSION__OVERRIDE_SEGMENTS_HPP_
