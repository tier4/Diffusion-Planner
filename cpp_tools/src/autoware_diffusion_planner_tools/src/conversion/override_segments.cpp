// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0.

#include "conversion/override_segments.hpp"

#include "nlohmann/json.hpp"
#include "rosbag/parsed_bag_data.hpp"

#include <autoware_vehicle_msgs/msg/control_mode_report.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

std::vector<OverrideSegment> build_override_segments(
  const std::vector<ControlModeSample> & control_modes)
{
  std::vector<OverrideSegment> segments;
  if (control_modes.empty()) {
    return segments;
  }

  int32_t current_mode = control_modes.front().mode;
  int64_t start_timestamp = control_modes.front().rosbag_time;
  int64_t previous_timestamp = control_modes.front().rosbag_time;
  for (size_t index = 1; index < control_modes.size(); ++index) {
    const auto & sample = control_modes[index];
    if (sample.mode != current_mode) {
      if (current_mode == autoware_vehicle_msgs::msg::ControlModeReport::MANUAL) {
        segments.push_back({start_timestamp, sample.rosbag_time});
      }
      current_mode = sample.mode;
      start_timestamp = sample.rosbag_time;
    }
    previous_timestamp = sample.rosbag_time;
  }
  if (current_mode == autoware_vehicle_msgs::msg::ControlModeReport::MANUAL) {
    segments.push_back({start_timestamp, previous_timestamp});
  }
  return segments;
}

void save_override_segments_json(
  const std::string & output_dir, const std::vector<OverrideSegment> & segments,
  const std::size_t control_mode_sample_count)
{
  nlohmann::json payload;
  payload["control_mode_sample_count"] = control_mode_sample_count;
  payload["override_segments"] = nlohmann::json::array();
  for (const auto & segment : segments) {
    payload["override_segments"].push_back(
      {{"start_timestamp_ns", segment.start_timestamp_ns},
       {"end_timestamp_ns", segment.end_timestamp_ns}});
  }
  std::filesystem::create_directories(output_dir);
  const std::filesystem::path output_path =
    std::filesystem::path(output_dir) / "control_mode_4_intervals.json";
  std::ofstream output_file(output_path);
  if (!output_file.is_open()) {
    throw std::runtime_error("failed to open override segment output: " + output_path.string());
  }
  output_file << std::setw(2) << payload << std::endl;
}
