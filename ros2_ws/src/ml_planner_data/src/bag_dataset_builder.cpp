// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "bag_dataset_builder.hpp"

#include "frame_data_cache.hpp"
#include "skip_index.hpp"
#include "topic_config.hpp"

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp/time.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace autoware::ml_planner::data {
namespace {

template <typename SampleT>
const SampleT *
latest_at_or_before(const std::vector<std::pair<double, SampleT>> &samples,
                    const double time, size_t &cursor) {
  while (cursor + 1 < samples.size() && samples[cursor + 1].first <= time) {
    ++cursor;
  }
  if (samples.empty() || samples[cursor].first > time) {
    return nullptr;
  }
  return &samples[cursor].second;
}

template <typename SampleT>
std::vector<double>
stamps_of(const std::vector<std::pair<double, SampleT>> &samples) {
  std::vector<double> stamps;
  stamps.reserve(samples.size());
  for (const auto &[stamp, unused] : samples) {
    (void)unused;
    stamps.push_back(stamp);
  }
  return stamps;
}

struct EgoSample {
  float speed_mps;
  float yaw_rate_rps;
  double x;
  double y;
};

struct CandidateFrame {
  BagFrameMetadata metadata;
};

struct CandidateResult {
  std::vector<CandidateFrame> frames;
  std::vector<std::string> warnings;
  size_t all_frames{0};
  size_t usable_frames{0};
  bool skipped{false};
};

CandidateResult collect_candidates(const std::string &bag_path,
                                   const TopicConfig &topics,
                                   const DatasetBuilderParam &param) {
  using autoware_perception_msgs::msg::TrackedObjects;
  using autoware_perception_msgs::msg::TrafficLightGroupArray;
  using autoware_planning_msgs::msg::LaneletRoute;
  using autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using nav_msgs::msg::Odometry;

  std::vector<std::pair<double, EgoSample>> ego_samples;
  std::vector<std::pair<double, uint8_t>> turn_samples;
  std::vector<std::pair<double, int32_t>> object_samples;
  std::vector<double> traffic_stamps;
  std::vector<double> route_stamps;

  rosbag2_cpp::Reader reader;
  reader.open(bag_path);
  rosbag2_storage::StorageFilter filter;
  filter.topics = {topics.kinematic_state, topics.tracked_objects,
                   topics.turn_indicators, topics.traffic_signals,
                   topics.route};
  reader.set_filter(filter);

  rclcpp::Serialization<Odometry> odom_serializer;
  rclcpp::Serialization<TrackedObjects> objects_serializer;
  rclcpp::Serialization<TurnIndicatorsReport> turn_serializer;
  rclcpp::Serialization<TrafficLightGroupArray> traffic_serializer;
  rclcpp::Serialization<LaneletRoute> route_serializer;

  while (reader.has_next()) {
    const auto bag_message = reader.read_next();
    rclcpp::SerializedMessage raw(*bag_message->serialized_data);
    const std::string &topic = bag_message->topic_name;
    if (topic == topics.kinematic_state) {
      Odometry message;
      odom_serializer.deserialize_message(&raw, &message);
      ego_samples.emplace_back(
          rclcpp::Time(message.header.stamp).seconds(),
          EgoSample{static_cast<float>(message.twist.twist.linear.x),
                    static_cast<float>(message.twist.twist.angular.z),
                    message.pose.pose.position.x,
                    message.pose.pose.position.y});
    } else if (topic == topics.tracked_objects) {
      TrackedObjects message;
      objects_serializer.deserialize_message(&raw, &message);
      object_samples.emplace_back(rclcpp::Time(message.header.stamp).seconds(),
                                  static_cast<int32_t>(message.objects.size()));
    } else if (topic == topics.turn_indicators) {
      TurnIndicatorsReport message;
      turn_serializer.deserialize_message(&raw, &message);
      turn_samples.emplace_back(rclcpp::Time(message.stamp).seconds(),
                                message.report);
    } else if (topic == topics.traffic_signals) {
      TrafficLightGroupArray message;
      traffic_serializer.deserialize_message(&raw, &message);
      traffic_stamps.push_back(rclcpp::Time(message.stamp).seconds());
    } else if (topic == topics.route) {
      LaneletRoute message;
      route_serializer.deserialize_message(&raw, &message);
      if (!message.segments.empty()) {
        route_stamps.push_back(rclcpp::Time(message.header.stamp).seconds());
      }
    }
  }

  CandidateResult result;
  if (ego_samples.empty()) {
    return result;
  }

  const auto by_stamp = [](const auto &left, const auto &right) {
    return left.first < right.first;
  };
  std::sort(ego_samples.begin(), ego_samples.end(), by_stamp);
  std::sort(turn_samples.begin(), turn_samples.end(), by_stamp);
  std::sort(object_samples.begin(), object_samples.end(), by_stamp);
  std::sort(traffic_stamps.begin(), traffic_stamps.end());
  std::sort(route_stamps.begin(), route_stamps.end());

  std::vector<std::pair<double, double>> ego_positions;
  ego_positions.reserve(ego_samples.size());
  for (const auto &sample : ego_samples) {
    ego_positions.emplace_back(sample.second.x, sample.second.y);
  }
  if (const auto warning =
          check_min_travel_distance(ego_positions, param.min_travel_distance)) {
    result.warnings.push_back(*warning);
    result.skipped = true;
    return result;
  }

  const double first_sec = ego_samples.front().first;
  const double last_sec = ego_samples.back().first;
  const auto num_frames =
      static_cast<size_t>(
          std::floor((last_sec - first_sec) / param.frame_interval_s)) +
      1;
  result.all_frames = num_frames;

  const std::vector<double> ego_stamps = stamps_of(ego_samples);
  const std::vector<double> turn_stamps = stamps_of(turn_samples);
  const std::vector<double> object_stamps = stamps_of(object_samples);
  const FrameRange frame_range = calculate_frame_range(
      topics, param, ego_stamps, turn_stamps, object_stamps, traffic_stamps,
      route_stamps, num_frames);
  result.usable_frames = frame_range.usable_frames;
  result.warnings = frame_range.warnings;

  size_t ego_cursor = 0;
  size_t turn_cursor = 0;
  size_t object_cursor = 0;
  size_t invalid_range_cursor = 0;
  result.frames.reserve(num_frames);
  for (size_t index = 0; index < num_frames; ++index) {
    const double time =
        first_sec + static_cast<double>(index) * param.frame_interval_s;
    const EgoSample *ego = latest_at_or_before(ego_samples, time, ego_cursor);
    const uint8_t *turn = latest_at_or_before(turn_samples, time, turn_cursor);
    const int32_t *objects =
        latest_at_or_before(object_samples, time, object_cursor);

    if (time > frame_range.last_valid_t) {
      break;
    }
    if (time < frame_range.first_valid_t ||
        is_frame_invalid(frame_range.invalid_ranges, time,
                         invalid_range_cursor)) {
      continue;
    }
    result.frames.push_back(
        {BagFrameMetadata{static_cast<int64_t>(std::llround(time * 1e9)),
                          ego != nullptr ? ego->speed_mps : 0.0F,
                          ego != nullptr ? ego->yaw_rate_rps : 0.0F,
                          turn != nullptr ? *turn : uint8_t{0},
                          objects != nullptr ? *objects : int32_t{0}}});
  }
  return result;
}

void validate_param(const DatasetBuilderParam &param) {
  if (!std::isfinite(param.frame_interval_s) || param.frame_interval_s <= 0.0) {
    throw std::invalid_argument(
        "frame_interval_s must be finite and greater than zero");
  }
  if (!std::isfinite(param.min_travel_distance) ||
      param.min_travel_distance < 0.0) {
    throw std::invalid_argument(
        "min_travel_distance must be finite and non-negative");
  }
  if (param.num_future_steps <= 0) {
    throw std::invalid_argument("num_future_steps must be positive");
  }
}

} // namespace

BagDataResult create_bag_frame_data(const std::string &bag_path,
                                    const std::string &map_path,
                                    const VehicleSpec &vehicle_spec,
                                    const DatasetBuilderParam &param,
                                    const TopicConfig &topics) {
  validate_param(param);
  CandidateResult candidates = collect_candidates(bag_path, topics, param);

  BagDataResult result;
  result.warnings = std::move(candidates.warnings);
  result.all_frames = candidates.all_frames;
  result.usable_frames = candidates.usable_frames;
  result.skipped = candidates.skipped;
  if (result.skipped || candidates.frames.empty()) {
    return result;
  }

  FrameDataCache cache(1, 1, topics, 5.0);
  result.frames.reserve(candidates.frames.size());
  result.metadata.reserve(candidates.frames.size());
  for (const CandidateFrame &candidate : candidates.frames) {
    FrameDataResult frame = cache.create_frame_data(
        bag_path, map_path, candidate.metadata.frame_time_ns, vehicle_spec,
        param.traffic_light_timeout_s, param.num_future_steps,
        param.neighbor_observation_timeout_s);
    if (!frame) {
      ++result.failed_frames;
      result.warnings.push_back(
          "frame " + std::to_string(candidate.metadata.frame_time_ns) +
          " could not be created: " + frame.error());
      continue;
    }
    result.frames.push_back(std::move(frame.value()));
    result.metadata.push_back(candidate.metadata);
  }
  return result;
}

} // namespace autoware::ml_planner::data
