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
#include <rcutils/error_handling.h>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
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
  double yaw;
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

struct DeserializationFailure {
  size_t count{0};
  std::string first_timestamp;
  std::string first_error;
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
  std::map<std::string, DeserializationFailure> deserialization_failures;

  while (reader.has_next()) {
    const auto bag_message = reader.read_next();
    rclcpp::SerializedMessage raw(*bag_message->serialized_data);
    const std::string &topic = bag_message->topic_name;
    const auto record_deserialization_failure =
        [&](const std::exception &error) {
          if (rcutils_error_is_set()) {
            rcutils_reset_error();
          }
          auto &[count, first_timestamp, first_error] =
              deserialization_failures[topic];
          ++count;
          if (count == 1) {
            first_timestamp = std::to_string(bag_message->time_stamp);
            first_error = error.what();
          }
        };
    if (topic == topics.kinematic_state) {
      Odometry message;
      try {
        odom_serializer.deserialize_message(&raw, &message);
      } catch (const std::exception &error) {
        record_deserialization_failure(error);
        continue;
      }
      const auto &orientation = message.pose.pose.orientation;
      const double yaw = std::atan2(
          2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
          1.0 - 2.0 * (orientation.y * orientation.y +
                       orientation.z * orientation.z));
      ego_samples.emplace_back(
          rclcpp::Time(message.header.stamp).seconds(),
          EgoSample{static_cast<float>(message.twist.twist.linear.x),
                    static_cast<float>(message.twist.twist.angular.z),
                    message.pose.pose.position.x, message.pose.pose.position.y,
                    yaw});
    } else if (topic == topics.tracked_objects) {
      TrackedObjects message;
      try {
        objects_serializer.deserialize_message(&raw, &message);
      } catch (const std::exception &error) {
        record_deserialization_failure(error);
        continue;
      }
      object_samples.emplace_back(rclcpp::Time(message.header.stamp).seconds(),
                                  static_cast<int32_t>(message.objects.size()));
    } else if (topic == topics.turn_indicators) {
      TurnIndicatorsReport message;
      try {
        turn_serializer.deserialize_message(&raw, &message);
      } catch (const std::exception &error) {
        record_deserialization_failure(error);
        continue;
      }
      turn_samples.emplace_back(rclcpp::Time(message.stamp).seconds(),
                                message.report);
    } else if (topic == topics.traffic_signals) {
      TrafficLightGroupArray message;
      try {
        traffic_serializer.deserialize_message(&raw, &message);
      } catch (const std::exception &error) {
        record_deserialization_failure(error);
        continue;
      }
      traffic_stamps.push_back(rclcpp::Time(message.stamp).seconds());
    } else if (topic == topics.route) {
      LaneletRoute message;
      try {
        route_serializer.deserialize_message(&raw, &message);
      } catch (const std::exception &error) {
        record_deserialization_failure(error);
        continue;
      }
      if (!message.segments.empty()) {
        route_stamps.push_back(rclcpp::Time(message.header.stamp).seconds());
      }
    }
  }

  CandidateResult result;
  for (const auto &[topic, failure] : deserialization_failures) {
    result.warnings.push_back(
        "skipped " + std::to_string(failure.count) + " incompatible " + topic +
        " message(s); first failure at bag timestamp " +
        failure.first_timestamp + ": " + failure.first_error);
  }
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
  DatasetBuilderParam frame_range_param = param;
  const auto disable_incompatible_topic = [&](const std::vector<double> &stamps,
                                              const std::string &topic,
                                              double &threshold) {
    const auto failure = deserialization_failures.find(topic);
    if (!stamps.empty() || failure == deserialization_failures.end()) {
      return;
    }
    threshold = 0.0;
    result.warnings.push_back(
        "proceeding without " + topic + " because all " +
        std::to_string(failure->second.count) +
        " message(s) were skipped as incompatible; dropout validation is "
        "disabled for this bag");
  };
  disable_incompatible_topic(
      turn_stamps, topics.turn_indicators,
      frame_range_param.topic_drop_thresholds.turn_indicators);
  disable_incompatible_topic(
      object_stamps, topics.tracked_objects,
      frame_range_param.topic_drop_thresholds.tracked_objects);
  disable_incompatible_topic(
      traffic_stamps, topics.traffic_signals,
      frame_range_param.topic_drop_thresholds.traffic_signals);
  const FrameRange frame_range = calculate_frame_range(
      topics, frame_range_param, ego_stamps, turn_stamps, object_stamps,
      traffic_stamps, route_stamps, num_frames);
  result.usable_frames = frame_range.usable_frames;
  result.warnings.insert(result.warnings.end(), frame_range.warnings.begin(),
                         frame_range.warnings.end());

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
    result.frames.push_back({BagFrameMetadata{
        static_cast<int64_t>(std::llround(time * 1e9)),
        ego != nullptr ? ego->x : 0.0, ego != nullptr ? ego->y : 0.0,
        ego != nullptr ? ego->yaw : 0.0, ego != nullptr ? ego->speed_mps : 0.0F,
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
