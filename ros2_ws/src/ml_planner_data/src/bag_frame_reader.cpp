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

#include "bag_frame_reader.hpp"

#include <rclcpp/serialized_message.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace autoware::ml_planner::data {

BagFrameReader::BagFrameReader(const std::string &bag_path,
                               const TopicConfig &topics)
    : bag_path_(bag_path), topics_(topics) {
  // Collect all route messages once (rare, usually one per bag).
  {
    rosbag2_cpp::Reader route_reader;
    route_reader.open(bag_path_);
    rosbag2_storage::StorageFilter filter;
    filter.topics = {topics_.route};
    route_reader.set_filter(filter);
    rclcpp::Serialization<LaneletRoute> serializer;
    while (route_reader.has_next()) {
      const auto bag_msg = route_reader.read_next();
      rclcpp::SerializedMessage raw(*bag_msg->serialized_data);
      LaneletRoute route;
      serializer.deserialize_message(&raw, &route);
      routes_.emplace_back(rclcpp::Time(route.header.stamp).seconds(), route);
    }
  }

  open_main_reader();
}

preprocess::InputBuilderResult BagFrameReader::create_input_data(
    const rclcpp::Time &frame_time,
    const preprocess::LaneSegmentContext &map_context,
    const VehicleSpec &vehicle_spec,
    const preprocess::InputBuilderParams &params) {
  const double frame_sec = frame_time.seconds();
  ensure_read_until(frame_sec);

  const auto ego_history = window_of(ego_buffer_, frame_sec);
  if (ego_history.empty()) {
    return tl::unexpected(
        "No ego odometry at or before the requested frame time in " +
        bag_path_);
  }
  const LaneletRoute *route = route_at(frame_sec);
  if (route == nullptr) {
    return tl::unexpected("No route message in " + bag_path_);
  }

  const preprocess::FrameInputs frame_inputs{
      frame_time,
      ego_history,
      window_of(turn_indicators_buffer_, frame_sec),
      window_of(objects_buffer_, frame_sec),
      window_of(traffic_signals_buffer_, frame_sec),
      *route};

  return preprocess::create_input_data_map(frame_inputs, map_context,
                                           vehicle_spec, params);
}

preprocess::TensorMapResult BagFrameReader::create_label_data(
    const rclcpp::Time &frame_time,
    const preprocess::LaneSegmentContext &map_context,
    const LabelBuilderParams &params,
    const std::vector<preprocess::SelectedAgent> &selected_agents) {
  const double frame_sec = frame_time.seconds();
  const double horizon_s =
      static_cast<double>(params.num_future_steps) * params.time_step_s;
  ensure_read_until(frame_sec + horizon_s);

  const LaneletRoute *route = route_at(frame_sec);
  if (route == nullptr) {
    return tl::unexpected("No route message in " + bag_path_);
  }

  // Windows must reach the end of the future grid; a small margin keeps the
  // bracketing message for interpolation at the last grid time.
  const double cutoff_sec = frame_sec + horizon_s + LOOKAHEAD_MARGIN_S;
  return create_label_data_map(frame_time, window_of(ego_buffer_, cutoff_sec),
                               window_of(objects_buffer_, cutoff_sec),
                               window_of(turn_indicators_buffer_, cutoff_sec),
                               window_of(traffic_signals_buffer_, cutoff_sec),
                               *route, map_context, selected_agents, params);
}

void BagFrameReader::open_main_reader() {
  reader_ = std::make_unique<rosbag2_cpp::Reader>();
  reader_->open(bag_path_);
  rosbag2_storage::StorageFilter filter;
  filter.topics = {topics_.kinematic_state, topics_.tracked_objects,
                   topics_.turn_indicators, topics_.traffic_signals};
  reader_->set_filter(filter);
  pending_message_.reset();
  ego_buffer_.clear();
  objects_buffer_.clear();
  turn_indicators_buffer_.clear();
  traffic_signals_buffer_.clear();
}

void BagFrameReader::ensure_read_until(const double target_sec) {
  // Backward jump: rebuild from a seek before the needed window.
  if (target_sec < last_target_sec_ - BUFFER_WINDOW_S + HISTORY_WINDOW_S) {
    open_main_reader();
    const auto seek_ns = static_cast<rcutils_time_point_value_t>(
        (target_sec - HISTORY_WINDOW_S - 1.0) * 1e9);
    reader_->seek(seek_ns);
    last_target_sec_ = target_sec;
  } else {
    last_target_sec_ = std::max(last_target_sec_, target_sec);
  }

  const double receive_time_limit = target_sec + LOOKAHEAD_MARGIN_S;
  while (pending_message_ != nullptr || reader_->has_next()) {
    const auto bag_msg = pending_message_ != nullptr
                             ? std::exchange(pending_message_, nullptr)
                             : reader_->read_next();
    const double receive_sec = static_cast<double>(bag_msg->time_stamp) * 1e-9;
    if (receive_sec > receive_time_limit) {
      pending_message_ = bag_msg;
      break;
    }
    rclcpp::SerializedMessage raw(*bag_msg->serialized_data);
    const std::string &topic = bag_msg->topic_name;

    if (topic == topics_.kinematic_state) {
      Odometry msg;
      odom_serializer_.deserialize_message(&raw, &msg);
      ego_buffer_.push_back(msg);
    } else if (topic == topics_.tracked_objects) {
      TrackedObjects msg;
      objects_serializer_.deserialize_message(&raw, &msg);
      objects_buffer_.push_back(msg);
    } else if (topic == topics_.turn_indicators) {
      TurnIndicatorsReport msg;
      turn_serializer_.deserialize_message(&raw, &msg);
      turn_indicators_buffer_.push_back(msg);
    } else if (topic == topics_.traffic_signals) {
      TrafficLightGroupArray msg;
      traffic_serializer_.deserialize_message(&raw, &msg);
      traffic_signals_buffer_.push_back(msg);
    }
  }
}

const BagFrameReader::LaneletRoute *
BagFrameReader::route_at(const double frame_sec) const {
  const LaneletRoute *result = nullptr;
  for (const auto &[stamp_sec, route] : routes_) {
    if (stamp_sec <= frame_sec || result == nullptr) {
      result = &route;
    }
  }
  return result;
}

} // namespace autoware::ml_planner::data
