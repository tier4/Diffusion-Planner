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

#ifndef ML_PLANNER_DATA__SRC__BAG_FRAME_READER_HPP_
#define ML_PLANNER_DATA__SRC__BAG_FRAME_READER_HPP_

#include "label_builder.hpp"
#include "topic_config.hpp"

#include "autoware/ml_planner/constants.hpp"
#include "autoware/ml_planner/dimensions.hpp"
#include "autoware/ml_planner/preprocessing/input_builder.hpp"
#include "autoware/ml_planner/utils/timed_buffer.hpp"

#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::ml_planner::data {

/**
 * @brief Sequential reader over one rosbag with time-windowed message buffers.
 *
 * Optimized for (near-)chronological access: consecutive frame times read the
 * bag forward incrementally. Requesting an older frame time triggers a seek
 * and buffer rebuild.
 */
class BagFrameReader {
public:
  BagFrameReader(const std::string &bag_path, const TopicConfig &topics);

  /**
   * @brief Build the FrameInputs message windows for the given frame time and
   * run the shared input builder.
   */
  preprocess::InputBuilderResult
  create_input_data(const rclcpp::Time &frame_time,
                    const preprocess::LaneSegmentContext &map_context,
                    const VehicleSpec &vehicle_spec,
                    const preprocess::InputBuilderParams &params);

  /**
   * @brief Build the label tensors (ego/neighbor futures, turn indicator
   * future) for the given frame time.
   */
  preprocess::TensorMapResult create_label_data(
      const rclcpp::Time &frame_time,
      const preprocess::LaneSegmentContext &map_context,
      const LabelBuilderParams &params,
      const std::vector<preprocess::SelectedAgent> &selected_agents);

private:
  using Odometry = nav_msgs::msg::Odometry;
  using TrackedObjects = autoware_perception_msgs::msg::TrackedObjects;
  using TurnIndicatorsReport = autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using TrafficLightGroupArray =
      autoware_perception_msgs::msg::TrafficLightGroupArray;
  using LaneletRoute = autoware_planning_msgs::msg::LaneletRoute;

  // Future horizon covered by the label grid.
  static constexpr double FUTURE_HORIZON_S =
      static_cast<double>(OUTPUT_T) * constants::PREDICTION_TIME_STEP_S;
  // Keep a margin over the model input window plus the label horizon so that
  // one buffer serves both input building (past window ending at the frame
  // time) and label building (future window), while the reader looks ahead.
  static constexpr double BUFFER_WINDOW_S =
      HISTORY_WINDOW_S + FUTURE_HORIZON_S + 3.0;
  // Reading stops once the bag receive time passes the frame time by this
  // margin; header stamps are never later than receive times by more than it.
  static constexpr double LOOKAHEAD_MARGIN_S = 0.5;

  void open_main_reader();
  void ensure_read_until(double target_sec);

  // Copy the sub-window of messages with stamp <= frame time. Messages newer
  // than the frame time must be excluded: the input builder treats the last
  // buffer entry as the current state.
  template <typename MsgT>
  preprocess::MessageView<MsgT>
  window_of(const utils::TimedBuffer<MsgT> &buffer,
            const double frame_sec) const {
    std::vector<const MsgT *> window;
    window.reserve(buffer.msgs().size());
    for (const MsgT &msg : buffer.msgs()) {
      if (stamp_sec_of(msg) <= frame_sec) {
        window.push_back(&msg);
      }
    }
    return preprocess::MessageView<MsgT>(std::move(window));
  }

  const LaneletRoute *route_at(double frame_sec) const;

  static double stamp_sec_of(const Odometry &msg) {
    return rclcpp::Time(msg.header.stamp).seconds();
  }
  static double stamp_sec_of(const TrackedObjects &msg) {
    return rclcpp::Time(msg.header.stamp).seconds();
  }
  static double stamp_sec_of(const TurnIndicatorsReport &msg) {
    return rclcpp::Time(msg.stamp).seconds();
  }
  static double stamp_sec_of(const TrafficLightGroupArray &msg) {
    return rclcpp::Time(msg.stamp).seconds();
  }

  std::string bag_path_;
  TopicConfig topics_;
  std::unique_ptr<rosbag2_cpp::Reader> reader_;
  rosbag2_storage::SerializedBagMessageSharedPtr pending_message_;

  rclcpp::Serialization<Odometry> odom_serializer_;
  rclcpp::Serialization<TrackedObjects> objects_serializer_;
  rclcpp::Serialization<TurnIndicatorsReport> turn_serializer_;
  rclcpp::Serialization<TrafficLightGroupArray> traffic_serializer_;

  utils::TimedBuffer<Odometry> ego_buffer_{
      BUFFER_WINDOW_S,
      [](const Odometry &m) { return rclcpp::Time(m.header.stamp); }};
  utils::TimedBuffer<TrackedObjects> objects_buffer_{
      BUFFER_WINDOW_S,
      [](const TrackedObjects &m) { return rclcpp::Time(m.header.stamp); }};
  utils::TimedBuffer<TurnIndicatorsReport> turn_indicators_buffer_{
      BUFFER_WINDOW_S,
      [](const TurnIndicatorsReport &m) { return rclcpp::Time(m.stamp); }};
  utils::TimedBuffer<TrafficLightGroupArray> traffic_signals_buffer_{
      BUFFER_WINDOW_S,
      [](const TrafficLightGroupArray &m) { return rclcpp::Time(m.stamp); }};

  std::vector<std::pair<double, LaneletRoute>> routes_;
  double last_target_sec_{-std::numeric_limits<double>::infinity()};
};

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__BAG_FRAME_READER_HPP_
