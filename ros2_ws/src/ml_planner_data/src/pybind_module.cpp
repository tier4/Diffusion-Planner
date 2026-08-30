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

// Python bindings to build ML planner model inputs directly from
// rosbags, sharing the exact preprocessing code used at inference time
// (autoware::ml_planner::preprocess::create_input_data_map).

#include "bag_dataset_builder.hpp"
#include "frame_data_cache.hpp"
#include "topic_config.hpp"

#include "autoware/ml_planner/dimensions.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rcutils/logging.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

namespace mpd = autoware::ml_planner::data;
using autoware::ml_planner::VehicleSpec;
namespace preprocess = autoware::ml_planner::preprocess;

py::dict to_numpy_dict(const preprocess::TensorMap &input_data_map) {
  py::dict result;
  for (const auto &[key, value] : input_data_map) {
    std::vector<py::ssize_t> shape;
    shape.reserve(value.dimension());
    for (const size_t dimension : value.shape()) {
      shape.push_back(static_cast<py::ssize_t>(dimension));
    }
    py::array_t<float> array(shape);
    std::copy(value.cbegin(), value.cend(), array.mutable_data());
    result[py::str(key)] = std::move(array);
  }
  return result;
}

py::dict stack_frames(const std::vector<mpd::FrameData> &frames) {
  py::dict result;
  if (frames.empty()) {
    return result;
  }
  const auto num_frames = static_cast<py::ssize_t>(frames.size());
  for (const auto &[key, first_value] : frames.front()) {
    std::vector<py::ssize_t> shape{num_frames};
    size_t elements_per_frame = 1;
    for (const size_t dimension : first_value.shape()) {
      shape.push_back(static_cast<py::ssize_t>(dimension));
      elements_per_frame *= dimension;
    }
    py::array_t<float> array(shape);
    float *output = array.mutable_data();
    for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
      const auto value_iterator = frames[frame_index].find(key);
      if (value_iterator == frames[frame_index].end()) {
        throw std::runtime_error("frame is missing tensor key: " + key);
      }
      const auto &value = value_iterator->second;
      if (value.shape() != first_value.shape()) {
        throw std::runtime_error("tensor shape changed between frames: " + key);
      }
      std::copy(value.cbegin(), value.cend(),
                output + frame_index * elements_per_frame);
    }
    result[py::str(key)] = std::move(array);
  }
  return result;
}

py::dict metadata_to_numpy(const std::vector<mpd::BagFrameMetadata> &metadata) {
  const std::vector<py::ssize_t> shape{
      static_cast<py::ssize_t>(metadata.size())};
  py::array_t<int64_t> frame_time_ns(shape);
  py::array_t<float> ego_speed_mps(shape);
  py::array_t<float> ego_yaw_rate_rps(shape);
  py::array_t<uint8_t> turn_indicator(shape);
  py::array_t<int32_t> num_objects(shape);
  for (size_t index = 0; index < metadata.size(); ++index) {
    frame_time_ns.mutable_data()[index] = metadata[index].frame_time_ns;
    ego_speed_mps.mutable_data()[index] = metadata[index].ego_speed_mps;
    ego_yaw_rate_rps.mutable_data()[index] = metadata[index].ego_yaw_rate_rps;
    turn_indicator.mutable_data()[index] = metadata[index].turn_indicator;
    num_objects.mutable_data()[index] = metadata[index].num_objects;
  }
  py::dict result;
  result["frame_time_ns"] = std::move(frame_time_ns);
  result["ego_speed_mps"] = std::move(ego_speed_mps);
  result["ego_yaw_rate_rps"] = std::move(ego_yaw_rate_rps);
  result["turn_indicator"] = std::move(turn_indicator);
  result["num_objects"] = std::move(num_objects);
  return result;
}

py::object create_frame_data(mpd::FrameDataCache &cache,
                             const std::string &bag_path,
                             const std::string &map_path,
                             const int64_t frame_time_ns,
                             const VehicleSpec &vehicle_spec,
                             const double traffic_light_timeout_s,
                             const int64_t num_future_steps,
                             const double neighbor_observation_timeout_s) {
  mpd::FrameDataResult result = tl::unexpected(std::string{"not created"});
  {
    py::gil_scoped_release release;
    result =
        cache.create_frame_data(bag_path, map_path, frame_time_ns, vehicle_spec,
                                traffic_light_timeout_s, num_future_steps,
                                neighbor_observation_timeout_s);
  }
  if (!result) {
    return py::none();
  }
  return to_numpy_dict(result.value());
}

py::dict create_bag_frame_data(const std::string &bag_path,
                               const std::string &map_path,
                               const VehicleSpec &vehicle_spec,
                               const mpd::DatasetBuilderParam &param,
                               const mpd::TopicConfig &topics) {
  mpd::BagDataResult bag_result;
  {
    py::gil_scoped_release release;
    bag_result = mpd::create_bag_frame_data(bag_path, map_path, vehicle_spec,
                                            param, topics);
  }
  py::dict stats;
  stats["all_frames"] = bag_result.all_frames;
  stats["usable_frames"] = bag_result.usable_frames;
  stats["created_frames"] = bag_result.frames.size();
  stats["failed_frames"] = bag_result.failed_frames;
  stats["skipped"] = bag_result.skipped;
  py::dict result;
  result["frames"] = stack_frames(bag_result.frames);
  result["metadata"] = metadata_to_numpy(bag_result.metadata);
  result["warnings"] = py::cast(bag_result.warnings);
  result["stats"] = std::move(stats);
  return result;
}

} // namespace

PYBIND11_MODULE(_ml_planner_data, m) {
  using autoware::ml_planner::HISTORY_WINDOW_S;
  using autoware::ml_planner::OUTPUT_T;

  m.doc() = "Build ML planner model inputs directly from rosbags";

  // Silence the per-file "Opened database ..." INFO logs from rosbag2.
  (void)rcutils_logging_initialize();
  (void)rcutils_logging_set_logger_level("rosbag2_storage",
                                         RCUTILS_LOG_SEVERITY_WARN);

  m.attr("HISTORY_WINDOW_S") = HISTORY_WINDOW_S;

  py::class_<mpd::TopicConfig>(m, "TopicConfig")
      .def(py::init<>())
      .def_readwrite("kinematic_state", &mpd::TopicConfig::kinematic_state)
      .def_readwrite("tracked_objects", &mpd::TopicConfig::tracked_objects)
      .def_readwrite("turn_indicators", &mpd::TopicConfig::turn_indicators)
      .def_readwrite("traffic_signals", &mpd::TopicConfig::traffic_signals)
      .def_readwrite("route", &mpd::TopicConfig::route);

  py::class_<mpd::TopicDropThresholds>(m, "TopicDropThresholds")
      .def(py::init<>())
      .def_readwrite("kinematic_state",
                     &mpd::TopicDropThresholds::kinematic_state)
      .def_readwrite("tracked_objects",
                     &mpd::TopicDropThresholds::tracked_objects)
      .def_readwrite("turn_indicators",
                     &mpd::TopicDropThresholds::turn_indicators)
      .def_readwrite("traffic_signals",
                     &mpd::TopicDropThresholds::traffic_signals);

  py::class_<mpd::DatasetBuilderParam>(m, "DatasetBuilderParam")
      .def(py::init<>())
      .def_readwrite("frame_interval_s",
                     &mpd::DatasetBuilderParam::frame_interval_s)
      .def_readwrite("min_travel_distance",
                     &mpd::DatasetBuilderParam::min_travel_distance)
      .def_readwrite("topic_drop_thresholds",
                     &mpd::DatasetBuilderParam::topic_drop_thresholds)
      .def_readwrite("traffic_light_timeout_s",
                     &mpd::DatasetBuilderParam::traffic_light_timeout_s)
      .def_readwrite("neighbor_observation_timeout_s",
                     &mpd::DatasetBuilderParam::neighbor_observation_timeout_s)
      .def_readwrite("num_future_steps",
                     &mpd::DatasetBuilderParam::num_future_steps);

  py::class_<VehicleSpec>(m, "VehicleSpec")
      .def(py::init<double, double, double>(), py::arg("base_link_to_front"),
           py::arg("vehicle_length"), py::arg("vehicle_width"))
      .def_readonly("base_link_to_front", &VehicleSpec::base_link_to_front)
      .def_readonly("vehicle_length", &VehicleSpec::vehicle_length)
      .def_readonly("vehicle_width", &VehicleSpec::vehicle_width);

  m.def("create_bag_frame_data", &create_bag_frame_data, py::arg("bag_path"),
        py::arg("map_path"), py::arg("vehicle_spec"),
        py::arg("param") = mpd::DatasetBuilderParam{},
        py::arg("topics") = mpd::load_topic_config(),
        "Build all usable model inputs and labels in one bag as stacked "
        "NumPy arrays");

  py::class_<mpd::FrameDataCache>(m, "FrameDataCache")
      .def(py::init<size_t, size_t, mpd::TopicConfig, double>(),
           py::arg("reader_capacity") = 16, py::arg("map_capacity") = 4,
           py::arg("topics") = mpd::load_topic_config(),
           py::arg("line_string_max_step_m") = 5.0)
      .def("create_frame_data", &create_frame_data, py::arg("bag_path"),
           py::arg("map_path"), py::arg("frame_time_ns"),
           py::arg("vehicle_spec"), py::arg("traffic_light_timeout_s") = 0.2,
           py::arg("num_future_steps") = OUTPUT_T,
           py::arg("neighbor_observation_timeout_s") = 0.3,
           "Build the single-batch model inputs and training labels for one "
           "frame as one dict (None "
           "if the frame is not usable)");
}
