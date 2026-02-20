// Copyright 2025 TIER IV, Inc. All rights reserved.
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

#ifndef PLANNING__AUTOWARE_DIFFUSION_PLANNER__TOOL__ROSBAG_PARSER_HPP_
#define PLANNING__AUTOWARE_DIFFUSION_PLANNER__TOOL__ROSBAG_PARSER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/converter.hpp>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_cpp/writer.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

namespace rosbag_parser
{

std::string determine_storage_id(const std::string & rosbag_path)
{
  std::filesystem::path path(rosbag_path);
  std::filesystem::path metadata_path = path / "metadata.yaml";

  // Check if metadata.yaml exists
  if (!std::filesystem::exists(metadata_path)) {
    // Fallback: check for .mcap files in the directory
    if (std::filesystem::is_directory(path)) {
      for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".mcap") {
          return "mcap";
        }
      }
    }
    // Default to sqlite3
    return "sqlite3";
  }

  // Read metadata.yaml file
  std::ifstream file(metadata_path);
  if (!file.is_open()) {
    return "sqlite3";  // Default fallback
  }

  std::string line;
  while (std::getline(file, line)) {
    // Look for storage_identifier field
    if (line.find("storage_identifier:") != std::string::npos) {
      std::istringstream iss(line);
      std::string key, value;
      if (iss >> key >> value) {
        // Remove quotes if present
        if (value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.length() - 2);
        }
        return value;
      }
    }
  }

  // Default to sqlite3 if not found
  return "sqlite3";
}

class RosbagParser
{
public:
  explicit RosbagParser(std::string rosbag_path) : rosbag_path_(std::move(rosbag_path))
  {
    std::string storage_id = determine_storage_id(rosbag_path_);
    storage_options_ = {rosbag_path_, storage_id};
    converter_options_ = {"cdr", "cdr"};

    rosbag2_cpp::SerializationFormatConverterFactory factory;
    deserializer_ = factory.load_deserializer("cdr");

    reader_.open(storage_options_, converter_options_);
    all_topic_data_ = reader_.get_all_topics_and_types();
    reader_.close();
  }
  ~RosbagParser() = default;

  void create_reader(const std::string & file_name)
  {
    rosbag2_storage::StorageOptions storage_options{};
    storage_options.uri = file_name;
    storage_options.storage_id = determine_storage_id(file_name);

    try {
      reader_.open(storage_options, converter_options_);
    } catch (std::exception & e) {
      throw std::runtime_error("Failed to open bag file: " + std::string(e.what()));
    }
  }

  std::vector<rosbag2_storage::TopicMetadata> get_all_topic_data() { return all_topic_data_; }

  void create_writer(const std::string & file_name)
  {
    rosbag2_storage::StorageOptions storage_options{};
    storage_options.uri = file_name;
    storage_options.storage_id = determine_storage_id(file_name);

    try {
      writer_.open(storage_options, converter_options_);
    } catch (const std::exception & e) {
      throw std::runtime_error("Failed to create bag file: " + std::string(e.what()));
    }
  }

  void create_topic(const rosbag2_storage::TopicMetadata & meta_data)
  {
    writer_.create_topic(meta_data);
  }

  void create_topic(const std::string & topic_name, const std::string & type)
  {
    rosbag2_storage::TopicMetadata meta_data;
    meta_data.name = topic_name;
    meta_data.type = type;
    meta_data.serialization_format = "cdr";
    writer_.create_topic(meta_data);
  }

  void write_topic(const rosbag2_storage::SerializedBagMessageSharedPtr & msg)
  {
    writer_.write(msg);
  }

  template <typename T>
  void write_topic(const T & msg, const rclcpp::Time & stamp, const std::string & topic_name)
  {
    rclcpp::Serialization<T> serialization;
    rclcpp::SerializedMessage serialized_msg;
    serialization.serialize_message(&msg, &serialized_msg);

    writer_.write(msg, topic_name, stamp);
  }

  bool has_next() { return reader_.has_next(); }

  rosbag2_storage::SerializedBagMessageSharedPtr read_next() { return reader_.read_next(); }

  template <typename T>
  T deserialize_message(const rosbag2_storage::SerializedBagMessageSharedPtr serialized_message)
  {
    T msg;
    rclcpp::Serialization<T> serialization;
    rclcpp::SerializedMessage extracted_serialized_msg(*serialized_message->serialized_data);
    serialization.deserialize_message(&extracted_serialized_msg, &msg);
    return msg;
  }

  template <typename T>
  std::optional<std::deque<T>> parse_topic(const std::string & topic_name)
  {
    try {
      create_reader(rosbag_path_);
    } catch (const std::exception & e) {
      return std::nullopt;
    }

    std::deque<T> topic_queue;
    while (reader_.has_next()) {
      auto serialized_message = reader_.read_next();
      if (serialized_message->topic_name == topic_name) {
        T msg;
        rclcpp::Serialization<T> serialization;
        rclcpp::SerializedMessage extracted_serialized_msg(*serialized_message->serialized_data);
        serialization.deserialize_message(&extracted_serialized_msg, &msg);
        topic_queue.push_back(msg);
      }
    }
    reader_.close();
    return topic_queue;
  }

private:
  rosbag2_cpp::Reader reader_;
  rosbag2_cpp::Writer writer_;
  std::unique_ptr<rosbag2_cpp::converter_interfaces::SerializationFormatDeserializer> deserializer_;
  rosbag2_storage::StorageOptions storage_options_;
  rosbag2_cpp::ConverterOptions converter_options_;

  std::string rosbag_path_;
  std::vector<rosbag2_storage::TopicMetadata> all_topic_data_;
};

}  // namespace rosbag_parser

#endif  // PLANNING__AUTOWARE_DIFFUSION_PLANNER__TOOL__ROSBAG_PARSER_HPP_
