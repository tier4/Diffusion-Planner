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

#ifndef IO__BAG_METADATA_HPP_
#define IO__BAG_METADATA_HPP_

#include <string>

struct BagMetadata
{
  std::string log_file_id;
  std::string vehicle_id;
  std::string project_id;
  std::string map_version_id;
  std::string date;
  std::string bag_time;
};

BagMetadata load_bag_metadata(const std::string & rosbag_path);

#endif  // IO__BAG_METADATA_HPP_
