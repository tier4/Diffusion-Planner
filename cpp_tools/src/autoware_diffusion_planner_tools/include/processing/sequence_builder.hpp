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

#ifndef PROCESSING__SEQUENCE_BUILDER_HPP_
#define PROCESSING__SEQUENCE_BUILDER_HPP_

#include "rosbag/parsed_bag_data.hpp"
#include "types/frame_data.hpp"

#include <cstdint>
#include <string>
#include <vector>

std::vector<SequenceData> build_sequences(ParsedBagData & data, const int64_t search_nearest_route);

#endif  // PROCESSING__SEQUENCE_BUILDER_HPP_
