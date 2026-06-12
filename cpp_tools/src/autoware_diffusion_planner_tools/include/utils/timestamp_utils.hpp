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

#ifndef UTILS__TIMESTAMP_UTILS_HPP_
#define UTILS__TIMESTAMP_UTILS_HPP_

#include <builtin_interfaces/msg/time.hpp>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

inline std::string create_token(const int64_t seq_id, const int64_t frame_id)
{
  std::ostringstream token_stream;
  token_stream << std::setfill('0') << std::setw(8) << seq_id << "_" << std::setw(8) << frame_id;
  return token_stream.str();
}

inline int64_t parse_timestamp(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

#endif  // UTILS__TIMESTAMP_UTILS_HPP_
