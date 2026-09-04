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

#ifndef PROCESSING__STOPPED_TAIL_HPP_
#define PROCESSING__STOPPED_TAIL_HPP_

// Does this sequence end standing still?
//
// Two things follow from that single judgement in process_sequence: the final pose is taken as
// the goal, and the GT ego future is allowed to hold that pose past the end of the recording. The
// second is what lets conversion run through the final stop: the future spans OUTPUT_T ticks, so
// the frame loop otherwise has to give up OUTPUT_T ticks early, dropping exactly the deceleration
// into the stop. A stopped ego really does stay at that pose, so the held future stays faithful;
// a sequence that ends while still moving has an unknown continuation and keeps the hard cut.

#include "types/frame_data.hpp"

#include <cmath>
#include <vector>

namespace stopped_tail
{

// Speed below which a sequence counts as ending standing still. The absolute speed is used so a
// reversing ego does not count as stopped.
inline constexpr double kEndStoppedSpeedMps = 0.5;

inline bool ends_stopped(const std::vector<FrameData> & data_list)
{
  if (data_list.empty()) {
    return false;
  }
  return std::abs(data_list.back().kinematic_state.twist.twist.linear.x) < kEndStoppedSpeedMps;
}

}  // namespace stopped_tail

#endif  // PROCESSING__STOPPED_TAIL_HPP_
