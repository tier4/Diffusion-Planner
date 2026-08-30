// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef ML_PLANNER_DATA__SRC__FRAME_DATA_HPP_
#define ML_PLANNER_DATA__SRC__FRAME_DATA_HPP_

#include "autoware/ml_planner/preprocessing/input_builder.hpp"

namespace autoware::ml_planner::data {

using FrameData = preprocess::TensorMap;
using FrameDataResult = preprocess::TensorMapResult;

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__FRAME_DATA_HPP_
