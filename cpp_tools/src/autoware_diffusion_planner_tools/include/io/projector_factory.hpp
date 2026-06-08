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

#ifndef IO__PROJECTOR_FACTORY_HPP_
#define IO__PROJECTOR_FACTORY_HPP_

#include <lanelet2_io/Projection.h>

#include <memory>
#include <string>

std::unique_ptr<lanelet::Projector> create_projector_from_yaml(const std::string & vector_map_path);

#endif  // IO__PROJECTOR_FACTORY_HPP_
