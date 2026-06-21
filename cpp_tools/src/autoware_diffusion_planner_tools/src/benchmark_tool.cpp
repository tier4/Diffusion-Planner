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
//
// Benchmark tool for diffusion planner TRT inference.
// Uses the actual TensorrtInference class from autoware_universe,
// so it automatically reflects any host-side optimizations in the real code.
//
// Usage:
//   ros2 run autoware_diffusion_planner_tools benchmark_tool
//   ros2 run autoware_diffusion_planner_tools benchmark_tool --runs 300 --warmup 50

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/inference/single_step_inference.hpp>
#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/parameter_map.hpp>

#include <rcl/allocator.h>
#include <rcl_yaml_param_parser/parser.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace autoware::diffusion_planner;

// --- Parameter helpers (same as inference_tool) ---
std::string resolve_substitutions(const std::string & str)
{
  std::string result = str;
  std::regex env_re(R"(\$\(env\s+(\w+)\))");
  std::smatch match;
  while (std::regex_search(result, match, env_re)) {
    const char * val = std::getenv(match[1].str().c_str());
    result = match.prefix().str() + (val ? val : "") + match.suffix().str();
  }
  std::regex pkg_re(R"(\$\(find-pkg-share\s+([\w-]+)\))");
  while (std::regex_search(result, match, pkg_re)) {
    const std::string pkg_dir = ament_index_cpp::get_package_share_directory(match[1].str());
    result = match.prefix().str() + pkg_dir + match.suffix().str();
  }
  return result;
}

using ParamMap = std::unordered_map<std::string, rclcpp::Parameter>;

ParamMap load_param_map(const std::string & yaml_path)
{
  rcl_params_t * params_st = rcl_yaml_node_struct_init(rcl_get_default_allocator());
  if (!rcl_parse_yaml_file(yaml_path.c_str(), params_st)) {
    std::cerr << "Failed to parse yaml: " << yaml_path << std::endl;
    std::exit(1);
  }
  const rclcpp::ParameterMap param_map = rclcpp::parameter_map_from(params_st, "");
  rcl_yaml_node_struct_fini(params_st);

  ParamMap flat_map;
  for (const auto & [ns, params] : param_map) {
    for (const auto & p : params) {
      flat_map[p.get_name()] = p;
    }
  }
  return flat_map;
}

template <typename T>
T get_param(const ParamMap & params, const std::string & name, const T & default_val)
{
  const auto it = params.find(name);
  if (it == params.end()) {
    return default_val;
  }
  return it->second.get_value<T>();
}

// --- Generate random InputDataMap matching model dimensions ---
preprocess::InputDataMap generate_random_inputs(std::mt19937 & gen)
{
  std::normal_distribution<float> dist(0.0f, 0.5f);

  const auto make_random = [&](const auto & shape) {
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      n *= static_cast<size_t>(shape[i]);
    }
    std::vector<float> data(n);
    for (auto & v : data) {
      v = dist(gen);
    }
    return data;
  };

  preprocess::InputDataMap input;
  input["sampled_trajectories"] = make_random(SAMPLED_TRAJECTORIES_SHAPE);
  input["ego_agent_past"] = make_random(EGO_HISTORY_SHAPE);
  input["ego_current_state"] = make_random(EGO_CURRENT_STATE_SHAPE);
  input["neighbor_agents_past"] = make_random(NEIGHBOR_SHAPE);
  input["static_objects"] = make_random(STATIC_OBJECTS_SHAPE);
  input["lanes"] = make_random(LANES_SHAPE);
  input["lanes_speed_limit"] = make_random(LANES_SPEED_LIMIT_SHAPE);
  input["route_lanes"] = make_random(ROUTE_LANES_SHAPE);
  input["route_lanes_speed_limit"] = make_random(ROUTE_LANES_SPEED_LIMIT_SHAPE);
  input["polygons"] = make_random(POLYGONS_SHAPE);
  input["line_strings"] = make_random(LINE_STRINGS_SHAPE);
  input["goal_pose"] = make_random(GOAL_POSE_SHAPE);
  input["ego_shape"] = make_random(EGO_SHAPE_SHAPE);
  input["turn_indicators"] = make_random(TURN_INDICATORS_SHAPE);
  return input;
}

void print_statistics(const std::vector<double> & raw)
{
  auto sorted = raw;
  std::sort(sorted.begin(), sorted.end());
  const int n = static_cast<int>(sorted.size());
  const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
  const double mean = sum / n;
  const double median = sorted[n / 2];
  const double p95 = sorted[static_cast<size_t>(n * 0.95)];
  const double p99 = sorted[static_cast<size_t>(n * 0.99)];
  double sq_sum = 0.0;
  for (const auto l : sorted) {
    sq_sum += (l - mean) * (l - mean);
  }
  const double stddev = std::sqrt(sq_sum / n);

  std::cout << "------------------------------------------------------\n";
  std::cout << "  RESULTS:\n";
  std::cout << "    Mean:   " << mean << " ms\n";
  std::cout << "    Median: " << median << " ms\n";
  std::cout << "    Min:    " << sorted.front() << " ms\n";
  std::cout << "    Max:    " << sorted.back() << " ms\n";
  std::cout << "    P95:    " << p95 << " ms\n";
  std::cout << "    P99:    " << p99 << " ms\n";
  std::cout << "    Std:    " << stddev << " ms\n";
  std::cout << "------------------------------------------------------\n";
}

int main(int argc, char ** argv)
{
  int warmup = 50;
  int runs = 300;
  std::string config_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--runs" && i + 1 < argc) {
      runs = std::stoi(argv[++i]);
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "Usage: benchmark_tool [options]\n"
                << "  --config PATH  Planner param yaml (default: from installed package)\n"
                << "  --warmup N     Warmup iterations (default: 50)\n"
                << "  --runs N       Benchmark iterations (default: 300)\n";
      return 0;
    }
  }

  // Load config
  if (config_path.empty()) {
    config_path = ament_index_cpp::get_package_share_directory("autoware_diffusion_planner") +
                  "/config/diffusion_planner.param.yaml";
  }

  std::cout << "======================================================\n";
  std::cout << "  Diffusion Planner TRT Benchmark\n";
  std::cout << "======================================================\n";
  std::cout << "  Config: " << config_path << "\n";

  const auto param_map = load_param_map(config_path);
  const std::string model_path =
    resolve_substitutions(get_param<std::string>(param_map, "onnx_model_path", ""));
  const std::string plugins_path =
    resolve_substitutions(get_param<std::string>(param_map, "plugins_path", ""));
  const int batch_size = static_cast<int>(get_param<int64_t>(param_map, "batch_size", 1));

  std::cout << "  Model:  " << model_path << "\n";
  std::cout << "  Batch:  " << batch_size << "\n";
  std::cout << "  Warmup: " << warmup << "\n";
  std::cout << "  Runs:   " << runs << "\n";
  std::cout << "------------------------------------------------------\n";

  // Create inference engine using the actual universe TensorrtInference.
  // All host-side optimizations come from the linked library automatically.
  std::cout << "  Loading engine..." << std::flush;
  auto t_load_start = std::chrono::high_resolution_clock::now();
  auto inference_ptr = std::make_unique<SingleStepInference>(model_path, plugins_path, batch_size);
  auto & inference = *inference_ptr;
  auto t_load_end = std::chrono::high_resolution_clock::now();
  const double load_s = std::chrono::duration<double>(t_load_end - t_load_start).count();
  std::cout << " done (" << load_s << "s)\n";

  // Generate random input data
  std::mt19937 gen(42);
  const auto input_data = generate_random_inputs(gen);

  // Warmup
  std::cout << "  Warming up (" << warmup << " iterations)..." << std::flush;
  for (int i = 0; i < warmup; ++i) {
    inference.infer(input_data);
  }
  std::cout << " done\n";

  // Benchmark
  std::cout << "  Benchmarking (" << runs << " iterations)..." << std::flush;
  std::vector<double> latencies(runs);
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    const auto result = inference.infer(input_data);
    auto t1 = std::chrono::high_resolution_clock::now();
    latencies[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!result.has_value()) {
      std::cerr << "\n  ERROR at iteration " << i << ": " << result.error() << "\n";
      return 1;
    }
  }
  std::cout << " done\n";

  // Results
  print_statistics(latencies);
  std::cout << "  Load time: " << load_s << " s\n";

  // GPU memory
  size_t gpu_free = 0, gpu_total = 0;
  cudaMemGetInfo(&gpu_free, &gpu_total);
  std::cout << "  GPU memory: ~" << (gpu_total - gpu_free) / (1024 * 1024) << " MiB\n";
  std::cout << "======================================================\n";

  return 0;
}
