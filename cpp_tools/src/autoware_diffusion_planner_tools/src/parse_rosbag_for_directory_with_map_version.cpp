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

/**
 * @file parse_rosbag_for_directory_with_map_version.cpp
 * @brief Convert rosbag directories to Diffusion Planner npz files with map version resolution.
 */

#include "cli/converter_options.hpp"
#include "conversion/data_converter.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace
{

/**
 * @brief Parsed command-line options for directory-based rosbag conversion.
 */
struct DirectoryOptions
{
  std::vector<fs::path> target_dir_list;
  fs::path save_root;
  ConverterOptions converter;
  int64_t num_workers{32};
};

/**
 * @brief Parse command-line arguments and build a DirectoryOptions.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Parsed options, or std::nullopt if parsing failed.
 */
std::optional<DirectoryOptions> parse_directory_arguments(int argc, char ** argv)
{
  DirectoryOptions options;
  options.converter = ConverterOptions::default_converter_options();
  std::vector<std::string> target_dir_list;
  std::string save_root;
  CLI::App app{"Convert rosbag directories to Diffusion Planner npz files"};
  app
    .add_option(
      "target_dir_list", target_dir_list,
      "Directories searched recursively for rosbag metadata.yaml files.")
    ->required()
    ->expected(1, -1);
  app
    .add_option(
      "--save_root", save_root,
      "Root directory where per-date and per-bag conversion outputs "
      "are written.")
    ->required();
  app.add_option("--num_workers", options.num_workers, "Number of worker threads to run.");
  options.converter.add_converter_options(app);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError & e) {
    app.exit(e);
    return std::nullopt;
  }
  for (const auto & target_dir : target_dir_list) {
    options.target_dir_list.emplace_back(target_dir);
  }
  options.save_root = save_root;

  if (options.target_dir_list.empty() || options.save_root.empty()) {
    return std::nullopt;
  }
  if (options.num_workers <= 0) {
    options.num_workers = static_cast<int64_t>(std::thread::hardware_concurrency());
    if (options.num_workers <= 0) {
      options.num_workers = 1;
    }
  }
  if (const auto err = validate_options(options.converter)) {
    std::cerr << *err << std::endl;
    return std::nullopt;
  }
  return options;
}

/**
 * @brief Recursively find rosbag directories by locating metadata.yaml files.
 * @param target_dir_list List of root directories to search.
 * @return Sorted, unique list of bag directory paths.
 */
std::vector<fs::path> find_bag_directories(const std::vector<fs::path> & target_dir_list)
{
  std::vector<fs::path> bag_dirs;
  for (const auto & target_dir : target_dir_list) {
    if (!fs::exists(target_dir)) {
      std::cerr << "Target directory does not exist: " << target_dir.string() << std::endl;
      continue;
    }
    for (const auto & entry : fs::recursive_directory_iterator(target_dir)) {
      if (entry.is_regular_file() && entry.path().filename() == "metadata.yaml") {
        bag_dirs.push_back(entry.path().parent_path());
      }
    }
  }
  std::sort(bag_dirs.begin(), bag_dirs.end());
  bag_dirs.erase(std::unique(bag_dirs.begin(), bag_dirs.end()), bag_dirs.end());
  return bag_dirs;
}

/**
 * @brief Sum the sizes of regular files directly inside a bag directory.
 * @param bag_dir Path to the bag directory.
 * @return Total byte size of top-level regular files (data files are at this level).
 */
uintmax_t bag_directory_size(const fs::path & bag_dir)
{
  uintmax_t total = 0;
  std::error_code ec;
  for (const auto & entry : fs::directory_iterator(bag_dir, ec)) {
    if (entry.is_regular_file()) {
      const uintmax_t sz = entry.file_size(ec);
      if (!ec) total += sz;
    }
  }
  return total;
}

/**
 * @brief Sort bag directories by descending size (LPT heuristic).
 *
 * Processing large bags first ensures that near the end of the run, only small
 * bags remain, minimising idle workers caused by a few slow stragglers.
 * Sizes are pre-computed once to avoid redundant filesystem calls during sort.
 *
 * @param bag_dirs List of bag directories to sort in-place.
 */
void sort_bags_lpt(std::vector<fs::path> & bag_dirs)
{
  std::vector<std::pair<uintmax_t, fs::path>> sized;
  sized.reserve(bag_dirs.size());
  for (auto & p : bag_dirs) {
    sized.emplace_back(bag_directory_size(p), std::move(p));
  }
  std::sort(
    sized.begin(), sized.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
  bag_dirs.clear();
  bag_dirs.reserve(sized.size());
  for (auto & [size, path] : sized) {
    bag_dirs.push_back(std::move(path));
  }
}

/**
 * @brief Extract area_map_version_id from a rosbag metadata.yaml file.
 * @param metadata_path Path to metadata.yaml.
 * @return The map version string, or std::nullopt if not found or unreadable.
 */
std::optional<std::string> load_map_version_from_metadata(const fs::path & metadata_path)
{
  if (!fs::is_regular_file(metadata_path)) {
    return std::nullopt;
  }

  try {
    const YAML::Node metadata = YAML::LoadFile(metadata_path.string());
    if (metadata["area_map_version_id"]) {
      return metadata["area_map_version_id"].as<std::string>();
    }
    if (
      metadata["rosbag2_bagfile_information"] &&
      metadata["rosbag2_bagfile_information"]["area_map_version_id"]) {
      return metadata["rosbag2_bagfile_information"]["area_map_version_id"].as<std::string>();
    }
  } catch (const YAML::Exception &) {
    return std::nullopt;
  }
  return std::nullopt;
}

/**
 * @brief Extract area_map_version_id from a log_file_info.json file.
 * @param info_path Path to log_file_info.json.
 * @return The map version string, or std::nullopt if not found or unreadable.
 */
std::optional<std::string> load_map_version_from_log_file_info(const fs::path & info_path)
{
  if (!fs::is_regular_file(info_path)) {
    return std::nullopt;
  }

  try {
    std::ifstream info_file(info_path);
    const nlohmann::json info = nlohmann::json::parse(info_file);
    if (info.contains("area_map_version_id") && info["area_map_version_id"].is_string()) {
      return info["area_map_version_id"].get<std::string>();
    }
  } catch (const nlohmann::json::exception &) {
    return std::nullopt;
  }
  return std::nullopt;
}

/**
 * @brief Load area_map_version_id, trying log_file_info.json first, then metadata.yaml.
 * @param bag_path Path to the bag directory.
 * @return The map version string, or std::nullopt if neither source has it.
 */
std::optional<std::string> load_map_version_id(const fs::path & bag_path)
{
  auto version = load_map_version_from_log_file_info(bag_path / "log_file_info.json");
  if (version) {
    return version;
  }
  return load_map_version_from_metadata(bag_path / "metadata.yaml");
}

/**
 * @brief Resolve the lanelet2_map.osm path for a bag using map version and date-based heuristics.
 * @param bag_path Path to the bag directory.
 * @return Path to the resolved lanelet2_map.osm file.
 * @throws std::runtime_error if no matching map file is found.
 */
fs::path resolve_vector_map_path(const fs::path & bag_path)
{
  const fs::path metadata_path = bag_path / "metadata.yaml";
  const fs::path log_file_info_path = bag_path / "log_file_info.json";
  const fs::path date = bag_path.parent_path().filename();
  const fs::path bag_time = bag_path.filename();

  const std::optional<std::string> map_version_id = load_map_version_id(bag_path);

  std::vector<fs::path> candidate_bases;
  fs::path parent = bag_path.parent_path();
  for (int i = 1; i < 6 && parent.has_parent_path(); ++i) {
    parent = parent.parent_path();
    if (
      std::find(candidate_bases.begin(), candidate_bases.end(), parent) == candidate_bases.end()) {
      candidate_bases.push_back(parent);
    }
  }

  std::vector<fs::path> candidate_paths;
  for (const auto & base : candidate_bases) {
    const fs::path map_dir = base / "map";
    if (!fs::is_directory(map_dir)) {
      continue;
    }

    if (map_version_id) {
      candidate_paths.push_back(map_dir / map_version_id.value() / "lanelet2_map.osm");
    }

    candidate_paths.push_back(map_dir / date / bag_time / "lanelet2_map.osm");
    candidate_paths.push_back(map_dir / date / "lanelet2_map.osm");
    candidate_paths.push_back(map_dir / bag_time / "lanelet2_map.osm");
    candidate_paths.push_back(map_dir / "lanelet2_map.osm");
  }

  for (const auto & path : candidate_paths) {
    if (fs::is_regular_file(path)) {
      return path;
    }
  }

  std::ostringstream searched;
  if (candidate_paths.empty()) {
    searched << "(no map dir found)";
  } else {
    for (const auto & path : candidate_paths) {
      searched << path.string() << "\n";
    }
  }

  std::ostringstream error;
  error << "lanelet2_map.osm was not found for bag: " << bag_path.string() << "\n"
        << "metadata: " << metadata_path.string() << "\n"
        << "log_file_info: " << log_file_info_path.string() << "\n"
        << "area_map_version_id: "
        << (map_version_id ? map_version_id.value() : std::string("(none)")) << "\n"
        << "searched:\n"
        << searched.str();
  throw std::runtime_error(error.str());
}

/**
 * @brief Format a duration as HH:MM:SS string.
 * @param elapsed The duration to format.
 * @return Formatted time string.
 */
std::string format_elapsed(std::chrono::steady_clock::duration elapsed)
{
  const auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
  const int64_t hours = total_seconds / 3600;
  const int64_t minutes = (total_seconds % 3600) / 60;
  const int64_t seconds = total_seconds % 60;
  return fmt::format("{:02}:{:02}:{:02}", hours, minutes, seconds);
}

/**
 * @brief Check that a relative path does not escape its root with ".." components.
 * @param relative_path The relative path to validate.
 * @return true if the path stays within the root, false otherwise.
 */
bool is_relative_inside_root(const fs::path & relative_path)
{
  if (relative_path.empty()) {
    return false;
  }
  for (const auto & part : relative_path) {
    if (part == "..") {
      return false;
    }
  }
  return true;
}

/**
 * @brief Compute the output directory for a bag by mirroring its relative path under save_root.
 * @param save_root Root directory for output.
 * @param bag_path Path to the bag directory.
 * @param target_dir_list List of target directories used to compute relative paths.
 * @return Output directory path for the bag.
 */
fs::path make_save_dir(
  const fs::path & save_root, const fs::path & bag_path,
  const std::vector<fs::path> & target_dir_list)
{
  const fs::path abs_bag_path = fs::absolute(bag_path).lexically_normal();
  fs::path best_relative;
  size_t best_root_len = 0;

  for (const auto & target_dir : target_dir_list) {
    const fs::path abs_target = fs::absolute(target_dir).lexically_normal();
    std::error_code ec;
    const fs::path relative_path = fs::relative(abs_bag_path, abs_target, ec);
    if (ec || !is_relative_inside_root(relative_path)) {
      continue;
    }

    const size_t root_len = abs_target.string().size();
    if (root_len > best_root_len) {
      best_root_len = root_len;
      best_relative = relative_path;
    }
  }

  if (!best_relative.empty()) {
    return fs::absolute(save_root / best_relative);
  }

  return fs::absolute(save_root / bag_path.filename());
}

/**
 * @brief Convert a single rosbag directory and write the output.
 * @param bag_path Path to the bag directory.
 * @param save_root Root directory for output.
 * @param base_options Converter options.
 * @param target_dir_list List of target directories used for relative path calculation.
 */
void process_single_bag(
  const fs::path & bag_path, const fs::path & save_root, const ConverterOptions & base_options,
  const std::vector<fs::path> & target_dir_list)
{
  const fs::path save_dir = make_save_dir(save_root, bag_path, target_dir_list);
  fs::create_directories(save_dir.parent_path());
  if (fs::is_directory(save_dir)) {
    return;
  }

  try {
    ConverterPaths paths;
    paths.rosbag_path = fs::absolute(bag_path).string();
    paths.vector_map_path = resolve_vector_map_path(bag_path).string();
    paths.save_dir = save_dir.string();

    run_data_converter(paths, base_options);
  } catch (const std::exception & e) {
    std::cerr << "Error processing " << bag_path.string() << ": " << e.what() << std::endl;
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  const auto start_time = std::chrono::steady_clock::now();

  const auto options_opt = parse_directory_arguments(argc, argv);
  if (!options_opt) {
    return 1;
  }
  const DirectoryOptions options = options_opt.value();

  const fs::path save_root = fs::absolute(options.save_root);
  fs::create_directories(save_root);

  std::vector<fs::path> bag_dir_list = find_bag_directories(options.target_dir_list);
  std::cout << "Found " << bag_dir_list.size() << " bag directories to process" << std::endl;
  sort_bags_lpt(bag_dir_list);
  std::cout << "Bags sorted by descending size (LPT)" << std::endl;
  std::cout << "Using " << options.num_workers << " parallel workers" << std::endl;

  std::atomic<size_t> next_index{0};
  const size_t worker_count =
    std::min(static_cast<size_t>(options.num_workers), std::max<size_t>(bag_dir_list.size(), 1));

  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&]() {
      while (true) {
        const size_t index = next_index.fetch_add(1);
        if (index >= bag_dir_list.size()) {
          break;
        }
        process_single_bag(
          bag_dir_list[index], save_root, options.converter, options.target_dir_list);
      }
    });
  }

  for (auto & worker : workers) {
    worker.join();
  }

  const std::string elapsed = format_elapsed(std::chrono::steady_clock::now() - start_time);
  std::cout << "Total elapsed time: " << elapsed << std::endl;

  std::ofstream summary_file(save_root / "processing_time.txt", std::ios::out);
  summary_file << "Total elapsed time: " << elapsed << "\n";

  return 0;
}
