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

#include "cli/converter_options.hpp"

#include <CLI/CLI.hpp>

#include <filesystem>
#include <string>

void ConverterOptions::add_converter_options(CLI::App & app)
{
  app.add_option("--step", step, "Frame sampling interval in 10 Hz ticks.");
  app.add_option(
    "--limit", limit,
    "Maximum number of rosbag messages to read. Use -1 to read "
    "all messages.");
  app.add_option(
    "--min_frames", min_frames,
    "Minimum number of assembled frames required to accept a sequence.");
  app.add_option(
    "--min_distance", min_distance,
    "Minimum traveled ego distance in meters required to accept a sequence.");
  app.add_option(
    "--search_nearest_route", search_nearest_route,
    "Use the latest route message at or before each frame "
    "timestamp when non-zero.");
  app.add_option(
    "--convert_yellow", convert_yellow,
    "Do not skip frames for yellow traffic lights when non-zero.");
  app.add_option(
    "--convert_red", convert_red, "Do not skip frames for red traffic lights when non-zero.");
  app.add_option_function<int64_t>(
    "--interpolation",
    [this](const int64_t value) {
      interpolation = value;
      use_interpolation = static_cast<bool>(interpolation);
    },
    "Use timestamp-based interpolation for ego past and future "
    "trajectories when non-zero.");
  app.add_option("--ego_wheel_base", ego_wheel_base, "Ego vehicle wheel base in meters.");
  app.add_option("--ego_length", ego_length, "Ego vehicle length in meters.");
  app.add_option("--ego_width", ego_width, "Ego vehicle width in meters.");
  app.add_option(
    "--static_object_margin", static_object_margin,
    "Additional margin in meters for static-object collision filtering.");
  app.add_option(
    "--neighbor_margin", neighbor_margin,
    "Additional margin in meters for neighbor-agent collision filtering.");
  app.add_option(
    "--road_border_margin", road_border_margin,
    "Additional margin in meters for road-border collision filtering.");
  app.add_option(
    "--collision_time_stride", collision_time_stride,
    "Time stride used when checking trajectory collision filters.");
  app.add_option(
    "--offlane_max_score", offlane_max_score,
    "Maximum average distance in meters from lane centerlines "
    "before a frame is skipped.");
  app.add_option(
    "--offlane_time_stride", offlane_time_stride,
    "Time stride used when checking the off-lane filter.");
  app.add_option(
    "--red_light_run_radius_m", red_light_run_radius_m,
    "Maximum distance in meters from a stop-line crossing point to a heading-aligned red "
    "route-lane entry for red-light-run skipping.");
  app.add_option(
    "--red_light_run_heading_tol_deg", red_light_run_heading_tol_deg,
    "Maximum heading difference in degrees between ego future and a red route lane when "
    "matching the ego's own signal.");
  app.add_option(
    "--green_stop_heading_tol_deg", green_stop_heading_tol_deg,
    "Maximum heading difference in degrees between ego heading and a green route lane.");
  app.add_option(
    "--green_stop_stay_radius_m", green_stop_stay_radius_m,
    "Maximum future spatial extent in meters for a green-stop stationary window.");
  app.add_option(
    "--green_stop_speed_max_mps", green_stop_speed_max_mps,
    "Maximum current ego speed in m/s for green-stop stationary detection.");
  app.add_option(
    "--green_stop_ahead_m", green_stop_ahead_m,
    "Maximum forward distance in meters to a heading-aligned green route-lane entry.");
  app.add_option(
    "--green_stop_lead_fwd_m", green_stop_lead_fwd_m,
    "Forward extent in meters of the green-stop lead-neighbor corridor.");
  app.add_option(
    "--green_stop_lead_lat_m", green_stop_lead_lat_m,
    "Half-width in meters of the green-stop lead-neighbor corridor.");
  app.add_option(
    "--write_skipped_npz", write_skipped_npz,
    "Also write npz files for skipped frames when non-zero. "
    "Intended for inspection.");
  app.add_flag(
    "--sidecar_only", sidecar_only,
    "Write ONLY the per-frame JSON sidecars (pose + neighbor_ids + is_skipped), skipping all "
    "npz output. Same pipeline/slot-ordering as a full convert, so the sidecars align to an "
    "existing npz set from this converter; faster than re-converting (no npz write).");
  app.add_flag(
    "--pack_sequence", pack_sequence,
    "Write ONE npz and ONE json per sequence (frames stacked along a leading frame axis) "
    "instead of one file per frame. Forces write_skipped_npz on so the packed sequence is "
    "gap-free. Mutually exclusive with --sidecar_only.");
  app.add_flag(
    "--extract_override_segments", extract_override_segments,
    "Read vehicle control-mode messages and write control_mode_4_intervals.json.");
}

std::string ConverterPaths::get_rosbag_dir_name() const
{
  return std::filesystem::path(rosbag_path).filename();
}

ConverterOptions ConverterOptions::default_converter_options()
{
  ConverterOptions options;
  options.step = 3;
  options.limit = -1;
  options.min_frames = 1700;
  options.search_nearest_route = 1;
  options.convert_yellow = 0;
  options.convert_red = 0;
  options.interpolation = 1;
  options.min_distance = 50.0;
  options.ego_wheel_base = -1.0f;
  options.ego_length = -1.0f;
  options.ego_width = -1.0f;

  // Collision-free filter defaults match filter_collision_free_npz.py.
  options.static_object_margin = 0.0f;
  options.neighbor_margin = 0.0f;
  options.road_border_margin = 0.0f;
  options.collision_time_stride = 5;

  // In-lanelet filter defaults match filter_in_lanelet_npz.py.
  options.offlane_max_score = 6.0f;
  options.offlane_time_stride = 1;

  // Red-light-run detector defaults match the integrated python heuristic.
  options.red_light_run_radius_m = 12.0f;
  options.red_light_run_heading_tol_deg = 45.0f;

  // Green-stop detector defaults match Sakayori's npz_cleansing reference.
  options.green_stop_heading_tol_deg = 45.0f;
  options.green_stop_stay_radius_m = 2.0f;
  options.green_stop_speed_max_mps = 1.0f;
  options.green_stop_ahead_m = 40.0f;
  options.green_stop_lead_fwd_m = 30.0f;
  options.green_stop_lead_lat_m = 2.0f;

  // Inspection-only: production keeps this off so skipped frames write no npz.
  options.write_skipped_npz = false;
  // Full conversion by default (write npz); --sidecar_only flips to sidecar-only output.
  options.sidecar_only = false;
  // One file per frame by default; --pack_sequence packs a whole sequence into one npz/json.
  options.pack_sequence = false;
  options.extract_override_segments = false;
  options.use_interpolation = static_cast<bool>(options.interpolation);
  return options;
}

void normalize_options(ConverterOptions & opts)
{
  if (opts.pack_sequence) {
    // Requirement (3): a packed sequence must be gap-free, so every frame is written.
    opts.write_skipped_npz = true;
  }
}

std::optional<std::string> validate_options(const ConverterOptions & opts)
{
  if (opts.ego_wheel_base < 0.0 || opts.ego_length < 0.0 || opts.ego_width < 0.0) {
    return "Ego vehicle dimensions must be non-negative.";
  }
  if (opts.pack_sequence && opts.sidecar_only) {
    return "--pack_sequence and --sidecar_only are mutually exclusive "
           "(pack_sequence writes one npz per sequence; sidecar_only writes no npz).";
  }
  if (opts.red_light_run_radius_m < 0.0f) {
    return "red_light_run_radius_m must be non-negative.";
  }
  if (opts.red_light_run_heading_tol_deg < 0.0f) {
    return "red_light_run_heading_tol_deg must be non-negative.";
  }
  if (opts.green_stop_heading_tol_deg < 0.0f) {
    return "green_stop_heading_tol_deg must be non-negative.";
  }
  if (opts.green_stop_stay_radius_m < 0.0f) {
    return "green_stop_stay_radius_m must be non-negative.";
  }
  if (opts.green_stop_speed_max_mps < 0.0f) {
    return "green_stop_speed_max_mps must be non-negative.";
  }
  if (opts.green_stop_ahead_m < 0.0f) {
    return "green_stop_ahead_m must be non-negative.";
  }
  if (opts.green_stop_lead_fwd_m < 0.0f) {
    return "green_stop_lead_fwd_m must be non-negative.";
  }
  if (opts.green_stop_lead_lat_m < 0.0f) {
    return "green_stop_lead_lat_m must be non-negative.";
  }
  return std::nullopt;
}
