// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0.

#include "conversion/override_segments.hpp"
#include "rosbag/parsed_bag_data.hpp"

#include <CLI/CLI.hpp>

#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char ** argv)
{
  std::string rosbag_path;
  std::string output_dir;
  int64_t limit = -1;
  CLI::App app{"Extract control-mode override intervals without generating NPZ files"};
  app.add_option("rosbag_path", rosbag_path, "Input ROSBag directory")->required();
  app.add_option("output_dir", output_dir, "Output directory")->required();
  app.add_option("--limit", limit, "Maximum number of ROSBag messages to read");
  CLI11_PARSE(app, argc, argv);

  const ParsedBagData bag_data = load_rosbag(rosbag_path, limit, true);
  const auto segments = build_override_segments(bag_data.control_modes);
  save_override_segments_json(output_dir, segments, bag_data.control_modes.size());
  std::cout << "Extracted " << segments.size() << " override intervals from " << rosbag_path
            << std::endl;
  return 0;
}
