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
#ifndef PLANNING__AUTOWARE_DIFFUSION_PLANNER_TOOLS_UTILS__TIMESTAMP_STATS_HPP_
#define PLANNING__AUTOWARE_DIFFUSION_PLANNER_TOOLS_UTILS__TIMESTAMP_STATS_HPP_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

namespace timestamp_stats
{
inline double mean(const std::vector<int64_t> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  int64_t sum = std::accumulate(values.begin(), values.end(), int64_t(0));
  return static_cast<double>(sum) / values.size();
}

inline double std_dev(const std::vector<int64_t> & values, double mean)
{
  if (values.size() < 2) {
    return 0.0;
  }
  double variance = 0.0;
  for (const auto & v : values) {
    variance += (v - mean) * (v - mean);
  }
  variance /= (values.size() - 1);
  return std::sqrt(variance);
}

inline int64_t min(const std::vector<int64_t> & values)
{
  if (values.empty()) {
    return 0;
  }
  return *std::min_element(values.begin(), values.end());
}

inline int64_t max(const std::vector<int64_t> & values)
{
  if (values.empty()) {
    return 0;
  }
  return *std::max_element(values.begin(), values.end());
}

class TimeStampStats
{
public:
  TimeStampStats(const std::string & topic) : topic_name_(topic) {}
  TimeStampStats() : topic_name_("") {}

  void add_header_timestamp(const int64_t timestamp) { header_timestamps_.push_back(timestamp); }
  void add_rosbag_timestamp(const int64_t timestamp) { rosbag_timestamps_.push_back(timestamp); }

  void calc_stats()
  {

    // check monotonicity
    is_monotonic_header_ = std::is_sorted(header_timestamps_.begin(), header_timestamps_.end());
    is_monotonic_rosbag_ = std::is_sorted(rosbag_timestamps_.begin(), rosbag_timestamps_.end());
  
    // check size
    if (header_timestamps_.size() < 1 || rosbag_timestamps_.size() < 1) {
      std::cerr << "Warning: Not enough timestamps to calculate stats for topic " << topic_name_ << std::endl;
      const std::vector<int64_t> empty_diffs;
      diff_stats_.calc_stats(empty_diffs);
      header_diff_stats_.calc_stats(empty_diffs);
      rosbag_diff_stats_.calc_stats(empty_diffs);
      return;
    }

    // calculate stats
    std::vector<int64_t> header_diffs;
    for (size_t i = 1; i < header_timestamps_.size(); ++i) {
        header_diffs.push_back(header_timestamps_[i] - header_timestamps_[i - 1]);
    }

    std::vector<int64_t> rosbag_diffs;
    for (size_t i = 1; i < rosbag_timestamps_.size(); ++i) {
        rosbag_diffs.push_back(rosbag_timestamps_[i] - rosbag_timestamps_[i - 1]);
    }
    
    std::vector<int64_t> diffs;
    for (size_t i = 0; i < std::min(header_timestamps_.size(), rosbag_timestamps_.size()); ++i) {
      diffs.push_back(rosbag_timestamps_[i] - header_timestamps_[i]);
    }

    diff_stats_.calc_stats(diffs);
    header_diff_stats_.calc_stats(header_diffs);
    rosbag_diff_stats_.calc_stats(rosbag_diffs);
  }

  bool is_monotonic_header() const { return is_monotonic_header_; }
  bool is_monotonic_rosbag() const { return is_monotonic_rosbag_; }

  double diff_mean() const
  {
    return get_mean(diff_stats_);
  }

  double diff_std_dev() const
  {
    return get_std_dev(diff_stats_);
  }

  int64_t diff_min() const
  {
    return get_min(diff_stats_);
  }

  int64_t diff_max() const
  {
    return get_max(diff_stats_);
  }

  double header_diff_mean() const
  {
    return get_mean(header_diff_stats_);
  }

  double header_diff_std_dev() const
  {
    return get_std_dev(header_diff_stats_);
  }

  int64_t header_diff_min() const
  {
    return get_min(header_diff_stats_);
  }

  int64_t header_diff_max() const
  {
    return get_max(header_diff_stats_);
  }

  double rosbag_diff_mean() const
  {
    return get_mean(rosbag_diff_stats_);
  }

  double rosbag_diff_std_dev() const
  {
    return get_std_dev(rosbag_diff_stats_);
  }

  int64_t rosbag_diff_min() const
  {
    return get_min(rosbag_diff_stats_);
  }

  int64_t rosbag_diff_max() const
  {
    return get_max(rosbag_diff_stats_);
  }

private:
  struct Stats {
    double mean_;
    double std_dev_;
    int64_t min_;
    int64_t max_;

    void calc_stats(const std::vector<int64_t> & values)
    {
      mean_ = mean(values);
      std_dev_ = std_dev(values, mean_);
      min_ = min(values);
      max_ = max(values);
    }
  };

  // Data source information
  std::string topic_name_;
  std::vector<int64_t> header_timestamps_;
  std::vector<int64_t> rosbag_timestamps_;

  // Statistics results
  bool is_monotonic_header_ = true;
  bool is_monotonic_rosbag_ = true;
  Stats diff_stats_;
  Stats header_diff_stats_;
  Stats rosbag_diff_stats_;

  double get_mean(const Stats & stats) const { return stats.mean_; }
  double get_std_dev(const Stats & stats) const { return stats.std_dev_; }
  int64_t get_min(const Stats & stats) const { return stats.min_; }
  int64_t get_max(const Stats & stats) const { return stats.max_; }

};

struct TimestampStatsMap
{
  TimestampStatsMap(const std::vector<std::string> & topics)
  {
    for (const auto & topic : topics) {
      stats_map.emplace(topic, TimeStampStats(topic));
    }
  }

  void add_timestamp(const std::string & topic, int64_t header_ts, int64_t rosbag_ts)
  {
    if (stats_map.find(topic) == stats_map.end()) {
      stats_map.emplace(topic, TimeStampStats(topic));
    }
    stats_map[topic].add_header_timestamp(header_ts);
    stats_map[topic].add_rosbag_timestamp(rosbag_ts);
  }

  void analyze_all()
  {
    for (auto & [topic, stats] : stats_map) {
      stats.calc_stats();
    }
  }

  std::unordered_map<std::string, TimeStampStats> stats_map;
};
} // namespace timestamp_stats

#endif  // PLANNING__AUTOWARE_DIFFUSION_PLANNER_TOOLS_UTILS__TIMESTAMP_STATS_HPP_
