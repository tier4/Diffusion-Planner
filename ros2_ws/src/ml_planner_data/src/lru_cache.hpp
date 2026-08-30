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

#ifndef ML_PLANNER_DATA__SRC__LRU_CACHE_HPP_
#define ML_PLANNER_DATA__SRC__LRU_CACHE_HPP_

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>

namespace autoware::ml_planner::data {

/**
 * @brief Simple LRU cache keyed by string.
 */
template <typename ValueT> class LruCache {
public:
  explicit LruCache(size_t capacity) : capacity_(capacity) {}

  // Returns the cached value, or creates it with factory() and caches it.
  template <typename FactoryT>
  ValueT &get_or_create(const std::string &key, FactoryT &&factory) {
    const auto it = index_.find(key);
    if (it != index_.end()) {
      entries_.splice(entries_.begin(), entries_, it->second);
      return entries_.front().second;
    }
    entries_.emplace_front(key, factory());
    index_[key] = entries_.begin();
    if (entries_.size() > capacity_) {
      index_.erase(entries_.back().first);
      entries_.pop_back();
    }
    return entries_.front().second;
  }

private:
  size_t capacity_;
  std::list<std::pair<std::string, ValueT>> entries_;
  std::unordered_map<std::string, typename decltype(entries_)::iterator> index_;
};

} // namespace autoware::ml_planner::data

#endif // ML_PLANNER_DATA__SRC__LRU_CACHE_HPP_
