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

#include "io/bag_metadata.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class BagMetadataTest : public ::testing::Test
{
protected:
  fs::path tmp_dir_;

  void SetUp() override
  {
    tmp_dir_ = fs::temp_directory_path() / "test_bag_metadata";
    fs::create_directories(tmp_dir_);
  }

  void TearDown() override { fs::remove_all(tmp_dir_); }

  void write_json(const std::string & content)
  {
    std::ofstream f(tmp_dir_ / "log_file_info.json");
    f << content;
  }
};

TEST_F(BagMetadataTest, ValidJsonAllFields)
{
  write_json(R"({
    "id": "1a5afeae-131a-4937-9c71-f9bd2a13e0dc",
    "vehicle_id": "cfa23601-97c4-4d47-a37e-edfc7e080d8c",
    "project_id": "x2_dev",
    "area_map_version_id": "1847-20250819054643702212",
    "t4_dataset_id": "ds-001",
    "t4_dataset_version_id": "7",
    "log_file_name": "cfa23601-97c4-4d47-a37e-edfc7e080d8c_2026-01-16-10-03-35_p0900_0.db3.zst"
  })");

  const BagMetadata meta = load_bag_metadata(tmp_dir_.string());

  EXPECT_EQ(meta.log_file_id, "1a5afeae-131a-4937-9c71-f9bd2a13e0dc");
  EXPECT_EQ(meta.vehicle_id, "cfa23601-97c4-4d47-a37e-edfc7e080d8c");
  EXPECT_EQ(meta.project_id, "x2_dev");
  EXPECT_EQ(meta.map_version_id, "1847-20250819054643702212");
  EXPECT_EQ(meta.date, "2026-01-16");
  EXPECT_EQ(meta.bag_time, "10-03-35");
  EXPECT_EQ(meta.t4_dataset_id, "ds-001");
  EXPECT_EQ(meta.t4_dataset_version_id, "7");
}

TEST_F(BagMetadataTest, MissingFileReturnsEmptyStrings)
{
  const BagMetadata meta = load_bag_metadata("/nonexistent/path");

  EXPECT_EQ(meta.log_file_id, "");
  EXPECT_EQ(meta.vehicle_id, "");
  EXPECT_EQ(meta.project_id, "");
  EXPECT_EQ(meta.map_version_id, "");
  EXPECT_EQ(meta.date, "");
  EXPECT_EQ(meta.bag_time, "");
  EXPECT_EQ(meta.t4_dataset_id, "");
  EXPECT_EQ(meta.t4_dataset_version_id, "");
}

TEST_F(BagMetadataTest, MalformedJsonReturnsEmptyStrings)
{
  write_json("this is not json {{{");

  const BagMetadata meta = load_bag_metadata(tmp_dir_.string());

  EXPECT_EQ(meta.log_file_id, "");
  EXPECT_EQ(meta.vehicle_id, "");
}

TEST_F(BagMetadataTest, MissingIndividualFieldsPartiallyPopulated)
{
  write_json(R"({
    "id": "abc-123",
    "project_id": "erga"
  })");

  const BagMetadata meta = load_bag_metadata(tmp_dir_.string());

  EXPECT_EQ(meta.log_file_id, "abc-123");
  EXPECT_EQ(meta.project_id, "erga");
  EXPECT_EQ(meta.vehicle_id, "");
  EXPECT_EQ(meta.map_version_id, "");
  EXPECT_EQ(meta.date, "");
  EXPECT_EQ(meta.bag_time, "");
  EXPECT_EQ(meta.t4_dataset_id, "");
  EXPECT_EQ(meta.t4_dataset_version_id, "");
}

TEST_F(BagMetadataTest, T4DatasetFieldsAreLoaded)
{
  write_json(R"({
    "id": "abc-123",
    "t4_dataset_id": "ds-001",
    "t4_dataset_version_id": "7"
  })");

  const BagMetadata meta = load_bag_metadata(tmp_dir_.string());

  EXPECT_EQ(meta.t4_dataset_id, "ds-001");
  EXPECT_EQ(meta.t4_dataset_version_id, "7");
}

TEST_F(BagMetadataTest, LogFileNameWithoutDateTimePattern)
{
  write_json(R"({
    "id": "abc",
    "log_file_name": "some_unusual_name.db3"
  })");

  const BagMetadata meta = load_bag_metadata(tmp_dir_.string());

  EXPECT_EQ(meta.log_file_id, "abc");
  EXPECT_EQ(meta.date, "");
  EXPECT_EQ(meta.bag_time, "");
}

TEST_F(BagMetadataTest, NonStringFieldReturnsEmpty)
{
  write_json(R"({
    "id": 42,
    "vehicle_id": true,
    "project_id": "valid_string"
  })");

  const BagMetadata meta = load_bag_metadata(tmp_dir_.string());

  EXPECT_EQ(meta.log_file_id, "");
  EXPECT_EQ(meta.vehicle_id, "");
  EXPECT_EQ(meta.project_id, "valid_string");
}
