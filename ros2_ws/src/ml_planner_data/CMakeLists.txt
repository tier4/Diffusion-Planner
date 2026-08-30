cmake_minimum_required(VERSION 3.14)
project(ml_planner_data)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Non-ament dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Python bindings (FrameDataCache)
pybind11_add_module(_ml_planner_data
  src/pybind_module.cpp
  src/bag_dataset_builder.cpp
  src/skip_index.cpp
  src/bag_frame_reader.cpp
  src/frame_data_cache.cpp
  src/label_builder.cpp
  src/topic_config.cpp
)
ament_target_dependencies(_ml_planner_data PUBLIC
  ${${PROJECT_NAME}_FOUND_BUILD_DEPENDS}
)
install(TARGETS _ml_planner_data
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

ament_auto_package(INSTALL_TO_SHARE config)
