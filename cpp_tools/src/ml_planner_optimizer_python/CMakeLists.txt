cmake_minimum_required(VERSION 3.16)
project(ml_planner_optimizer_python)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# autoware_ml_planner is compiled from its own sources, which stay outside the colcon base path:
# it lives in tier4/autoware_universe, whose packages share names with the autoware_universe this
# workspace already imports.
set(ML_PLANNER_SOURCE_DIR "$ENV{ML_PLANNER_SOURCE_DIR}" CACHE PATH
  "autoware_ml_planner package directory (tier4/autoware_universe planning/autoware_ml_planner)")
set(ACADOS_SOURCE_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "acados installation")
if(NOT ACADOS_SOURCE_DIR)
  set(ACADOS_SOURCE_DIR /opt/acados CACHE PATH "acados installation" FORCE)
endif()

# Without both there is nothing to build; the evaluation runs without the optimizer then.
if(NOT EXISTS "${ML_PLANNER_SOURCE_DIR}/scripts/generate_solver.py")
  message(WARNING "ML_PLANNER_SOURCE_DIR does not point at autoware_ml_planner; skipping")
  ament_auto_package()
  return()
endif()
set(acados_python "${ACADOS_SOURCE_DIR}/.venv/bin/python3")
set(acados_libs acados hpipm blasfeo qpOASES_e)
foreach(lib IN LISTS acados_libs)
  find_library(ACADOS_LIB_${lib} ${lib} HINTS "${ACADOS_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
  if(NOT ACADOS_LIB_${lib})
    message(FATAL_ERROR "lib${lib} not found under ${ACADOS_SOURCE_DIR}/lib")
  endif()
endforeach()
if(NOT EXISTS "${acados_python}")
  message(FATAL_ERROR "${acados_python} not found; acados_template generates the solver")
endif()

# The solver is generated from the package's own OCP definition, as its AddAcadosSolver.cmake does.
set(acados_model ml_planner_optimizer)
set(generated_root "${CMAKE_CURRENT_BINARY_DIR}/acados_generated")
set(generated_dir "${generated_root}/c_generated_code")
file(MAKE_DIRECTORY "${generated_root}")
set(generated_files
  "${generated_dir}/acados_solver_${acados_model}.c"
  "${generated_dir}/${acados_model}_model/${acados_model}_expl_ode_fun.c"
  "${generated_dir}/${acados_model}_model/${acados_model}_expl_vde_forw.c"
  "${generated_dir}/${acados_model}_model/${acados_model}_expl_vde_adj.c"
  "${generated_dir}/${acados_model}_constraints/${acados_model}_constr_h_fun.c"
  "${generated_dir}/${acados_model}_constraints/${acados_model}_constr_h_fun_jac_uxt_zt.c"
)
add_custom_command(
  OUTPUT ${generated_files} "${generated_dir}/acados_solver_${acados_model}.h"
  COMMAND ${CMAKE_COMMAND} -E env
    "ACADOS_SOURCE_DIR=${ACADOS_SOURCE_DIR}"
    "PYTHONPATH=${ACADOS_SOURCE_DIR}/interfaces/acados_template:${ML_PLANNER_SOURCE_DIR}/scripts"
    "${acados_python}" "${ML_PLANNER_SOURCE_DIR}/scripts/generate_solver.py"
  WORKING_DIRECTORY "${generated_root}"
  DEPENDS
    "${ML_PLANNER_SOURCE_DIR}/scripts/generate_solver.py"
    "${ML_PLANNER_SOURCE_DIR}/scripts/vehicle_model.py"
  COMMENT "Generating the ml_planner acados solver"
  VERBATIM
)
set_source_files_properties(${generated_files} PROPERTIES
  COMPILE_OPTIONS "-Wno-unused-variable;-Wno-unused-parameter;-Wno-unused-function")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

pybind11_add_module(ml_planner_optimizer
  src/ml_planner_optimizer.cpp
  "${ML_PLANNER_SOURCE_DIR}/src/optimization/acados_solver_wrapper.cpp"
  "${ML_PLANNER_SOURCE_DIR}/src/optimization/trajectory_optimizer.cpp"
  "${ML_PLANNER_SOURCE_DIR}/src/postprocessing/road_border_avoidance.cpp"
  ${generated_files}
)
target_compile_definitions(ml_planner_optimizer PRIVATE AUTOWARE_ML_PLANNER_USE_ACADOS)
target_include_directories(ml_planner_optimizer PRIVATE
  "${ML_PLANNER_SOURCE_DIR}/include"
  "${generated_root}"
  "${generated_dir}"
  "${ACADOS_SOURCE_DIR}/include"
  "${ACADOS_SOURCE_DIR}/include/acados"
  "${ACADOS_SOURCE_DIR}/include/acados_c"
  "${ACADOS_SOURCE_DIR}/include/blasfeo/include"
  "${ACADOS_SOURCE_DIR}/include/hpipm/include"
)
ament_target_dependencies(ml_planner_optimizer PUBLIC
  autoware_planning_msgs
  autoware_utils
  autoware_utils_geometry
  autoware_vehicle_info_utils
  geometry_msgs
  lanelet2_core
  nav_msgs
  rclcpp)
# Every acados library is a direct dependency of the module, so the module's RUNPATH finds all of
# them; libacados itself carries no RUNPATH to find its own.
foreach(lib IN LISTS acados_libs)
  target_link_libraries(ml_planner_optimizer PRIVATE
    "-Wl,--no-as-needed" "${ACADOS_LIB_${lib}}" "-Wl,--as-needed")
endforeach()
target_link_libraries(ml_planner_optimizer PRIVATE Eigen3::Eigen yaml-cpp)

# Shipped next to the module: acados, so a host needs no acados install to run it, and the two
# autoware libraries the simulator's runtime install does not carry. What /opt/ros/humble or the
# system provides (rclcpp, lanelet2_core, yaml-cpp) is left to the host.
set(site_packages "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set_target_properties(ml_planner_optimizer PROPERTIES
  BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib"
  INSTALL_RPATH "$ORIGIN/../..")
install(TARGETS ml_planner_optimizer DESTINATION "${site_packages}")
set(shipped)
foreach(lib IN LISTS acados_libs)
  list(APPEND shipped "${ACADOS_LIB_${lib}}")
endforeach()
find_library(VEHICLE_INFO_UTILS_LIB vehicle_info_utils
  HINTS "${autoware_vehicle_info_utils_DIR}/../../../lib" NO_DEFAULT_PATH REQUIRED)
find_library(AUTOWARE_UTILS_GEOMETRY_LIB autoware_utils_geometry
  HINTS "${autoware_utils_geometry_DIR}/../../../lib" NO_DEFAULT_PATH REQUIRED)
list(APPEND shipped "${VEHICLE_INFO_UTILS_LIB}" "${AUTOWARE_UTILS_GEOMETRY_LIB}")
foreach(path IN LISTS shipped)
  get_filename_component(real "${path}" REALPATH)
  execute_process(COMMAND objdump -p "${real}" OUTPUT_VARIABLE dynamic)
  string(REGEX MATCH "SONAME +([^\n ]+)" _ "${dynamic}")
  install(FILES "${real}" DESTINATION lib RENAME "${CMAKE_MATCH_1}")
endforeach()

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_ml_planner_optimizer test/test_ml_planner_optimizer.py
    PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
    APPEND_LIBRARY_DIRS "${ACADOS_SOURCE_DIR}/lib")
endif()

ament_auto_package(INSTALL_TO_SHARE config)
