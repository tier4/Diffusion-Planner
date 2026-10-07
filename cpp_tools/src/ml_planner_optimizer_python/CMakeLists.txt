cmake_minimum_required(VERSION 3.18)
project(ml_planner_optimizer_python)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# autoware_ml_planner builds its optimizer only into its node, which needs CUDA and TensorRT, so the
# two stages are compiled here from its sources. build.sh leaves this package out; it is built only
# when selected, with ML_PLANNER_SOURCE_DIR set.
set(ML_PLANNER_SOURCE_DIR "$ENV{ML_PLANNER_SOURCE_DIR}")
set(ACADOS_SOURCE_DIR /opt/acados)

# Mirrors autoware_ml_planner's AddAcadosSolver.cmake, which is bound to its own source directory.
set(acados_model ml_planner_optimizer)
set(generated_root "${CMAKE_CURRENT_BINARY_DIR}/acados_generated")
set(generated_dir "${generated_root}/c_generated_code")
file(MAKE_DIRECTORY "${generated_root}")
set(generated_files
  "${generated_dir}/acados_solver_${acados_model}.h"
  "${generated_dir}/acados_solver_${acados_model}.c"
  "${generated_dir}/${acados_model}_model/${acados_model}_expl_ode_fun.c"
  "${generated_dir}/${acados_model}_model/${acados_model}_expl_vde_forw.c"
  "${generated_dir}/${acados_model}_model/${acados_model}_expl_vde_adj.c"
  "${generated_dir}/${acados_model}_constraints/${acados_model}_constr_h_fun.c"
  "${generated_dir}/${acados_model}_constraints/${acados_model}_constr_h_fun_jac_uxt_zt.c"
)
add_custom_command(
  OUTPUT ${generated_files}
  COMMAND ${CMAKE_COMMAND} -E env
    "ACADOS_SOURCE_DIR=${ACADOS_SOURCE_DIR}"
    "PYTHONPATH=${ACADOS_SOURCE_DIR}/interfaces/acados_template:${ML_PLANNER_SOURCE_DIR}/scripts"
    "${ACADOS_SOURCE_DIR}/.venv/bin/python3" "${ML_PLANNER_SOURCE_DIR}/scripts/generate_solver.py"
  WORKING_DIRECTORY "${generated_root}"
  DEPENDS
    "${ML_PLANNER_SOURCE_DIR}/scripts/generate_solver.py"
    "${ML_PLANNER_SOURCE_DIR}/scripts/vehicle_model.py"
  COMMENT "Generating the ml_planner acados solver"
  VERBATIM
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(ml_planner_optimizer
  src/ml_planner_optimizer.cpp
  "${ML_PLANNER_SOURCE_DIR}/src/optimization/acados_solver_wrapper.cpp"
  "${ML_PLANNER_SOURCE_DIR}/src/optimization/trajectory_optimizer.cpp"
  "${ML_PLANNER_SOURCE_DIR}/src/postprocessing/road_border_avoidance.cpp"
  ${generated_files}
)
target_compile_features(ml_planner_optimizer PRIVATE cxx_std_17)
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
# PUBLIC: pybind11_add_module links with the keyword signature, which every later call must match.
ament_target_dependencies(ml_planner_optimizer PUBLIC
  autoware_planning_msgs
  autoware_utils
  autoware_utils_geometry
  autoware_vehicle_info_utils
  geometry_msgs
  lanelet2_core
  nav_msgs
  rclcpp)
# libacados has no RUNPATH; linking its dependencies directly lets the module's RUNPATH load them.
target_link_directories(ml_planner_optimizer PRIVATE "${ACADOS_SOURCE_DIR}/lib")
target_link_libraries(ml_planner_optimizer PRIVATE
  "-Wl,--no-as-needed" acados hpipm blasfeo qpOASES_e "-Wl,--as-needed" Eigen3::Eigen yaml-cpp)
set_target_properties(ml_planner_optimizer PROPERTIES
  BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib"
  INSTALL_RPATH "${ACADOS_SOURCE_DIR}/lib")
install(TARGETS ml_planner_optimizer
  DESTINATION "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
install(FILES "${ML_PLANNER_SOURCE_DIR}/config/ml_planner.param.yaml"
  DESTINATION "share/${PROJECT_NAME}/config")

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_ml_planner_optimizer test/test_ml_planner_optimizer.py
    PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
    ENV "ML_PLANNER_SOURCE_DIR=${ML_PLANNER_SOURCE_DIR}"
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()

ament_auto_package()
