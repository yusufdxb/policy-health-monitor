cmake_minimum_required(VERSION 3.16)
project(phm_go2 VERSION 0.2.0 LANGUAGES CXX)

# GO2 integration. Three layers, each built when its dependencies exist:
#   - ROS-free (always): observation assembly + shadow tick logic, the session
#     reduction tools (phm_go2_summarize, phm_go2_plot) and their tests;
#   - ONNX Runtime (ONNXRUNTIME_ROOT or a system install): the policy runner
#     and the offline replay tool (phoenix_shadow_replay);
#   - ROS 2 (colcon, not PHM_STANDALONE): the probe (phm_go2_probe), and the
#     shadow embedder node when both ONNX Runtime and Unitree's unitree_go
#     messages are found.
# Nothing in this package publishes a command.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT PHM_STANDALONE)
  find_package(ament_cmake QUIET)
endif()
if(ament_cmake_FOUND AND NOT PHM_STANDALONE)
  set(PHM_GO2_AMENT ON)
  set(PHM_GO2_BIN lib/${PROJECT_NAME})
else()
  set(PHM_GO2_AMENT OFF)
  set(PHM_GO2_BIN bin)
endif()
if(NOT TARGET phm_core::phm_core)
  find_package(phm_core REQUIRED)
endif()
if(NOT TARGET phm_tools::phm_tools)
  find_package(phm_tools REQUIRED)
endif()

# ---------------------------------------------------------------------------
# ROS-free layer
# ---------------------------------------------------------------------------
add_library(phm_go2_shadow STATIC src/observation.cpp)
target_include_directories(phm_go2_shadow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(phm_go2_shadow PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # The gravity terms are evaluated in double exactly as Python did; no FMA.
  target_compile_options(phm_go2_shadow PRIVATE -ffp-contract=off)
endif()

add_library(phm_go2_session STATIC tools/session.cpp)
target_include_directories(phm_go2_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools)
target_link_libraries(phm_go2_session PUBLIC phm_core::phm_core phm_tools::phm_tools)

add_executable(phm_go2_summarize tools/summarize_session.cpp)
target_link_libraries(phm_go2_summarize phm_go2_session)
add_executable(phm_go2_plot tools/plot_session.cpp)
target_link_libraries(phm_go2_plot phm_go2_session)
install(TARGETS phm_go2_summarize phm_go2_plot DESTINATION ${PHM_GO2_BIN})

# ---------------------------------------------------------------------------
# ONNX Runtime layer
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
  "ONNX Runtime install prefix (an onnxruntime-linux-<arch>-<version> release directory)")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
set(PHM_GO2_HAVE_ORT FALSE)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  set(PHM_GO2_HAVE_ORT TRUE)
  get_filename_component(PHM_ORT_LIBDIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
  message(STATUS "phm_go2: ONNX Runtime ${ONNXRUNTIME_LIBRARY}")
  add_library(phm_go2_policy STATIC src/policy_runner.cpp)
  target_include_directories(phm_go2_policy SYSTEM PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(phm_go2_policy PUBLIC phm_go2_shadow ${ONNXRUNTIME_LIBRARY})
  set_target_properties(phm_go2_policy PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_executable(phoenix_shadow_replay src/shadow_replay.cpp)
  target_link_libraries(phoenix_shadow_replay phm_go2_policy phm_core::phm_core)
  set_target_properties(phoenix_shadow_replay PROPERTIES
    BUILD_RPATH "${PHM_ORT_LIBDIR}" INSTALL_RPATH "${PHM_ORT_LIBDIR}")
  install(TARGETS phoenix_shadow_replay DESTINATION ${PHM_GO2_BIN})
else()
  message(STATUS "phm_go2: ONNX Runtime not found (set ONNXRUNTIME_ROOT); the shadow "
    "embedder and phoenix_shadow_replay are not built")
endif()

# ---------------------------------------------------------------------------
# ROS 2 layer
# ---------------------------------------------------------------------------
if(PHM_GO2_AMENT)
  find_package(phm_msgs REQUIRED)
  find_package(rclcpp REQUIRED)

  add_executable(phm_go2_probe src/probe.cpp)
  target_link_libraries(phm_go2_probe phm_core::phm_core phm_tools::phm_tools)
  ament_target_dependencies(phm_go2_probe phm_msgs rclcpp)
  install(TARGETS phm_go2_probe DESTINATION ${PHM_GO2_BIN})

  find_package(unitree_go QUIET)
  if(PHM_GO2_HAVE_ORT AND unitree_go_FOUND)
    add_executable(phoenix_shadow_embedder src/shadow_embedder_node.cpp)
    target_link_libraries(phoenix_shadow_embedder phm_go2_policy phm_core::phm_core)
    ament_target_dependencies(phoenix_shadow_embedder phm_msgs rclcpp unitree_go)
    set_target_properties(phoenix_shadow_embedder PROPERTIES
      BUILD_RPATH "${PHM_ORT_LIBDIR}" INSTALL_RPATH "${PHM_ORT_LIBDIR}")
    install(TARGETS phoenix_shadow_embedder DESTINATION ${PHM_GO2_BIN})
  else()
    message(STATUS "phm_go2: phoenix_shadow_embedder needs ONNX Runtime and unitree_go "
      "(source the unitree_ros2 workspace); not built")
  endif()
  install(FILES phm_go2_detectors.yaml DESTINATION share/${PROJECT_NAME})
  install(PROGRAMS onboard_session.sh DESTINATION share/${PROJECT_NAME})
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PHM_GO2_AMENT)
  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_observation test/test_observation.cpp
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
    target_link_libraries(test_observation phm_go2_shadow phm_core::phm_core)
    ament_add_gtest(test_session_tools test/test_session_tools.cpp)
    target_link_libraries(test_session_tools phm_go2_session)
  endif()
  ament_package()
else()
  include(CTest)
  if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(test_observation test/test_observation.cpp)
    target_link_libraries(test_observation phm_go2_shadow phm_core::phm_core GTest::gtest
      GTest::gtest_main)
    add_test(NAME phm_go2.test_observation COMMAND test_observation
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
    add_executable(test_session_tools test/test_session_tools.cpp)
    target_link_libraries(test_session_tools phm_go2_session GTest::gtest GTest::gtest_main)
    add_test(NAME phm_go2.test_session_tools COMMAND test_session_tools)
  endif()
endif()
