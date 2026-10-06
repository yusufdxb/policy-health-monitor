cmake_minimum_required(VERSION 3.16)
project(policy_health_monitor_standalone LANGUAGES CXX)

# ROS-independent superbuild: the phm_core library, the offline tools, the
# benchmark, the demo and the GO2 session tools, with their tests.
#   cmake -S standalone -B build/standalone -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/standalone -j
#   ctest --test-dir build/standalone --output-on-failure
# Requirements: a C++17 compiler, CMake >= 3.16, zlib, Eigen 3, GoogleTest.
# Pass -DONNXRUNTIME_ROOT=<onnxruntime release dir> to also build the offline
# shadow-policy replay tool. The ROS 2 nodes are built with colcon instead.

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(PHM_STANDALONE ON)
enable_testing()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/phm_core phm_core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/phm_tools phm_tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../benchmark benchmark)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../vla_monitor_demo vla_monitor_demo)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../integrations/go2 phm_go2)
