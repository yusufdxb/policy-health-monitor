cmake_minimum_required(VERSION 3.16)
project(phm_benchmark LANGUAGES CXX)

# Offline reliability benchmark (no ROS). Build from the repository root with
# the superbuild in standalone/, or alone:
#   cmake -S benchmark -B build/benchmark && cmake --build build/benchmark
# It needs phm_core, phm_tools (taken from src/ when not already defined) and
# Eigen 3.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(PHM_STANDALONE ON)
if(NOT TARGET phm_core::phm_core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/phm_core ${CMAKE_BINARY_DIR}/phm_core)
endif()
if(NOT TARGET phm_tools::phm_tools)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src/phm_tools ${CMAKE_BINARY_DIR}/phm_tools)
endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(phm_bench STATIC src/bench.cpp)
target_include_directories(phm_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(phm_bench PUBLIC phm_core::phm_core phm_tools::phm_tools Eigen3::Eigen)
target_compile_options(phm_bench PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(phm_benchmark src/run_benchmark.cpp)
target_link_libraries(phm_benchmark phm_bench)
target_compile_options(phm_benchmark PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS phm_benchmark DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  foreach(t test_metrics test_generator test_detectors)
    add_executable(bench_${t} test/${t}.cpp)
    target_link_libraries(bench_${t} phm_bench GTest::gtest GTest::gtest_main)
    add_test(NAME phm_benchmark.${t} COMMAND bench_${t})
  endforeach()
endif()
