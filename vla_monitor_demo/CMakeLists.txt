cmake_minimum_required(VERSION 3.16)
project(vla_monitor_demo LANGUAGES CXX)

# Offline demo (no ROS). Build with the superbuild in standalone/, or alone:
#   cmake -S vla_monitor_demo -B build/demo && cmake --build build/demo
# Needs phm_core, phm_tools (taken from src/ when not already defined) and
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

add_library(vla_demo STATIC src/harness.cpp)
target_include_directories(vla_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vla_demo PUBLIC phm_core::phm_core phm_tools::phm_tools Eigen3::Eigen)
target_compile_options(vla_demo PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(vla_monitor_demo src/run_demo.cpp)
target_link_libraries(vla_monitor_demo vla_demo)
target_compile_options(vla_monitor_demo PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS vla_monitor_demo DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  add_executable(test_vla_harness test/test_harness.cpp)
  target_link_libraries(test_vla_harness vla_demo GTest::gtest GTest::gtest_main)
  add_test(NAME vla_monitor_demo.test_harness COMMAND test_vla_harness)
endif()
