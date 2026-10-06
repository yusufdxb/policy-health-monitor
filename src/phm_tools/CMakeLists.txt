cmake_minimum_required(VERSION 3.16)
project(phm_tools VERSION 0.2.0 LANGUAGES CXX)

# Header-only helpers for the offline tools: JSON with Python-compatible
# output and a small SVG chart writer. Builds under colcon (ament_cmake) and
# as a plain CMake project (PHM_STANDALONE, e.g. from standalone/).

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT PHM_STANDALONE)
  find_package(ament_cmake QUIET)
endif()
if(ament_cmake_FOUND AND NOT PHM_STANDALONE)
  set(PHM_TOOLS_AMENT ON)
else()
  set(PHM_TOOLS_AMENT OFF)
endif()
if(NOT TARGET phm_core::phm_core)
  find_package(phm_core REQUIRED)
endif()

add_library(phm_tools INTERFACE)
add_library(phm_tools::phm_tools ALIAS phm_tools)
target_include_directories(phm_tools INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(phm_tools INTERFACE phm_core::phm_core)
target_compile_features(phm_tools INTERFACE cxx_std_17)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS phm_tools EXPORT phm_toolsTargets)

if(PHM_TOOLS_AMENT)
  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
    find_package(ament_cmake_gtest REQUIRED)
    foreach(t test_json test_svg)
      ament_add_gtest(${t} test/${t}.cpp)
      target_link_libraries(${t} phm_tools)
    endforeach()
  endif()
  ament_export_targets(phm_toolsTargets)
  ament_export_dependencies(phm_core)
  ament_package()
else()
  include(CTest)
  if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    foreach(t test_json test_svg)
      add_executable(${t} test/${t}.cpp)
      target_link_libraries(${t} phm_tools GTest::gtest GTest::gtest_main)
      add_test(NAME phm_tools.${t} COMMAND ${t})
    endforeach()
  endif()
endif()
