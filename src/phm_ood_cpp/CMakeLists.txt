cmake_minimum_required(VERSION 3.16)
project(phm_ood_cpp)

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

find_package(ament_cmake REQUIRED)
find_package(phm_core REQUIRED)
find_package(phm_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# The OOD lifecycle node as a library, shared with the phm_ood package (same
# implementation, phm_ood interface profile). The rolling-spread kernels and
# the decision logic live in phm_core.
add_library(phm_ood_lifecycle SHARED src/ood_lifecycle_node.cpp)
target_include_directories(phm_ood_lifecycle PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(phm_ood_lifecycle phm_core::phm_core)
ament_target_dependencies(phm_ood_lifecycle phm_msgs rcl_interfaces rclcpp rclcpp_lifecycle)

add_executable(ood_node src/ood_node.cpp)
target_link_libraries(ood_node phm_ood_lifecycle)

# Detector-core latency micro-benchmark (no ROS graph).
add_executable(bench_latency src/bench_latency.cpp)
target_link_libraries(bench_latency phm_core::phm_core)

install(TARGETS phm_ood_lifecycle EXPORT phm_ood_lifecycleTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS ood_node bench_latency DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ood_node test/test_ood_node.cpp TIMEOUT 120 ENV ROS_DOMAIN_ID=71)
  if(TARGET test_ood_node)
    target_link_libraries(test_ood_node phm_ood_lifecycle)
    ament_target_dependencies(test_ood_node lifecycle_msgs)
  endif()
endif()

ament_export_targets(phm_ood_lifecycleTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(phm_core phm_msgs rcl_interfaces rclcpp rclcpp_lifecycle)
ament_package()
