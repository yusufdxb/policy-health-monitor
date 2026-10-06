cmake_minimum_required(VERSION 3.16)
project(phm_recovery)

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
find_package(geometry_msgs REQUIRED)
find_package(phm_core REQUIRED)
find_package(phm_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)

# The node as a library, so a host stack can embed it and register a rewind
# callback.
add_library(phm_recovery_lib SHARED src/recovery_node.cpp)
target_include_directories(phm_recovery_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(phm_recovery_lib phm_core::phm_core)
ament_target_dependencies(phm_recovery_lib geometry_msgs phm_msgs rcl_interfaces rclcpp)

add_executable(recovery_node src/main.cpp)
target_link_libraries(recovery_node phm_recovery_lib)

install(TARGETS phm_recovery_lib EXPORT phm_recoveryTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS recovery_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_recovery_node test/test_recovery_node.cpp TIMEOUT 120
    ENV ROS_DOMAIN_ID=74)
  if(TARGET test_recovery_node)
    target_link_libraries(test_recovery_node phm_recovery_lib)
  endif()
endif()

ament_export_targets(phm_recoveryTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(geometry_msgs phm_core phm_msgs rcl_interfaces rclcpp)
ament_package()
