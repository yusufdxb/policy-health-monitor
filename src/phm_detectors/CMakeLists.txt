cmake_minimum_required(VERSION 3.16)
project(phm_detectors)

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

add_library(phm_detectors_lib SHARED src/detectors_node.cpp)
target_include_directories(phm_detectors_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(phm_detectors_lib phm_core::phm_core)
ament_target_dependencies(phm_detectors_lib phm_msgs rcl_interfaces rclcpp)

add_executable(phm_detectors_node src/main.cpp)
target_link_libraries(phm_detectors_node phm_detectors_lib)

install(TARGETS phm_detectors_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS phm_detectors_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  find_package(std_msgs REQUIRED)
  ament_add_gtest(test_detectors_node test/test_detectors_node.cpp TIMEOUT 120
    ENV ROS_DOMAIN_ID=72)
  if(TARGET test_detectors_node)
    target_link_libraries(test_detectors_node phm_detectors_lib)
    ament_target_dependencies(test_detectors_node std_msgs)
  endif()
endif()

ament_package()
