cmake_minimum_required(VERSION 3.16)
project(phm_arbiter)

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
find_package(rclcpp REQUIRED)

add_library(phm_arbiter_lib SHARED src/arbiter_node.cpp)
target_include_directories(phm_arbiter_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(phm_arbiter_lib phm_core::phm_core)
ament_target_dependencies(phm_arbiter_lib phm_msgs rclcpp)

add_executable(phm_arbiter src/main.cpp)
target_link_libraries(phm_arbiter phm_arbiter_lib)

install(TARGETS phm_arbiter_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS phm_arbiter DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_arbiter_node test/test_arbiter_node.cpp TIMEOUT 120
    ENV ROS_DOMAIN_ID=73)
  if(TARGET test_arbiter_node)
    target_link_libraries(test_arbiter_node phm_arbiter_lib)
  endif()
endif()

ament_package()
