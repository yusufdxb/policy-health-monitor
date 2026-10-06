cmake_minimum_required(VERSION 3.16)
project(phm_sim)

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
find_package(std_srvs REQUIRED)

add_executable(embedding_publisher src/embedding_publisher.cpp)
target_link_libraries(embedding_publisher phm_core::phm_core)
ament_target_dependencies(embedding_publisher phm_msgs rcl_interfaces rclcpp std_srvs)

# End-to-end timing of a running PHM graph (measurement tool).
add_executable(phm_chain_bench src/chain_bench.cpp)
target_link_libraries(phm_chain_bench phm_core::phm_core)
ament_target_dependencies(phm_chain_bench geometry_msgs phm_msgs rclcpp)

install(TARGETS embedding_publisher phm_chain_bench DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
