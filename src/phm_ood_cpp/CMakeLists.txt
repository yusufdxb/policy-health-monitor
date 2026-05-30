cmake_minimum_required(VERSION 3.8)
project(phm_ood_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(phm_msgs REQUIRED)

# --------------------------------------------------------------------------
# Backend selection. The plain std::vector backend is ALWAYS compiled so this
# package builds with zero math dependencies. Eigen is opportunistic. LibTorch
# is opt-in behind PHM_WITH_LIBTORCH (default OFF).
# --------------------------------------------------------------------------
set(PHM_BACKEND_SRCS
  src/backend_plain.cpp
  src/ood_core.cpp
)

# Eigen: enabled automatically if the system headers exist
# (/usr/include/eigen3, confirmed present on this workstation).
option(PHM_WITH_EIGEN "Build the Eigen rolling-spread backend if Eigen is found" ON)
set(PHM_EIGEN_FOUND FALSE)
if(PHM_WITH_EIGEN)
  find_package(Eigen3 QUIET)
  if(Eigen3_FOUND)
    set(PHM_EIGEN_FOUND TRUE)
    list(APPEND PHM_BACKEND_SRCS src/backend_eigen.cpp)
    message(STATUS "phm_ood_cpp: Eigen3 found, building Eigen backend")
  elseif(EXISTS "/usr/include/eigen3/Eigen/Dense")
    # Fall back to the raw include dir if no CMake config package is installed.
    set(PHM_EIGEN_FOUND TRUE)
    set(PHM_EIGEN_INCLUDE_DIR "/usr/include/eigen3")
    list(APPEND PHM_BACKEND_SRCS src/backend_eigen.cpp)
    message(STATUS "phm_ood_cpp: Eigen headers at /usr/include/eigen3, building Eigen backend")
  else()
    message(STATUS "phm_ood_cpp: Eigen not found, Eigen backend disabled")
  endif()
endif()

# LibTorch: opt-in. Caller passes -DPHM_WITH_LIBTORCH=ON. We resolve Torch_DIR
# from the system python torch install if not already provided.
option(PHM_WITH_LIBTORCH "Build the LibTorch rolling-spread backend" OFF)
set(PHM_TORCH_FOUND FALSE)
if(PHM_WITH_LIBTORCH)
  if(NOT DEFINED Torch_DIR AND NOT DEFINED CMAKE_PREFIX_PATH_TORCH)
    execute_process(
      COMMAND /usr/bin/python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE PHM_TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(PHM_TORCH_PREFIX)
      list(APPEND CMAKE_PREFIX_PATH "${PHM_TORCH_PREFIX}")
      message(STATUS "phm_ood_cpp: trying Torch cmake prefix ${PHM_TORCH_PREFIX}")
    endif()
  endif()
  find_package(Torch QUIET)
  if(Torch_FOUND)
    set(PHM_TORCH_FOUND TRUE)
    list(APPEND PHM_BACKEND_SRCS src/backend_torch.cpp)
    message(STATUS "phm_ood_cpp: Torch found, building LibTorch backend")
  else()
    message(WARNING "phm_ood_cpp: PHM_WITH_LIBTORCH=ON but Torch not found; "
                    "LibTorch backend disabled, plain/eigen path still builds")
  endif()
endif()

# --------------------------------------------------------------------------
# Core library (ROS-free math + decision logic).
# --------------------------------------------------------------------------
add_library(phm_ood_core ${PHM_BACKEND_SRCS})
target_include_directories(phm_ood_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(PHM_EIGEN_FOUND)
  target_compile_definitions(phm_ood_core PUBLIC PHM_HAVE_EIGEN)
  if(Eigen3_FOUND)
    target_link_libraries(phm_ood_core Eigen3::Eigen)
  elseif(DEFINED PHM_EIGEN_INCLUDE_DIR)
    target_include_directories(phm_ood_core PRIVATE ${PHM_EIGEN_INCLUDE_DIR})
  endif()
endif()

if(PHM_TORCH_FOUND)
  target_compile_definitions(phm_ood_core PUBLIC PHM_WITH_LIBTORCH)
  target_link_libraries(phm_ood_core ${TORCH_LIBRARIES})
  target_include_directories(phm_ood_core PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

# --------------------------------------------------------------------------
# Node executable.
# --------------------------------------------------------------------------
add_executable(ood_node src/ood_node.cpp)
target_link_libraries(ood_node phm_ood_core)
ament_target_dependencies(ood_node rclcpp phm_msgs)

# --------------------------------------------------------------------------
# Latency micro-benchmark (standalone, no ROS).
# --------------------------------------------------------------------------
add_executable(bench_latency src/bench_latency.cpp)
target_link_libraries(bench_latency phm_ood_core)

# When built with the LibTorch backend, the executables pull libtorch .so files
# transitively through the static core but get no RUNPATH to them, so they fail
# at runtime with "libtorch.so: cannot open shared object file". Add the torch
# lib dir to their build- and install-RPATH so they run from the install tree.
if(PHM_TORCH_FOUND)
  set_target_properties(ood_node bench_latency PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH ON)
endif()

install(TARGETS ood_node bench_latency
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS phm_ood_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/
  DESTINATION include)

# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ood_core test/test_ood_core.cpp)
  if(TARGET test_ood_core)
    target_link_libraries(test_ood_core phm_ood_core)
  endif()
endif()

ament_package()
