cmake_minimum_required(VERSION 3.16)
project(phm_core VERSION 0.2.0 LANGUAGES CXX)

# phm_core builds two ways from this file:
#   - as an ament_cmake package under colcon (ament_cmake found, and
#     PHM_STANDALONE not set), exporting phm_core::phm_core to other packages;
#   - as a plain CMake project with no ROS at all (cmake -S src/phm_core, or the
#     superbuild in standalone/), installing a CMake package config.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(PHM_WITH_EIGEN "Build the Eigen rolling-spread backend if Eigen 3 is found" ON)
option(PHM_WITH_LIBTORCH "Build the LibTorch rolling-spread backend (large dependency)" OFF)

if(NOT PHM_STANDALONE)
  find_package(ament_cmake QUIET)
endif()
if(ament_cmake_FOUND AND NOT PHM_STANDALONE)
  set(PHM_CORE_AMENT ON)
else()
  set(PHM_CORE_AMENT OFF)
endif()

find_package(ZLIB REQUIRED)

set(PHM_CORE_SOURCES
  src/adapters.cpp
  src/arbiter.cpp
  src/backend_plain.cpp
  src/calibration.cpp
  src/format.cpp
  src/npy.cpp
  src/numerics.cpp
  src/numpy_random.cpp
  src/ood_core.cpp
  src/recovery.cpp
  src/severity.cpp
  src/sim.cpp
  src/system_metrics.cpp
  src/ziggurat_tables.cpp
)

set(PHM_EIGEN_FOUND FALSE)
if(PHM_WITH_EIGEN)
  find_package(Eigen3 3.3 QUIET NO_MODULE)
  if(TARGET Eigen3::Eigen)
    set(PHM_EIGEN_FOUND TRUE)
    list(APPEND PHM_CORE_SOURCES src/backend_eigen.cpp)
    message(STATUS "phm_core: Eigen ${Eigen3_VERSION} found, building the eigen backend")
  else()
    message(STATUS "phm_core: Eigen not found, eigen backend disabled")
  endif()
endif()

set(PHM_TORCH_FOUND FALSE)
if(PHM_WITH_LIBTORCH)
  find_package(Torch QUIET)
  if(Torch_FOUND)
    set(PHM_TORCH_FOUND TRUE)
    list(APPEND PHM_CORE_SOURCES src/backend_torch.cpp)
    message(STATUS "phm_core: Torch found, building the libtorch backend")
  else()
    message(WARNING "phm_core: PHM_WITH_LIBTORCH=ON but Torch was not found "
      "(set Torch_DIR); the libtorch backend is disabled")
  endif()
endif()

add_library(phm_core SHARED ${PHM_CORE_SOURCES})
add_library(phm_core::phm_core ALIAS phm_core)
target_compile_features(phm_core PUBLIC cxx_std_17)
target_include_directories(phm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(phm_core PRIVATE ZLIB::ZLIB)
set_target_properties(phm_core PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -ffp-contract=off: no fused multiply-add, so the NumPy-order kernels give
  # the same bits on x86-64 and aarch64 (where GCC would otherwise contract).
  target_compile_options(phm_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()
if(PHM_EIGEN_FOUND)
  target_compile_definitions(phm_core PRIVATE PHM_HAVE_EIGEN)
  target_link_libraries(phm_core PRIVATE Eigen3::Eigen)
endif()
if(PHM_TORCH_FOUND)
  target_compile_definitions(phm_core PRIVATE PHM_WITH_LIBTORCH)
  target_link_libraries(phm_core PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(phm_core PRIVATE ${TORCH_INCLUDE_DIRS})
  set_target_properties(phm_core PROPERTIES INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

# The README's concrete example, built so its printed numbers stay verified.
add_executable(phm_collapse_example examples/collapse_example.cpp)
target_link_libraries(phm_collapse_example phm_core)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS phm_core EXPORT phm_coreTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests (GoogleTest). Under ament the vendored gtest is used; standalone uses
# the system GTest package.
# ---------------------------------------------------------------------------
set(PHM_CORE_TESTS
  test_adapters
  test_arbiter
  test_calibration
  test_format
  test_npy
  test_numpy_random
  test_ood_core
  test_recovery
  test_severity_hysteresis
  test_sim
  test_system_metrics
)

if(PHM_CORE_AMENT)
  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
    find_package(ament_cmake_gtest REQUIRED)
    foreach(t ${PHM_CORE_TESTS})
      ament_add_gtest(${t} test/${t}.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
      target_link_libraries(${t} phm_core)
      target_compile_options(${t} PRIVATE -ffp-contract=off)
    endforeach()
  endif()
  ament_export_targets(phm_coreTargets HAS_LIBRARY_TARGET)
  ament_export_dependencies(ZLIB)
  ament_package()
else()
  include(CMakePackageConfigHelpers)
  install(EXPORT phm_coreTargets NAMESPACE phm_core:: DESTINATION lib/cmake/phm_core)
  configure_package_config_file(cmake/phm_coreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/phm_coreConfig.cmake
    INSTALL_DESTINATION lib/cmake/phm_core)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/phm_coreConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/phm_coreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/phm_coreConfigVersion.cmake
    DESTINATION lib/cmake/phm_core)

  include(CTest)
  if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    foreach(t ${PHM_CORE_TESTS})
      add_executable(${t} test/${t}.cpp)
      target_link_libraries(${t} phm_core GTest::gtest GTest::gtest_main)
      target_compile_options(${t} PRIVATE -ffp-contract=off)
      add_test(NAME phm_core.${t} COMMAND ${t}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
    endforeach()
    add_test(NAME phm_core.collapse_example COMMAND phm_collapse_example)
    set_tests_properties(phm_core.collapse_example PROPERTIES PASS_REGULAR_EXPRESSION
      "calibrated threshold   56.86.*first alarm at/after collapse   frame 200.*2 / 181")
  endif()
endif()
