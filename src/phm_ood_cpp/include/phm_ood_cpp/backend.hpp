// Copyright 2026 Yusuf Guenena. MIT License.
// Pluggable backend interface for the rolling-spread compute.
//
// The OOD node holds a Backend* and never touches the math directly, so the
// numerical kernel (sum over dims of the windowed population variance) can be
// swapped between a dependency-free std::vector implementation (always built),
// an Eigen implementation (built when /usr/include/eigen3 exists), and a
// LibTorch implementation (behind the PHM_WITH_LIBTORCH CMake option).
//
// Math contract (identical for every backend), ported from:
//   phm_core/phm_core/calibration.py:25-36  rolling_spread()
//   which is byte-faithful to phantom-braking/src/e6_detector.py:16-23
//
//   rolling_spread over a (window, D) block = sum_d Var_pop(block[:, d])
//   where Var_pop is the population (ddof=0, biased) variance, matching
//   numpy.var default. Only the trailing full window is scored by this node.
#ifndef PHM_OOD_CPP__BACKEND_HPP_
#define PHM_OOD_CPP__BACKEND_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace phm_ood_cpp
{

// One contiguous (window x dim) block, row-major: frame f, dim d at
// block[f * dim + d]. This is the input the node hands to a backend.
class Backend
{
public:
  virtual ~Backend() = default;

  // Sum over dims of the population variance of each column.
  // block.size() must equal window * dim. Returns the scalar rolling spread.
  virtual double rolling_spread(
    const std::vector<double> & block, std::size_t window, std::size_t dim) const = 0;

  // Human-readable backend name for logging ("plain", "eigen", "libtorch").
  virtual std::string name() const = 0;
};

// Dependency-free std::vector implementation. Always available.
std::unique_ptr<Backend> make_plain_backend();

#ifdef PHM_HAVE_EIGEN
// Eigen implementation, built only when Eigen headers were found at configure.
std::unique_ptr<Backend> make_eigen_backend();
#endif

#ifdef PHM_WITH_LIBTORCH
// LibTorch implementation, built only when -DPHM_WITH_LIBTORCH=ON and Torch found.
std::unique_ptr<Backend> make_torch_backend();
#endif

// Select the best backend available at build time. Preference order:
// libtorch (if enabled) > eigen (if found) > plain. Honors the env var
// PHM_BACKEND ("plain" | "eigen" | "libtorch") to force a specific one;
// falls back to plain if the requested backend was not compiled in.
std::unique_ptr<Backend> make_default_backend();

}  // namespace phm_ood_cpp

#endif  // PHM_OOD_CPP__BACKEND_HPP_
