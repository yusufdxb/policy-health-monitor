// Copyright 2026 Yusuf Guenena. MIT License.
// The NumPy-order rolling-spread kernel shared by the plain backend (float32
// ring frames) and calibration (float64 matrix rows). Private header.
//
// np.var(block, axis=0).sum() on a C-contiguous (W, D) block evaluates, in the
// block's own dtype T:
//   mean_d = (sum over frames of column d) / W
//   var_d  = (sum over frames of (x - mean_d)**2) / W
//   spread = pairwise sum over the D variances
// For D > 1 NumPy's reduction loop runs over the columns and accumulates each
// column frame by frame in order; for D == 1 the column axis vanishes and the
// frame axis itself is reduced with pairwise summation. Both are reproduced
// here, with the column loop innermost so the compiler can vectorize it.
#ifndef SPREAD_KERNEL_HPP_
#define SPREAD_KERNEL_HPP_

#include <cstddef>
#include <vector>

#include "phm_core/numerics.hpp"

namespace phm_core
{
namespace detail
{

// frame(k) returns a pointer to the D values of frame k (0 = oldest), of any
// element type convertible to T. scratch grows on first use and is reused.
template<typename T, typename FrameFn>
T numpy_window_spread(
  FrameFn frame, std::size_t window, std::size_t dim, std::vector<T> & scratch)
{
  const T w = static_cast<T>(window);
  if (dim == 1) {
    if (scratch.size() < window) {
      scratch.assign(window, T(0));
    }
    T * col = scratch.data();
    for (std::size_t k = 0; k < window; ++k) {
      col[k] = static_cast<T>(frame(k)[0]);
    }
    const T mean = numpy_sum(col, window) / w;
    for (std::size_t k = 0; k < window; ++k) {
      const T x = col[k] - mean;
      col[k] = x * x;
    }
    const T var = numpy_sum(col, window) / w;
    return numpy_sum(&var, 1);
  }
  if (scratch.size() < 2 * dim) {
    scratch.assign(2 * dim, T(0));
  }
  T * mean = scratch.data();
  T * ss = scratch.data() + dim;
  for (std::size_t d = 0; d < dim; ++d) {
    mean[d] = T(0);
    ss[d] = T(0);
  }
  for (std::size_t k = 0; k < window; ++k) {
    const auto * row = frame(k);
    for (std::size_t d = 0; d < dim; ++d) {
      mean[d] += static_cast<T>(row[d]);
    }
  }
  for (std::size_t d = 0; d < dim; ++d) {
    mean[d] /= w;
  }
  for (std::size_t k = 0; k < window; ++k) {
    const auto * row = frame(k);
    for (std::size_t d = 0; d < dim; ++d) {
      const T x = static_cast<T>(row[d]) - mean[d];
      ss[d] += x * x;
    }
  }
  for (std::size_t d = 0; d < dim; ++d) {
    ss[d] /= w;
  }
  return numpy_sum(ss, dim);
}

}  // namespace detail
}  // namespace phm_core

#endif  // SPREAD_KERNEL_HPP_
