// Copyright 2026 Yusuf Guenena. MIT License.
// NumPy-compatible reductions. Evaluation order follows NumPy 1.26:
// numpy/core/src/umath/loops_utils.h.src (pairwise_sum) and
// numpy/lib/function_base.py (_quantile, _lerp, linear method).
#include "phm_core/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace phm_core
{
namespace
{

constexpr std::size_t kPairwiseBlock = 128;

// DOUBLE_pairwise_sum / FLOAT_pairwise_sum for a contiguous vector.
template<typename T>
T pairwise_sum(const T * a, std::size_t n)
{
  if (n < 8) {
    T res = T(0);
    for (std::size_t i = 0; i < n; ++i) {
      res += a[i];
    }
    return res;
  }
  if (n <= kPairwiseBlock) {
    T r[8];
    for (std::size_t j = 0; j < 8; ++j) {
      r[j] = a[j];
    }
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (std::size_t j = 0; j < 8; ++j) {
        r[j] += a[i + j];
      }
    }
    T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) {
      res += a[i];
    }
    return res;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// numpy.lib.function_base._lerp(a, b, t) for scalars.
double lerp(double a, double b, double t)
{
  const double diff_b_a = b - a;
  if (t >= 0.5) {
    return b - diff_b_a * (1.0 - t);
  }
  return a + diff_b_a * t;
}

double linear_quantile(std::vector<double> values, double quantile_fraction)
{
  const std::size_t n = values.size();
  if (n == 0) {
    throw std::invalid_argument("quantile of an empty array");
  }
  // NumPy's 'linear' method uses (n - 1) * q directly (not the generic
  // _compute_virtual_index form) "to avoid some rounding issues".
  const double last = static_cast<double>(n - 1);
  const double virtual_index = last * quantile_fraction;
  std::size_t lo;
  std::size_t hi;
  double previous_for_gamma;
  if (virtual_index >= last) {
    // _get_indexes maps an index at or above the end to -1 (the last value);
    // gamma is still computed against that -1.
    lo = hi = n - 1;
    previous_for_gamma = -1.0;
  } else {
    previous_for_gamma = std::floor(virtual_index);
    lo = static_cast<std::size_t>(previous_for_gamma);
    hi = lo + 1;
  }
  const double gamma = virtual_index - previous_for_gamma;
  std::nth_element(values.begin(), values.begin() + lo, values.end());
  const double a = values[lo];
  double b = a;
  if (hi != lo) {
    b = *std::min_element(values.begin() + lo + 1, values.end());
  }
  return lerp(a, b, gamma);
}

}  // namespace

double numpy_sum(const double * a, std::size_t n)
{
  if (n == 0) {
    return 0.0;
  }
  // np.add.reduce starts from the identity 0.0 and adds one pairwise_sum over
  // all n elements (so a sum of -0.0 values is +0.0, as in NumPy).
  return 0.0 + pairwise_sum(a, n);
}

float numpy_sum(const float * a, std::size_t n)
{
  if (n == 0) {
    return 0.0f;
  }
  return 0.0f + pairwise_sum(a, n);
}

double numpy_mean(const double * a, std::size_t n)
{
  if (n == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return numpy_sum(a, n) / static_cast<double>(n);
}

double percentile(std::vector<double> values, double q)
{
  if (!(q >= 0.0 && q <= 100.0)) {
    throw std::invalid_argument("percentile q must be in [0, 100]");
  }
  // np.percentile divides q by 100 before computing the virtual index.
  return linear_quantile(std::move(values), q / 100.0);
}

double quantile(std::vector<double> values, double q)
{
  if (!(q >= 0.0 && q <= 1.0)) {
    throw std::invalid_argument("quantile q must be in [0, 1]");
  }
  return linear_quantile(std::move(values), q);
}

}  // namespace phm_core
