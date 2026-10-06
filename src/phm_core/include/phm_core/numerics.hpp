// Copyright 2026 Yusuf Guenena. MIT License.
// NumPy-compatible reductions used by calibration, the OOD spread kernel and
// the benchmark. They reproduce NumPy's floating-point evaluation order, so a
// threshold calibrated here equals the one NumPy produced for the same data:
//
//   numpy_sum()  np.add.reduce over a contiguous float64 vector: NumPy's
//                8-way unrolled pairwise summation (blocks of 128) added to
//                the identity 0.0.
//   percentile() np.percentile(a, q) with the default 'linear' method.
//   quantile()   np.quantile(a, q) with the default 'linear' method.
//
// Callers must not compile these with floating-point contraction (FMA); the
// phm_core build passes -ffp-contract=off.
#ifndef PHM_CORE__NUMERICS_HPP_
#define PHM_CORE__NUMERICS_HPP_

#include <cstddef>
#include <vector>

namespace phm_core
{

// np.sum of a contiguous vector (np.add.reduce evaluation order), in the
// vector's own precision as NumPy does for float64 and float32.
double numpy_sum(const double * a, std::size_t n);
float numpy_sum(const float * a, std::size_t n);

// np.mean of a contiguous float64 vector: numpy_sum / n (NaN for n == 0).
double numpy_mean(const double * a, std::size_t n);

// np.percentile(values, q) for q in [0, 100], linear interpolation. Throws
// std::invalid_argument on an empty input or q outside [0, 100].
double percentile(std::vector<double> values, double q);

// np.quantile(values, q) for q in [0, 1], linear interpolation.
double quantile(std::vector<double> values, double q);

}  // namespace phm_core

#endif  // PHM_CORE__NUMERICS_HPP_
