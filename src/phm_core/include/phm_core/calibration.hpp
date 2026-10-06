// Copyright 2026 Yusuf Guenena. MIT License.
// Rolling-spread OOD calibration, ported from the public supercombo-blindspot
// src/e6_detector.py (rolling_spread, calibrate_threshold, loco_fpr).
//
// The OOD signal is the per-frame trace of the rolling covariance of a
// policy's hidden state. When the hidden state collapses (freezes to a point)
// the windowed variance sum drops toward zero, so a spread below a threshold
// calibrated on in-distribution data flags out-of-distribution behavior.
//
// Results are bit-identical to the NumPy implementation this replaces (same
// evaluation order; see numerics.hpp). loco_fpr takes a list of corpora and
// keys folds by index, as the Python port did (the source keyed a dict).
#ifndef PHM_CORE__CALIBRATION_HPP_
#define PHM_CORE__CALIBRATION_HPP_

#include <cstddef>
#include <vector>

#include "phm_core/matrix.hpp"

namespace phm_core
{

// Per-frame rolling spread of hidden (T x D); NaN until the window fills, so
// out[t] covers frames (t - window, t]. window must be >= 1.
std::vector<double> rolling_spread(const Matrix & hidden, std::size_t window);

// Spread of one block of frames (all rows of `block`).
double block_spread(const Matrix & block);

// rolling_spread of float32 frames (rows x cols, row-major) evaluated in
// float32 arithmetic, as NumPy evaluates np.var on a float32 array; each value
// is widened to double afterwards, as float(np.float32(x)) is.
std::vector<double> rolling_spread_f32(
  const float * hidden, std::size_t rows, std::size_t cols, std::size_t window);

// Below this spread = OOD: the percentile-th percentile (linear method) of the
// non-NaN spreads. Throws std::invalid_argument if no non-NaN value remains.
double calibrate_threshold(const std::vector<double> & spreads, double percentile = 1.0);

struct LocoFold
{
  double threshold = 0.0;
  double fpr = 0.0;                      // NaN if the held-out corpus has no valid spread
  std::vector<std::size_t> calibrated_on;
};

struct LocoResult
{
  std::vector<LocoFold> folds;           // folds[i] holds corpus i out
  double fpr_mean = 0.0;
  double fpr_max = 0.0;
};

// Leave-one-corpus-out false-positive rate: calibrate on the other corpora,
// measure the fraction of the held-out corpus's valid spreads below the
// threshold, and report each fold plus the mean and max.
LocoResult loco_fpr(const std::vector<Matrix> & corpora, std::size_t window, double percentile);

}  // namespace phm_core

#endif  // PHM_CORE__CALIBRATION_HPP_
