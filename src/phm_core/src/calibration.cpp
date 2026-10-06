// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "phm_core/numerics.hpp"
#include "spread_kernel.hpp"

namespace phm_core
{

std::vector<double> rolling_spread(const Matrix & hidden, std::size_t window)
{
  if (window < 1) {
    throw std::invalid_argument("rolling_spread: window must be >= 1");
  }
  const std::size_t t_total = hidden.rows;
  const std::size_t dim = hidden.cols;
  std::vector<double> out(t_total, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> scratch;
  for (std::size_t t = window; t <= t_total; ++t) {
    const std::size_t start = t - window;
    out[t - 1] = detail::numpy_window_spread(
      [&hidden, start](std::size_t k) {return hidden.row(start + k);}, window, dim, scratch);
  }
  return out;
}

double block_spread(const Matrix & block)
{
  if (block.rows == 0) {
    throw std::invalid_argument("block_spread: empty block");
  }
  std::vector<double> scratch;
  return detail::numpy_window_spread(
    [&block](std::size_t k) {return block.row(k);}, block.rows, block.cols, scratch);
}

std::vector<double> rolling_spread_f32(
  const float * hidden, std::size_t rows, std::size_t cols, std::size_t window)
{
  if (window < 1) {
    throw std::invalid_argument("rolling_spread_f32: window must be >= 1");
  }
  std::vector<double> out(rows, std::numeric_limits<double>::quiet_NaN());
  std::vector<float> scratch;
  for (std::size_t t = window; t <= rows; ++t) {
    const float * start = hidden + (t - window) * cols;
    out[t - 1] = static_cast<double>(
      detail::numpy_window_spread(
        [start, cols](std::size_t k) {return start + k * cols;}, window, cols, scratch));
  }
  return out;
}

double calibrate_threshold(const std::vector<double> & spreads, double percentile_value)
{
  std::vector<double> valid;
  valid.reserve(spreads.size());
  for (double s : spreads) {
    if (!std::isnan(s)) {
      valid.push_back(s);
    }
  }
  if (valid.empty()) {
    throw std::invalid_argument("calibrate_threshold: no non-NaN spread to calibrate on");
  }
  return percentile(std::move(valid), percentile_value);
}

LocoResult loco_fpr(const std::vector<Matrix> & corpora, std::size_t window, double pct)
{
  LocoResult result;
  const std::size_t n = corpora.size();
  std::vector<double> fprs;
  for (std::size_t held_out = 0; held_out < n; ++held_out) {
    LocoFold fold;
    std::vector<const Matrix *> calib;
    for (std::size_t k = 0; k < n; ++k) {
      if (k != held_out) {
        fold.calibrated_on.push_back(k);
        calib.push_back(&corpora[k]);
      }
    }
    const Matrix calib_hidden = vstack(calib);
    fold.threshold = calibrate_threshold(rolling_spread(calib_hidden, window), pct);
    const std::vector<double> held = rolling_spread(corpora[held_out], window);
    std::vector<double> below;
    for (double s : held) {
      if (!std::isnan(s)) {
        below.push_back(s < fold.threshold ? 1.0 : 0.0);
      }
    }
    fold.fpr = below.empty() ? std::numeric_limits<double>::quiet_NaN() :
      numpy_mean(below.data(), below.size());
    fprs.push_back(fold.fpr);
    result.folds.push_back(fold);
  }
  result.fpr_mean = numpy_mean(fprs.data(), fprs.size());
  // np.max propagates NaN.
  double mx = -std::numeric_limits<double>::infinity();
  for (double f : fprs) {
    if (std::isnan(f)) {
      mx = f;
      break;
    }
    mx = std::max(mx, f);
  }
  result.fpr_max = fprs.empty() ? std::numeric_limits<double>::quiet_NaN() : mx;
  return result;
}

}  // namespace phm_core
