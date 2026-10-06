// Copyright 2026 Yusuf Guenena. MIT License.
#include "vla_demo/harness.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/numpy_random.hpp"

namespace vla_demo
{

namespace
{

using phm_tools::torch_mlp::Activation;
using phm_tools::torch_mlp::Mlp;

// float32 mean of 8 contiguous values as NumPy reduces them (pairwise block).
float mean8(const float * a)
{
  return phm_core::numpy_sum(a, 8) / 8.0f;
}

MatrixF standard_normal_f32(int rows, int cols, phm_core::numpy_random::Generator & rng)
{
  MatrixF x(rows, cols);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    x.data()[i] = static_cast<float>(rng.standard_normal());
  }
  return x;
}

}  // namespace

MatrixF true_control(const MatrixF & x)
{
  MatrixF y(x.rows(), kOutDim);
  for (Eigen::Index r = 0; r < x.rows(); ++r) {
    const float * row = x.row(r).data();
    y(r, 0) = std::tanh(mean8(row) * 2.0f);
    y(r, 1) = std::sin(mean8(row + 8) * 2.0f);
  }
  return y;
}

Dataset make_dataset(int n, uint64_t seed)
{
  phm_core::numpy_random::Generator rng(seed);
  Dataset d;
  d.x = standard_normal_f32(n, kInDim, rng);
  d.y = true_control(d.x);
  return d;
}

TrainedPolicy train_policy(int n_train, int epochs, double lr, uint64_t seed)
{
  const Dataset data = make_dataset(n_train, seed);
  auto gen = phm_tools::torch_mlp::manual_seed(seed);
  TrainedPolicy p{Mlp({kInDim, kHiddenDim, kHiddenDim, kOutDim}, Activation::kTanh, gen), 0.0f};
  for (int e = 0; e < epochs; ++e) {
    p.final_mse = p.mlp.train_step(data.x, data.y, lr);
  }
  return p;
}

MatrixF perturb_inputs(const MatrixF & x, double alpha, uint64_t noise_seed)
{
  if (alpha == 0.0) {
    return x;
  }
  phm_core::numpy_random::Generator rng(noise_seed);
  const float a = static_cast<float>(alpha * 1.0);
  // direction = ones / sqrt(in_dim) = 0.25; shift = (alpha * 3) * direction.
  const float shift = static_cast<float>(alpha * 3.0) *
    (1.0f / static_cast<float>(std::sqrt(static_cast<double>(kInDim))));
  MatrixF out(x.rows(), x.cols());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const float noise = static_cast<float>(rng.standard_normal()) * a;
    out.data()[i] = (x.data()[i] + noise) + shift;
  }
  return out;
}

double first_crossing_up(
  const std::vector<double> & x, const std::vector<double> & y, double level)
{
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (y[i] >= level) {
      return x[i];
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

SweepResult run_sweep(
  const phm_tools::torch_mlp::Mlp & policy, int n_eval, int n_alphas, int window,
  double percentile, double output_degradation_level, double monitor_fire_fraction,
  uint64_t seed)
{
  SweepResult res;
  res.window = window;
  res.output_degradation_level = output_degradation_level;
  res.monitor_fire_fraction = monitor_fire_fraction;
  const MatrixF x_clean = make_dataset(n_eval, seed).x;
  MatrixF hid_clean;
  const MatrixF act_clean = policy.forward(x_clean, &hid_clean);

  const auto spreads = [window](const MatrixF & hid) {
      return phm_core::rolling_spread_f32(
        hid.data(), static_cast<std::size_t>(hid.rows()), static_cast<std::size_t>(hid.cols()),
        static_cast<std::size_t>(window));
    };
  const auto valid_of = [](const std::vector<double> & s) {
      std::vector<double> v;
      for (double x : s) {
        if (!std::isnan(x)) {
          v.push_back(x);
        }
      }
      return v;
    };
  const auto fraction_below = [](const std::vector<double> & v, double thr) {
      std::vector<double> flags;
      for (double x : v) {
        flags.push_back(x < thr ? 1.0 : 0.0);
      }
      return phm_core::numpy_mean(flags.data(), flags.size());
    };

  const std::vector<double> clean_spread = spreads(hid_clean);
  res.threshold = phm_core::calibrate_threshold(clean_spread, percentile);
  const std::vector<double> clean_valid = valid_of(clean_spread);
  res.clean_fpr = clean_valid.empty() ? std::numeric_limits<double>::quiet_NaN() :
    fraction_below(clean_valid, res.threshold);

  for (int i = 0; i < n_alphas; ++i) {
    // np.linspace(0, 1, n): i * step, with the last point exactly 1.
    const double alpha = i + 1 == n_alphas ? 1.0 :
      static_cast<double>(i) * (1.0 / static_cast<double>(n_alphas - 1));
    res.alphas.push_back(alpha);
    // A fresh noise stream per alpha, independent of sweep order.
    MatrixF hid;
    const MatrixF act = policy.forward(
      perturb_inputs(x_clean, alpha, 1000 + static_cast<uint64_t>(i)), &hid);
    std::vector<float> norms(static_cast<std::size_t>(act.rows()));
    for (Eigen::Index r = 0; r < act.rows(); ++r) {
      const float d0 = act(r, 0) - act_clean(r, 0);
      const float d1 = act(r, 1) - act_clean(r, 1);
      norms[static_cast<std::size_t>(r)] = std::sqrt((0.0f + d0 * d0) + d1 * d1);
    }
    res.raw_output_error.push_back(static_cast<double>(
        phm_core::numpy_sum(norms.data(), norms.size()) / static_cast<float>(norms.size())));
    const std::vector<double> valid = valid_of(spreads(hid));
    res.ood_score.push_back(
      valid.empty() ? std::numeric_limits<double>::quiet_NaN() :
      phm_core::numpy_mean(valid.data(), valid.size()));
    res.fired_fraction.push_back(
      valid.empty() ? std::numeric_limits<double>::quiet_NaN() :
      fraction_below(valid, res.threshold));
  }
  double err_max = 0.0;
  for (double e : res.raw_output_error) {
    err_max = std::max(err_max, e);
  }
  if (!(err_max > 0.0)) {
    err_max = 1.0;
  }
  for (double e : res.raw_output_error) {
    res.output_error.push_back(e / err_max);
  }
  res.monitor_fires_at = first_crossing_up(res.alphas, res.fired_fraction, monitor_fire_fraction);
  res.output_collapses_at =
    first_crossing_up(res.alphas, res.output_error, output_degradation_level);
  res.lead_time = res.output_collapses_at - res.monitor_fires_at;
  return res;
}

}  // namespace vla_demo
