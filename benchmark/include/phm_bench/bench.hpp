// Copyright 2026 Yusuf Guenena. MIT License.
// Reliability benchmark: the PHM rolling-spread OOD score against canonical
// feature-space OOD baselines on synthetic in-distribution vs OOD embedding
// streams, with threshold-free metrics and stratified bootstrap CIs.
//
// Seeds mean what they meant in the original NumPy harness: every random draw
// goes through phm_core's NumPy-compatible streams (default_rng for the
// generator, legacy RandomState for k-means init, the closed-form RND target
// and the bootstrap), and every reduction that feeds a rank or a threshold
// uses NumPy's evaluation order. Dense linear algebra (Cholesky, inverse,
// solve, matrix products) runs on Eigen rather than NumPy's BLAS/LAPACK, so
// those values agree with NumPy to rounding (about 1e-15 relative), not bit
// for bit.
#ifndef PHM_BENCH__BENCH_HPP_
#define PHM_BENCH__BENCH_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "phm_core/matrix.hpp"

namespace phm_bench
{

using phm_core::Matrix;

// ---------------------------------------------------------------------------
// Streams (benchmark/lib/generator.py)
// ---------------------------------------------------------------------------
struct StreamSpec
{
  int dim = 64;
  int n_id = 600;
  int n_ood = 600;
  std::string ood_mode = "collapse";  // "collapse" | "shift"
  double in_dist_scale = 1.0;
  double ood_scale = 0.02;            // collapse: near-zero variance
  double ood_shift = 4.0;             // shift: mean offset in sigma units
  double ar_rho = 0.85;               // AR(1) temporal correlation of the ID stream
  uint64_t seed = 42;
};

struct Streams
{
  Matrix id;   // n_id x dim: stable, temporally correlated, healthy spread
  Matrix ood;  // n_ood x dim: collapsed around the last ID frame, or shifted
};

// ID frames: AR(1) x_t = mean + rho (x_{t-1} - mean) + sqrt(1 - rho^2) L eps_t
// over a random non-isotropic covariance L L^T. OOD "collapse": the last ID
// frame plus N(0, ood_scale^2) noise. OOD "shift": the AR(1) process around
// mean + ood_shift * sigma with half the covariance. Throws on another mode.
Streams generate_stream(const StreamSpec & spec);

// Mean over valid frames of the windowed spread (a scalar spread summary).
double rolling_spread_trace(const Matrix & frames, std::size_t window);

// ---------------------------------------------------------------------------
// Detectors. Each returns one score per test frame, higher = more OOD.
// ---------------------------------------------------------------------------

// PHM: threshold - rolling_spread(test), threshold calibrated on the ID
// frames' spread at `percentile`; NaN until the window fills.
std::vector<double> phm_scores(
  const Matrix & fid, const Matrix & test, std::size_t window = 20, double percentile = 1.0);

// Squared Mahalanobis distance to a ridge-regularized Gaussian fit of the ID
// features (Lee et al. 2018).
std::vector<double> mahalanobis(const Matrix & fid, const Matrix & test, double reg = 0.1);

// Relative Mahalanobis (Ren et al. 2021): min over a 2-component k-means
// background fit minus the single-Gaussian distance.
std::vector<double> relative_mahalanobis(
  const Matrix & fid, const Matrix & test, double reg = 0.1);

// Distance to the k-th nearest ID neighbour (Sun et al. 2022), optionally on
// L2-normalized features.
std::vector<double> knn_distance(
  const Matrix & fid, const Matrix & test, int k = 50, bool normalize = true);

// Random Network Distillation (Burda et al. 2019), closed form: a fixed random
// ReLU target and a ridge-regression predictor; per-frame squared error.
std::vector<double> rnd_closed_form(
  const Matrix & fid, const Matrix & test, int proj_dim = 128, double reg = 1e-2,
  uint32_t seed = 0);

// Random Network Distillation with a gradient-trained predictor: a fixed
// random MLP target and an MLP predictor (Linear(D,128)-ReLU-Linear(128,128))
// trained by Adam (lr 1e-3, full batch, `epochs` steps) on standardized ID
// features, initialized exactly as torch.manual_seed(seed) would.
struct RndMlpResult
{
  std::vector<double> scores;
  float final_train_mse = 0.0f;
};
RndMlpResult rnd_mlp(const Matrix & fid, const Matrix & test, int epochs = 300, uint64_t seed = 0);

// ---------------------------------------------------------------------------
// Metrics (benchmark/lib/metrics.py; sklearn definitions). Non-finite scores
// are dropped first. A single-class input gives NaN.
// ---------------------------------------------------------------------------
double auroc(const std::vector<double> & scores, const std::vector<int> & labels);
double aupr(const std::vector<double> & scores, const std::vector<int> & labels);
double fpr_at_tpr(
  const std::vector<double> & scores, const std::vector<int> & labels, double tpr = 0.95);

struct Ci
{
  double mean;
  double lo;
  double hi;
};
using MetricFn = std::function<double(const std::vector<double> &, const std::vector<int> &)>;
// Stratified bootstrap: resample OOD and ID frames independently with
// replacement (numpy.random.RandomState(seed).choice), keep finite values,
// report (mean, quantile(alpha), quantile(1 - alpha)).
Ci bootstrap_ci(
  const MetricFn & metric, const std::vector<double> & scores, const std::vector<int> & labels,
  int n_bootstrap = 1000, double ci = 0.95, uint32_t seed = 42);

}  // namespace phm_bench

#endif  // PHM_BENCH__BENCH_HPP_
