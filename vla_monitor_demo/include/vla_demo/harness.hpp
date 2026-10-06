// Copyright 2026 Yusuf Guenena. MIT License.
// Monitor-before-output-collapse demo: a small stand-in policy is trained
// in-process, then an input-perturbation strength alpha is swept from 0 (clean)
// to 1 (full shift). At each alpha the harness measures the policy's output
// error against the clean output and the PHM rolling-spread OOD score of the
// policy's hidden layer, calibrated on the clean hidden features at the 1st
// percentile. Lead time = alpha where the output collapses - alpha where the
// monitor fires; positive means the monitor fires first.
//
// The stand-in policy is a 16 -> 48 -> 48 -> 2 tanh MLP, NOT a VLA, trained
// with Adam on a synthetic control law. Its initialization reproduces
// torch.manual_seed(0) exactly and the data draws reproduce NumPy's
// default_rng streams; training arithmetic differs from PyTorch's in the last
// bits (README.md reports the measured effect). The OOD math is phm_core's
// rolling spread (evaluated in float32, as NumPy evaluated it on the float32
// hidden activations) and calibrate_threshold.
#ifndef VLA_DEMO__HARNESS_HPP_
#define VLA_DEMO__HARNESS_HPP_

#include <cstdint>
#include <vector>

#include "phm_tools/torch_mlp.hpp"

namespace vla_demo
{

using phm_tools::torch_mlp::MatrixF;

constexpr int kInDim = 16;
constexpr int kHiddenDim = 48;
constexpr int kOutDim = 2;

// Ground-truth control law: (tanh(2 mean(x[:8])), sin(2 mean(x[8:]))).
MatrixF true_control(const MatrixF & x);

// n x 16 standard normals from a NumPy default_rng(seed) stream, as float32,
// and their labels.
struct Dataset
{
  MatrixF x;
  MatrixF y;
};
Dataset make_dataset(int n, uint64_t seed);

struct TrainedPolicy
{
  phm_tools::torch_mlp::Mlp mlp;
  float final_mse;
};
// Train the stand-in policy (8192 samples, 350 full-batch Adam steps, lr 1e-2).
TrainedPolicy train_policy(
  int n_train = 8192, int epochs = 350, double lr = 1e-2, uint64_t seed = 0);

// x + alpha * N(0,1) noise (from default_rng(noise_seed)) + alpha * 3 along
// the all-ones direction / sqrt(16), in float32; alpha == 0 returns x.
MatrixF perturb_inputs(const MatrixF & x, double alpha, uint64_t noise_seed);

struct SweepResult
{
  std::vector<double> alphas;
  std::vector<double> output_error;      // normalized to the sweep maximum
  std::vector<double> raw_output_error;  // mean L2 action error vs the clean output
  std::vector<double> ood_score;         // mean rolling spread of the hidden features
  std::vector<double> fired_fraction;    // fraction of windows below the threshold
  double threshold = 0.0;
  int window = 30;
  double clean_fpr = 0.0;
  double output_degradation_level = 0.5;
  double monitor_fire_fraction = 0.05;
  double monitor_fires_at = 0.0;         // NaN if never
  double output_collapses_at = 0.0;      // NaN if never
  double lead_time = 0.0;
};

SweepResult run_sweep(
  const phm_tools::torch_mlp::Mlp & policy, int n_eval = 600, int n_alphas = 21,
  int window = 30, double percentile = 1.0, double output_degradation_level = 0.5,
  double monitor_fire_fraction = 0.05, uint64_t seed = 7);

// Smallest x[i] with y[i] >= level, NaN if none.
double first_crossing_up(
  const std::vector<double> & x, const std::vector<double> & y, double level);

}  // namespace vla_demo

#endif  // VLA_DEMO__HARNESS_HPP_
