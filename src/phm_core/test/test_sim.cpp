// Copyright 2026 Yusuf Guenena. MIT License.
// Synthetic streams (ported from phm_sim/tests/test_sim_core.py), with the
// first frame pinned to NumPy's default_rng(42) output.
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/sim.hpp"

using phm_core::EmbeddingStream;
using phm_core::generate_embeddings;
using phm_core::Matrix;

namespace
{
double variance(const std::vector<double> & v)
{
  const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  double ss = 0.0;
  for (double x : v) {
    ss += (x - mean) * (x - mean);
  }
  return ss / v.size();
}
std::vector<double> valid(const std::vector<double> & s)
{
  std::vector<double> out;
  for (double x : s) {
    if (!std::isnan(x)) {
      out.push_back(x);
    }
  }
  return out;
}
double mean(const std::vector<double> & v)
{
  return phm_core::numpy_mean(v.data(), v.size());
}
}  // namespace

TEST(GenerateEmbeddings, ShapesSeedAndNumpyParity)
{
  const auto b = generate_embeddings(32, 50);
  EXPECT_EQ(b.in_dist.rows, 50u);
  EXPECT_EQ(b.in_dist.cols, 32u);
  EXPECT_EQ(b.ood.rows, 50u);
  const auto a = generate_embeddings(64, 200, 1.0, 0.01, 42);
  EXPECT_EQ(a.in_dist(0, 0), 0x1.3807c1104fc6bp-2);
  EXPECT_EQ(a.in_dist(0, 1), -0x1.0a3c65fca9a7ep+0);
  EXPECT_EQ(a.ood(199, 63), 0x1.a9880bc76651cp-10);
  EXPECT_EQ(generate_embeddings(16, 10, 1.0, 0.01, 7).in_dist.data,
    generate_embeddings(16, 10, 1.0, 0.01, 7).in_dist.data);
  EXPECT_NE(generate_embeddings(16, 10, 1.0, 0.01, 1).in_dist.data,
    generate_embeddings(16, 10, 1.0, 0.01, 2).in_dist.data);
}

TEST(GenerateEmbeddings, InDistHasFarHigherVarianceAndSpread)
{
  const auto b = generate_embeddings(64, 500, 1.0, 0.01, 0);
  EXPECT_GT(variance(b.in_dist.data), variance(b.ood.data) * 100);
  const auto c = generate_embeddings(64, 300, 1.0, 0.01, 42);
  const auto in_s = valid(phm_core::rolling_spread(c.in_dist, 20));
  const auto ood_s = valid(phm_core::rolling_spread(c.ood, 20));
  EXPECT_GT(mean(in_s) / mean(ood_s), 100.0);
  const double thr = phm_core::calibrate_threshold(in_s, 1.0);
  EXPECT_GT(thr, 0.0);
  int above = 0;
  int below = 0;
  for (double s : in_s) {
    above += s >= thr;
  }
  for (double s : ood_s) {
    below += s < thr;
  }
  EXPECT_GE(static_cast<double>(above) / in_s.size(), 0.90);
  EXPECT_GE(static_cast<double>(below) / ood_s.size(), 0.90);
}

TEST(Stream, PhasesTriggerResetAndValidation)
{
  EmbeddingStream s(8, 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(s.is_ood_phase());
    EXPECT_EQ(s.frame_index(), static_cast<std::size_t>(i));
    EXPECT_EQ(s.next_frame().size(), 8u);
  }
  EXPECT_TRUE(s.is_ood_phase());
  const auto idx = s.frame_index();
  s.trigger_ood();
  EXPECT_EQ(s.frame_index(), idx);
  s.reset();
  EXPECT_EQ(s.frame_index(), 0u);
  EXPECT_FALSE(s.is_ood_phase());
  EmbeddingStream t(8, 100);
  t.trigger_ood();
  EXPECT_TRUE(t.is_ood_phase());
  EXPECT_THROW(EmbeddingStream(8, 0), std::invalid_argument);
  EXPECT_EQ(EmbeddingStream().policy_id(), "phm_sim");
}

TEST(Stream, SeedReproducibilityAndPhaseVariance)
{
  EmbeddingStream a(16, 10, 1.0, 0.01, "p", 99);
  EmbeddingStream b(16, 10, 1.0, 0.01, "p", 99);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(a.next_frame(), b.next_frame());
  }
  EmbeddingStream c(64, 200, 1.0, 0.01, "p", 0);
  std::vector<double> in;
  std::vector<double> ood;
  for (int i = 0; i < 200; ++i) {
    const auto f = c.next_frame();
    in.insert(in.end(), f.begin(), f.end());
  }
  for (int i = 0; i < 200; ++i) {
    const auto f = c.next_frame();
    ood.insert(ood.end(), f.begin(), f.end());
  }
  EXPECT_GT(variance(in), variance(ood) * 100);
  // The stream is the same NumPy draw sequence: frame 0 of seed 42.
  EmbeddingStream d(64, 100, 1.0, 0.01, "p", 42);
  EXPECT_EQ(d.next_frame()[0], 0x1.3807c1104fc6bp-2);
}
