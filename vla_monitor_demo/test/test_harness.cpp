// Copyright 2026 Yusuf Guenena. MIT License.
// The demo harness (ported from test_harness.py): the policy learns, the
// threshold is sane, the OOD signal rises with alpha, the lead-time formula
// holds, and the shipped configuration's measured lead time is non-negative
// (so README.md cannot silently overclaim).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "phm_core/calibration.hpp"
#include "vla_demo/harness.hpp"

namespace
{
struct Fixture
{
  vla_demo::TrainedPolicy policy = vla_demo::train_policy();
  vla_demo::SweepResult result = vla_demo::run_sweep(policy.mlp);
};
const Fixture & fixture()
{
  static const Fixture f;
  return f;
}

std::vector<double> ranks(const std::vector<double> & v)
{
  std::vector<std::size_t> order(v.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&v](auto a, auto b) {return v[a] < v[b];});
  std::vector<double> r(v.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    r[order[i]] = static_cast<double>(i);
  }
  return r;
}
}  // namespace

TEST(Harness, PolicyLearnsAndExposesHiddenLayer)
{
  const auto & f = fixture();
  EXPECT_TRUE(std::isfinite(f.policy.final_mse));
  EXPECT_LT(f.policy.final_mse, 0.05f);
  vla_demo::MatrixF hidden;
  const auto action = f.policy.mlp.forward(vla_demo::MatrixF::Zero(4, vla_demo::kInDim), &hidden);
  EXPECT_EQ(action.rows(), 4);
  EXPECT_EQ(action.cols(), vla_demo::kOutDim);
  EXPECT_EQ(hidden.cols(), vla_demo::kHiddenDim);
}

TEST(Harness, CalibrationIsFiniteAndCleanFprSmall)
{
  const auto & r = fixture().result;
  EXPECT_TRUE(std::isfinite(r.threshold));
  EXPECT_GT(r.threshold, 0.0);
  EXPECT_GE(r.clean_fpr, 0.0);
  EXPECT_LE(r.clean_fpr, 0.05);
  EXPECT_GT(r.monitor_fire_fraction, r.clean_fpr);
}

TEST(Harness, OodSignalRisesAndSpreadFallsWithAlpha)
{
  const auto & r = fixture().result;
  EXPECT_GT(r.fired_fraction.back(), r.fired_fraction.front());
  EXPECT_LT(r.ood_score.back(), r.ood_score.front());
  const auto a = ranks(r.alphas);
  const auto f = ranks(r.fired_fraction);
  const double ma = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
  const double mf = std::accumulate(f.begin(), f.end(), 0.0) / f.size();
  double num = 0.0;
  double da = 0.0;
  double df = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - ma) * (f[i] - mf);
    da += (a[i] - ma) * (a[i] - ma);
    df += (f[i] - mf) * (f[i] - mf);
  }
  EXPECT_GT(num / std::sqrt(da * df), 0.7);
}

TEST(Harness, LeadTimeFormulaAndBothEventsInsideSweep)
{
  const auto & r = fixture().result;
  EXPECT_DOUBLE_EQ(r.lead_time, r.output_collapses_at - r.monitor_fires_at);
  EXPECT_TRUE(std::isfinite(r.monitor_fires_at));
  EXPECT_TRUE(std::isfinite(r.output_collapses_at));
  EXPECT_GE(r.monitor_fires_at, 0.0);
  EXPECT_LE(r.output_collapses_at, 1.0);
}

TEST(Harness, HeadlineNonNegativeLeadTime)
{
  const auto & r = fixture().result;
  EXPECT_LE(r.monitor_fires_at, r.output_collapses_at);
  EXPECT_GE(r.lead_time, 0.0);
}

TEST(Harness, PerturbationIdentityAtZeroAndMovesAtOne)
{
  const auto x = vla_demo::make_dataset(10, 0).x;
  EXPECT_EQ(vla_demo::perturb_inputs(x, 0.0, 0), x);
  const auto moved = vla_demo::perturb_inputs(x, 1.0, 0);
  EXPECT_NE(moved, x);
  EXPECT_GT((moved - x).cwiseAbs().mean(), 0.1f);
}

TEST(Harness, FrozenHiddenStateHasZeroSpread)
{
  std::vector<float> frozen(100 * 8, 1.0f);
  for (double s : phm_core::rolling_spread_f32(frozen.data(), 100, 8, 30)) {
    if (!std::isnan(s)) {
      EXPECT_EQ(s, 0.0);
    }
  }
}
