// Copyright 2026 Yusuf Guenena. MIT License.
// Metrics against hand-computed fixtures (ported from tests/test_metrics.py).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "phm_bench/bench.hpp"
#include "phm_core/numpy_random.hpp"

using phm_bench::aupr;
using phm_bench::auroc;
using phm_bench::fpr_at_tpr;

TEST(Metrics, AurocSeparationKnownValueAndTies)
{
  EXPECT_EQ(auroc({0.1, 0.2, 0.3, 0.9, 1.0, 1.1}, {0, 0, 0, 1, 1, 1}), 1.0);
  EXPECT_EQ(auroc({0.9, 1.0, 1.1, 0.1, 0.2, 0.3}, {0, 0, 0, 1, 1, 1}), 0.0);
  // OOD=2 beats both ID, OOD=0.5 beats ID=0 only: 3 / 4.
  EXPECT_NEAR(auroc({1.0, 0.0, 2.0, 0.5}, {0, 0, 1, 1}), 0.75, 1e-12);
  // A tie between an ID and an OOD frame counts half: 3.5 / 4.
  EXPECT_NEAR(auroc({1.0, 0.0, 1.0, 2.0}, {0, 0, 1, 1}), 0.875, 1e-12);
}

TEST(Metrics, AuprKnownValues)
{
  EXPECT_NEAR(aupr({0.1, 0.2, 0.9, 1.0}, {0, 0, 1, 1}), 1.0, 1e-12);
  // Ranked [3,2,1,0] / labels [1,0,1,0]: 1.0*0.5 + (2/3)*0.5.
  EXPECT_NEAR(aupr({3.0, 2.0, 1.0, 0.0}, {1, 0, 1, 0}), 0.5 + 1.0 / 3.0, 1e-12);
}

TEST(Metrics, FprAtTpr)
{
  EXPECT_EQ(fpr_at_tpr({0.1, 0.2, 0.3, 0.9, 1.0, 1.1}, {0, 0, 0, 1, 1, 1}, 0.95), 0.0);
  EXPECT_NEAR(
    fpr_at_tpr({4, 3, 2, 1, 3.5, 2.5, 1.5, 0.5}, {1, 1, 1, 1, 0, 0, 0, 0}, 0.75), 0.5, 1e-12);
}

TEST(Metrics, SingleClassIsNanAndNonFiniteScoresAreDropped)
{
  EXPECT_TRUE(std::isnan(auroc({1.0, 2.0, 3.0}, {0, 0, 0})));
  EXPECT_TRUE(std::isnan(aupr({1.0, 2.0, 3.0}, {0, 0, 0})));
  EXPECT_TRUE(std::isnan(fpr_at_tpr({1.0, 2.0, 3.0}, {0, 0, 0})));
  EXPECT_EQ(auroc({std::nan(""), 0.1, 0.9}, {1, 0, 1}), 1.0);
}

TEST(Metrics, BootstrapCiBracketsPointEstimate)
{
  phm_core::numpy_random::Generator rng(0);
  std::vector<double> s;
  std::vector<int> y;
  for (int i = 0; i < 200; ++i) {
    s.push_back(rng.normal(0.0, 1.0));
    y.push_back(0);
  }
  for (int i = 0; i < 200; ++i) {
    s.push_back(rng.normal(2.0, 1.0));
    y.push_back(1);
  }
  const double point = auroc(s, y);
  const auto ci = phm_bench::bootstrap_ci(auroc, s, y, 300);
  EXPECT_LE(ci.lo, point);
  EXPECT_LE(point, ci.hi);
  EXPECT_LT(ci.lo, ci.hi);
}
