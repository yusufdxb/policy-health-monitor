// Copyright 2026 Yusuf Guenena. MIT License.
// Each detector separates the failure mode it targets (ported from
// tests/test_detectors.py; coarse AUROC bounds, not pinned numbers).
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "phm_bench/bench.hpp"

using phm_bench::Matrix;

namespace
{
struct Eval
{
  Matrix fid;
  Matrix id;
  Matrix ood;
};

Eval streams(const char * mode, uint64_t seed = 11)
{
  phm_bench::StreamSpec spec;
  spec.n_id = 400;
  spec.n_ood = 400;
  spec.ood_mode = mode;
  spec.seed = seed;
  Eval e;
  e.fid = phm_bench::generate_stream(spec).id;
  spec.seed = seed + 1;
  const auto s = phm_bench::generate_stream(spec);
  e.id = s.id;
  e.ood = s.ood;
  return e;
}

double auroc_of(
  const std::function<std::vector<double>(const Matrix &, const Matrix &)> & f, const Eval & e)
{
  auto s = f(e.fid, e.id);
  const auto o = f(e.fid, e.ood);
  std::vector<int> y(s.size(), 0);
  s.insert(s.end(), o.begin(), o.end());
  y.resize(s.size(), 1);
  return phm_bench::auroc(s, y);
}
}  // namespace

TEST(Detectors, PhmDetectsCollapse)
{
  const auto e = streams("collapse");
  EXPECT_GT(auroc_of([](const Matrix & a, const Matrix & b) {
      return phm_bench::phm_scores(a, b, 20);
    }, e), 0.9);
}

TEST(Detectors, LocationBaselinesDetectShift)
{
  const auto e = streams("shift");
  EXPECT_GT(auroc_of([](const Matrix & a, const Matrix & b) {
      return phm_bench::mahalanobis(a, b);
    }, e), 0.9);
  EXPECT_GT(auroc_of([](const Matrix & a, const Matrix & b) {
      return phm_bench::knn_distance(a, b, 50, false);
    }, e), 0.9);
  // L2 normalization discards the radial magnitude the shift lives in.
  EXPECT_LT(auroc_of([](const Matrix & a, const Matrix & b) {
      return phm_bench::knn_distance(a, b, 50, true);
    }, e), 0.5);
  EXPECT_GT(auroc_of([](const Matrix & a, const Matrix & b) {
      return phm_bench::rnd_closed_form(a, b);
    }, e), 0.7);
}

TEST(Detectors, RndScoresFiniteAndPhmHasNanPrefix)
{
  const auto e = streams("shift");
  const auto s = phm_bench::rnd_closed_form(e.fid, e.ood);
  ASSERT_EQ(s.size(), e.ood.rows);
  for (double v : s) {
    EXPECT_TRUE(std::isfinite(v));
  }
  const auto c = streams("collapse");
  const auto p = phm_bench::phm_scores(c.fid, c.ood, 20);
  ASSERT_EQ(p.size(), c.ood.rows);
  for (std::size_t i = 0; i < p.size(); ++i) {
    EXPECT_EQ(std::isnan(p[i]), i < 19) << i;
  }
}
