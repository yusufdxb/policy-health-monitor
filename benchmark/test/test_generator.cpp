// Copyright 2026 Yusuf Guenena. MIT License.
// Stream generator properties (ported from tests/test_generator.py).
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "phm_bench/bench.hpp"

using phm_bench::generate_stream;
using phm_bench::rolling_spread_trace;
using phm_bench::StreamSpec;

TEST(Generator, CollapseHasFarLowerSpread)
{
  StreamSpec spec;
  spec.n_id = 300;
  spec.n_ood = 300;
  spec.seed = 1;
  const auto s = generate_stream(spec);
  const double id = rolling_spread_trace(s.id, 20);
  const double ood = rolling_spread_trace(s.ood, 20);
  EXPECT_GT(id, ood);
  EXPECT_LT(ood, 0.1 * id);
}

TEST(Generator, ShapesAndShiftAndDeterminism)
{
  StreamSpec spec;
  spec.dim = 32;
  spec.n_id = 120;
  spec.n_ood = 80;
  spec.ood_mode = "shift";
  spec.seed = 2;
  const auto s = generate_stream(spec);
  EXPECT_EQ(s.id.rows, 120u);
  EXPECT_EQ(s.id.cols, 32u);
  EXPECT_EQ(s.ood.rows, 80u);

  StreamSpec sh;
  sh.dim = 48;
  sh.n_id = 400;
  sh.n_ood = 400;
  sh.ood_mode = "shift";
  sh.seed = 3;
  const auto t = generate_stream(sh);
  double sep = 0.0;
  for (std::size_t c = 0; c < t.id.cols; ++c) {
    double a = 0.0;
    double b = 0.0;
    for (std::size_t r = 0; r < t.id.rows; ++r) {
      a += t.id(r, c) / t.id.rows;
      b += t.ood(r, c) / t.ood.rows;
    }
    sep += (a - b) * (a - b);
  }
  EXPECT_GT(std::sqrt(sep), std::sqrt(rolling_spread_trace(t.id, 20)));

  StreamSpec d;
  d.seed = 7;
  EXPECT_EQ(generate_stream(d).id.data, generate_stream(d).id.data);
  EXPECT_EQ(generate_stream(d).ood.data, generate_stream(d).ood.data);
  StreamSpec bogus;
  bogus.ood_mode = "bogus";
  EXPECT_THROW(generate_stream(bogus), std::invalid_argument);
}
