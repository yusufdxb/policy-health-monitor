// Copyright 2026 Yusuf Guenena. MIT License.
// Rolling-spread calibration (ported from phm_core/tests/test_calibration.py)
// plus bit-exact parity with NumPy on the README example.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/numpy_random.hpp"
#include "phm_core/sim.hpp"

using phm_core::calibrate_threshold;
using phm_core::Matrix;
using phm_core::rolling_spread;

namespace
{
Matrix make(std::size_t r, std::size_t c, const std::vector<double> & v)
{
  Matrix m(r, c);
  m.data = v;
  return m;
}

// T=4, D=2 fixture of test_calibration.py.
const Matrix kHidden = make(4, 2, {0.0, 1.0, 2.0, 1.0, 2.0, 4.0, 5.0, 4.0});
}  // namespace

TEST(RollingSpread, MatchesHandComputation)
{
  // t=2: var([0,2])=1 + var([1,1])=0; t=3: 0 + 2.25; t=4: 2.25 + 0.
  const auto out = rolling_spread(kHidden, 2);
  ASSERT_EQ(out.size(), 4u);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_NEAR(out[1], 1.0, 1e-12);
  EXPECT_NEAR(out[2], 2.25, 1e-12);
  EXPECT_NEAR(out[3], 2.25, 1e-12);
}

TEST(RollingSpread, NanUntilWindowFills)
{
  const auto out = rolling_spread(kHidden, 3);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_FALSE(std::isnan(out[2]));
  EXPECT_FALSE(std::isnan(out[3]));
  // Window longer than the stream: all NaN.
  for (double s : rolling_spread(kHidden, 5)) {
    EXPECT_TRUE(std::isnan(s));
  }
}

TEST(RollingSpread, PopulationVarianceOfSecondFixture)
{
  // test_adapters.py fixture: window 3 gives 2/3 then 8/3.
  const Matrix h = make(4, 2, {0, 0, 1, 0, 2, 0, 3, 3});
  const auto s = rolling_spread(h, 3);
  EXPECT_DOUBLE_EQ(s[2], 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(s[3], 8.0 / 3.0);
}

TEST(CalibrateThreshold, FirstPercentileLinearInterpolation)
{
  // Valid spreads [1.0, 2.25, 2.25]; 1st percentile = 1.0 + 0.02 * 1.25.
  EXPECT_NEAR(calibrate_threshold(rolling_spread(kHidden, 2), 1.0), 1.025, 1e-12);
}

TEST(CalibrateThreshold, IgnoresNan)
{
  const std::vector<double> spreads = {std::nan(""), 1.0, 2.25, 2.25};
  EXPECT_NEAR(calibrate_threshold(spreads, 50.0), 2.25, 1e-12);
  EXPECT_THROW(calibrate_threshold({std::nan("")}, 1.0), std::invalid_argument);
}

TEST(LocoFpr, TwoCorpora)
{
  const Matrix c1 = make(4, 2, {0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1});
  const auto res = phm_core::loco_fpr({kHidden, c1}, 2, 1.0);
  ASSERT_EQ(res.folds.size(), 2u);
  EXPECT_NEAR(res.folds[0].fpr, 0.0, 1e-12);
  EXPECT_NEAR(res.folds[1].fpr, 1.0, 1e-12);
  EXPECT_EQ(res.folds[0].calibrated_on, (std::vector<std::size_t>{1}));
  EXPECT_EQ(res.folds[1].calibrated_on, (std::vector<std::size_t>{0}));
  EXPECT_NEAR(res.folds[1].threshold, 1.025, 1e-12);
  EXPECT_NEAR(res.fpr_mean, 0.5, 1e-12);
  EXPECT_NEAR(res.fpr_max, 1.0, 1e-12);
}

TEST(LocoFpr, MatchesNumpyOnRandomCorpora)
{
  // rng = np.random.default_rng(0); corpora of (40,3), (35,3), (50,3).
  phm_core::numpy_random::Generator rng(0);
  std::vector<Matrix> corpora;
  for (std::size_t rows : {40u, 35u, 50u}) {
    Matrix m(rows, 3);
    rng.normal(0.0, 1.0, m.data.data(), m.data.size());
    corpora.push_back(m);
  }
  const auto res = phm_core::loco_fpr(corpora, 5, 1.0);
  EXPECT_EQ(res.folds[0].threshold, 0x1.edca71f4bbbfep-1);
  EXPECT_EQ(res.folds[1].threshold, 0x1.e4d71f474cafep-1);
  EXPECT_EQ(res.folds[2].threshold, 0x1.23d518af921ffp+0);
  EXPECT_DOUBLE_EQ(res.folds[0].fpr, 0.027777777777777776);
  EXPECT_DOUBLE_EQ(res.folds[1].fpr, 0.0);
  EXPECT_DOUBLE_EQ(res.folds[2].fpr, 0.06521739130434782);
  EXPECT_DOUBLE_EQ(res.fpr_mean, 0.030998389694041867);
  EXPECT_DOUBLE_EQ(res.fpr_max, 0.06521739130434782);
}

// The README example, bit-for-bit against the NumPy implementation: healthy
// stream then a collapse at frame 200, threshold at the 1st percentile.
TEST(ReadmeExample, ReproducesNumpyNumbers)
{
  const auto batch = phm_core::generate_embeddings(64, 200, 1.0, 0.01, 42);
  const double thr = calibrate_threshold(rolling_spread(batch.in_dist, 20), 1.0);
  EXPECT_EQ(thr, 0x1.c6e60a05a04dcp+5);  // 56.86232380290468
  const Matrix stream = phm_core::vstack({&batch.in_dist, &batch.ood});
  const auto spread = rolling_spread(stream, 20);
  EXPECT_EQ(spread[19], 0x1.e23161ace0441p+5);
  EXPECT_EQ(spread[399], 0x1.8d58d638c7cf0p-8);
  int false_alarms = 0;
  int valid = 0;
  int first_alarm = -1;
  for (std::size_t t = 0; t < spread.size(); ++t) {
    if (std::isnan(spread[t])) {
      continue;
    }
    if (t < 200) {
      ++valid;
      false_alarms += spread[t] < thr ? 1 : 0;
    } else if (first_alarm < 0 && spread[t] < thr) {
      first_alarm = static_cast<int>(t);
    }
  }
  EXPECT_EQ(first_alarm, 200);
  EXPECT_EQ(false_alarms, 2);
  EXPECT_EQ(valid, 181);
}

TEST(Numerics, PercentileEdges)
{
  EXPECT_DOUBLE_EQ(phm_core::percentile({3.0}, 1.0), 3.0);
  EXPECT_DOUBLE_EQ(phm_core::percentile({1.0, 2.0, 3.0}, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(phm_core::percentile({1.0, 2.0, 3.0}, 100.0), 3.0);
  EXPECT_DOUBLE_EQ(phm_core::percentile({4.0, 1.0, 3.0, 2.0}, 50.0), 2.5);
  EXPECT_THROW(phm_core::percentile({}, 1.0), std::invalid_argument);
  EXPECT_THROW(phm_core::percentile({1.0}, 101.0), std::invalid_argument);
}
