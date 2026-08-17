// Copyright 2026 Yusuf Guenena. MIT License.
// gtest for the C++ OOD core. Pins the rolling-spread math against values
// hand-computed the same way phm_core/phm_core/calibration.py:32-35 does
// (sum over dims of population/ddof=0 variance), and checks the verdict +
// hysteresis behavior against phm_ood/phm_ood/_core.py.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "phm_ood_cpp/backend.hpp"
#include "phm_ood_cpp/ood_core.hpp"
#include "phm_ood_cpp/severity.hpp"

using phm_ood_cpp::Backend;
using phm_ood_cpp::Hysteresis;
using phm_ood_cpp::OodCore;
using phm_ood_cpp::VerdictData;
using phm_ood_cpp::make_plain_backend;

namespace
{

double spread_via_backend(
  const std::unique_ptr<Backend> & b,
  const std::vector<std::vector<float>> & frames)
{
  const std::size_t window = frames.size();
  const std::size_t dim = frames[0].size();
  std::vector<double> flat(window * dim);
  for (std::size_t f = 0; f < window; ++f) {
    for (std::size_t d = 0; d < dim; ++d) {
      flat[f * dim + d] = static_cast<double>(frames[f][d]);
    }
  }
  return b->rolling_spread(flat, window, dim);
}

}  // namespace

// Constant block -> zero variance in every dim -> spread 0
// (matches test_calibration.py constant-is-zero check).
TEST(RollingSpread, ConstantBlockIsZero)
{
  auto b = make_plain_backend();
  std::vector<std::vector<float>> frames(10, std::vector<float>{1.0f, 2.0f, 3.0f});
  EXPECT_NEAR(spread_via_backend(b, frames), 0.0, 1e-12);
}

// Hand-computed: 2 frames, 2 dims.
// dim0 values {0,2}: mean 1, var = ((0-1)^2+(2-1)^2)/2 = 1
// dim1 values {0,4}: mean 2, var = (4+4)/2 = 4
// sum = 5. (population variance, ddof=0, == numpy.var default.)
TEST(RollingSpread, HandComputedTwoByTwo)
{
  auto b = make_plain_backend();
  std::vector<std::vector<float>> frames = {{0.0f, 0.0f}, {2.0f, 4.0f}};
  EXPECT_NEAR(spread_via_backend(b, frames), 5.0, 1e-12);
}

// Eigen backend, when compiled in, must agree with plain to ~1e-9.
#ifdef PHM_HAVE_EIGEN
TEST(RollingSpread, EigenMatchesPlain)
{
  auto plain = make_plain_backend();
  auto eigen = phm_ood_cpp::make_eigen_backend();
  std::vector<std::vector<float>> frames = {
    {1.0f, -2.0f, 3.5f}, {0.5f, 2.0f, -1.0f}, {-3.0f, 4.0f, 0.0f}, {2.0f, 1.0f, 1.0f}};
  std::vector<double> flat;
  for (auto & r : frames) {for (float x : r) {flat.push_back(x);}}
  const double sp = plain->rolling_spread(flat, 4, 3);
  const double se = eigen->rolling_spread(flat, 4, 3);
  EXPECT_NEAR(sp, se, 1e-9);
}
#endif

// Hysteresis: fire only after min_consecutive in a row, reset on healthy.
TEST(HysteresisTest, FiresAfterRunAndResets)
{
  Hysteresis h(2);
  EXPECT_FALSE(h.observe(true));   // 1
  EXPECT_TRUE(h.observe(true));    // 2 -> fire
  EXPECT_TRUE(h.observe(true));    // 3 -> still firing
  EXPECT_FALSE(h.observe(false));  // healthy resets
  EXPECT_FALSE(h.observe(true));   // 1 again, not yet
}

TEST(HysteresisTest, RejectsZeroMinConsecutive)
{
  EXPECT_THROW(Hysteresis(0), std::invalid_argument);
}

// OodCore warms up until the buffer fills, then returns a real verdict.
TEST(OodCoreTest, WarmsUpThenComputes)
{
  OodCore core(3, /*threshold=*/0.5, /*min_consecutive=*/1,
               /*compute_every=*/1, make_plain_backend());
  std::vector<float> e1{1.0f, 2.0f};
  auto v1 = core.update(e1, "p");
  EXPECT_FALSE(v1.violating);
  EXPECT_NE(v1.reason.find("warming up"), std::string::npos);
  core.update(e1, "p");  // 2/3
  auto v3 = core.update(e1, "p");  // 3/3, full
  // Constant frames -> spread 0 < threshold -> OOD; threshold>0 so score=1.0,
  // min_consecutive=1 fires immediately, score 1.0 -> STOP_AND_HOLD.
  EXPECT_TRUE(v3.violating);
  EXPECT_NEAR(v3.score, 1.0, 1e-9);
  EXPECT_EQ(v3.suggested_action, phm_ood_cpp::ACTION_STOP_AND_HOLD);
  EXPECT_EQ(v3.source, std::string("phm_ood_cpp"));
}

// In-distribution (spread >= threshold) -> non-violating, score 0, action NONE.
TEST(OodCoreTest, InDistributionIsHealthy)
{
  OodCore core(2, /*threshold=*/0.001, /*min_consecutive=*/1,
               /*compute_every=*/1, make_plain_backend());
  // Two distinct frames -> nonzero spread well above the tiny threshold.
  core.update(std::vector<float>{0.0f, 0.0f}, "p");
  auto v = core.update(std::vector<float>{2.0f, 4.0f}, "p");  // spread 5.0 >= 0.001
  EXPECT_FALSE(v.violating);
  EXPECT_NEAR(v.score, 0.0, 1e-12);
  EXPECT_EQ(v.suggested_action, phm_ood_cpp::ACTION_NONE);
}

// reset() drops the rolling window, the hysteresis run, and the pinned
// dimension, so a re-activated node warms up again instead of blending a stale
// context into the new one. Config (window/threshold) is preserved.
TEST(OodCoreTest, ResetClearsRollingState)
{
  OodCore core(3, /*threshold=*/0.5, /*min_consecutive=*/1,
               /*compute_every=*/1, make_plain_backend());
  const std::vector<float> e{1.0f, 2.0f};
  core.update(e, "p");
  core.update(e, "p");
  const auto full = core.update(e, "p");  // buffer full -> real verdict
  EXPECT_TRUE(full.violating);

  // Without a reset, changing the embedding dimension is a hard error.
  EXPECT_THROW(core.update(std::vector<float>{1.0f, 2.0f, 3.0f}, "p"), std::invalid_argument);

  core.reset();
  EXPECT_EQ(core.window(), static_cast<std::size_t>(3));  // config preserved

  // Post-reset the buffer is empty again, so the next frame must warm up, and
  // the pinned dimension is cleared so a new dimension is accepted.
  const auto after = core.update(std::vector<float>{1.0f, 2.0f, 3.0f}, "p");
  EXPECT_FALSE(after.violating);
  EXPECT_NE(after.reason.find("warming up: 1/3"), std::string::npos);
}

// A window containing NaN/Inf must not come back healthy.
//
// Regression: the threshold test is `spread < threshold_`, and every
// comparison against NaN is false, so a corrupt embedding previously produced
// a normal healthy verdict with score 0.0. That is a fail-OPEN in a safety
// monitor, on exactly the input class a broken upstream policy emits.
// Mirrors test_ood_core.py::test_non_finite_embedding_does_not_report_healthy.
TEST(NonFiniteInput, DoesNotReportHealthy)
{
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()})
  {
    OodCore core(3, /*threshold=*/0.5, /*min_consecutive=*/1,
      /*compute_every=*/1, make_plain_backend());
    core.update(std::vector<float>{1.0f, 2.0f, 3.0f}, "p");
    core.update(std::vector<float>{4.0f, 0.0f, 1.0f}, "p");
    const auto v = core.update(std::vector<float>{bad, 1.0f, 2.0f}, "p");

    EXPECT_DOUBLE_EQ(v.score, phm_ood_cpp::BAD_INPUT_SCORE);
    EXPECT_FALSE(v.violating);
    EXPECT_EQ(v.suggested_action, phm_ood_cpp::ACTION_NONE);
    EXPECT_NE(v.reason.find("non-finite"), std::string::npos);
  }
}

// Once a corrupt window is computed the degraded verdict must stick, including
// across frames skipped by the frequency gate. Without caching, a corrupt
// stream would alternate between degraded and a stale healthy verdict.
TEST(NonFiniteInput, PersistsThroughFrequencyGate)
{
  OodCore core(3, /*threshold=*/0.5, /*min_consecutive=*/1,
    /*compute_every=*/2, make_plain_backend());
  for (int i = 0; i < 6; ++i) {
    core.update(std::vector<float>{1.0f * i, 2.0f, 3.0f}, "p");
  }
  const std::vector<float> bad{std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f};

  bool seen_degraded = false;
  for (int i = 0; i < 6; ++i) {
    const auto v = core.update(bad, "p");
    const bool degraded = v.reason.find("non-finite") != std::string::npos;
    if (degraded) {seen_degraded = true;}
    // Never revert to healthy once the fault has been observed.
    if (seen_degraded) {EXPECT_TRUE(degraded);}
  }
  EXPECT_TRUE(seen_degraded);
}

// The guard must not perturb any verdict on well-formed input.
TEST(NonFiniteInput, FiniteInputUnaffected)
{
  OodCore core(2, /*threshold=*/0.001, /*min_consecutive=*/1,
    /*compute_every=*/1, make_plain_backend());
  core.update(std::vector<float>{0.0f, 0.0f}, "p");
  const auto v = core.update(std::vector<float>{2.0f, 4.0f}, "p");
  EXPECT_EQ(v.reason.find("non-finite"), std::string::npos);
  EXPECT_NE(v.score, phm_ood_cpp::BAD_INPUT_SCORE);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
