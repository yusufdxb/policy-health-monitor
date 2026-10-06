// Copyright 2026 Yusuf Guenena. MIT License.
// OodCore: ports of phm_ood/tests/test_ood_core.py and
// phm_ood_cpp/test/test_ood_core.cpp, plus backend and NumPy parity.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/numpy_random.hpp"
#include "phm_core/ood_core.hpp"
#include "phm_core/spread_backend.hpp"

using phm_core::OodConfig;
using phm_core::OodCore;
using phm_core::VerdictData;

namespace
{

constexpr std::size_t kDim = 16;
constexpr std::size_t kWindow = 5;
using Frames = std::vector<std::vector<float>>;

// Fixtures of test_ood_core.py from np.random.default_rng(42): a unit-normal
// stream and a near-constant (collapsed) stream around 0.5.
struct Fixtures
{
  Frames in_dist;
  Frames collapsed;
  phm_core::numpy_random::Generator rng{42};
  Fixtures()
  {
    in_dist.assign(200, std::vector<float>(kDim));
    collapsed.assign(200, std::vector<float>(kDim));
    for (auto & row : in_dist) {
      for (auto & x : row) {
        x = static_cast<float>(rng.normal(0.0, 1.0));
      }
    }
    for (auto & row : collapsed) {
      for (auto & x : row) {
        x = static_cast<float>(0.5 + rng.normal(0.0, 1e-6));
      }
    }
  }
  std::vector<float> healthy(double scale)
  {
    std::vector<float> v(kDim);
    for (auto & x : v) {
      x = static_cast<float>(rng.normal(0.0, scale));
    }
    return v;
  }
};

OodCore make_core(
  double threshold = 0.05, int min_consecutive = 3, int compute_every = 1,
  std::size_t embed_dim = 0, std::size_t window = kWindow)
{
  OodConfig c;
  c.window = window;
  c.threshold = threshold;
  c.min_consecutive = min_consecutive;
  c.compute_every = compute_every;
  c.embed_dim = embed_dim;
  return OodCore(c, phm_core::make_plain_backend());
}

}  // namespace

TEST(OodCore, InDistributionStreamStaysOk)
{
  Fixtures f;
  auto core = make_core();
  for (const auto & v : f.in_dist) {
    EXPECT_FALSE(core.update(v, "").violating);
  }
}

TEST(OodCore, CollapsedStreamGoesViolatingAfterHysteresis)
{
  Fixtures f;
  auto core = make_core(0.05, 3);
  VerdictData last;
  int violating_after_warmup = 0;
  for (std::size_t i = 0; i < f.collapsed.size(); ++i) {
    last = core.update(f.collapsed[i], "");
    if (i >= kWindow + 3 && last.violating) {
      ++violating_after_warmup;
    }
  }
  EXPECT_GT(violating_after_warmup, 0);
  EXPECT_TRUE(last.violating) << last.reason;
}

TEST(OodCore, WarmupReturnsOk)
{
  Fixtures f;
  auto core = make_core();
  for (std::size_t i = 0; i + 1 < kWindow; ++i) {
    const auto & v = core.update(f.collapsed[i], "");
    EXPECT_FALSE(v.violating);
    EXPECT_EQ(v.score, 0.0);
    EXPECT_EQ(v.reason, "warming up: " + std::to_string(i + 1) + "/5 frames");
  }
}

TEST(OodCore, PreHysteresisThenFire)
{
  // The last warm-up frame fills the window and is the first computed frame
  // (count 1); the next is count 2 (still pre-hysteresis), then count 3 fires.
  Fixtures f;
  auto core = make_core(0.05, 3);
  for (std::size_t i = 0; i < kWindow; ++i) {
    core.update(f.collapsed[i], "");
  }
  const VerdictData v1 = core.update(f.collapsed[kWindow], "");
  const VerdictData v2 = core.update(f.collapsed[kWindow + 1], "");
  EXPECT_FALSE(v1.violating) << v1.reason;
  EXPECT_NE(v1.reason.find("(pre-hysteresis)"), std::string::npos);
  EXPECT_TRUE(v2.violating) << v2.reason;
}

TEST(OodCore, RecoveryAfterViolatingStream)
{
  Fixtures f;
  auto core = make_core(0.05, 2);
  for (std::size_t i = 0; i < kWindow + 5; ++i) {
    core.update(f.collapsed[i], "");
  }
  EXPECT_TRUE(core.update(f.collapsed[kWindow + 5], "").violating);
  // One healthy vector fed `window` times, as the Python test does: the first
  // mixed window resets the hysteresis run, and the final all-identical window
  // is only the first below-threshold frame of a new run.
  const auto healthy = f.healthy(2.0);
  VerdictData v;
  for (std::size_t i = 0; i < kWindow; ++i) {
    v = core.update(healthy, "");
  }
  EXPECT_FALSE(v.violating) << v.reason;
}

TEST(OodCore, FrequencyGateReplaysLastComputedVerdict)
{
  Fixtures f;
  auto core = make_core(0.05, 3, 2);
  for (std::size_t i = 0; i < kWindow; ++i) {
    core.update(f.in_dist[i], "");
  }
  const VerdictData even = core.update(f.in_dist[kWindow], "");
  const double spread_even = core.last_spread();
  const VerdictData odd = core.update(f.in_dist[kWindow + 1], "");
  EXPECT_EQ(odd.reason, even.reason);
  EXPECT_EQ(core.last_spread(), spread_even);  // gated frame computed nothing
}

TEST(OodCore, CalibrateFromDataSetsThreshold)
{
  Fixtures f;
  OodConfig c;
  c.window = kWindow;
  OodCore core(c, phm_core::make_plain_backend());
  phm_core::Matrix m(f.in_dist.size(), kDim);
  for (std::size_t r = 0; r < f.in_dist.size(); ++r) {
    for (std::size_t d = 0; d < kDim; ++d) {
      m(r, d) = f.in_dist[r][d];
    }
  }
  const double thr = core.calibrate_from_data(m, 1.0);
  EXPECT_GT(thr, 0.0);
  EXPECT_EQ(core.threshold(), thr);
  int violating = 0;
  int n = 0;
  for (std::size_t i = kWindow; i < f.in_dist.size(); ++i, ++n) {
    violating += core.update(f.in_dist[i], "").violating ? 1 : 0;
  }
  EXPECT_LT(static_cast<double>(violating) / n, 0.05);
}

TEST(OodCore, EmbedDimMismatchReturnsBadInputVerdict)
{
  auto core = make_core(0.05, 3, 1, kDim);
  const std::vector<float> wrong(kDim + 5, 1.0f);
  const VerdictData v = core.update(wrong, "");
  EXPECT_EQ(v.reason, "dim mismatch: expected 16, got 21");
  EXPECT_FALSE(v.violating);
  EXPECT_EQ(v.score, phm_core::BAD_INPUT_SCORE);
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kDimMismatch);
}

TEST(OodCore, DimensionChangeWithoutEmbedDimIsBadInputNotCrash)
{
  auto core = make_core(0.5, 1, 1, 0, 3);
  core.update(std::vector<float>{1.0f, 2.0f}, "p");
  const VerdictData v = core.update(std::vector<float>{1.0f, 2.0f, 3.0f}, "p");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kDimMismatch);
  EXPECT_EQ(v.reason, "dim mismatch: expected 2, got 3");
  // The bad frame did not enter the window: two more good frames fill it.
  core.update(std::vector<float>{1.0f, 2.0f}, "p");
  const VerdictData full = core.update(std::vector<float>{1.0f, 2.0f}, "p");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kOk);
  EXPECT_TRUE(full.violating);  // constant frames: spread 0 < 0.5
}

TEST(OodCore, ScoreAtZeroSpreadIsOneAndAboveThresholdIsZero)
{
  Fixtures f;
  auto core = make_core(0.05, 1);
  const std::vector<float> zero(kDim, 0.0f);
  VerdictData v;
  for (std::size_t i = 0; i <= kWindow; ++i) {
    v = core.update(zero, "");
  }
  EXPECT_NEAR(v.score, 1.0, 1e-12);
  EXPECT_EQ(v.suggested_action, phm_core::ACTION_STOP_AND_HOLD);
  auto core2 = make_core(0.05, 1);
  for (std::size_t i = 0; i < kWindow + 5; ++i) {
    v = core2.update(f.in_dist[i], "");
  }
  EXPECT_EQ(v.score, 0.0);
}

TEST(OodCore, SourceIsConfigurable)
{
  Fixtures f;
  auto core = make_core();
  for (std::size_t i = 0; i < 30; ++i) {
    EXPECT_EQ(core.update(i < 10 ? f.in_dist[i] : f.collapsed[i], "").source, "phm_ood");
  }
  OodConfig c;
  c.window = 3;
  c.source = "phm_ood_cpp";
  OodCore cpp(c, phm_core::make_plain_backend());
  EXPECT_EQ(cpp.update(f.in_dist[0], "").source, "phm_ood_cpp");
}

TEST(OodCore, ReasonFormatMatchesPythonVerdicts)
{
  Fixtures f;
  auto core = make_core(0.05, 1);
  VerdictData v;
  for (std::size_t i = 0; i <= kWindow; ++i) {
    v = core.update(f.collapsed[i], "phoenix");
  }
  EXPECT_EQ(v.reason, "[phoenix] ood: rolling-spread 0.0000 < thr 0.0500 for 2 frame(s) [stop]");
  auto core2 = make_core(0.05, 1);
  for (std::size_t i = 0; i < kWindow; ++i) {
    v = core2.update(f.in_dist[i], "");
  }
  EXPECT_EQ(v.reason.rfind("ood: rolling-spread ", 0), 0u);
  EXPECT_NE(v.reason.find(" >= thr 0.0500"), std::string::npos);
}

TEST(OodCore, ConstructorRejectsInvalidParameters)
{
  EXPECT_THROW(make_core(0.05, 3, 1, 0, 1), std::invalid_argument);
  EXPECT_THROW(make_core(0.05, 0), std::invalid_argument);
  EXPECT_THROW(make_core(0.05, 3, 0), std::invalid_argument);
  EXPECT_THROW(OodCore(OodConfig{}, nullptr), std::invalid_argument);
  // window * dim beyond the buffer limit, including sizes whose product would
  // overflow size_t, is refused up front.
  EXPECT_THROW(
    make_core(0.05, 3, 1, 0, std::size_t{1} << 62), std::invalid_argument);
  EXPECT_THROW(
    make_core(0.05, 3, 1, phm_core::kMaxWindowFloats, 2), std::invalid_argument);
}

TEST(OodCore, OversizedFirstFrameIsBadInputAndDoesNotPin)
{
  const std::size_t window = std::size_t{1} << 20;
  auto core = make_core(0.05, 3, 1, 0, window);
  const std::vector<float> big(phm_core::kMaxWindowFloats / window + 1, 1.0f);
  const VerdictData v = core.update(big, "");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kDimMismatch);
  EXPECT_EQ(v.score, phm_core::BAD_INPUT_SCORE);
  EXPECT_FALSE(v.violating);
  EXPECT_NE(v.reason.find("embedding too large"), std::string::npos);
  const VerdictData ok = core.update(std::vector<float>{1.0f, 2.0f}, "");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kOk);
  EXPECT_EQ(ok.reason, "warming up: 1/" + std::to_string(window) + " frames");
}

TEST(OodCore, EmptyFirstFrameDoesNotPinTheDimension)
{
  // An empty frame used to pin the dimension to 0, after which every real
  // frame was rejected as a dim mismatch until reset. The first non-empty
  // frame now pins the dimension and restarts the window.
  auto core = make_core(0.5, 1, 1, 0, 3);
  EXPECT_EQ(core.update(nullptr, 0, "").reason, "warming up: 1/3 frames");
  const std::vector<float> e{1.0f, 2.0f};
  EXPECT_EQ(core.update(e, "").reason, "warming up: 1/3 frames");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kOk);
  EXPECT_EQ(core.update(e, "").reason, "warming up: 2/3 frames");
  const VerdictData v = core.update(e, "");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kOk);
  EXPECT_TRUE(v.violating);  // three identical frames: zero spread
  EXPECT_EQ(core.last_spread(), 0.0);
  // Once pinned, an empty frame is a dim mismatch.
  EXPECT_EQ(core.update(nullptr, 0, "").reason, "dim mismatch: expected 2, got 0");
}

// Severity floor and action banding (LOCKED decisions 2 and 3): with
// threshold 1, spread = 1 - target score.
TEST(OodCore, SeverityFloorAndActionBanding)
{
  auto core = make_core(1.0, 1);
  const auto at = [&core](double score) {
      VerdictData v;
      core.make_verdict(1.0 * (1.0 - score), true, true, "", v);
      return v;
    };
  const VerdictData floor = at(0.10);
  EXPECT_FALSE(floor.violating);
  EXPECT_EQ(floor.suggested_action, phm_core::ACTION_NONE);
  EXPECT_NE(floor.reason.find("(below severity floor)"), std::string::npos);
  EXPECT_TRUE(at(0.35).violating);
  EXPECT_EQ(at(0.35).suggested_action, phm_core::ACTION_LOG_ONLY);
  EXPECT_EQ(at(0.25).suggested_action, phm_core::ACTION_LOG_ONLY);
  EXPECT_EQ(at(0.65).suggested_action, phm_core::ACTION_HOLD);
  EXPECT_EQ(at(0.90).suggested_action, phm_core::ACTION_STOP_AND_HOLD);
}

// A window containing NaN or Inf must not come back healthy (fail closed).
TEST(OodCore, NonFiniteEmbeddingDoesNotReportHealthy)
{
  Fixtures f;
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
  {
    auto core = make_core();
    for (std::size_t i = 0; i + 1 < kWindow; ++i) {
      core.update(f.in_dist[0], "");
    }
    auto frame = f.in_dist[1];
    frame[3] = bad;
    const VerdictData v = core.update(frame, "");
    EXPECT_EQ(v.score, phm_core::BAD_INPUT_SCORE);
    EXPECT_EQ(v.reason, "non-finite spread: embedding contains NaN or Inf");
    EXPECT_EQ(v.suggested_action, phm_core::ACTION_NONE);
    EXPECT_FALSE(v.violating);
  }
}

TEST(OodCore, NonFiniteVerdictPersistsThroughFrequencyGate)
{
  Fixtures f;
  auto core = make_core(0.05, 3, 2, kDim);
  for (std::size_t i = 0; i < kWindow * 2; ++i) {
    core.update(f.in_dist[i], "");
  }
  auto bad = f.in_dist[0];
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  std::vector<bool> degraded;
  for (int i = 0; i < 6; ++i) {
    degraded.push_back(core.update(bad, "").reason.find("non-finite") != std::string::npos);
  }
  std::size_t first = 0;
  while (first < degraded.size() && !degraded[first]) {
    ++first;
  }
  ASSERT_LT(first, 2u);
  for (std::size_t i = first; i < degraded.size(); ++i) {
    EXPECT_TRUE(degraded[i]);
  }
}

TEST(OodCore, ResetClearsRollingStateAndPinnedDim)
{
  auto core = make_core(0.5, 1, 1, 0, 3);
  const std::vector<float> e{1.0f, 2.0f};
  core.update(e, "p");
  core.update(e, "p");
  EXPECT_TRUE(core.update(e, "p").violating);
  core.reset();
  EXPECT_EQ(core.window(), 3u);
  const VerdictData after = core.update(std::vector<float>{1.0f, 2.0f, 3.0f}, "p");
  EXPECT_EQ(core.last_status(), phm_core::FrameStatus::kOk);
  EXPECT_EQ(after.reason, "warming up: 1/3 frames");
}

TEST(OodCore, EmptyEmbeddingsHaveZeroSpreadAsInNumpy)
{
  // np.var over a (W, 0) block sums to 0.0, so an all-empty stream reads as
  // a full collapse, exactly as the NumPy core scored it.
  auto core = make_core(0.5, 1, 1, 0, 2);
  core.update(nullptr, 0, "");
  const VerdictData v = core.update(nullptr, 0, "");
  EXPECT_TRUE(v.violating);
  EXPECT_EQ(core.last_spread(), 0.0);
}

TEST(SpreadBackends, PlainMatchesNumpyOrderAndEigenAgrees)
{
  phm_core::numpy_random::Generator rng(5);
  const std::size_t w = 30;
  const std::size_t d = 384;
  std::vector<float> ring(w * d);
  for (auto & x : ring) {
    x = static_cast<float>(rng.normal(0.3, 2.0));
  }
  phm_core::Matrix m(w, d);
  for (std::size_t i = 0; i < ring.size(); ++i) {
    m.data[i] = ring[i];
  }
  const double reference = phm_core::block_spread(m);
  auto plain = phm_core::make_plain_backend();
  EXPECT_EQ(plain->spread(phm_core::WindowView{ring.data(), w, d, 0}), reference);
  for (const auto & name : phm_core::available_backends()) {
    auto backend = phm_core::make_backend(name);
    ASSERT_TRUE(backend) << name;
    EXPECT_NEAR(backend->spread(phm_core::WindowView{ring.data(), w, d, 0}), reference,
      1e-12 * reference) << name;
  }
  EXPECT_EQ(phm_core::make_backend("no-such-backend"), nullptr);
}

TEST(SpreadBackends, RingOrderMatchesChronologicalWindow)
{
  // Feed 7 frames through a window of 4: the core's ring wraps, and its spread
  // must equal block_spread of the last 4 frames in time order.
  phm_core::numpy_random::Generator rng(9);
  auto core = make_core(1e-9, 1, 1, 0, 4);
  phm_core::Matrix frames(7, 6);
  for (std::size_t r = 0; r < 7; ++r) {
    std::vector<float> f(6);
    for (std::size_t c = 0; c < 6; ++c) {
      f[c] = static_cast<float>(rng.normal(0.0, 1.0));
      frames(r, c) = f[c];
    }
    core.update(f, "");
  }
  EXPECT_EQ(core.last_spread(), phm_core::block_spread(frames.slice_rows(3, 7)));
}
