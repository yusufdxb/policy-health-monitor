// Copyright 2026 Yusuf Guenena. MIT License.
// Health detectors (ported from phm_detectors/tests/test_adapters.py).
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "phm_core/adapters.hpp"
#include "phm_core/calibration.hpp"
#include "phm_core/numpy_random.hpp"

using phm_core::DeadTopicAdapter;
using phm_core::DeadTopicSample;
using phm_core::FrequencyDropAdapter;
using phm_core::FrequencySample;
using phm_core::RecurrentSpreadSample;
using phm_core::RecurrentTemporalSpreadAdapter;
using phm_core::StaticThresholdAdapter;
using phm_core::ThresholdSample;

namespace
{
constexpr char kTopic[] = "/scan";
constexpr double kBaseline = 10.0;

void feed_learning(FrequencyDropAdapter & a, double hz = kBaseline)
{
  for (int i = 0; i < phm_core::kFrequencyLearningSamples; ++i) {
    a.update({kTopic, hz});
  }
}
}  // namespace

TEST(FrequencyDrop, LearningPhaseAndWrongTopicReturnNothing)
{
  FrequencyDropAdapter a(kTopic, 20.0, 2);
  for (int i = 0; i < phm_core::kFrequencyLearningSamples - 1; ++i) {
    EXPECT_FALSE(a.update({kTopic, kBaseline}).has_value());
  }
  EXPECT_FALSE(a.update({"/odom", 5.0}).has_value());
}

TEST(FrequencyDrop, HealthyAfterLearning)
{
  FrequencyDropAdapter a(kTopic, 20.0, 2);
  feed_learning(a);
  const auto v = a.update({kTopic, kBaseline});
  ASSERT_TRUE(v);
  EXPECT_FALSE(v->violating);
  EXPECT_EQ(v->source, "freq:/scan");
  EXPECT_EQ(v->reason, "freq:/scan 10.00 Hz healthy (baseline 10.00 Hz)");
}

TEST(FrequencyDrop, ConsecutiveDropsViolate)
{
  FrequencyDropAdapter a(kTopic, 20.0, 2);
  feed_learning(a);
  const auto v1 = a.update({kTopic, 3.0});
  const auto v2 = a.update({kTopic, 3.0});
  EXPECT_FALSE(v1->violating);
  EXPECT_TRUE(v2->violating);
  EXPECT_GT(v2->score, 0.0);
  EXPECT_EQ(
    v2->reason,
    "freq:/scan 3.00 Hz below floor 8.00 Hz (baseline 10.00 Hz, tolerance 20.0%)");
}

TEST(FrequencyDrop, ScoreAndActionBands)
{
  FrequencyDropAdapter a(kTopic, 20.0, 1);
  feed_learning(a);
  EXPECT_NEAR(a.update({kTopic, 0.0})->score, 1.0, 1e-12);
  FrequencyDropAdapter b(kTopic, 20.0, 1);
  feed_learning(b);
  EXPECT_EQ(b.update({kTopic, 5.0})->suggested_action, phm_core::ACTION_HOLD);
  // Over-rate stays healthy at score 0.
  EXPECT_EQ(b.update({kTopic, 50.0})->score, 0.0);
}

TEST(FrequencyDrop, RecoveryResetsHysteresis)
{
  FrequencyDropAdapter a(kTopic, 20.0, 2);
  feed_learning(a);
  a.update({kTopic, 1.0});
  a.update({kTopic, 1.0});
  EXPECT_FALSE(a.update({kTopic, kBaseline})->violating);
  EXPECT_FALSE(a.update({kTopic, 1.0})->violating);
}

TEST(StaticThreshold, BreachHysteresisScoreAndActions)
{
  StaticThresholdAdapter a("system:cpu", "cpu_percent", 2);
  EXPECT_FALSE(a.update({"memory_percent", 95.0, 85.0}).has_value());
  const auto ok = a.update({"cpu_percent", 50.0, 80.0});
  EXPECT_FALSE(ok->violating);
  EXPECT_EQ(ok->source, "threshold:cpu_percent");
  EXPECT_EQ(ok->reason, "threshold:cpu_percent 50.00 within limit 80.00");
  const auto b1 = a.update({"cpu_percent", 95.0, 80.0});
  const auto b2 = a.update({"cpu_percent", 95.0, 80.0});
  EXPECT_FALSE(b1->violating);
  EXPECT_TRUE(b2->violating);
  EXPECT_EQ(b2->reason, "threshold:cpu_percent 95.00 exceeds limit 80.00");
  EXPECT_FALSE(a.update({"cpu_percent", 30.0, 80.0})->violating);
  EXPECT_FALSE(a.update({"cpu_percent", 95.0, 80.0})->violating);

  StaticThresholdAdapter c("system:cpu", "cpu_percent", 1);
  EXPECT_NEAR(c.update({"cpu_percent", 160.0, 80.0})->score, 1.0, 1e-12);
  const auto at = c.update({"cpu_percent", 80.0, 80.0});
  EXPECT_NEAR(at->score, 0.5, 1e-12);
  EXPECT_EQ(at->suggested_action, phm_core::ACTION_HOLD);
  EXPECT_EQ(c.update({"cpu_percent", 20.0, 80.0})->suggested_action, phm_core::ACTION_NONE);
}

TEST(DeadTopic, SilenceBoundariesAndRecovery)
{
  DeadTopicAdapter a("/imu/data", 5.0);
  EXPECT_FALSE(a.update({"/odom", 0.0, 10.0}).has_value());
  const auto alive = a.update({"/imu/data", 100.0, 101.0});
  EXPECT_FALSE(alive->violating);
  EXPECT_EQ(alive->source, "dead:/imu/data");
  EXPECT_EQ(alive->reason, "dead:/imu/data alive (last seen 1.00s ago)");
  const auto dead = a.update({"/imu/data", 0.0, 10.0});
  EXPECT_TRUE(dead->violating);
  EXPECT_EQ(dead->suggested_action, phm_core::ACTION_STOP_AND_HOLD);
  EXPECT_EQ(dead->reason, "dead:/imu/data silent 10.0s (timeout 5.0s)");
  EXPECT_NEAR(dead->score, 1.0, 1e-12);
  EXPECT_TRUE(a.alerted());
  a.mark_alive(10.0);
  EXPECT_FALSE(a.alerted());
  EXPECT_FALSE(a.update({"/imu/data", 10.0, 10.5})->violating);
  // elapsed == timeout is not dead (strict >); just over is.
  EXPECT_FALSE(a.update({"/imu/data", 95.0, 100.0})->violating);
  EXPECT_TRUE(a.update({"/imu/data", 94.9, 100.0})->violating);
}

TEST(Integration, HealthyFaultsAndRecovery)
{
  FrequencyDropAdapter freq(kTopic, 20.0, 2);
  StaticThresholdAdapter thresh("system:cpu", "cpu_percent", 2);
  DeadTopicAdapter dead("/imu/data", 5.0);
  feed_learning(freq);
  double now = 1000.0;
  for (int i = 0; i < 5; ++i, now += 1.0) {
    EXPECT_FALSE(freq.update({kTopic, kBaseline})->violating);
    EXPECT_FALSE(thresh.update({"cpu_percent", 40.0, 80.0})->violating);
    EXPECT_FALSE(dead.update({"/imu/data", now - 0.5, now})->violating);
  }
  for (int i = 0; i < 2; ++i) {
    freq.update({kTopic, 1.0});
    thresh.update({"cpu_percent", 95.0, 80.0});
  }
  EXPECT_TRUE(freq.update({kTopic, 1.0})->violating);
  EXPECT_TRUE(thresh.update({"cpu_percent", 95.0, 80.0})->violating);
  EXPECT_TRUE(dead.update({"/imu/data", 0.0, 200.0})->violating);
  dead.mark_alive(300.0);
  EXPECT_FALSE(freq.update({kTopic, kBaseline})->violating);
  EXPECT_FALSE(thresh.update({"cpu_percent", 30.0, 80.0})->violating);
  EXPECT_FALSE(dead.update({"/imu/data", 300.05, 300.1})->violating);
}

namespace
{
constexpr char kEmb[] = "/policy/embedding";
RecurrentSpreadSample emb(std::vector<double> v) {return {kEmb, std::move(v)};}
}  // namespace

TEST(RecurrentSpread, NameTopicAndWarmup)
{
  RecurrentTemporalSpreadAdapter a(kEmb, 3, 1.0, 2);
  EXPECT_EQ(a.name(), "recurrent_temporal_spread:/policy/embedding");
  EXPECT_FALSE(a.update({"/other", {0.0, 0.0}}).has_value());
  for (int i = 0; i < 2; ++i) {
    const auto v = a.update(emb({static_cast<double>(i), 0.0}));
    EXPECT_FALSE(v->violating);
    EXPECT_EQ(v->score, 0.0);
    EXPECT_NE(v->reason.find("warming up"), std::string::npos);
  }
  EXPECT_THROW(RecurrentTemporalSpreadAdapter(kEmb, 1), std::invalid_argument);
}

TEST(RecurrentSpread, LastSpreadMatchesRollingSpread)
{
  phm_core::numpy_random::Generator rng(7);
  phm_core::Matrix frames(8, 5);
  rng.normal(0.0, 1.0, frames.data.data(), frames.data.size());
  RecurrentTemporalSpreadAdapter a(kEmb, 8, 0.0);
  for (std::size_t r = 0; r < 8; ++r) {
    a.update(emb(std::vector<double>(frames.row(r), frames.row(r) + 5)));
  }
  EXPECT_EQ(a.last_spread(), phm_core::rolling_spread(frames, 8)[7]);
}

TEST(RecurrentSpread, CalibrationIsASanePercentile)
{
  phm_core::numpy_random::Generator rng(0);
  phm_core::Matrix ref(500, 8);
  rng.normal(0.0, 1.0, ref.data.data(), ref.data.size());
  RecurrentTemporalSpreadAdapter a(kEmb, 30);
  const double thr = a.calibrate_from_data(ref, 1.0);
  EXPECT_EQ(a.threshold(), thr);
  int below = 0;
  int valid = 0;
  double lo = 1e300;
  double hi = -1e300;
  for (double s : phm_core::rolling_spread(ref, 30)) {
    if (std::isnan(s)) {
      continue;
    }
    ++valid;
    below += s < thr;
    lo = std::min(lo, s);
    hi = std::max(hi, s);
  }
  EXPECT_LE(lo, thr);
  EXPECT_LE(thr, hi);
  EXPECT_LE(static_cast<double>(below) / valid, 0.05);
}

TEST(RecurrentSpread, CollapseFiresAfterHysteresisAndRecoveryResets)
{
  RecurrentTemporalSpreadAdapter a(kEmb, 3, 1.0, 2);
  a.update(emb({0.0, 0.0}));
  a.update(emb({5.0, 0.0}));
  const auto healthy = a.update(emb({10.0, 0.0}));
  EXPECT_FALSE(healthy->violating);
  EXPECT_NE(healthy->reason.find(">="), std::string::npos);
  a.update(emb({2.0, 2.0}));
  a.update(emb({2.0, 2.0}));
  const auto pre = a.update(emb({2.0, 2.0}));
  const auto fire = a.update(emb({2.0, 2.0}));
  EXPECT_FALSE(pre->violating);
  EXPECT_TRUE(fire->violating);
  EXPECT_NEAR(fire->score, 1.0, 1e-12);
  EXPECT_EQ(fire->suggested_action, phm_core::ACTION_STOP_AND_HOLD);
  EXPECT_NE(fire->reason.find("<"), std::string::npos);

  RecurrentTemporalSpreadAdapter b(kEmb, 2, 1.0, 2);
  b.update(emb({1.0, 1.0}));
  EXPECT_FALSE(b.update(emb({1.0, 1.0}))->violating);
  EXPECT_TRUE(b.update(emb({1.0, 1.0}))->violating);
  b.update(emb({0.0, 0.0}));
  EXPECT_FALSE(b.update(emb({10.0, 10.0}))->violating);
  b.update(emb({1.0, 1.0}));
  EXPECT_FALSE(b.update(emb({1.0, 1.0}))->violating);
}

TEST(RecurrentSpread, ScoreRangeAndDirection)
{
  RecurrentTemporalSpreadAdapter a(kEmb, 2, 2.0, 1);
  phm_core::numpy_random::Generator rng(3);
  for (int i = 0; i < 20; ++i) {
    std::vector<double> v(4);
    rng.normal(0.0, 1.0, v.data(), v.size());
    const auto out = a.update(emb(v));
    EXPECT_GE(out->score, 0.0);
    EXPECT_LE(out->score, 1.0);
  }
  RecurrentTemporalSpreadAdapter b(kEmb, 2, 1.0, 1);
  b.update(emb({0.0, 0.0}));
  EXPECT_FALSE(b.update(emb({10.0, 10.0}))->violating);
  b.update(emb({4.0, 4.0}));
  EXPECT_TRUE(b.update(emb({4.0, 4.0}))->violating);
}
