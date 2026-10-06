// Copyright 2026 Yusuf Guenena. MIT License.
// Worst-wins arbitration (ported from phm_arbiter/tests/test_arbiter.py and
// test_arbiter_safety_logic.py) and the arbiter -> recovery seam
// (phm_recovery/tests/test_arbiter_recovery_integration.py).
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "phm_core/arbiter.hpp"
#include "phm_core/recovery.hpp"
#include "phm_core/severity.hpp"

using phm_core::arbitrate;
using phm_core::ArbiterInput;

namespace
{
constexpr double kNow = 100.0;

ArbiterInput verdict(
  const std::string & source, double score, bool violating, const std::string & reason,
  uint8_t action = phm_core::ACTION_NONE, double age = 0.0)
{
  ArbiterInput v;
  v.source = source;
  v.score = score;
  v.violating = violating;
  v.reason = reason;
  v.suggested_action = action;
  v.timestamp = kNow - age;
  return v;
}
}  // namespace

TEST(Arbiter, NoVerdictsIsNominalOk)
{
  const auto r = arbitrate({}, kNow);
  EXPECT_EQ(r.state, phm_core::STATE_OK);
  EXPECT_EQ(r.score, 0.0);
  EXPECT_EQ(r.suggested_action, phm_core::ACTION_NONE);
  EXPECT_EQ(r.source, "");
  EXPECT_EQ(r.reason, "all detectors nominal");
}

TEST(Arbiter, SingleVerdictBands)
{
  EXPECT_EQ(arbitrate({verdict("a", 0.1, false, "fine")}, kNow).state, phm_core::STATE_OK);
  const auto d = arbitrate(
    {verdict("freq:/scan", 0.3, true, "freq low", phm_core::ACTION_LOG_ONLY)}, kNow);
  EXPECT_EQ(d.state, phm_core::STATE_DEGRADED);
  EXPECT_DOUBLE_EQ(d.score, 0.3);
  EXPECT_EQ(d.source, "freq:/scan");
  EXPECT_EQ(d.reason, "freq low");
  EXPECT_EQ(d.suggested_action, phm_core::ACTION_LOG_ONLY);
  EXPECT_EQ(
    arbitrate({verdict("phm_ood", 0.65, true, "c", phm_core::ACTION_HOLD)}, kNow).state,
    phm_core::STATE_INTERVENE);
  const auto s = arbitrate(
    {verdict("threshold:cpu", 0.9, true, "cpu", phm_core::ACTION_STOP_AND_HOLD)}, kNow);
  EXPECT_EQ(s.state, phm_core::STATE_STOP);
  EXPECT_EQ(s.suggested_action, phm_core::ACTION_STOP_AND_HOLD);
}

TEST(Arbiter, WorstWinsOrdering)
{
  const auto r = arbitrate(
    {verdict("src_a", 0.3, true, "a", phm_core::ACTION_LOG_ONLY),
      verdict("src_b", 0.55, true, "b", phm_core::ACTION_HOLD),
      verdict("src_c", 0.85, true, "c", phm_core::ACTION_STOP_AND_HOLD)}, kNow);
  EXPECT_EQ(r.state, phm_core::STATE_STOP);
  EXPECT_EQ(r.source, "src_c");
  EXPECT_EQ(r.reason, "c");
  EXPECT_DOUBLE_EQ(r.score, 0.85);
  EXPECT_EQ(
    arbitrate(
      {verdict("freq", 0.35, true, "f"), verdict("ood", 0.55, true, "o")}, kNow).source, "ood");
}

TEST(Arbiter, StaleNonViolatingFloorsAtDegraded)
{
  const auto r = arbitrate({verdict("freq:/scan", 0.1, false, "ok", 0, 5.0)}, kNow, 1.0);
  EXPECT_EQ(r.state, phm_core::STATE_DEGRADED);
  EXPECT_EQ(r.reason, "stale:freq:/scan");
  EXPECT_EQ(r.source, "freq:/scan");
  EXPECT_DOUBLE_EQ(r.score, 0.25);
  EXPECT_EQ(r.suggested_action, phm_core::ACTION_LOG_ONLY);
}

TEST(Arbiter, StalenessNeverDeescalatesAViolatingVerdict)
{
  const auto r = arbitrate({verdict("phm_ood", 0.65, true, "ood", 0, 2.0)}, kNow, 1.0);
  EXPECT_EQ(r.state, phm_core::STATE_INTERVENE);
  EXPECT_DOUBLE_EQ(r.score, 0.65);
  EXPECT_EQ(r.reason, "stale:phm_ood");
  const auto stop = arbitrate(
    {verdict("threshold:cpu", 0.95, true, "cpu", phm_core::ACTION_STOP_AND_HOLD, 1.1)}, kNow);
  EXPECT_EQ(stop.state, phm_core::STATE_STOP);
  EXPECT_GE(stop.score, 0.80);
  EXPECT_EQ(stop.reason, "stale:threshold:cpu");
}

TEST(Arbiter, StaleAndFreshMixes)
{
  EXPECT_EQ(
    arbitrate(
      {verdict("stale_src", 0.65, true, "s", 0, 2.0),
        verdict("fresh_src", 0.85, true, "cpu", phm_core::ACTION_STOP_AND_HOLD)}, kNow).source,
    "fresh_src");
  const auto r = arbitrate(
    {verdict("stale_src", 0.1, false, "ok", 0, 2.0), verdict("fresh_src", 0.05, false, "fine")},
    kNow);
  EXPECT_EQ(r.state, phm_core::STATE_DEGRADED);
  EXPECT_EQ(r.source, "stale_src");
  const auto both = arbitrate(
    {verdict("src_a", 0.1, false, "ok", 0, 3.0), verdict("src_b", 0.9, true, "bad", 0, 5.0)},
    kNow);
  EXPECT_EQ(both.state, phm_core::STATE_STOP);
  EXPECT_EQ(both.source, "src_b");
  EXPECT_DOUBLE_EQ(both.score, 0.9);
  EXPECT_EQ(both.reason, "stale:src_b");
}

TEST(Arbiter, StalenessBoundaryIsStrict)
{
  EXPECT_EQ(arbitrate({verdict("src", 0.9, true, "bad", 0, 1.0)}, kNow, 1.0).reason, "bad");
  EXPECT_EQ(arbitrate({verdict("src", 0.9, true, "bad", 0, 1.001)}, kNow, 1.0).reason,
    "stale:src");
  EXPECT_EQ(arbitrate({verdict("src", 0.05, false, "ok", 0, 1.001)}, kNow, 1.0).state,
    phm_core::STATE_DEGRADED);
  EXPECT_EQ(arbitrate({verdict("src", 0.9, true, "bad", 0, 0.999)}, kNow, 1.0).state,
    phm_core::STATE_STOP);
  EXPECT_EQ(arbitrate({verdict("src", 0.9, true, "bad", 0, 0.0)}, kNow, 0.0).reason, "bad");
  EXPECT_EQ(arbitrate({verdict("src", 0.9, true, "bad", 0, 0.001)}, kNow, 0.0).reason,
    "stale:src");
}

TEST(Arbiter, TieBreaksOnScoreThenInputOrder)
{
  const auto higher = arbitrate(
    {verdict("src_a", 0.55, true, "a low", phm_core::ACTION_HOLD),
      verdict("src_b", 0.75, true, "b high", phm_core::ACTION_HOLD)}, kNow);
  EXPECT_EQ(higher.source, "src_b");
  EXPECT_EQ(higher.reason, "b high");
  const auto equal = arbitrate(
    {verdict("src_a", 0.6, true, "a"), verdict("src_b", 0.6, true, "b")}, kNow);
  EXPECT_EQ(equal.source, "src_a");  // first of equals, as Python's max()
}

TEST(Arbiter, ActionAllowlist)
{
  for (uint8_t a = 0; a <= phm_core::ACTION_REWIND; ++a) {
    EXPECT_EQ(arbitrate({verdict("src", 0.65, true, "r", a)}, kNow).suggested_action, a);
  }
  EXPECT_EQ(
    arbitrate({verdict("src", 0.65, true, "r", 99)}, kNow).suggested_action,
    phm_core::ACTION_NONE);
}

TEST(Arbiter, FreshNonViolatingIsInvisibleAndTimestamplessIsFresh)
{
  const auto r = arbitrate(
    {verdict("a", 0.8, false, "fine"), verdict("b", 0.9, false, "fine")}, kNow);
  EXPECT_EQ(r.state, phm_core::STATE_OK);
  ArbiterInput no_ts = verdict("phm_ood", 0.85, true, "bad", phm_core::ACTION_STOP_AND_HOLD);
  no_ts.has_timestamp = false;
  no_ts.timestamp = -1e9;
  EXPECT_EQ(arbitrate({no_ts}, kNow, 1.0).state, phm_core::STATE_STOP);
}

TEST(Arbiter, ManySources)
{
  std::vector<ArbiterInput> v;
  for (int i = 0; i < 20; ++i) {
    v.push_back(verdict("src_" + std::to_string(i), 0.1 + i * 0.02, true, "r"));
  }
  const auto r = arbitrate(v, kNow);
  EXPECT_EQ(r.source, "src_19");
  EXPECT_NEAR(r.score, 0.48, 1e-12);
  v.push_back(verdict("src_stop", 0.95, true, "critical"));
  EXPECT_EQ(arbitrate(v, kNow).source, "src_stop");
}

TEST(ArbiterSafety, ViolatingNeverResolvesToOk)
{
  EXPECT_GE(arbitrate({verdict("ood", 0.10, true, "sub")}, kNow).state, phm_core::STATE_DEGRADED);
  EXPECT_GE(arbitrate({verdict("src", 0.0, true, "zero")}, kNow).state, phm_core::STATE_DEGRADED);
}

TEST(ArbiterSafety, NonFiniteScoresAreBoundedAndDeterministic)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto r = arbitrate({verdict("phm_ood", nan, true, "poisoned")}, kNow);
  EXPECT_NE(r.state, phm_core::STATE_STOP);
  EXPECT_TRUE(std::isfinite(r.score));
  EXPECT_EQ(r.reason, "bad-score:phm_ood");
  const auto r2 = arbitrate({verdict("phm_ood", nan, true, "poisoned")}, kNow);
  EXPECT_EQ(r.state, r2.state);
  EXPECT_EQ(r.score, r2.score);
  const auto real = arbitrate(
    {verdict("phm_ood", nan, true, "p"), verdict("threshold:cpu", 0.92, true, "cpu")}, kNow);
  EXPECT_EQ(real.state, phm_core::STATE_STOP);
  EXPECT_EQ(real.source, "threshold:cpu");
  const auto inf = arbitrate(
    {verdict("src", std::numeric_limits<double>::infinity(), true, "inf")}, kNow);
  EXPECT_NE(inf.state, phm_core::STATE_STOP);
  EXPECT_EQ(inf.reason, "bad-score:src");
}

// No arbitrated state >= INTERVENE may ever leave the recovery hold released,
// for any score band, suggested action, or violating flag.
TEST(ArbiterRecoverySeam, NoInterveneOrStopEverReleasesTheHold)
{
  for (double score : {0.0, 0.10, 0.30, 0.65, 0.95}) {
    for (uint8_t action = 0; action <= phm_core::ACTION_REWIND; ++action) {
      for (bool violating : {true, false}) {
        const auto health = arbitrate({verdict("phm_ood", score, violating, "seam", action)}, kNow);
        phm_core::HealthToActionMapper mapper;
        const auto d = mapper.map(health.state, health.suggested_action, health.source,
            health.reason);
        if (health.state >= phm_core::STATE_INTERVENE) {
          EXPECT_TRUE(d.hold_active) << score << " " << int(action);
        }
      }
    }
  }
}

TEST(ArbiterRecoverySeam, InterveneRewindAndStopCompose)
{
  const auto rewind = arbitrate(
    {verdict("phm_ood", 0.65, true, "rewind", phm_core::ACTION_REWIND)}, kNow);
  EXPECT_EQ(rewind.state, phm_core::STATE_INTERVENE);
  EXPECT_EQ(rewind.suggested_action, phm_core::ACTION_REWIND);
  phm_core::HealthToActionMapper m1;
  const auto d1 = m1.map(rewind.state, rewind.suggested_action, rewind.source, rewind.reason);
  EXPECT_EQ(d1.action, phm_core::ACTION_REWIND);
  EXPECT_TRUE(d1.hold_active);

  const auto none = arbitrate({verdict("phm_ood", 0.55, true, "ood")}, kNow);
  phm_core::HealthToActionMapper m2;
  EXPECT_TRUE(m2.map(none.state, none.suggested_action, none.source, none.reason).hold_active);

  const auto stop = arbitrate(
    {verdict("threshold:cpu", 0.95, true, "cpu", phm_core::ACTION_STOP_AND_HOLD)}, kNow);
  phm_core::HealthToActionMapper m3;
  const auto d3 = m3.map(stop.state, stop.suggested_action, stop.source, stop.reason);
  EXPECT_EQ(d3.action, phm_core::ACTION_STOP_AND_HOLD);
  EXPECT_TRUE(d3.hold_active);
}
