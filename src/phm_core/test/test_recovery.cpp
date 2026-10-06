// Copyright 2026 Yusuf Guenena. MIT License.
// SafetyEnvelope, HealthToActionMapper and RewindHook (ported from
// phm_recovery/tests/test_safety_envelope.py and test_health_action_mapper.py).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "phm_core/recovery.hpp"
#include "phm_core/severity.hpp"

using phm_core::EnvelopeStatus;
using phm_core::HealthToActionMapper;
using phm_core::RewindHook;
using phm_core::SafetyEnvelope;

TEST(Envelope, Allowlist)
{
  SafetyEnvelope env(true, 5.0);
  const auto unknown = env.evaluate(99, "src", 0.0);
  EXPECT_EQ(unknown.status, EnvelopeStatus::kSuppressedAllowlist);
  EXPECT_FALSE(unknown.publish);
  EXPECT_EQ(unknown.reason, "action 99 not in allowlist");
  for (int a = 0; a <= phm_core::ACTION_REWIND; ++a) {
    EXPECT_EQ(SafetyEnvelope(true, 5.0).evaluate(a, "src", 0.0).status, EnvelopeStatus::kAccepted);
  }
  EXPECT_TRUE(SafetyEnvelope(true, 5.0).evaluate(phm_core::ACTION_HOLD, "s", 0).publish);
  EXPECT_TRUE(SafetyEnvelope(true, 5.0).evaluate(phm_core::ACTION_STOP_AND_HOLD, "s", 0).publish);
  EXPECT_TRUE(SafetyEnvelope(true, 5.0).evaluate(phm_core::ACTION_REWIND, "s", 0).publish);
  EXPECT_FALSE(SafetyEnvelope(true, 5.0).evaluate(phm_core::ACTION_NONE, "s", 0).publish);
  EXPECT_FALSE(SafetyEnvelope(true, 5.0).evaluate(phm_core::ACTION_LOG_ONLY, "s", 0).publish);
}

TEST(Envelope, CooldownDampsNewHoldsPerActionAndKey)
{
  SafetyEnvelope env(true, 5.0);
  EXPECT_EQ(env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 0.0).status,
    EnvelopeStatus::kAccepted);
  const auto second = env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 2.0);
  EXPECT_EQ(second.status, EnvelopeStatus::kSuppressedCooldown);
  EXPECT_FALSE(second.publish);
  EXPECT_EQ(second.reason, "cooldown active for fault_a (2.00s < 5.00s)");
  // Strict <: exactly the cooldown later is accepted; later still too.
  EXPECT_EQ(env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 5.0).status,
    EnvelopeStatus::kAccepted);
  SafetyEnvelope env2(true, 5.0);
  env2.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 0.0);
  EXPECT_EQ(env2.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 5.01).status,
    EnvelopeStatus::kAccepted);
  EXPECT_EQ(env2.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_b", 5.02).status,
    EnvelopeStatus::kAccepted);
  SafetyEnvelope env3(true, 5.0);
  env3.evaluate(phm_core::ACTION_HOLD, "fault_a", 0.0);
  EXPECT_EQ(env3.evaluate(phm_core::ACTION_HOLD, "fault_a", 2.0).status,
    EnvelopeStatus::kSuppressedCooldown);
  EXPECT_EQ(env3.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 1.0).status,
    EnvelopeStatus::kAccepted);
  SafetyEnvelope env4(true, 5.0);
  env4.evaluate(phm_core::ACTION_REWIND, "fault_a", 0.0);
  EXPECT_EQ(env4.evaluate(phm_core::ACTION_REWIND, "fault_a", 0.1).status,
    EnvelopeStatus::kAccepted);
}

TEST(Envelope, ContinuedHoldIsCooldownExempt)
{
  SafetyEnvelope env(true, 5.0);
  EXPECT_TRUE(env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 0.0).publish);
  const auto cont = env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 2.0, true);
  EXPECT_EQ(cont.status, EnvelopeStatus::kAccepted);
  EXPECT_TRUE(cont.publish);
  EXPECT_EQ(cont.reason, "continue active hold (STOP_AND_HOLD, cooldown-exempt)");
  EXPECT_FALSE(env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 2.0).publish);
  SafetyEnvelope env2(true, 5.0);
  env2.evaluate(phm_core::ACTION_HOLD, "fault_a", 0.0);
  for (double t : {0.5, 1.0, 1.5, 2.0, 2.5}) {
    EXPECT_TRUE(env2.evaluate(phm_core::ACTION_HOLD, "fault_a", t, true).publish) << t;
  }
}

TEST(Envelope, ResumeIsNeverRateLimitedAndDisabledSuppressesAll)
{
  SafetyEnvelope env(true, 5.0);
  env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", 0.0);
  const auto resume = env.evaluate_resume("fault_a", 0.0001);
  EXPECT_EQ(resume.status, EnvelopeStatus::kAccepted);
  EXPECT_TRUE(resume.publish);
  EXPECT_EQ(resume.reason, "RESUME accepted for fault_a (cooldown exempt)");
  for (int t = 0; t < 5; ++t) {
    env.evaluate(phm_core::ACTION_STOP_AND_HOLD, "fault_a", t * 10.0);
  }
  EXPECT_TRUE(env.evaluate_resume("fault_a", 41.0).publish);

  SafetyEnvelope off(false, 5.0);
  for (int a = 0; a <= phm_core::ACTION_REWIND; ++a) {
    const auto r = off.evaluate(a, "fault_a", 0.0);
    EXPECT_EQ(r.status, EnvelopeStatus::kSuppressedDisabled);
    EXPECT_FALSE(r.publish);
  }
  EXPECT_EQ(off.evaluate_resume("fault_a", 0.0).status, EnvelopeStatus::kSuppressedDisabled);
  EXPECT_STREQ(phm_core::to_string(EnvelopeStatus::kSuppressedCooldown), "SUPPRESSED_COOLDOWN");
}

TEST(Mapper, StateMappings)
{
  using phm_core::ACTION_HOLD;
  using phm_core::ACTION_LOG_ONLY;
  using phm_core::ACTION_NONE;
  using phm_core::ACTION_REWIND;
  using phm_core::ACTION_STOP_AND_HOLD;
  HealthToActionMapper m;
  auto d = m.map(phm_core::STATE_STOP, ACTION_LOG_ONLY, "phm_ood", "ood");
  EXPECT_EQ(d.action, ACTION_STOP_AND_HOLD);
  EXPECT_TRUE(d.hold_active);
  EXPECT_EQ(d.reason, "STATE_STOP from phm_ood: ood");
  EXPECT_EQ(HealthToActionMapper().map(phm_core::STATE_INTERVENE, ACTION_HOLD, "f", "").action,
    ACTION_HOLD);
  EXPECT_EQ(
    HealthToActionMapper().map(phm_core::STATE_INTERVENE, ACTION_STOP_AND_HOLD, "f", "").action,
    ACTION_HOLD);
  const auto rw = HealthToActionMapper().map(phm_core::STATE_INTERVENE, ACTION_REWIND, "o", "r");
  EXPECT_EQ(rw.action, ACTION_REWIND);
  EXPECT_TRUE(rw.hold_active);
  const auto lo = HealthToActionMapper().map(phm_core::STATE_INTERVENE, ACTION_LOG_ONLY, "c", "");
  EXPECT_EQ(lo.action, ACTION_HOLD);
  EXPECT_TRUE(lo.hold_active);
  EXPECT_TRUE(m.map(phm_core::STATE_INTERVENE, ACTION_LOG_ONLY, "cpu", "").hold_active);
  d = m.map(phm_core::STATE_OK, ACTION_NONE, "ood", "recovered");
  EXPECT_EQ(d.action, ACTION_NONE);
  EXPECT_FALSE(d.hold_active);
  EXPECT_FALSE(m.hold_active());
  EXPECT_EQ(d.reason, "hold cleared: state=0 from ood");
  m.map(phm_core::STATE_STOP, ACTION_NONE, "ood", "");
  EXPECT_FALSE(m.map(phm_core::STATE_DEGRADED, ACTION_LOG_ONLY, "ood", "").hold_active);
  const auto idle = m.map(phm_core::STATE_OK, ACTION_NONE, "ood", "all good");
  EXPECT_EQ(idle.action, ACTION_NONE);
  EXPECT_EQ(idle.reason, "state 0 from ood: no actuation");
  EXPECT_EQ(m.map(phm_core::STATE_DEGRADED, ACTION_LOG_ONLY, "x", "").action, ACTION_LOG_ONLY);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(m.map(phm_core::STATE_STOP, ACTION_NONE, "ood", "").hold_active);
  }
  m.clear_hold();
  EXPECT_FALSE(m.hold_active());
  const auto unknown = m.map(99, ACTION_NONE, "unknown", "");
  EXPECT_EQ(unknown.action, ACTION_STOP_AND_HOLD);
  EXPECT_TRUE(unknown.hold_active);
}

TEST(Rewind, DefaultLogsAndRegisteredCallbackRuns)
{
  std::vector<std::string> logs;
  RewindHook hook;
  hook.set_logger([&logs](phm_core::LogLevel, const std::string & m) {logs.push_back(m);});
  hook.trigger();
  ASSERT_EQ(logs.size(), 1u);
  EXPECT_NE(logs[0].find("REWIND"), std::string::npos);
  int a = 0;
  int b = 0;
  hook.register_callback([&a]() {++a;});
  hook.trigger();
  hook.trigger();
  EXPECT_EQ(a, 2);
  hook.register_callback([&b]() {++b;});
  hook.trigger();
  EXPECT_EQ(a, 2);
  EXPECT_EQ(b, 1);
}
