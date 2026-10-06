// Copyright 2026 Yusuf Guenena. MIT License.
// Observation parity and shadow tick semantics, without ONNX Runtime or ROS.
//
// data/observation_fixture.npz holds 64 random /lowstate readings and previous
// actions (float32, Unitree motor order, quaternions w-first, row 0 the
// identity) and the 48-D observation go2-phoenix's
// phoenix.sim2real.observation.assemble_policy_observation produced for each
// (go2-phoenix commit 23c6fb5, flat-terrain stand-v3 joint order and default
// pose, zero base_lin_vel and velocity_command). The C++ assembly must match
// it bit for bit.
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "phm_core/npy.hpp"
#include "phm_go2/shadow_policy.hpp"

using phm_go2::kJoints;
using phm_go2::kObsDim;

TEST(Observation, MatchesGo2PhoenixAssemblerBitForBit)
{
  const auto z = phm_core::npy::load_npz("data/observation_fixture.npz");
  const auto q = z.at("q").as_doubles();
  const auto dq = z.at("dq").as_doubles();
  const auto quat = z.at("quat_wxyz").as_doubles();
  const auto gyro = z.at("gyro").as_doubles();
  const auto last = z.at("last_action").as_doubles();
  const auto expected = z.at("obs").as_doubles();
  const std::size_t n = q.size() / kJoints;
  ASSERT_EQ(n, 64u);
  for (std::size_t t = 0; t < n; ++t) {
    phm_go2::LowStateSample s;
    std::array<float, kJoints> la{};
    for (std::size_t j = 0; j < kJoints; ++j) {
      s.q[j] = static_cast<float>(q[t * kJoints + j]);
      s.dq[j] = static_cast<float>(dq[t * kJoints + j]);
      la[j] = static_cast<float>(last[t * kJoints + j]);
    }
    for (std::size_t j = 0; j < 4; ++j) {
      s.quat_wxyz[j] = static_cast<float>(quat[t * 4 + j]);
    }
    for (std::size_t j = 0; j < 3; ++j) {
      s.gyro[j] = static_cast<float>(gyro[t * 3 + j]);
    }
    std::array<float, kObsDim> obs{};
    phm_go2::assemble_observation(s, la, obs.data());
    for (std::size_t k = 0; k < kObsDim; ++k) {
      const float e = static_cast<float>(expected[t * kObsDim + k]);
      EXPECT_EQ(std::memcmp(&obs[k], &e, sizeof(float)), 0)
        << "row " << t << " element " << k << ": " << obs[k] << " vs " << e;
    }
  }
}

TEST(Observation, IdentityQuaternionGivesGravityDown)
{
  const auto g = phm_go2::projected_gravity(0.0, 0.0, 0.0, 1.0);
  EXPECT_EQ(g[0], 0.0f);
  EXPECT_EQ(g[1], 0.0f);
  EXPECT_EQ(g[2], -1.0f);
}

namespace
{
// Deterministic stand-in policy: action = first 12 obs terms * 0.5 + 1,
// latent = the full observation.
class FakePolicy : public phm_go2::Policy
{
public:
  std::size_t latent_dim() const override {return kObsDim;}
  void run(const float * obs) override
  {
    ++runs;
    for (std::size_t k = 0; k < kObsDim; ++k) {
      latent_[k] = obs[k];
    }
    for (std::size_t j = 0; j < kJoints; ++j) {
      action_[j] = obs[j + 12] * 0.5f + 1.0f;
    }
  }
  const float * action() const override {return action_.data();}
  const float * latent() const override {return latent_.data();}
  int runs = 0;

private:
  std::array<float, kJoints> action_{};
  std::array<float, kObsDim> latent_{};
};

phm_go2::LowStateSample reading(float v)
{
  phm_go2::LowStateSample s;
  for (std::size_t j = 0; j < kJoints; ++j) {
    s.q[j] = v + static_cast<float>(j);
    s.dq[j] = -v;
  }
  s.quat_wxyz = {1.0f, 0.0f, 0.0f, 0.0f};
  s.gyro = {v, 0.0f, 0.0f};
  return s;
}
}  // namespace

TEST(Stepper, NoDataThenLastActionFeedsTheNextObservation)
{
  FakePolicy p;
  phm_go2::ShadowStepper st(p, phm_go2::Fault::kNone);
  EXPECT_EQ(st.step(nullptr, false), phm_go2::StepOutcome::kNoData);
  EXPECT_EQ(p.runs, 0);
  const auto a = reading(0.1f);
  ASSERT_EQ(st.step(&a, false), phm_go2::StepOutcome::kPublished);
  for (std::size_t j = 0; j < kJoints; ++j) {
    EXPECT_EQ(st.obs()[36 + j], 0.0f);  // zero action before the first tick
  }
  const auto prev_action = st.last_action();
  const auto b = reading(0.2f);
  st.step(&b, false);
  for (std::size_t j = 0; j < kJoints; ++j) {
    EXPECT_EQ(st.obs()[36 + j], prev_action[j]);
  }
}

TEST(Stepper, FreezeObsHoldsTheWholeObservationIncludingLastAction)
{
  FakePolicy p;
  phm_go2::ShadowStepper st(p, phm_go2::Fault::kFreezeObs);
  const auto a = reading(0.1f);
  st.step(&a, false);
  const auto b = reading(0.2f);
  st.step(&b, true);  // first faulted tick: assembled from b, then frozen
  EXPECT_TRUE(st.fault_just_injected());
  const auto frozen = st.obs();
  EXPECT_EQ(frozen[3], 0.2f);  // gyro x of reading b
  for (float v : {0.3f, 0.4f, 0.5f}) {
    const auto c = reading(v);
    EXPECT_EQ(st.step(&c, true), phm_go2::StepOutcome::kPublished);
    EXPECT_FALSE(st.fault_just_injected());
    EXPECT_EQ(st.obs(), frozen);
    EXPECT_EQ(std::memcmp(st.latent(), frozen.data(), sizeof(float) * kObsDim), 0);
  }
}

TEST(Stepper, StopPublishesNothingAfterTheFault)
{
  FakePolicy p;
  phm_go2::ShadowStepper st(p, phm_go2::Fault::kStop);
  const auto a = reading(0.1f);
  EXPECT_EQ(st.step(&a, false), phm_go2::StepOutcome::kPublished);
  EXPECT_EQ(st.step(&a, true), phm_go2::StepOutcome::kStopped);
  EXPECT_TRUE(st.fault_just_injected());
  EXPECT_EQ(st.step(&a, true), phm_go2::StepOutcome::kStopped);
  EXPECT_FALSE(st.fault_just_injected());
  EXPECT_EQ(p.runs, 1);
  EXPECT_THROW(phm_go2::parse_fault("explode"), std::invalid_argument);
}

namespace
{
// Deterministic policy whose action changes on every call (a function of the
// call count only), so a live last-action term is visibly different each tick.
class CountingPolicy : public phm_go2::Policy
{
public:
  std::size_t latent_dim() const override {return kObsDim;}
  void run(const float * obs) override
  {
    ++runs;
    for (std::size_t k = 0; k < kObsDim; ++k) {
      latent_[k] = obs[k];
    }
    for (std::size_t j = 0; j < kJoints; ++j) {
      action_[j] = static_cast<float>(runs) + 0.01f * static_cast<float>(j);
    }
  }
  const float * action() const override {return action_.data();}
  const float * latent() const override {return latent_.data();}
  int runs = 0;

private:
  std::array<float, kJoints> action_{};
  std::array<float, kObsDim> latent_{};
};
}  // namespace

TEST(Stepper, ParseFaultAcceptsAllFourNamesAndListsThemInTheError)
{
  EXPECT_EQ(phm_go2::parse_fault("none"), phm_go2::Fault::kNone);
  EXPECT_EQ(phm_go2::parse_fault("freeze_obs"), phm_go2::Fault::kFreezeObs);
  EXPECT_EQ(phm_go2::parse_fault("freeze_sensors"), phm_go2::Fault::kFreezeSensors);
  EXPECT_EQ(phm_go2::parse_fault("stop"), phm_go2::Fault::kStop);
  try {
    phm_go2::parse_fault("freeze");
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument & e) {
    const std::string msg = e.what();
    for (const char * name : {"none", "freeze_obs", "freeze_sensors", "stop"}) {
      EXPECT_NE(msg.find(std::string("'") + name + "'"), std::string::npos) << msg;
    }
  }
}

TEST(Stepper, FreezeSensorsKeepsSensorTermsAndFollowsTheLiveLastAction)
{
  CountingPolicy p;
  phm_go2::ShadowStepper st(p, phm_go2::Fault::kFreezeSensors);
  const auto a = reading(0.1f);
  ASSERT_EQ(st.step(&a, false), phm_go2::StepOutcome::kPublished);
  EXPECT_FALSE(st.fault_just_injected());
  const auto b = reading(0.2f);
  std::array<float, kJoints> prev_action = st.last_action();
  ASSERT_EQ(st.step(&b, true), phm_go2::StepOutcome::kPublished);  // first faulted tick
  EXPECT_TRUE(st.fault_just_injected());
  std::array<float, kObsDim> expected_b{};
  phm_go2::assemble_observation(b, prev_action, expected_b.data());
  const std::array<float, kObsDim> faulted = st.obs();
  EXPECT_EQ(faulted, expected_b);  // the faulted tick itself is a normal assembly
  prev_action = st.last_action();
  for (int i = 0; i < 4; ++i) {
    const auto c = reading(0.3f + 0.1f * static_cast<float>(i));  // different live readings
    ASSERT_EQ(st.step(&c, true), phm_go2::StepOutcome::kPublished);
    EXPECT_FALSE(st.fault_just_injected());
    for (std::size_t k = 0; k < 36; ++k) {
      EXPECT_EQ(st.obs()[k], faulted[k]) << "sensor term " << k << " changed on tick " << i;
    }
    for (std::size_t j = 0; j < kJoints; ++j) {
      EXPECT_EQ(st.obs()[36 + j], prev_action[j]) << "last_action term " << j;
    }
    // The last-action term really moves: the policy action differs every tick.
    EXPECT_NE(st.last_action()[0], prev_action[0]);
    prev_action = st.last_action();
    // The latent is the observation here, so it carries the live last action too.
    EXPECT_EQ(std::memcmp(st.latent(), st.obs().data(), sizeof(float) * kObsDim), 0);
  }
  // Consecutive faulted observations differ only in the last-action block.
  const std::array<float, kObsDim> before = st.obs();
  const auto d = reading(9.0f);
  st.step(&d, true);
  EXPECT_NE(st.obs(), before);
  EXPECT_EQ(std::memcmp(st.obs().data(), before.data(), sizeof(float) * 36), 0);
  EXPECT_NE(std::memcmp(st.obs().data() + 36, before.data() + 36, sizeof(float) * kJoints), 0);
}

TEST(Stepper, FreezeSensorsBehavesLikeNoFaultBeforeTheFault)
{
  CountingPolicy p_none;
  CountingPolicy p_fs;
  phm_go2::ShadowStepper none(p_none, phm_go2::Fault::kNone);
  phm_go2::ShadowStepper fs(p_fs, phm_go2::Fault::kFreezeSensors);
  for (int i = 0; i < 5; ++i) {
    const auto r = reading(0.1f * static_cast<float>(i + 1));
    ASSERT_EQ(none.step(&r, false), phm_go2::StepOutcome::kPublished);
    ASSERT_EQ(fs.step(&r, false), phm_go2::StepOutcome::kPublished);
    EXPECT_EQ(none.obs(), fs.obs());
    EXPECT_FALSE(fs.fault_just_injected());
  }
}

TEST(Stepper, FreezeObsStaysFullyConstantWithAChangingPolicyAction)
{
  CountingPolicy p;
  phm_go2::ShadowStepper st(p, phm_go2::Fault::kFreezeObs);
  const auto a = reading(0.1f);
  st.step(&a, true);
  const auto frozen = st.obs();
  for (int i = 0; i < 4; ++i) {
    const auto c = reading(0.5f + static_cast<float>(i));
    st.step(&c, true);
    EXPECT_EQ(st.obs(), frozen);
  }
  EXPECT_NE(st.last_action()[0], 0.0f);  // the action still changes; the observation does not
}
