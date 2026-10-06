// Copyright 2026 Yusuf Guenena. MIT License.
// Observation assembly and the shadow tick logic (no ONNX Runtime, no ROS).
// Built with -ffp-contract=off so the double-precision gravity terms round
// exactly as Python evaluated them on every architecture.
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "phm_go2/shadow_policy.hpp"

namespace phm_go2
{

std::array<float, 3> projected_gravity(double x, double y, double z, double w)
{
  // g_body = R(q)^T (0, 0, -1), as go2-phoenix projected_gravity_from_quat.
  const double gx = -2.0 * (x * z - w * y);
  const double gy = -2.0 * (y * z + w * x);
  const double gz = -(1.0 - 2.0 * (x * x + y * y));
  return {static_cast<float>(gx), static_cast<float>(gy), static_cast<float>(gz)};
}

void assemble_observation(
  const LowStateSample & s, const std::array<float, kJoints> & last_action, float * obs)
{
  std::size_t k = 0;
  for (int i = 0; i < 3; ++i) {
    obs[k++] = 0.0f;  // base_lin_vel: zeros source
  }
  for (int i = 0; i < 3; ++i) {
    obs[k++] = s.gyro[static_cast<std::size_t>(i)];  // base_ang_vel
  }
  // The quaternion components are float32 values widened to double.
  const auto g = projected_gravity(
    static_cast<double>(s.quat_wxyz[1]), static_cast<double>(s.quat_wxyz[2]),
    static_cast<double>(s.quat_wxyz[3]), static_cast<double>(s.quat_wxyz[0]));
  for (float v : g) {
    obs[k++] = v;
  }
  for (int i = 0; i < 3; ++i) {
    obs[k++] = 0.0f;  // velocity_command
  }
  for (std::size_t j = 0; j < kJoints; ++j) {
    obs[k++] = s.q[kMotorToPolicy[j]] - kDefaultJointPos[j];  // float32 subtraction
  }
  for (std::size_t j = 0; j < kJoints; ++j) {
    obs[k++] = s.dq[kMotorToPolicy[j]];
  }
  for (std::size_t j = 0; j < kJoints; ++j) {
    obs[k++] = last_action[j];
  }
}

Fault parse_fault(const std::string & name)
{
  if (name == "none") {
    return Fault::kNone;
  }
  if (name == "freeze_obs") {
    return Fault::kFreezeObs;
  }
  if (name == "freeze_sensors") {
    return Fault::kFreezeSensors;
  }
  if (name == "stop") {
    return Fault::kStop;
  }
  throw std::invalid_argument(
    "fault must be one of ('none', 'freeze_obs', 'freeze_sensors', 'stop'), got '" + name +
    "'");
}

ShadowStepper::ShadowStepper(Policy & policy, Fault fault)
: runner_(policy), fault_(fault)
{
}

StepOutcome ShadowStepper::step(const LowStateSample * latest, bool fault_active)
{
  just_injected_ = false;
  if (latest == nullptr) {
    return StepOutcome::kNoData;
  }
  const bool on = fault_active && fault_ != Fault::kNone;
  if (on && fault_ == Fault::kStop) {
    if (!fault_logged_) {
      fault_logged_ = true;
      just_injected_ = true;
    }
    return StepOutcome::kStopped;
  }
  if (on && fault_ == Fault::kFreezeSensors) {
    if (!frozen_) {
      frozen_sample_ = *latest;
      frozen_ = true;
      just_injected_ = true;
    }
    // Stale sensors, live last action.
    assemble_observation(frozen_sample_, last_action_, obs_.data());
  } else if (!(on && fault_ == Fault::kFreezeObs && frozen_)) {
    assemble_observation(*latest, last_action_, obs_.data());
    if (on && fault_ == Fault::kFreezeObs) {
      frozen_ = true;
      just_injected_ = true;
    }
  }
  runner_.run(obs_.data());
  const float * a = runner_.action();
  for (std::size_t j = 0; j < kJoints; ++j) {
    last_action_[j] = a[j];
  }
  return StepOutcome::kPublished;
}

}  // namespace phm_go2
