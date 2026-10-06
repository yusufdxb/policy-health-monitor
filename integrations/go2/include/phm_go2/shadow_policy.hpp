// Copyright 2026 Yusuf Guenena. MIT License.
// Shadow-mode policy for the GO2 integration: observation assembly, ONNX
// Runtime inference, and the per-tick fault logic, without ROS.
//
// Observation (48 float32, the go2-phoenix flat-terrain contract, term order
// of Isaac Lab's PolicyCfg, no scaling):
//   base_lin_vel (3)        zeros: the stand-v3 deploy's operator-selected
//                           "zeros" source (go2-phoenix resolve_base_lin_vel)
//   base_ang_vel (3)        IMU gyroscope
//   projected_gravity (3)   world (0, 0, -1) rotated into the body frame
//   velocity_command (3)    zeros
//   joint_pos - default (12) policy joint order
//   joint_vel (12)          policy joint order
//   last_action (12)        the policy's own previous action
// This reproduces go2-phoenix's assemble_policy_observation() for this policy
// (pinned by test_observation.cpp against values that function produced).
//
// The action output is used only as the next tick's last_action term. Nothing
// here, or in the node that uses it, publishes a command.
#ifndef PHM_GO2__SHADOW_POLICY_HPP_
#define PHM_GO2__SHADOW_POLICY_HPP_

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace phm_go2
{

constexpr std::size_t kJoints = 12;
constexpr std::size_t kObsDim = 48;

// Unitree motor index (lowstate motor_state[0..11]: FR, FL, RR, RL legs, each
// hip / thigh / calf) of each joint in policy order (FL, FR, RL, RR hips, then
// thighs, then calves).
constexpr std::array<std::size_t, kJoints> kMotorToPolicy = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};
// Training default pose in policy order (deploy_stand_v3_shielded.yaml).
constexpr std::array<float, kJoints> kDefaultJointPos = {
  0.1f, -0.1f, 0.1f, -0.1f, 0.8f, 0.8f, 1.0f, 1.0f, -1.5f, -1.5f, -1.5f, -1.5f};

// One /lowstate reading in Unitree order.
struct LowStateSample
{
  std::array<float, kJoints> q{};          // motor_state[i].q
  std::array<float, kJoints> dq{};         // motor_state[i].dq
  std::array<float, 4> quat_wxyz{};        // imu_state.quaternion (w, x, y, z)
  std::array<float, 3> gyro{};             // imu_state.gyroscope
};

// World gravity (0, 0, -1) rotated into the body frame for quaternion
// (x, y, z, w), computed in double and rounded to float32.
std::array<float, 3> projected_gravity(double x, double y, double z, double w);

// Fill obs[0..48) from a reading and the previous action.
void assemble_observation(
  const LowStateSample & s, const std::array<float, kJoints> & last_action, float * obs);

// A policy that maps a 48-D observation to a 12-D action and a latent.
class Policy
{
public:
  virtual ~Policy() = default;
  virtual std::size_t latent_dim() const = 0;
  // Run on obs[0..48); action() and latent() then hold the outputs.
  virtual void run(const float * obs) = 0;
  virtual const float * action() const = 0;
  virtual const float * latent() const = 0;
};

// ONNX Runtime session with inputs "obs" [1, 48] and outputs "action" [1, 12]
// and "latent" [1, D]. CPU execution provider. Tensors are bound to buffers
// allocated once, so run() allocates nothing.
class PolicyRunner : public Policy
{
public:
  explicit PolicyRunner(const std::string & onnx_path, int intra_op_threads = 1);
  ~PolicyRunner() override;
  PolicyRunner(const PolicyRunner &) = delete;
  PolicyRunner & operator=(const PolicyRunner &) = delete;

  std::size_t latent_dim() const override;
  void run(const float * obs) override;
  const float * action() const override;
  const float * latent() const override;
  // Wall time of the last run() in milliseconds (ONNX Runtime only).
  double last_run_ms() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

enum class Fault { kNone, kFreezeObs, kFreezeSensors, kStop };
// "none" | "freeze_obs" | "freeze_sensors" | "stop"; throws std::invalid_argument otherwise.
Fault parse_fault(const std::string & name);

enum class StepOutcome
{
  kNoData,      // no /lowstate yet: nothing to do
  kStopped,     // stop fault active: publish nothing
  kPublished,   // inference ran; publish the latent
};

// One shadow tick, shared by the live node and the offline replay so both
// apply identical fault semantics:
//   - stop: once the fault is active, nothing more is computed or published.
//   - freeze_obs: the first faulted tick assembles the observation as usual
//     and freezes it; every later tick feeds that same observation, last-action
//     term included, so the latent becomes constant.
//   - freeze_sensors: the first faulted tick keeps a copy of the reading it
//     used; every later tick assembles the observation from that frozen
//     reading plus the live last_action (the policy's own previous action),
//     runs the policy and updates last_action. The sensor terms obs[0..36)
//     stay constant while obs[36..48) keep following the policy's own actions.
//   - otherwise the observation is assembled from the latest reading, and the
//     action becomes the next tick's last_action.
class ShadowStepper
{
public:
  ShadowStepper(Policy & policy, Fault fault);

  // `latest` is nullptr until the first /lowstate arrives.
  StepOutcome step(const LowStateSample * latest, bool fault_active);

  // True exactly once, on the tick the fault first took effect.
  bool fault_just_injected() const {return just_injected_;}
  const std::array<float, kObsDim> & obs() const {return obs_;}
  const std::array<float, kJoints> & last_action() const {return last_action_;}
  const float * latent() const {return runner_.latent();}
  std::size_t latent_dim() const {return runner_.latent_dim();}
  Fault fault() const {return fault_;}

private:
  Policy & runner_;
  Fault fault_;
  std::array<float, kObsDim> obs_{};
  std::array<float, kJoints> last_action_{};
  LowStateSample frozen_sample_{};  // freeze_sensors: the reading kept at the fault
  bool frozen_ = false;
  bool fault_logged_ = false;
  bool just_injected_ = false;
};

}  // namespace phm_go2

#endif  // PHM_GO2__SHADOW_POLICY_HPP_
