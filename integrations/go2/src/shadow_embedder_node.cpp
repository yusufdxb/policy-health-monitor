// Copyright 2026 Yusuf Guenena. MIT License.
// phoenix_shadow_embedder: the go2-phoenix stand-v3 locomotion policy in shadow
// mode on a live Unitree GO2, publishing its latent for PHM.
//
// Runs the policy ONNX (exported with --emit-latent) on the robot's real
// /lowstate at rate_hz (50 Hz, the trained rate) and publishes the latent as
// phm_msgs/PolicyEmbedding on embedding_topic. It never commands the robot:
// its publishers are the embedding topic and, with stale_check, one
// DetectorVerdict stream on /phm/verdicts. The action output
// is used only as the next observation's last_action term, so the policy runs
// open loop on that term; the joint, IMU and gravity terms are the robot's real
// state.
//
// Fault injection (for an induced-failure run, still without actuation):
//   fault = "none"        no fault (default)
//   fault = "freeze_obs"  after fault_after_sec, keep feeding the last
//                         observation, as a policy wired to a stale snapshot;
//                         the latent stops varying
//   fault = "freeze_sensors"
//                         after fault_after_sec, keep feeding the last sensor
//                         reading but assemble the last-action term live, as a
//                         stale sensor feeding a policy that still acts; the
//                         sensor terms of the observation stay constant
//   fault = "stop"        after fault_after_sec, stop publishing, as a crashed
//                         policy process; the embedding topic goes silent
//   fault = "gyro_stuck"  after fault_after_sec, hold the gyroscope reading
//   fault = "joint_stuck" after fault_after_sec, hold q and dq of the FL calf
//                         (partial sensor faults; every other term stays live)
// Each logs one "FAULT INJECTED" warning, the time detection latencies are
// measured from.
//
// stale_check (default false, so the original detector set is unchanged):
// a phm_core::StaleSignalAdapter on the reading each observation was
// assembled from, faults applied. Groups: the 3 gyro axes together, and q with
// dq of each motor. A group is stale on a tick when all its channels equal the
// previous tick's; it fires after more than stale_gyro_max_repeats /
// stale_joint_max_repeats stale ticks in a row. The verdict goes to
// /phm/verdicts every published tick.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/adapters.hpp"
#include "phm_core/numerics.hpp"
#include "phm_go2/shadow_policy.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_state.hpp"

namespace
{

double mono()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

class PhoenixShadowEmbedder : public rclcpp::Node
{
public:
  PhoenixShadowEmbedder()
  : rclcpp::Node("phoenix_shadow_embedder")
  {
    const auto onnx_path = declare_parameter<std::string>("onnx_path", "");
    const auto phoenix_src = declare_parameter<std::string>("phoenix_src", "");
    const double rate_hz = declare_parameter<double>("rate_hz", 50.0);
    const auto topic = declare_parameter<std::string>("embedding_topic", "/policy/embedding");
    policy_id_ = declare_parameter<std::string>("policy_id", "phoenix-stand-v3");
    const auto fault_name = declare_parameter<std::string>("fault", "none");
    fault_after_ = declare_parameter<double>("fault_after_sec", -1.0);
    stats_every_ = declare_parameter<double>("stats_every_sec", 10.0);
    const bool stale_check = declare_parameter<bool>("stale_check", false);
    const int gyro_max = static_cast<int>(declare_parameter<int64_t>("stale_gyro_max_repeats", 5));
    const int joint_max =
      static_cast<int>(declare_parameter<int64_t>("stale_joint_max_repeats", 12));

    const phm_go2::Fault fault = phm_go2::parse_fault(fault_name);
    if (onnx_path.empty() || !std::ifstream(onnx_path).good()) {
      throw std::runtime_error("onnx_path '" + onnx_path + "' does not exist");
    }
    if (!phoenix_src.empty()) {
      RCLCPP_WARN(
        get_logger(), "phoenix_src is ignored: observation assembly is built into this node");
    }
    runner_ = std::make_unique<phm_go2::PolicyRunner>(onnx_path, 1);
    stepper_ = std::make_unique<phm_go2::ShadowStepper>(*runner_, fault);
    msg_.embedding.resize(runner_->latent_dim());
    msg_.dim = static_cast<uint32_t>(runner_->latent_dim());
    msg_.policy_id = policy_id_;
    t0_ = mono();
    stats_t_ = t0_;
    infer_ms_.reserve(1024);
    tick_ms_.reserve(1024);

    const auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
    sub_ = create_subscription<unitree_go::msg::LowState>(
      "/lowstate", sensor_qos,
      [this](unitree_go::msg::LowState::ConstSharedPtr m) {on_lowstate(*m);});
    pub_ = create_publisher<phm_msgs::msg::PolicyEmbedding>(topic, sensor_qos);
    if (stale_check) {
      // Channels: gyro 0..2, q 3..14, dq 15..26 (Unitree motor order).
      std::vector<phm_core::StaleGroup> groups{{"gyro", {0, 1, 2}, gyro_max}};
      for (std::size_t i = 0; i < phm_go2::kJoints; ++i) {
        groups.push_back({"motor" + std::to_string(i), {3 + i, 15 + i}, joint_max});
      }
      stale_ = std::make_unique<phm_core::StaleSignalAdapter>("/lowstate", std::move(groups));
      stale_->set_logger(
        [this](phm_core::LogLevel, const std::string & m) {
          RCLCPP_WARN(get_logger(), "%s", m.c_str());
        });
      stale_sample_.topic = "/lowstate";
      stale_sample_.values.resize(3 + 2 * phm_go2::kJoints);
      // Same QoS as phm_detectors' verdicts: reliable, volatile.
      verdict_pub_ = create_publisher<phm_msgs::msg::DetectorVerdict>(
        "/phm/verdicts", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
    }
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_hz), [this]() {tick();});
    RCLCPP_INFO(
      get_logger(),
      "shadow policy %s (%s) latent_dim=%zu at %.0f Hz; fault=%s after %.1fs. NO actuation: "
      "only publisher is the embedding.", policy_id_.c_str(), onnx_path.c_str(),
      runner_->latent_dim(), rate_hz, fault_name.c_str(), fault_after_);
    if (stale_) {
      RCLCPP_INFO(
        get_logger(), "stale_check on: gyro max %d, joint max %d repeated ticks", gyro_max,
        joint_max);
    }
  }

private:
  void on_lowstate(const unitree_go::msg::LowState & m)
  {
    for (std::size_t i = 0; i < phm_go2::kJoints; ++i) {
      latest_.q[i] = m.motor_state[i].q;
      latest_.dq[i] = m.motor_state[i].dq;
    }
    for (std::size_t i = 0; i < 4; ++i) {
      latest_.quat_wxyz[i] = m.imu_state.quaternion[i];
    }
    for (std::size_t i = 0; i < 3; ++i) {
      latest_.gyro[i] = m.imu_state.gyroscope[i];
    }
    have_latest_ = true;
    latest_rx_ = mono();
  }

  void tick()
  {
    const auto t_start = std::chrono::steady_clock::now();
    const double now = mono();
    const bool fault_active = fault_after_ >= 0.0 && now - t0_ >= fault_after_;
    const auto outcome = stepper_->step(have_latest_ ? &latest_ : nullptr, fault_active);
    if (stepper_->fault_just_injected()) {
      if (stepper_->fault() == phm_go2::Fault::kStop) {
        RCLCPP_WARN(get_logger(), "FAULT INJECTED: stop (policy stops publishing)");
      } else if (stepper_->fault() == phm_go2::Fault::kGyroStuck) {
        RCLCPP_WARN(get_logger(), "FAULT INJECTED: gyro_stuck (gyroscope held, rest live)");
      } else if (stepper_->fault() == phm_go2::Fault::kJointStuck) {
        RCLCPP_WARN(get_logger(), "FAULT INJECTED: joint_stuck (FL calf q/dq held, rest live)");
      } else if (stepper_->fault() == phm_go2::Fault::kFreezeSensors) {
        RCLCPP_WARN(
          get_logger(),
          "FAULT INJECTED: freeze_sensors (policy sees stale sensors; its own last action "
          "stays live)");
      } else {
        RCLCPP_WARN(get_logger(), "FAULT INJECTED: freeze_obs (policy sees a stale snapshot)");
      }
    }
    if (outcome != phm_go2::StepOutcome::kPublished) {
      return;
    }
    infer_ms_.push_back(runner_->last_run_ms());
    const float * latent = stepper_->latent();
    for (std::size_t i = 0; i < msg_.embedding.size(); ++i) {
      msg_.embedding[i] = latent[i];
    }
    msg_.header.stamp = get_clock()->now();
    pub_->publish(msg_);
    ++n_pub_;
    if (stale_) {
      publish_stale(msg_.header.stamp);
    }
    const auto elapsed = std::chrono::steady_clock::now() - t_start;
    tick_ms_.push_back(std::chrono::duration<double, std::milli>(elapsed).count());

    if (now - stats_t_ >= stats_every_) {
      RCLCPP_INFO(
        get_logger(),
        "published %zu; onnx p50 %.3f ms p99 %.3f ms; tick p50 %.3f ms p99 %.3f ms; lowstate "
        "age %.1f ms", n_pub_, phm_core::percentile(infer_ms_, 50.0),
        phm_core::percentile(infer_ms_, 99.0), phm_core::percentile(tick_ms_, 50.0),
        phm_core::percentile(tick_ms_, 99.0), (now - latest_rx_) * 1e3);
      infer_ms_.clear();
      tick_ms_.clear();
      stats_t_ = now;
    }
  }

  void publish_stale(const builtin_interfaces::msg::Time & stamp)
  {
    const auto & in = stepper_->input();
    auto & x = stale_sample_.values;
    for (std::size_t i = 0; i < 3; ++i) {
      x[i] = in.gyro[i];
    }
    for (std::size_t i = 0; i < phm_go2::kJoints; ++i) {
      x[3 + i] = in.q[i];
      x[15 + i] = in.dq[i];
    }
    const auto v = stale_->update(stale_sample_);
    if (!v) {
      return;
    }
    phm_msgs::msg::DetectorVerdict out;
    out.header.stamp = stamp;
    out.source = v->source;
    out.score = static_cast<float>(v->score);
    out.violating = v->violating;
    out.reason = v->reason;
    out.suggested_action = v->suggested_action;
    verdict_pub_->publish(out);
  }

  std::string policy_id_;
  double fault_after_ = -1.0;
  double stats_every_ = 10.0;
  std::unique_ptr<phm_go2::PolicyRunner> runner_;
  std::unique_ptr<phm_go2::ShadowStepper> stepper_;
  phm_go2::LowStateSample latest_;
  bool have_latest_ = false;
  double latest_rx_ = 0.0;
  double t0_ = 0.0;
  double stats_t_ = 0.0;
  std::size_t n_pub_ = 0;
  std::vector<double> infer_ms_;
  std::vector<double> tick_ms_;
  phm_msgs::msg::PolicyEmbedding msg_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
  rclcpp::Publisher<phm_msgs::msg::PolicyEmbedding>::SharedPtr pub_;
  std::unique_ptr<phm_core::StaleSignalAdapter> stale_;
  phm_core::StaleSignalSample stale_sample_;
  rclcpp::Publisher<phm_msgs::msg::DetectorVerdict>::SharedPtr verdict_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<PhoenixShadowEmbedder>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("phoenix_shadow_embedder"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
