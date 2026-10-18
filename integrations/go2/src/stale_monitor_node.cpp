// Copyright 2026 Yusuf Guenena. MIT License.
// phm_go2_stale_monitor: the stale-input check on a live Unitree GO2's /lowstate,
// with no policy of its own.
//
// For a session where the controlling policy publishes its own latent (Phoenix
// with --phm-embedding-topic), the shadow embedder's policy is not wanted; this
// node keeps only its stale_check: the same phm_core::StaleSignalAdapter, the
// same channel groups (the 3 gyro axes together, q with dq of each motor), the
// same limits (stale_gyro_max_repeats / stale_joint_max_repeats, defaults 5 and
// 12) and the same sampling, the latest reading at each rate_hz tick. A group
// is stale on a tick when all its channels equal the previous tick's; it fires
// after more than its limit of stale ticks in a row. One DetectorVerdict per
// tick goes to /phm/verdicts. It never commands the robot.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/adapters.hpp"
#include "phm_go2/shadow_policy.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_state.hpp"

namespace
{

double mono()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

class StaleMonitor : public rclcpp::Node
{
public:
  StaleMonitor()
  : rclcpp::Node("phm_go2_stale_monitor")
  {
    const double rate_hz = declare_parameter<double>("rate_hz", 50.0);
    const int gyro_max = static_cast<int>(declare_parameter<int64_t>("stale_gyro_max_repeats", 5));
    const int joint_max =
      static_cast<int>(declare_parameter<int64_t>("stale_joint_max_repeats", 12));
    stats_every_ = declare_parameter<double>("stats_every_sec", 10.0);

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
    sample_.topic = "/lowstate";
    sample_.values.resize(3 + 2 * phm_go2::kJoints);

    const auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
    sub_ = create_subscription<unitree_go::msg::LowState>(
      "/lowstate", sensor_qos,
      [this](unitree_go::msg::LowState::ConstSharedPtr m) {on_lowstate(*m);});
    // Same QoS as phm_detectors' verdicts: reliable, volatile.
    pub_ = create_publisher<phm_msgs::msg::DetectorVerdict>(
      "/phm/verdicts", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_hz), [this]() {tick();});
    stats_t_ = mono();
    RCLCPP_INFO(
      get_logger(),
      "stale check on /lowstate at %.0f Hz: gyro max %d, joint max %d repeated ticks. NO "
      "actuation: only publisher is /phm/verdicts.", rate_hz, gyro_max, joint_max);
  }

private:
  void on_lowstate(const unitree_go::msg::LowState & m)
  {
    auto & x = sample_.values;
    for (std::size_t i = 0; i < 3; ++i) {
      x[i] = m.imu_state.gyroscope[i];
    }
    for (std::size_t i = 0; i < phm_go2::kJoints; ++i) {
      x[3 + i] = m.motor_state[i].q;
      x[15 + i] = m.motor_state[i].dq;
    }
    have_ = true;
    rx_ = mono();
  }

  void tick()
  {
    if (!have_) {
      return;
    }
    const auto v = stale_->update(sample_);
    ++n_ticks_;
    if (v) {
      phm_msgs::msg::DetectorVerdict out;
      out.header.stamp = get_clock()->now();
      out.source = v->source;
      out.score = static_cast<float>(v->score);
      out.violating = v->violating;
      out.reason = v->reason;
      out.suggested_action = v->suggested_action;
      pub_->publish(out);
      n_violating_ += v->violating ? 1 : 0;
    }
    const double now = mono();
    if (now - stats_t_ >= stats_every_) {
      RCLCPP_INFO(
        get_logger(), "ticks %zu, violating %zu; lowstate age %.1f ms", n_ticks_, n_violating_,
        (now - rx_) * 1e3);
      stats_t_ = now;
    }
  }

  double stats_every_ = 10.0;
  std::unique_ptr<phm_core::StaleSignalAdapter> stale_;
  phm_core::StaleSignalSample sample_;
  bool have_ = false;
  double rx_ = 0.0;
  double stats_t_ = 0.0;
  std::size_t n_ticks_ = 0;
  std::size_t n_violating_ = 0;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
  rclcpp::Publisher<phm_msgs::msg::DetectorVerdict>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<StaleMonitor>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("phm_go2_stale_monitor"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
