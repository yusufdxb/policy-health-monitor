// Copyright 2026 Yusuf Guenena. MIT License.
// phm_arbiter node: fuses phm_msgs/DetectorVerdict from /phm/verdicts into one
// phm_msgs/PolicyHealthStatus on /phm/health with phm_core::arbitrate
// (worst-wins, stale-never-de-escalates, bad-score sentinel).
//
// Exactly one instance publishes /phm/health. Each source's latest verdict is
// kept with its monotonic receive time, which is the staleness timestamp (the
// detectors' header stamps may be unset). A timer on the node clock
// (timer_period, default 0.05 s = 20 Hz) arbitrates and publishes.
//
// QoS: /phm/verdicts reliable keep-last 10 volatile; /phm/health reliable
// keep-last 1 transient-local, so a late subscriber (the recovery node) gets
// the current state immediately; it must also be transient-local.
#ifndef PHM_ARBITER__ARBITER_NODE_HPP_
#define PHM_ARBITER__ARBITER_NODE_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "phm_core/arbiter.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_health_status.hpp"
#include "rclcpp/rclcpp.hpp"

namespace phm_arbiter
{

class ArbiterNode : public rclcpp::Node
{
public:
  explicit ArbiterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // Arbitrate and publish now (the timer callback; exposed for tests).
  void arbitrate_and_publish();
  const phm_core::HealthStatusData & last_result() const {return result_;}

private:
  void on_verdict(const phm_msgs::msg::DetectorVerdict::ConstSharedPtr & msg);
  static double monotonic_now();

  double staleness_sec_;
  // Latest verdict per source, in first-seen order (tie-break order).
  std::vector<phm_core::ArbiterInput> latest_;
  std::unordered_map<std::string, std::size_t> index_;
  phm_core::HealthStatusData result_;
  phm_msgs::msg::PolicyHealthStatus out_;

  rclcpp::Subscription<phm_msgs::msg::DetectorVerdict>::SharedPtr sub_;
  rclcpp::Publisher<phm_msgs::msg::PolicyHealthStatus>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace phm_arbiter

#endif  // PHM_ARBITER__ARBITER_NODE_HPP_
