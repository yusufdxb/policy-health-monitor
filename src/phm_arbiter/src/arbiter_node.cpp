// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_arbiter/arbiter_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <string>

#include "phm_core/format.hpp"
#include "phm_core/severity.hpp"

namespace phm_arbiter
{

double ArbiterNode::monotonic_now()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

ArbiterNode::ArbiterNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("phm_arbiter", options)
{
  staleness_sec_ = declare_parameter<double>("staleness_sec", 1.0);
  const double timer_period = declare_parameter<double>("timer_period", 0.05);

  sub_ = create_subscription<phm_msgs::msg::DetectorVerdict>(
    "/phm/verdicts", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile(),
    std::bind(&ArbiterNode::on_verdict, this, std::placeholders::_1));
  pub_ = create_publisher<phm_msgs::msg::PolicyHealthStatus>(
    "/phm/health", rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Duration::from_seconds(timer_period),
    [this]() {arbitrate_and_publish();});

  RCLCPP_INFO(
    get_logger(), "phm_arbiter started (staleness=%ss, period=%ss)",
    phm_core::fmt::repr(staleness_sec_).c_str(), phm_core::fmt::repr(timer_period).c_str());
}

void ArbiterNode::on_verdict(const phm_msgs::msg::DetectorVerdict::ConstSharedPtr & msg)
{
  // Last write wins per source. Assigning into the stored entry reuses its
  // string capacity, so steady state does not allocate.
  auto it = index_.find(msg->source);
  if (it == index_.end()) {
    it = index_.emplace(msg->source, latest_.size()).first;
    latest_.emplace_back();
    latest_.back().source = msg->source;
  }
  phm_core::ArbiterInput & v = latest_[it->second];
  v.score = static_cast<double>(msg->score);
  v.violating = msg->violating;
  v.reason = msg->reason;
  v.suggested_action = msg->suggested_action;
  v.timestamp = monotonic_now();
  v.has_timestamp = true;
}

void ArbiterNode::arbitrate_and_publish()
{
  for (const auto & v : latest_) {
    if (!std::isfinite(v.score)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "non-finite score %f from source '%s'; treating detector as DEGRADED (bad-score "
        "sentinel)", v.score, v.source.c_str());
    }
  }
  result_ = phm_core::arbitrate(latest_, monotonic_now(), staleness_sec_);
  // Belt and braces: never publish a non-finite score.
  if (!std::isfinite(result_.score)) {
    result_.score = 0.5;
  }
  out_.header.stamp = get_clock()->now();
  out_.state = result_.state;
  out_.score = static_cast<float>(result_.score);
  out_.reason = result_.reason;
  out_.source = result_.source;
  out_.suggested_action = result_.suggested_action;
  pub_->publish(out_);

  if (result_.state != phm_core::STATE_OK) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "health=%d score=%.3f source='%s' reason='%s'",
      result_.state, result_.score, result_.source.c_str(), result_.reason.c_str());
  }
}

}  // namespace phm_arbiter
