// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_recovery/recovery_node.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "phm_core/format.hpp"
#include "phm_core/severity.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"

namespace phm_recovery
{

namespace
{
rcl_interfaces::msg::ParameterDescriptor describe(const std::string & text)
{
  rcl_interfaces::msg::ParameterDescriptor d;
  d.description = text;
  d.read_only = false;
  return d;
}
}  // namespace

RecoveryNode::RecoveryNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("phm_recovery_node", options)
{
  const bool enabled = declare_parameter<bool>(
    "recovery.enabled", false,
    describe("Master enable for actuation. False -> no commands published."));
  const double cooldown = declare_parameter<double>(
    "recovery.cooldown_seconds", 5.0,
    describe("Minimum seconds between repeated HOLD/STOP_AND_HOLD for the same fault_key."));
  const double publish_hz = declare_parameter<double>(
    "recovery.publish_hz", 20.0,
    describe("Rate (Hz) at which zero-velocity Twist is re-published while hold is active."));

  controller_ = std::make_unique<phm_core::RecoveryController>(enabled, cooldown);
  controller_->rewind_hook().set_logger(
    [this](phm_core::LogLevel, const std::string & m) {
      RCLCPP_WARN(get_logger(), "%s", m.c_str());
    });

  sub_ = create_subscription<phm_msgs::msg::PolicyHealthStatus>(
    "/phm/health", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local(),
    std::bind(&RecoveryNode::on_health, this, std::placeholders::_1));
  pub_ = create_publisher<geometry_msgs::msg::Twist>(
    "/phm/cmd_vel", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
  // Guard against a zero or negative rate.
  const double period = 1.0 / std::max(publish_hz, 0.1);
  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Duration::from_seconds(period), [this]() {on_publish_tick();});

  RCLCPP_INFO(
    get_logger(), "PHM RecoveryNode started: enabled=%s cooldown=%ss publish_hz=%sHz",
    enabled ? "True" : "False", phm_core::fmt::repr(cooldown).c_str(),
    phm_core::fmt::repr(publish_hz).c_str());
}

void RecoveryNode::on_health(const phm_msgs::msg::PolicyHealthStatus::ConstSharedPtr & msg)
{
  const double now = get_clock()->now().seconds();
  const auto step = controller_->on_health(
    msg->state, msg->suggested_action, msg->source, msg->reason, now);
  if (step.resume_evaluated) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "RESUME: %s", step.result->reason.c_str());
    return;
  }
  if (!step.result) {
    return;  // ACTION_NONE
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 1000, "envelope: action=%d status=%s reason=%s",
    step.decision.action, phm_core::to_string(step.result->status),
    step.result->reason.c_str());
  if (!step.result->publish) {
    return;
  }
  const uint8_t action = step.decision.action;
  if (action == phm_core::ACTION_HOLD || action == phm_core::ACTION_STOP_AND_HOLD) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000, "HOLD active: %s", step.decision.reason.c_str());
  } else if (action == phm_core::ACTION_REWIND) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000, "REWIND triggered: %s", step.decision.reason.c_str());
  } else if (action == phm_core::ACTION_LOG_ONLY) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "LOG_ONLY: %s", step.decision.reason.c_str());
  }
}

void RecoveryNode::on_publish_tick()
{
  // Zero velocity follows the envelope-coupled actuation state, so the
  // cooldown gate decides what is actually published.
  if (controller_->hold_actuating()) {
    pub_->publish(zero_);
  }
}

}  // namespace phm_recovery
