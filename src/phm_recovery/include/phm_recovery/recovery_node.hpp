// Copyright 2026 Yusuf Guenena. MIT License.
// phm_recovery node: turns INTERVENE / STOP on /phm/health into zero-velocity
// holds on /phm/cmd_vel (geometry_msgs/Twist), re-published at publish_hz while
// the hold is actuating. All decisions are phm_core::RecoveryController
// (SafetyEnvelope + HealthToActionMapper + RewindHook).
//
// Actuation is OFF by default (recovery.enabled: false): the node then
// subscribes and logs but never publishes a command. Publishing STOP on
// /phm/health alone does not stop a robot; only this node, enabled, does.
//
// QoS: /phm/health reliable keep-last 10 transient-local (must match the
// arbiter's transient-local publisher or every message is dropped);
// /phm/cmd_vel reliable keep-last 10 volatile.
//
// A host stack embedding this node registers its return-to-last-safe-waypoint
// behavior with rewind_hook().register_callback(); without one, REWIND logs a
// warning and holds.
#ifndef PHM_RECOVERY__RECOVERY_NODE_HPP_
#define PHM_RECOVERY__RECOVERY_NODE_HPP_

#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "phm_core/recovery.hpp"
#include "phm_msgs/msg/policy_health_status.hpp"
#include "rclcpp/rclcpp.hpp"

namespace phm_recovery
{

class RecoveryNode : public rclcpp::Node
{
public:
  explicit RecoveryNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  phm_core::RewindHook & rewind_hook() {return controller_->rewind_hook();}
  bool hold_actuating() const {return controller_->hold_actuating();}

private:
  void on_health(const phm_msgs::msg::PolicyHealthStatus::ConstSharedPtr & msg);
  void on_publish_tick();

  std::unique_ptr<phm_core::RecoveryController> controller_;
  rclcpp::Subscription<phm_msgs::msg::PolicyHealthStatus>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  geometry_msgs::msg::Twist zero_;
};

}  // namespace phm_recovery

#endif  // PHM_RECOVERY__RECOVERY_NODE_HPP_
