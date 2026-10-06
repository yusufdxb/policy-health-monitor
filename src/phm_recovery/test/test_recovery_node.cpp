// Copyright 2026 Yusuf Guenena. MIT License.
// The recovery node on a real rclcpp graph: zero-velocity holds on
// /phm/cmd_vel only when enabled, release on recovery, cooldown damping of a
// new hold, the rewind hook, and the transient-local health subscription.
#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "phm_msgs/msg/policy_health_status.hpp"
#include "phm_recovery/recovery_node.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using geometry_msgs::msg::Twist;
using phm_msgs::msg::PolicyHealthStatus;

class RecoveryGraph : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void start(bool enabled, double cooldown = 5.0)
  {
    rclcpp::NodeOptions opts;
    opts.parameter_overrides(
    {
      {"recovery.enabled", enabled},
      {"recovery.cooldown_seconds", cooldown},
      {"recovery.publish_hz", 50.0},
    });
    node_ = std::make_shared<phm_recovery::RecoveryNode>(opts);
    peer_ = std::make_shared<rclcpp::Node>("recovery_test_peer");
    pub_ = peer_->create_publisher<PolicyHealthStatus>(
      "/phm/health", rclcpp::QoS(1).reliable().transient_local());
    sub_ = peer_->create_subscription<Twist>(
      "/phm/cmd_vel", rclcpp::QoS(10).reliable(),
      [this](Twist::ConstSharedPtr m) {twists_.push_back(*m);});
    exec_.add_node(node_);
    exec_.add_node(peer_);
    spin(300ms);
  }

  void TearDown() override
  {
    if (node_) {
      exec_.remove_node(node_);
      exec_.remove_node(peer_);
    }
  }

  void spin(std::chrono::milliseconds d)
  {
    const auto end = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < end) {
      exec_.spin_some(5ms);
    }
  }

  void health(uint8_t state, uint8_t action, const std::string & source = "phm_ood")
  {
    PolicyHealthStatus h;
    h.state = state;
    h.suggested_action = action;
    h.source = source;
    h.reason = "test";
    pub_->publish(h);
    spin(200ms);
  }

  rclcpp::executors::SingleThreadedExecutor exec_;
  std::shared_ptr<phm_recovery::RecoveryNode> node_;
  rclcpp::Node::SharedPtr peer_;
  rclcpp::Publisher<PolicyHealthStatus>::SharedPtr pub_;
  rclcpp::Subscription<Twist>::SharedPtr sub_;
  std::vector<Twist> twists_;
};

TEST_F(RecoveryGraph, DisabledNeverPublishes)
{
  start(false);
  health(3, 3);
  spin(300ms);
  EXPECT_FALSE(node_->hold_actuating());
  EXPECT_TRUE(twists_.empty());
}

TEST_F(RecoveryGraph, StopHoldsWithZeroVelocityUntilOk)
{
  start(true);
  health(3, 3);
  ASSERT_TRUE(node_->hold_actuating());
  spin(300ms);
  ASSERT_GE(twists_.size(), 5u);  // re-published at publish_hz
  for (const auto & t : twists_) {
    EXPECT_EQ(t.linear.x, 0.0);
    EXPECT_EQ(t.angular.z, 0.0);
  }
  health(0, 0);
  EXPECT_FALSE(node_->hold_actuating());
  twists_.clear();
  spin(300ms);
  EXPECT_TRUE(twists_.empty());
}

TEST_F(RecoveryGraph, NewHoldWithinCooldownIsDampedButContinuationIsNot)
{
  start(true, 5.0);
  health(2, 2, "freq:/x");
  EXPECT_TRUE(node_->hold_actuating());
  health(2, 2, "freq:/x");  // continuation of the active hold
  EXPECT_TRUE(node_->hold_actuating());
  health(0, 0, "freq:/x");
  EXPECT_FALSE(node_->hold_actuating());
  health(2, 2, "freq:/x");  // a new hold within 5 s of the last one: damped
  EXPECT_FALSE(node_->hold_actuating());
}

TEST_F(RecoveryGraph, RewindInvokesHookAndHolds)
{
  start(true);
  int calls = 0;
  node_->rewind_hook().register_callback([&calls]() {++calls;});
  health(2, 4);
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(node_->hold_actuating());
}
