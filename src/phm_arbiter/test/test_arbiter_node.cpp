// Copyright 2026 Yusuf Guenena. MIT License.
// The arbiter on a real rclcpp graph. Regression for the first live-graph
// failure: the Python arbiter set an attribute on the received (slotted)
// message and died on the first verdict, so /phm/health was never published;
// unit tests with plain objects could not see it. Here real DetectorVerdict
// messages travel over DDS and /phm/health must come out, including to a
// transient-local subscriber that joins late.
#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "phm_arbiter/arbiter_node.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_health_status.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;
using phm_msgs::msg::DetectorVerdict;
using phm_msgs::msg::PolicyHealthStatus;

class ArbiterGraph : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    rclcpp::NodeOptions opts;
    opts.parameter_overrides({{"staleness_sec", 0.3}, {"timer_period", 0.02}});
    node_ = std::make_shared<phm_arbiter::ArbiterNode>(opts);
    peer_ = std::make_shared<rclcpp::Node>("arbiter_test_peer");
    pub_ = peer_->create_publisher<DetectorVerdict>("/phm/verdicts", rclcpp::QoS(10).reliable());
    sub_ = peer_->create_subscription<PolicyHealthStatus>(
      "/phm/health", rclcpp::QoS(1).reliable().transient_local(),
      [this](PolicyHealthStatus::ConstSharedPtr m) {health_.push_back(*m);});
    exec_.add_node(node_);
    exec_.add_node(peer_);
    spin_until([this]() {return pub_->get_subscription_count() > 0;});
  }

  void TearDown() override
  {
    exec_.remove_node(node_);
    exec_.remove_node(peer_);
  }

  bool spin_until(const std::function<bool()> & done, std::chrono::milliseconds limit = 5s)
  {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < end) {
      exec_.spin_some(10ms);
      if (done()) {
        return true;
      }
    }
    return false;
  }

  void send(const std::string & source, float score, bool violating, uint8_t action = 0)
  {
    DetectorVerdict v;
    v.source = source;
    v.score = score;
    v.violating = violating;
    v.reason = source + " test";
    v.suggested_action = action;
    pub_->publish(v);
  }

  rclcpp::executors::SingleThreadedExecutor exec_;
  std::shared_ptr<phm_arbiter::ArbiterNode> node_;
  rclcpp::Node::SharedPtr peer_;
  rclcpp::Publisher<DetectorVerdict>::SharedPtr pub_;
  rclcpp::Subscription<PolicyHealthStatus>::SharedPtr sub_;
  std::vector<PolicyHealthStatus> health_;
};

TEST_F(ArbiterGraph, LiveVerdictReachesHealthOutput)
{
  send("freq:/lowstate", 0.0f, false);
  send("phm_ood_cpp", 0.9f, true, 3);
  ASSERT_TRUE(
    spin_until([this]() {return !health_.empty() && health_.back().state == 3;}));
  EXPECT_EQ(health_.back().source, "phm_ood_cpp");
  EXPECT_EQ(health_.back().suggested_action, 3);
  EXPECT_NE(health_.back().header.stamp.sec, 0);
}

TEST_F(ArbiterGraph, HealthyVerdictsPublishOkThenGoStale)
{
  send("freq:/lowstate", 0.0f, false);
  ASSERT_TRUE(spin_until([this]() {return !health_.empty() && health_.back().state == 0;}));
  EXPECT_EQ(health_.back().reason, "all detectors nominal");
  // No fresh verdict for longer than staleness_sec: DEGRADED, never dropped.
  ASSERT_TRUE(spin_until([this]() {return health_.back().state == 1;}, 3s));
  EXPECT_EQ(health_.back().reason, "stale:freq:/lowstate");
}

TEST_F(ArbiterGraph, NonFiniteScoreIsDegradedWithFiniteScore)
{
  send("phm_ood", std::numeric_limits<float>::quiet_NaN(), true);
  ASSERT_TRUE(spin_until([this]() {return !health_.empty() && health_.back().state == 1;}));
  EXPECT_EQ(health_.back().reason, "bad-score:phm_ood");
  EXPECT_FLOAT_EQ(health_.back().score, 0.5f);
}

TEST_F(ArbiterGraph, LateTransientLocalSubscriberGetsCurrentState)
{
  send("phm_ood_cpp", 0.9f, true, 3);
  ASSERT_TRUE(spin_until([this]() {return !health_.empty() && health_.back().state == 3;}));
  std::vector<PolicyHealthStatus> late;
  auto late_sub = peer_->create_subscription<PolicyHealthStatus>(
    "/phm/health", rclcpp::QoS(1).reliable().transient_local(),
    [&late](PolicyHealthStatus::ConstSharedPtr m) {late.push_back(*m);});
  ASSERT_TRUE(spin_until([&late]() {return !late.empty();}));
  EXPECT_EQ(late.front().state, 3);
}
