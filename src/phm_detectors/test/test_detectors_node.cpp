// Copyright 2026 Yusuf Guenena. MIT License.
// The detectors node on a real rclcpp graph: generic (raw) subscriptions bind
// to topics present on the graph, absent topics stay unbound and read as dead,
// a stopped publisher trips the frequency detector, and the optional startup
// grace holds back dead verdicts for topics not seen yet.
#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "phm_detectors/detectors_node.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using phm_msgs::msg::DetectorVerdict;

class DetectorsGraph : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    peer_ = std::make_shared<rclcpp::Node>("detectors_test_peer");
    pub_ = peer_->create_publisher<std_msgs::msg::String>("/phm_test/alive", 10);
    sub_ = peer_->create_subscription<DetectorVerdict>(
      "/phm/verdicts", rclcpp::QoS(200).reliable(),
      [this](DetectorVerdict::ConstSharedPtr m) {verdicts_.push_back(*m);});
    exec_.add_node(peer_);
    // Let the peer's publisher reach the graph before the node looks it up.
    pump(200ms, true);
    start_node(0.0);
  }

  // (Re)create the node under test with the given dead_startup_grace_sec.
  void start_node(double grace)
  {
    if (node_) {
      exec_.remove_node(node_);
    }
    rclcpp::NodeOptions opts;
    opts.parameter_overrides(
    {
      {"freq_topics", std::vector<std::string>{"/phm_test/alive"}},
      {"dead_topics", std::vector<std::string>{"/phm_test/alive", "/phm_test/absent"}},
      {"dead_timeout_sec", 0.3},
      {"dead_startup_grace_sec", grace},
      {"freq_min_consecutive", 2},
    });
    node_ = std::make_shared<phm_detectors::DetectorsNode>(opts);
    node_->set_metrics_reader(phm_core::SystemMetricsReader("/nonexistent", "/nonexistent"));
    exec_.add_node(node_);
    verdicts_.clear();
  }

  void TearDown() override
  {
    exec_.remove_node(node_);
    exec_.remove_node(peer_);
  }

  // Spin for `d`, publishing on /phm_test/alive at ~200 Hz when `publish`.
  void pump(std::chrono::milliseconds d, bool publish)
  {
    const auto end = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < end) {
      if (publish) {
        pub_->publish(std_msgs::msg::String());
      }
      exec_.spin_some(5ms);
    }
  }

  std::vector<DetectorVerdict> from(const std::string & source) const
  {
    std::vector<DetectorVerdict> out;
    for (const auto & v : verdicts_) {
      if (v.source == source) {
        out.push_back(v);
      }
    }
    return out;
  }

  rclcpp::executors::SingleThreadedExecutor exec_;
  rclcpp::Node::SharedPtr peer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::Subscription<DetectorVerdict>::SharedPtr sub_;
  std::shared_ptr<phm_detectors::DetectorsNode> node_;
  std::vector<DetectorVerdict> verdicts_;
};

TEST_F(DetectorsGraph, BindsPresentTopicAndReportsAbsentTopicDead)
{
  EXPECT_EQ(node_->bound_topics(), 1u);  // /phm_test/absent is not on the graph
  pump(500ms, true);
  node_->tick();
  pump(200ms, true);
  const auto alive = from("dead:/phm_test/alive");
  const auto absent = from("dead:/phm_test/absent");
  ASSERT_FALSE(alive.empty());
  ASSERT_FALSE(absent.empty());
  EXPECT_FALSE(alive.back().violating) << alive.back().reason;
  EXPECT_TRUE(absent.back().violating) << absent.back().reason;
  EXPECT_EQ(absent.back().suggested_action, 3);
}

TEST_F(DetectorsGraph, StoppedPublisherTripsFrequencyDetector)
{
  // Learn the baseline over 10 short windows while the topic publishes.
  for (int i = 0; i < 12; ++i) {
    pump(100ms, true);
    node_->tick();
  }
  pump(100ms, true);
  const auto healthy = from("freq:/phm_test/alive");
  ASSERT_FALSE(healthy.empty());
  EXPECT_FALSE(healthy.back().violating) << healthy.back().reason;
  // Silence: two consecutive empty windows fire the detector.
  for (int i = 0; i < 3; ++i) {
    pump(100ms, false);
    node_->tick();
  }
  pump(100ms, false);
  const auto dropped = from("freq:/phm_test/alive");
  EXPECT_TRUE(dropped.back().violating) << dropped.back().reason;
  EXPECT_NE(dropped.back().reason.find("below floor"), std::string::npos);
}

TEST_F(DetectorsGraph, StartupGraceHoldsBackUnseenTopicThenReportsItDead)
{
  start_node(1.0);
  // Past the 0.3 s timeout but inside the 1 s grace: the absent topic gets no
  // dead verdict, the topic that is arriving is judged normally.
  pump(500ms, true);
  node_->tick();
  pump(100ms, true);
  EXPECT_TRUE(from("dead:/phm_test/absent").empty());
  const auto alive = from("dead:/phm_test/alive");
  ASSERT_FALSE(alive.empty());
  EXPECT_FALSE(alive.back().violating) << alive.back().reason;
  // After the grace the never-seen topic is dead, timed from startup.
  pump(600ms, true);
  node_->tick();
  pump(100ms, true);
  const auto absent = from("dead:/phm_test/absent");
  ASSERT_FALSE(absent.empty());
  EXPECT_TRUE(absent.back().violating) << absent.back().reason;
  EXPECT_EQ(absent.back().suggested_action, 3);
}

TEST_F(DetectorsGraph, StartupGraceDoesNotMaskATopicThatWentSilent)
{
  start_node(5.0);
  // Seen, then silent longer than the timeout: dead even inside the grace.
  pump(300ms, true);
  pump(500ms, false);
  node_->tick();
  pump(100ms, false);
  const auto alive = from("dead:/phm_test/alive");
  ASSERT_FALSE(alive.empty());
  EXPECT_TRUE(alive.back().violating) << alive.back().reason;
}
