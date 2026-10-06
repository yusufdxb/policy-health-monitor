// Copyright 2026 Yusuf Guenena. MIT License.
// The OOD lifecycle node on a real rclcpp graph (both interface profiles):
// lifecycle gating, verdict publication, stamps, malformed-frame handling,
// deactivate semantics and calibration-file loading.
#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lifecycle_msgs/msg/state.hpp"
#include "phm_core/npy.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"
#include "phm_ood_cpp/ood_lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"

using phm_msgs::msg::DetectorVerdict;
using phm_msgs::msg::PolicyEmbedding;
using phm_ood_cpp::OodLifecycleNode;
using phm_ood_cpp::OodNodeProfile;
using namespace std::chrono_literals;

namespace
{

class Graph : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    peer_ = std::make_shared<rclcpp::Node>("ood_test_peer");
    pub_ = peer_->create_publisher<PolicyEmbedding>(
      "/policy/embedding", rclcpp::QoS(10).reliable());
    sub_ = peer_->create_subscription<DetectorVerdict>(
      "/phm/verdicts", rclcpp::QoS(100).reliable(),
      [this](DetectorVerdict::ConstSharedPtr m) {verdicts_.push_back(*m);});
    exec_.add_node(peer_);
  }

  void TearDown() override
  {
    exec_.remove_node(peer_);
    if (node_) {
      exec_.remove_node(node_->get_node_base_interface());
    }
  }

  std::shared_ptr<OodLifecycleNode> make(
    OodNodeProfile profile, std::vector<rclcpp::Parameter> params)
  {
    rclcpp::NodeOptions opts;
    opts.parameter_overrides(std::move(params));
    node_ = std::make_shared<OodLifecycleNode>(std::move(profile), opts);
    exec_.add_node(node_->get_node_base_interface());
    return node_;
  }

  bool spin_until(const std::function<bool()> & done, std::chrono::milliseconds limit = 5s)
  {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < end) {
      exec_.spin_some(20ms);
      if (done()) {
        return true;
      }
    }
    return false;
  }

  // Publish frames until the node has matched our publisher, then a burst.
  void send(const std::vector<std::vector<float>> & frames, const std::string & pid = "")
  {
    spin_until([this]() {return pub_->get_subscription_count() > 0;}, 5s);
    int32_t sec = 100;
    for (const auto & f : frames) {
      PolicyEmbedding m;
      m.header.stamp.sec = sec++;
      m.header.frame_id = "base";
      m.embedding = f;
      m.dim = static_cast<uint32_t>(f.size());
      m.policy_id = pid;
      pub_->publish(m);
      exec_.spin_some(5ms);
    }
  }

  rclcpp::executors::SingleThreadedExecutor exec_;
  rclcpp::Node::SharedPtr peer_;
  rclcpp::Publisher<PolicyEmbedding>::SharedPtr pub_;
  rclcpp::Subscription<DetectorVerdict>::SharedPtr sub_;
  std::shared_ptr<OodLifecycleNode> node_;
  std::vector<DetectorVerdict> verdicts_;
};

std::vector<std::vector<float>> constant(std::size_t n)
{
  return std::vector<std::vector<float>>(n, std::vector<float>{1.0f, 2.0f, 3.0f});
}

}  // namespace

TEST_F(Graph, CppProfileGatesOnLifecycleAndCarriesInputStamp)
{
  auto node = make(
    OodNodeProfile::phm_ood_cpp(),
    {{"window", 3}, {"threshold", 0.5}, {"min_consecutive", 1}});
  // Unconfigured: no endpoint, nothing published.
  send(constant(3));
  exec_.spin_some(50ms);
  EXPECT_TRUE(verdicts_.empty());

  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  send(constant(3));  // inactive: subscription exists, frames are not scored
  exec_.spin_some(50ms);
  EXPECT_TRUE(verdicts_.empty());

  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  send(constant(3), "p");
  ASSERT_TRUE(spin_until([this]() {return verdicts_.size() >= 3;}));
  EXPECT_EQ(verdicts_[0].source, "phm_ood_cpp");
  EXPECT_EQ(verdicts_[0].reason, "warming up: 1/3 frames");
  EXPECT_TRUE(verdicts_[2].violating);  // constant frames: spread 0 < 0.5
  EXPECT_EQ(verdicts_[2].suggested_action, 3);
  EXPECT_EQ(verdicts_[2].header.stamp.sec, verdicts_[0].header.stamp.sec + 2);
  EXPECT_EQ(verdicts_[2].header.frame_id, "base");

  // Malformed frames are dropped: dim field mismatch and dimension change.
  const std::size_t before = verdicts_.size();
  PolicyEmbedding bad;
  bad.embedding = {1.0f, 2.0f, 3.0f};
  bad.dim = 7;
  pub_->publish(bad);
  send({{1.0f, 2.0f}});
  exec_.spin_some(100ms);
  EXPECT_EQ(verdicts_.size(), before);

  // Deactivate resets the window: after re-activation it warms up again.
  ASSERT_EQ(node->deactivate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  verdicts_.clear();
  send(constant(1));
  ASSERT_TRUE(spin_until([this]() {return !verdicts_.empty();}));
  EXPECT_EQ(verdicts_[0].reason, "warming up: 1/3 frames");
}

TEST_F(Graph, CppProfileRejectsOutOfRangeWindowAtConfigureTime)
{
  // A window of 1 is outside the declared range, so the override is refused
  // when the node declares its parameters.
  EXPECT_THROW(
    make(OodNodeProfile::phm_ood_cpp(), {{"window", 1}}), std::exception);
  node_.reset();
}

TEST_F(Graph, PhmOodProfileStampsNowPublishesBadInputAndKeepsWindow)
{
  auto node = make(
    OodNodeProfile::phm_ood(),
    {{"window", 3}, {"threshold", 0.5}, {"hysteresis_count", 1}, {"embed_dim", 3}});
  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  send(constant(2));
  send({{1.0f, 2.0f, 3.0f, 4.0f}});
  ASSERT_TRUE(spin_until([this]() {return verdicts_.size() >= 3;}));
  EXPECT_EQ(verdicts_[0].source, "phm_ood");
  EXPECT_NE(verdicts_[0].header.stamp.sec, 100);  // stamped now(), not the input
  EXPECT_EQ(verdicts_[0].header.frame_id, "");
  EXPECT_EQ(verdicts_[2].reason, "dim mismatch: expected 3, got 4");
  EXPECT_FLOAT_EQ(verdicts_[2].score, 0.5f);
  EXPECT_FALSE(verdicts_[2].violating);

  // Deactivate keeps the window: two frames before, one after, fills it.
  ASSERT_EQ(node->deactivate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  verdicts_.clear();
  send(constant(1));
  ASSERT_TRUE(spin_until([this]() {return !verdicts_.empty();}));
  EXPECT_TRUE(verdicts_[0].violating) << verdicts_[0].reason;

  // Cleanup then configure again works (parameters are declared once).
  ASSERT_EQ(node->deactivate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node->cleanup().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
}

TEST_F(Graph, PhmOodProfileLoadsCalibrationFileAndFailsBadWindow)
{
  char path[] = "/tmp/phm_ood_calib_XXXXXX";
  const int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  close(fd);
  phm_core::npy::save_npz(path, {{"threshold", phm_core::npy::Array::scalar_double(0.25)}});
  auto node = make(
    OodNodeProfile::phm_ood(), {{"window", 2}, {"calibration_file", std::string(path)}});
  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_DOUBLE_EQ(node->core()->threshold(), 0.25);
  std::remove(path);
  exec_.remove_node(node->get_node_base_interface());

  auto bad = make(OodNodeProfile::phm_ood(), {{"window", 1}});
  EXPECT_EQ(bad->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  exec_.remove_node(bad->get_node_base_interface());

  // phm_ood declares its parameters without ranges (as the Python node did),
  // so configure enforces the bounds: a window too large to ever fill, and
  // int64 values that would wrap when narrowed to int, fail configure.
  for (const auto & p : std::vector<rclcpp::Parameter>{
      {"window", int64_t{1} << 62}, {"window", int64_t{100001}},
      {"hysteresis_count", (int64_t{1} << 32) + 3}, {"compute_every", int64_t{0}}})
  {
    auto n = make(OodNodeProfile::phm_ood(), {p});
    EXPECT_EQ(n->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED)
      << p.get_name() << "=" << p.value_to_string();
    exec_.remove_node(n->get_node_base_interface());
  }
  node_.reset();  // already removed from the executor above
}
