// Copyright 2026 Yusuf Guenena. MIT License.
// rclcpp node: subscribes phm_msgs/PolicyEmbedding on /policy/embedding,
// maintains a rolling window, computes rolling_spread in C++ via a pluggable
// Backend, applies a calibrated threshold with consecutive-hysteresis, and
// publishes phm_msgs/DetectorVerdict on /phm/verdicts with source "phm_ood_cpp".
//
// Decision logic lives in OodCore (ood_core.hpp), which mirrors the Python rclpy
// detector phm_ood/phm_ood/_core.py. This file is the ROS adapter only.
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"

#include "phm_ood_cpp/backend.hpp"
#include "phm_ood_cpp/ood_core.hpp"

namespace phm_ood_cpp
{

class OodNode : public rclcpp::Node
{
public:
  OodNode()
  : rclcpp::Node("phm_ood_cpp")
  {
    // Parameters (calibrated threshold is a param, per spec).
    const int window = static_cast<int>(declare_parameter<int64_t>("window", 30));
    const double threshold = declare_parameter<double>("threshold", 0.0);
    const int min_consecutive =
      static_cast<int>(declare_parameter<int64_t>("min_consecutive", 2));
    const int compute_every =
      static_cast<int>(declare_parameter<int64_t>("compute_every", 1));

    core_ = std::make_unique<OodCore>(
      static_cast<std::size_t>(window), threshold, min_consecutive,
      compute_every, make_default_backend());

    // Explicit QoS. Embeddings are a high-rate sensor-like stream: keep-last
    // depth 10, reliable, volatile. Verdicts: keep-last 10, reliable, volatile
    // so a late-joining arbiter does not replay stale faults.
    rclcpp::QoS emb_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    rclcpp::QoS verdict_qos =
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    pub_ = create_publisher<phm_msgs::msg::DetectorVerdict>("/phm/verdicts", verdict_qos);
    sub_ = create_subscription<phm_msgs::msg::PolicyEmbedding>(
      "/policy/embedding", emb_qos,
      std::bind(&OodNode::on_embedding, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "phm_ood_cpp up: backend=%s window=%d threshold=%.4f min_consecutive=%d "
      "compute_every=%d, /policy/embedding -> /phm/verdicts",
      core_->backend_name().c_str(), window, threshold, min_consecutive, compute_every);
  }

private:
  void on_embedding(const phm_msgs::msg::PolicyEmbedding::SharedPtr msg)
  {
    // Validate against the message's own dim field (PolicyEmbedding.msg:3).
    if (msg->dim != msg->embedding.size()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "PolicyEmbedding dim=%u != embedding length=%zu; dropping frame",
        msg->dim, msg->embedding.size());
      return;
    }
    if (msg->embedding.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "empty embedding; dropping frame");
      return;
    }

    VerdictData v;
    try {
      v = core_->update(msg->embedding, msg->policy_id);
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "core->update failed: %s", e.what());
      return;
    }

    phm_msgs::msg::DetectorVerdict out;
    // Carry the embedding's stamp through so downstream latency is measurable;
    // fall back to now() if the producer left it unstamped.
    if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
      out.header.stamp = now();
    } else {
      out.header.stamp = msg->header.stamp;
    }
    out.header.frame_id = msg->header.frame_id;
    out.source = v.source;
    out.score = static_cast<float>(v.score);
    out.violating = v.violating;
    out.reason = v.reason;
    out.suggested_action = v.suggested_action;
    pub_->publish(out);
  }

  std::unique_ptr<OodCore> core_;
  rclcpp::Publisher<phm_msgs::msg::DetectorVerdict>::SharedPtr pub_;
  rclcpp::Subscription<phm_msgs::msg::PolicyEmbedding>::SharedPtr sub_;
};

}  // namespace phm_ood_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<phm_ood_cpp::OodNode>());
  rclcpp::shutdown();
  return 0;
}
