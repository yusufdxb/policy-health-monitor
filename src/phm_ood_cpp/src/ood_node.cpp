// Copyright 2026 Yusuf Guenena. MIT License.
// Managed-lifecycle rclcpp node: subscribes phm_msgs/PolicyEmbedding on
// /policy/embedding, maintains a rolling window, computes rolling_spread in C++
// via a pluggable Backend, applies a calibrated threshold with
// consecutive-hysteresis, and publishes phm_msgs/DetectorVerdict on
// /phm/verdicts with source "phm_ood_cpp".
//
// This is a LifecycleNode, not a plain Node, because a safety monitor must have
// deterministic bring-up: parameters and the detector core are built in
// on_configure, the verdict publisher only emits while ACTIVE, and a supervisor
// can deactivate the monitor without tearing down the process. Frames that
// arrive while the node is not ACTIVE are dropped rather than silently buffered,
// so the rolling window never mixes pre- and post-activation state.
//
// Decision logic lives in OodCore (ood_core.hpp), which mirrors the Python rclpy
// detector phm_ood/phm_ood/_core.py. This file is the ROS adapter only.
#include <memory>
#include <string>

#include "rcl_interfaces/msg/floating_point_range.hpp"
#include "rcl_interfaces/msg/integer_range.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"

#include "phm_ood_cpp/backend.hpp"
#include "phm_ood_cpp/ood_core.hpp"

namespace phm_ood_cpp
{

using CallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using LifecycleState = rclcpp_lifecycle::State;

class OodNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit OodNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp_lifecycle::LifecycleNode("phm_ood_cpp", options)
  {
    // Parameters are declared (not consumed) in the constructor so they are
    // introspectable before configuration. They are read in on_configure.
    rcl_interfaces::msg::ParameterDescriptor window_desc;
    window_desc.description =
      "Frames in the rolling covariance window (>= 2). Read at configure time.";
    window_desc.read_only = true;
    rcl_interfaces::msg::IntegerRange window_range;
    window_range.from_value = 2;
    window_range.to_value = 100000;
    window_range.step = 1;
    window_desc.integer_range.push_back(window_range);
    declare_parameter<int64_t>("window", 30, window_desc);

    rcl_interfaces::msg::ParameterDescriptor thr_desc;
    thr_desc.description =
      "Calibrated rolling-spread threshold; spread < threshold flags OOD.";
    declare_parameter<double>("threshold", 0.0, thr_desc);

    rcl_interfaces::msg::ParameterDescriptor cons_desc;
    cons_desc.description =
      "Consecutive violating frames required to confirm a fault (hysteresis).";
    rcl_interfaces::msg::IntegerRange cons_range;
    cons_range.from_value = 1;
    cons_range.to_value = 1000;
    cons_range.step = 1;
    cons_desc.integer_range.push_back(cons_range);
    declare_parameter<int64_t>("min_consecutive", 2, cons_desc);

    rcl_interfaces::msg::ParameterDescriptor every_desc;
    every_desc.description =
      "Frequency gate: recompute the spread only every Nth full-buffer frame.";
    rcl_interfaces::msg::IntegerRange every_range;
    every_range.from_value = 1;
    every_range.to_value = 10000;
    every_range.step = 1;
    every_desc.integer_range.push_back(every_range);
    declare_parameter<int64_t>("compute_every", 1, every_desc);
  }

  CallbackReturn on_configure(const LifecycleState &) override
  {
    const int window = static_cast<int>(get_parameter("window").as_int());
    const double threshold = get_parameter("threshold").as_double();
    const int min_consecutive =
      static_cast<int>(get_parameter("min_consecutive").as_int());
    const int compute_every =
      static_cast<int>(get_parameter("compute_every").as_int());

    try {
      core_ = std::make_unique<OodCore>(
        static_cast<std::size_t>(window), threshold, min_consecutive,
        compute_every, make_default_backend());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "on_configure: failed to build OodCore: %s", e.what());
      return CallbackReturn::FAILURE;
    }

    // Explicit QoS. Embeddings are a high-rate sensor-like stream: keep-last
    // depth 10, reliable, volatile. Verdicts: keep-last 10, reliable, volatile
    // so a late-joining arbiter does not replay stale faults.
    const rclcpp::QoS emb_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    const rclcpp::QoS verdict_qos =
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    pub_ = create_publisher<phm_msgs::msg::DetectorVerdict>("/phm/verdicts", verdict_qos);
    sub_ = create_subscription<phm_msgs::msg::PolicyEmbedding>(
      "/policy/embedding", emb_qos,
      std::bind(&OodNode::on_embedding, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "configured: backend=%s window=%d threshold=%.4f min_consecutive=%d "
      "compute_every=%d, /policy/embedding -> /phm/verdicts",
      core_->backend_name().c_str(), window, threshold, min_consecutive, compute_every);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const LifecycleState & state) override
  {
    LifecycleNode::on_activate(state);  // activates managed publishers
    RCLCPP_INFO(get_logger(), "activated: publishing verdicts");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const LifecycleState & state) override
  {
    LifecycleNode::on_deactivate(state);  // deactivates managed publishers
    // Drop rolling state so a re-activation starts from a clean window rather
    // than blending an old context with the new one.
    if (core_) {
      core_->reset();
    }
    RCLCPP_INFO(get_logger(), "deactivated: dropping frames, window reset");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_cleanup(const LifecycleState &) override
  {
    sub_.reset();
    pub_.reset();
    core_.reset();
    RCLCPP_INFO(get_logger(), "cleaned up");
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_shutdown(const LifecycleState &) override
  {
    sub_.reset();
    pub_.reset();
    core_.reset();
    RCLCPP_INFO(get_logger(), "shut down");
    return CallbackReturn::SUCCESS;
  }

private:
  void on_embedding(const phm_msgs::msg::PolicyEmbedding::SharedPtr msg)
  {
    // Only run while ACTIVE. When inactive the publisher would no-op anyway, but
    // gating here also stops the rolling window from advancing off-duty.
    if (!core_ || !pub_ || !pub_->is_activated()) {
      return;
    }

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
  rclcpp_lifecycle::LifecyclePublisher<phm_msgs::msg::DetectorVerdict>::SharedPtr pub_;
  rclcpp::Subscription<phm_msgs::msg::PolicyEmbedding>::SharedPtr sub_;
};

}  // namespace phm_ood_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<phm_ood_cpp::OodNode>();
  executor.add_node(node->get_node_base_interface());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
