// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_sim embedding_publisher: streams synthetic
// phm_msgs/PolicyEmbedding on /policy/embedding (node "phm_sim"), n_in_dist
// healthy frames then a collapsed (OOD) phase forever, for end-to-end tests
// without a policy. Frames come from phm_core::EmbeddingStream, so a seed
// gives the same frames as the original NumPy generator.
//
// QoS reliable keep-last 10 volatile: the OOD node needs every frame to count
// hysteresis runs. Service /phm_sim/trigger_ood (std_srvs/Trigger) switches to
// the OOD phase immediately.
#include <cinttypes>
#include <memory>
#include <string>
#include <vector>

#include "phm_core/format.hpp"
#include "phm_core/sim.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace
{

rcl_interfaces::msg::ParameterDescriptor describe(const std::string & text, bool read_only)
{
  rcl_interfaces::msg::ParameterDescriptor d;
  d.description = text;
  d.read_only = read_only;
  return d;
}

class EmbeddingPublisher : public rclcpp::Node
{
public:
  EmbeddingPublisher()
  : rclcpp::Node("phm_sim")
  {
    const int64_t dim = declare_parameter<int64_t>(
      "dim", 64, describe("Embedding dimensionality.", true));
    const int64_t n_in_dist = declare_parameter<int64_t>(
      "n_in_dist", 100,
      describe("Number of in-distribution frames before switching to OOD phase.", true));
    const double in_scale = declare_parameter<double>(
      "in_dist_scale", 1.0,
      describe("Gaussian std-dev for the in-distribution (healthy) phase.", true));
    const double ood_scale = declare_parameter<double>(
      "ood_scale", 0.01, describe("Gaussian std-dev for the OOD (collapse) phase.", true));
    const double rate_hz = declare_parameter<double>(
      "publish_rate_hz", 10.0, describe("Publish frequency in Hz.", false));
    const std::string policy_id = declare_parameter<std::string>(
      "policy_id", "phm_sim", describe("Label forwarded to PolicyEmbedding.policy_id.", true));
    const int64_t seed = declare_parameter<int64_t>(
      "seed", 42, describe("RNG seed for reproducibility (NumPy default_rng stream).", true));

    stream_ = std::make_unique<phm_core::EmbeddingStream>(
      static_cast<std::size_t>(dim), static_cast<std::size_t>(n_in_dist), in_scale, ood_scale,
      policy_id, static_cast<uint64_t>(seed));
    frame_.resize(static_cast<std::size_t>(dim));
    msg_.embedding.resize(static_cast<std::size_t>(dim));
    msg_.dim = static_cast<uint32_t>(dim);
    msg_.policy_id = policy_id;

    pub_ = create_publisher<phm_msgs::msg::PolicyEmbedding>(
      "/policy/embedding", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
    srv_ = create_service<std_srvs::srv::Trigger>(
      "/phm_sim/trigger_ood",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        const bool was = stream_->is_ood_phase();
        stream_->trigger_ood();
        response->message = was ? std::string("OOD phase was already active.") :
        "Switched to OOD phase at frame " + std::to_string(stream_->frame_index()) + ".";
        response->success = true;
        RCLCPP_INFO(get_logger(), "[phm_sim] trigger_ood: %s", response->message.c_str());
      });
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_hz), [this]() {tick();});
    RCLCPP_INFO(
      get_logger(),
      "phm_sim ready: dim=%" PRId64 ", n_in_dist=%" PRId64
      ", in_dist_scale=%s, ood_scale=%s, rate=%s Hz, policy_id='%s'", dim, n_in_dist,
      phm_core::fmt::repr(in_scale).c_str(), phm_core::fmt::repr(ood_scale).c_str(),
      phm_core::fmt::repr(rate_hz).c_str(), policy_id.c_str());
  }

private:
  void tick()
  {
    stream_->next_frame(frame_.data());
    for (std::size_t i = 0; i < frame_.size(); ++i) {
      msg_.embedding[i] = static_cast<float>(frame_[i]);
    }
    msg_.header.stamp = get_clock()->now();
    pub_->publish(msg_);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "[phm_sim] frame=%zu phase=%s", stream_->frame_index(),
      stream_->is_ood_phase() ? "OOD" : "in_dist");
  }

  std::unique_ptr<phm_core::EmbeddingStream> stream_;
  std::vector<double> frame_;
  phm_msgs::msg::PolicyEmbedding msg_;
  rclcpp::Publisher<phm_msgs::msg::PolicyEmbedding>::SharedPtr pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EmbeddingPublisher>());
  rclcpp::shutdown();
  return 0;
}
