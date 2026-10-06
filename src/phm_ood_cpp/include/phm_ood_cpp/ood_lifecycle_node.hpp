// Copyright 2026 Yusuf Guenena. MIT License.
// Managed-lifecycle rclcpp node running the shared phm_core::OodCore:
// phm_msgs/PolicyEmbedding in, phm_msgs/DetectorVerdict on /phm/verdicts out.
//
// One implementation serves two public entry points whose ROS interfaces
// differ, so each keeps the contract it shipped with:
//
//   ros2 run phm_ood_cpp ood_node   node "phm_ood_cpp", source "phm_ood_cpp",
//     params window / threshold / min_consecutive (2) / compute_every; topic
//     fixed to /policy/embedding; endpoints created at configure; a malformed
//     frame (dim field mismatch, empty, dimension change) is dropped; the
//     verdict carries the embedding's stamp and frame_id; deactivate resets the
//     rolling window.
//
//   ros2 run phm_ood phm_ood_node   node "phm_ood", source "phm_ood", params
//     embedding_topic / window / threshold / calibration_file (.npz key
//     "threshold") / hysteresis_count (3) / compute_every / embed_dim;
//     endpoints created at activate and destroyed at deactivate; the array is
//     trusted over the dim field and a dimension mismatch yields a published
//     bad-input verdict; the verdict is stamped now() with an empty frame_id;
//     deactivate keeps the rolling window, cleanup resets it.
//
// Both: QoS best-effort keep-last 10 volatile in, reliable keep-last 10
// volatile out; frames that arrive while not ACTIVE are not scored; a
// non-positive threshold logs that the detector is inert.
#ifndef PHM_OOD_CPP__OOD_LIFECYCLE_NODE_HPP_
#define PHM_OOD_CPP__OOD_LIFECYCLE_NODE_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "phm_core/ood_core.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace phm_ood_cpp
{

struct OodNodeProfile
{
  std::string node_name;
  std::string source;
  std::string hysteresis_param;      // "min_consecutive" or "hysteresis_count"
  int64_t hysteresis_default = 2;
  bool topic_param = false;          // declare embedding_topic
  bool calibration_file_param = false;
  bool embed_dim_param = false;
  bool stamp_from_input = true;      // copy the embedding header into the verdict
  bool drop_malformed = true;        // drop bad frames instead of publishing bad-input verdicts
  bool endpoints_on_activate = false;
  bool reset_on_deactivate = true;
  // phm_ood_cpp declares integer ranges (window read-only); phm_ood leaves
  // validation to configure, where a bad value fails the transition.
  bool ranged_params = true;

  static OodNodeProfile phm_ood_cpp();
  static OodNodeProfile phm_ood();
};

class OodLifecycleNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit OodLifecycleNode(
    OodNodeProfile profile, const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

  const phm_core::OodCore * core() const {return core_.get();}

private:
  void create_endpoints();
  void destroy_endpoints();
  void on_embedding(const phm_msgs::msg::PolicyEmbedding::ConstSharedPtr & msg);

  OodNodeProfile profile_;
  std::string topic_;
  std::unique_ptr<phm_core::OodCore> core_;
  rclcpp_lifecycle::LifecyclePublisher<phm_msgs::msg::DetectorVerdict>::SharedPtr pub_;
  rclcpp::Subscription<phm_msgs::msg::PolicyEmbedding>::SharedPtr sub_;
  phm_msgs::msg::DetectorVerdict out_;  // reused so steady state does not reallocate
};

}  // namespace phm_ood_cpp

#endif  // PHM_OOD_CPP__OOD_LIFECYCLE_NODE_HPP_
