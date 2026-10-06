// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_ood_cpp/ood_lifecycle_node.hpp"

#include <cinttypes>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "phm_core/npy.hpp"
#include "phm_core/spread_backend.hpp"
#include "rcl_interfaces/msg/integer_range.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"

namespace phm_ood_cpp
{

namespace
{
constexpr char kDefaultTopic[] = "/policy/embedding";
constexpr char kVerdictTopic[] = "/phm/verdicts";
constexpr int64_t kMaxWindow = 100000;

rcl_interfaces::msg::ParameterDescriptor describe(
  const std::string & text, bool read_only = false)
{
  rcl_interfaces::msg::ParameterDescriptor d;
  d.description = text;
  d.read_only = read_only;
  return d;
}

rcl_interfaces::msg::ParameterDescriptor ranged(
  const std::string & text, int64_t lo, int64_t hi, bool read_only = false)
{
  auto d = describe(text, read_only);
  rcl_interfaces::msg::IntegerRange r;
  r.from_value = lo;
  r.to_value = hi;
  r.step = 1;
  d.integer_range.push_back(r);
  return d;
}
}  // namespace

OodNodeProfile OodNodeProfile::phm_ood_cpp()
{
  OodNodeProfile p;
  p.node_name = "phm_ood_cpp";
  p.source = "phm_ood_cpp";
  p.hysteresis_param = "min_consecutive";
  p.hysteresis_default = 2;
  return p;
}

OodNodeProfile OodNodeProfile::phm_ood()
{
  OodNodeProfile p;
  p.node_name = "phm_ood";
  p.source = "phm_ood";
  p.hysteresis_param = "hysteresis_count";
  p.hysteresis_default = 3;
  p.topic_param = true;
  p.calibration_file_param = true;
  p.embed_dim_param = true;
  p.stamp_from_input = false;
  p.drop_malformed = false;
  p.endpoints_on_activate = true;
  p.reset_on_deactivate = false;
  p.ranged_params = false;
  return p;
}

OodLifecycleNode::OodLifecycleNode(OodNodeProfile profile, const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode(profile.node_name, options), profile_(std::move(profile))
{
  // Declared here so they are introspectable before configuration; read once
  // in on_configure.
  if (profile_.topic_param) {
    declare_parameter<std::string>(
      "embedding_topic", kDefaultTopic,
      describe("Topic to subscribe to for PolicyEmbedding messages.", true));
  }
  const bool r = profile_.ranged_params;
  const std::string window_text =
    "Frames in the rolling covariance window (>= 2). Read at configure time.";
  declare_parameter<int64_t>(
    "window", 30, r ? ranged(window_text, 2, kMaxWindow, true) : describe(window_text));
  declare_parameter<double>(
    "threshold", 0.0,
    describe("Calibrated rolling-spread threshold; spread < threshold flags OOD."));
  if (profile_.calibration_file_param) {
    declare_parameter<std::string>(
      "calibration_file", "",
      describe(
        "Path to a .npz file with key 'threshold'. If non-empty, loads the threshold from "
        "the file and ignores the 'threshold' parameter.", true));
  }
  const std::string hyst_text =
    "Consecutive violating frames required to confirm a fault (hysteresis), >= 1.";
  declare_parameter<int64_t>(
    profile_.hysteresis_param, profile_.hysteresis_default,
    r ? ranged(hyst_text, 1, 1000) : describe(hyst_text));
  const std::string every_text =
    "Frequency gate: recompute the spread only every Nth full-buffer frame, >= 1.";
  declare_parameter<int64_t>(
    "compute_every", 1, r ? ranged(every_text, 1, 10000) : describe(every_text));
  if (profile_.embed_dim_param) {
    declare_parameter<int64_t>(
      "embed_dim", 0, describe("Expected embedding dimension for input validation. 0 = skip."));
  }
}

OodLifecycleNode::CallbackReturn OodLifecycleNode::on_configure(const rclcpp_lifecycle::State &)
{
  topic_ = profile_.topic_param ? get_parameter("embedding_topic").as_string() : kDefaultTopic;
  const int64_t window = get_parameter("window").as_int();
  double threshold = get_parameter("threshold").as_double();
  const int64_t hysteresis = get_parameter(profile_.hysteresis_param).as_int();
  const int64_t compute_every = get_parameter("compute_every").as_int();
  const int64_t embed_dim = profile_.embed_dim_param ? get_parameter("embed_dim").as_int() : 0;

  // Both profiles get the same bounds, whether or not their parameters carry
  // declared ranges: the int64 values are narrowed below, and a window beyond
  // kMaxWindow could never fill in practice.
  if (window < 2 || window > kMaxWindow) {
    RCLCPP_ERROR(
      get_logger(), "Parameter 'window' must be in [2, %" PRId64 "], got %" PRId64
      ". Failing configure.", kMaxWindow, window);
    return CallbackReturn::FAILURE;
  }
  if (hysteresis < 1 || hysteresis > std::numeric_limits<int>::max()) {
    RCLCPP_ERROR(
      get_logger(), "Parameter '%s' must be >= 1 and fit in an int, got %" PRId64
      ". Failing configure.", profile_.hysteresis_param.c_str(), hysteresis);
    return CallbackReturn::FAILURE;
  }
  if (compute_every < 1 || compute_every > std::numeric_limits<int>::max()) {
    RCLCPP_ERROR(
      get_logger(), "Parameter 'compute_every' must be >= 1 and fit in an int, got %" PRId64
      ". Failing configure.", compute_every);
    return CallbackReturn::FAILURE;
  }
  if (profile_.calibration_file_param) {
    const std::string calib = get_parameter("calibration_file").as_string();
    if (!calib.empty()) {
      try {
        threshold = phm_core::npy::load_npz(calib).at("threshold").scalar();
        RCLCPP_INFO(get_logger(), "Loaded threshold %.6f from %s", threshold, calib.c_str());
      } catch (const std::exception & e) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to load calibration_file '%s': %s. Falling back to 'threshold' parameter.",
          calib.c_str(), e.what());
      }
    }
  }
  // spread is a sum of variances and never negative, so a non-positive
  // threshold can never fire: the node would publish healthy verdicts forever
  // while monitoring nothing, indistinguishable from a working monitor.
  if (threshold <= 0.0) {
    RCLCPP_WARN(
      get_logger(),
      "threshold=%.6f is non-positive: the detector is INERT and will never flag OOD. Set the "
      "'threshold' parameter%s from a calibration run.",
      threshold, profile_.calibration_file_param ? " or 'calibration_file'" : "");
  }

  phm_core::OodConfig cfg;
  cfg.window = static_cast<std::size_t>(window);
  cfg.threshold = threshold;
  cfg.min_consecutive = static_cast<int>(hysteresis);
  cfg.compute_every = static_cast<int>(compute_every);
  cfg.embed_dim = embed_dim > 0 ? static_cast<std::size_t>(embed_dim) : 0;
  cfg.source = profile_.source;
  try {
    core_ = std::make_unique<phm_core::OodCore>(cfg, phm_core::make_default_backend());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "on_configure: failed to build OodCore: %s", e.what());
    return CallbackReturn::FAILURE;
  }
  if (!profile_.endpoints_on_activate) {
    create_endpoints();
  }
  RCLCPP_INFO(
    get_logger(),
    "configured: backend=%s window=%" PRId64 " threshold=%.6f %s=%" PRId64
    " compute_every=%" PRId64 " embed_dim=%" PRId64 ", %s -> %s",
    core_->backend_name(), window, threshold, profile_.hysteresis_param.c_str(), hysteresis,
    compute_every, embed_dim, topic_.c_str(), kVerdictTopic);
  return CallbackReturn::SUCCESS;
}

void OodLifecycleNode::create_endpoints()
{
  // Embeddings are a high-rate sensor-like stream: best-effort keep-last 10.
  // A best-effort subscriber matches both best-effort and reliable
  // publishers; a reliable one would silently receive nothing from a
  // best-effort publisher. Verdicts drive safety decisions: reliable, volatile
  // so a late-joining arbiter does not replay stale faults.
  const auto emb_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  const auto verdict_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  pub_ = create_publisher<phm_msgs::msg::DetectorVerdict>(kVerdictTopic, verdict_qos);
  sub_ = create_subscription<phm_msgs::msg::PolicyEmbedding>(
    topic_, emb_qos, std::bind(&OodLifecycleNode::on_embedding, this, std::placeholders::_1));
}

void OodLifecycleNode::destroy_endpoints()
{
  sub_.reset();
  pub_.reset();
}

OodLifecycleNode::CallbackReturn OodLifecycleNode::on_activate(
  const rclcpp_lifecycle::State & state)
{
  if (!core_) {
    RCLCPP_ERROR(get_logger(), "Core not initialized; call configure first.");
    return CallbackReturn::FAILURE;
  }
  if (profile_.endpoints_on_activate) {
    create_endpoints();
  }
  LifecycleNode::on_activate(state);  // activates the managed publisher
  RCLCPP_INFO(get_logger(), "activated: %s -> %s", topic_.c_str(), kVerdictTopic);
  return CallbackReturn::SUCCESS;
}

OodLifecycleNode::CallbackReturn OodLifecycleNode::on_deactivate(
  const rclcpp_lifecycle::State & state)
{
  LifecycleNode::on_deactivate(state);
  if (profile_.endpoints_on_activate) {
    destroy_endpoints();
  }
  if (profile_.reset_on_deactivate && core_) {
    // A re-activation starts from a clean window instead of blending an old
    // context with the new one.
    core_->reset();
  }
  RCLCPP_INFO(
    get_logger(), "deactivated%s", profile_.reset_on_deactivate ? ": window reset" : "");
  return CallbackReturn::SUCCESS;
}

OodLifecycleNode::CallbackReturn OodLifecycleNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  destroy_endpoints();
  core_.reset();
  RCLCPP_INFO(get_logger(), "cleaned up");
  return CallbackReturn::SUCCESS;
}

OodLifecycleNode::CallbackReturn OodLifecycleNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  destroy_endpoints();
  core_.reset();
  RCLCPP_INFO(get_logger(), "shut down");
  return CallbackReturn::SUCCESS;
}

void OodLifecycleNode::on_embedding(const phm_msgs::msg::PolicyEmbedding::ConstSharedPtr & msg)
{
  // Score only while ACTIVE, so the window never advances off duty.
  if (!core_ || !pub_ || !pub_->is_activated()) {
    return;
  }
  const std::size_t n = msg->embedding.size();
  if (profile_.drop_malformed) {
    if (msg->dim != n) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "PolicyEmbedding dim=%u != embedding length=%zu; dropping frame", msg->dim, n);
      return;
    }
    if (n == 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "empty embedding; dropping frame");
      return;
    }
  } else if (msg->dim > 0 && msg->dim != n) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "PolicyEmbedding.dim=%u != len(embedding)=%zu; trusting the array.", msg->dim, n);
  }

  const phm_core::VerdictData & v = core_->update(msg->embedding.data(), n, msg->policy_id);
  if (core_->last_status() != phm_core::FrameStatus::kOk && profile_.drop_malformed) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000, "dropping frame: %s", v.reason.c_str());
    return;
  }

  if (profile_.stamp_from_input) {
    // Carry the embedding's stamp so downstream latency is measurable; fall
    // back to now() when the producer left it unstamped.
    if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
      out_.header.stamp = now();
    } else {
      out_.header.stamp = msg->header.stamp;
    }
    out_.header.frame_id = msg->header.frame_id;
  } else {
    out_.header.stamp = now();
    out_.header.frame_id.clear();
  }
  out_.source = v.source;
  out_.score = static_cast<float>(v.score);
  out_.violating = v.violating;
  out_.reason = v.reason;
  out_.suggested_action = v.suggested_action;
  pub_->publish(out_);

  if (core_->has_spread()) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "OOD: spread=%.4f thr=%.4f violating=%s score=%.3f",
      core_->last_spread(), core_->threshold(), v.violating ? "True" : "False", v.score);
  }
}

}  // namespace phm_ood_cpp
