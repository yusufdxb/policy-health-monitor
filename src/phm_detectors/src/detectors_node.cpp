// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_detectors/detectors_node.hpp"

#include <chrono>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rcl_interfaces/msg/parameter_descriptor.hpp"

namespace phm_detectors
{

namespace
{

rcl_interfaces::msg::ParameterDescriptor describe(const std::string & text)
{
  rcl_interfaces::msg::ParameterDescriptor d;
  d.description = text;
  return d;
}

}  // namespace

double DetectorsNode::monotonic_now()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

DetectorsNode::DetectorsNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("phm_detectors", options)
{
  const auto freq_topics = declare_parameter<std::vector<std::string>>(
    "freq_topics", std::vector<std::string>{},
    describe("Topic names to monitor for frequency drops (string array)"));
  const double freq_tol = declare_parameter<double>(
    "freq_tolerance_percent", 20.0,
    describe("Frequency must stay above baseline*(1-tol/100) to be healthy"));
  const int64_t freq_min = declare_parameter<int64_t>(
    "freq_min_consecutive", 2,
    describe("Hysteresis window: consecutive violations before freq fires"));
  const double cpu_lim = declare_parameter<double>(
    "threshold_cpu_limit", 80.0, describe("cpu_percent upper bound (default 80%)"));
  const double mem_lim = declare_parameter<double>(
    "threshold_mem_limit", 85.0, describe("memory_percent upper bound (default 85%)"));
  const double temp_lim = declare_parameter<double>(
    "threshold_temp_limit", 85.0, describe("gpu_temp_c upper bound (default 85 C)"));
  const int64_t thresh_min = declare_parameter<int64_t>(
    "threshold_min_consecutive", 2,
    describe("Hysteresis window: consecutive violations before threshold fires"));
  const double dead_timeout = declare_parameter<double>(
    "dead_timeout_sec", 5.0, describe("Silence after which a topic is dead (sec)"));
  const auto dead_topics = declare_parameter<std::vector<std::string>>(
    "dead_topics", std::vector<std::string>{},
    describe("Topic names to monitor for dead-topic (string array)"));

  const double now = monotonic_now();
  // One adapter per distinct topic, kept in parameter order (first occurrence).
  for (const auto & t : dead_topics) {
    Watch & w = watches_[t];
    w.topic = t;
    if (w.dead == nullptr) {
      dead_.push_back(std::make_unique<phm_core::DeadTopicAdapter>(t, dead_timeout));
      dead_.back()->set_logger(adapter_logger(dead_.back()->name()));
      w.dead = dead_.back().get();
      w.last_seen = now;
    }
  }
  for (const auto & t : freq_topics) {
    Watch & w = watches_[t];
    w.topic = t;
    if (w.freq == nullptr) {
      freq_.push_back(
        std::make_unique<phm_core::FrequencyDropAdapter>(
          t, freq_tol, static_cast<int>(freq_min)));
      freq_.back()->set_logger(adapter_logger(freq_.back()->name()));
      w.freq = freq_.back().get();
    }
  }
  const std::pair<const char *, std::pair<const char *, double>> metrics[] = {
    {"cpu_percent", {"system:cpu", cpu_lim}},
    {"memory_percent", {"system:mem", mem_lim}},
    {"gpu_temp_c", {"system:gpu", temp_lim}},
  };
  for (const auto & [metric, spec] : metrics) {
    Threshold t{metric, spec.second,
      std::make_unique<phm_core::StaticThresholdAdapter>(
        spec.first, metric, static_cast<int>(thresh_min))};
    t.adapter->set_logger(adapter_logger(t.adapter->name()));
    thresholds_.push_back(std::move(t));
  }

  // Verdicts are control-critical: reliable, so a STOP is never dropped;
  // volatile, matching the arbiter's verdict subscription.
  pub_ = create_publisher<phm_msgs::msg::DetectorVerdict>(
    "/phm/verdicts", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());

  try_bind_watch_subs();
  if (bound_topics() < watches_.size()) {
    resolve_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(2.0), [this]() {try_bind_watch_subs();});
  }
  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Duration::from_seconds(1.0), [this]() {tick();});

  RCLCPP_INFO(
    get_logger(), "phm_detectors_node ready: %zu freq, %zu threshold, %zu dead-topic adapters",
    freq_.size(), thresholds_.size(), dead_.size());
}

phm_core::LogFn DetectorsNode::adapter_logger(const std::string & name)
{
  return [this, name](phm_core::LogLevel level, const std::string & msg) {
           if (level == phm_core::LogLevel::kInfo) {
             RCLCPP_INFO(get_logger(), "%s", msg.c_str());
           } else if (level == phm_core::LogLevel::kWarn) {
             // At most one warning per adapter every 5 s.
             const double now = monotonic_now();
             auto it = last_log_.find(name);
             if (it == last_log_.end() || now - it->second >= 5.0) {
               last_log_[name] = now;
               RCLCPP_WARN(get_logger(), "%s", msg.c_str());
             }
           }
         };
}

std::size_t DetectorsNode::bound_topics() const
{
  std::size_t n = 0;
  for (const auto & kv : watches_) {
    n += kv.second.sub ? 1 : 0;
  }
  return n;
}

void DetectorsNode::try_bind_watch_subs()
{
  std::map<std::string, std::vector<std::string>> graph;
  try {
    graph = get_topic_names_and_types();
  } catch (const std::exception & e) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000, "topic graph query failed (%s); will retry", e.what());
    return;
  }
  // Only arrival timing is needed, so best-effort matches typical sensor
  // publishers (and accepts reliable ones too).
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  for (auto & [topic, w] : watches_) {
    if (w.sub) {
      continue;
    }
    const auto it = graph.find(topic);
    if (it == graph.end() || it->second.empty()) {
      continue;  // not on the graph yet
    }
    const std::string & type = it->second.front();
    try {
      Watch * wp = &w;
      w.sub = create_generic_subscription(
        topic, type, qos,
        [this, wp](std::shared_ptr<rclcpp::SerializedMessage>) {on_message(*wp);});
      RCLCPP_INFO(
        get_logger(), "bound liveness/frequency subscription on %s (%s)", topic.c_str(),
        type.c_str());
    } catch (const std::exception & e) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000, "could not subscribe to %s (%s): %s; will retry",
        topic.c_str(), type.c_str(), e.what());
    }
  }
  if (resolve_timer_ && bound_topics() >= watches_.size()) {
    resolve_timer_->cancel();
  }
}

void DetectorsNode::on_message(Watch & w)
{
  const double now = monotonic_now();
  if (w.dead != nullptr) {
    w.dead->mark_alive(now);
    w.last_seen = now;
  }
  if (w.freq != nullptr) {
    if (!w.window_open) {
      w.window_open = true;
      w.count = 0;
      w.window_start = now;
    }
    ++w.count;
  }
}

void DetectorsNode::tick()
{
  const double now = monotonic_now();

  for (const auto & adapter : dead_) {
    const Watch & w = watches_.at(adapter->target_topic());
    if (auto v = adapter->update({w.topic, w.last_seen, now})) {
      publish(*v);
    }
  }

  // Host metrics; a metric the host cannot report emits nothing.
  const phm_core::SystemMetrics m = metrics_.read();
  const std::optional<double> values[] = {m.cpu_percent, m.memory_percent, m.gpu_temp_c};
  for (std::size_t i = 0; i < thresholds_.size(); ++i) {
    if (!values[i]) {
      continue;
    }
    const Threshold & t = thresholds_[i];
    if (auto v = t.adapter->update({t.metric, *values[i], t.limit})) {
      publish(*v);
    }
  }

  // Frequency: messages over the elapsed window, then start the next window.
  for (const auto & adapter : freq_) {
    Watch & w = watches_.at(adapter->target_topic());
    const int64_t count = w.window_open ? w.count : 0;
    const double t0 = w.window_open ? w.window_start : now;
    const double elapsed = now - t0;
    const double hz = elapsed > 0 ? static_cast<double>(count) / elapsed : 0.0;
    if (auto v = adapter->update({w.topic, hz})) {
      publish(*v);
    }
    w.window_open = true;
    w.count = 0;
    w.window_start = now;
  }
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "phm_detectors tick");
}

void DetectorsNode::publish(const phm_core::VerdictData & v)
{
  out_.header.stamp = get_clock()->now();
  out_.source = v.source;
  out_.score = static_cast<float>(v.score);
  out_.violating = v.violating;
  out_.reason = v.reason;
  out_.suggested_action = v.suggested_action;
  pub_->publish(out_);
}

}  // namespace phm_detectors
