// Copyright 2026 Yusuf Guenena. MIT License.
// phm_detectors node: runs the phm_core health detectors off the live ROS
// graph and publishes phm_msgs/DetectorVerdict on /phm/verdicts.
//
//   - Watched topics (freq_topics and dead_topics) are subscribed with
//     rclcpp generic subscriptions: the detectors only need arrival times, so
//     messages are never deserialized. The message type is resolved from the
//     graph; a topic that is not on the graph yet is retried every 2 s.
//   - A 1 Hz timer (node clock) drives, in this order: the dead-topic checks
//     (dead_topics order), the host-metric checks (cpu, memory, GPU
//     temperature, each only when readable), and the frequency checks
//     (freq_topics order, rate = messages / seconds in the last tick window).
//   - Elapsed-time math uses a monotonic clock.
//
// Parameters (config/detectors.yaml): freq_topics, freq_tolerance_percent,
// freq_min_consecutive, threshold_cpu_limit, threshold_mem_limit,
// threshold_temp_limit, threshold_min_consecutive, dead_timeout_sec,
// dead_topics.
#ifndef PHM_DETECTORS__DETECTORS_NODE_HPP_
#define PHM_DETECTORS__DETECTORS_NODE_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/adapters.hpp"
#include "phm_core/system_metrics.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "rclcpp/rclcpp.hpp"

namespace phm_detectors
{

class DetectorsNode : public rclcpp::Node
{
public:
  explicit DetectorsNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // Run one 1 Hz tick now (exposed for tests).
  void tick();
  // Number of watched topics with a bound subscription.
  std::size_t bound_topics() const;
  // Replace the host-metric reader (tests point it at a fake /proc tree).
  void set_metrics_reader(phm_core::SystemMetricsReader reader) {metrics_ = std::move(reader);}

private:
  struct Watch
  {
    std::string topic;
    phm_core::FrequencyDropAdapter * freq = nullptr;
    phm_core::DeadTopicAdapter * dead = nullptr;
    bool window_open = false;  // a frequency window has started
    double window_start = 0.0;
    int64_t count = 0;
    double last_seen = 0.0;
    rclcpp::GenericSubscription::SharedPtr sub;
  };
  struct Threshold
  {
    std::string metric;
    double limit;
    std::unique_ptr<phm_core::StaticThresholdAdapter> adapter;
  };

  void try_bind_watch_subs();
  void on_message(Watch & w);
  void publish(const phm_core::VerdictData & v);
  phm_core::LogFn adapter_logger(const std::string & name);
  static double monotonic_now();

  std::vector<std::unique_ptr<phm_core::FrequencyDropAdapter>> freq_;
  std::vector<std::unique_ptr<phm_core::DeadTopicAdapter>> dead_;
  std::vector<Threshold> thresholds_;
  std::map<std::string, Watch> watches_;
  phm_core::SystemMetricsReader metrics_;
  std::map<std::string, double> last_log_;  // per-adapter warning throttle

  rclcpp::Publisher<phm_msgs::msg::DetectorVerdict>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr resolve_timer_;
  phm_msgs::msg::DetectorVerdict out_;
};

}  // namespace phm_detectors

#endif  // PHM_DETECTORS__DETECTORS_NODE_HPP_
