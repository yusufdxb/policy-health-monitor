// Copyright 2026 Yusuf Guenena. MIT License.
// ros2 run phm_sim phm_chain_bench: end-to-end timing of a running PHM graph.
//
// Publishes synthetic phm_msgs/PolicyEmbedding at rate_hz and times what comes
// back from the separately running nodes:
//   embedding -> /phm/verdicts   per frame; the frame's sequence number rides
//                                in policy_id and returns in the OOD verdict's
//                                reason ("[seqN] ood: ..."), so latency is
//                                matched per message for any OOD node profile
//   fault -> first violating OOD verdict, first /phm/health >= INTERVENE, first
//            /phm/health STOP, first /phm/cmd_vel Twist (if recovery runs)
// Cycles: `healthy_sec` of healthy frames, then `fault_sec` of the fault
// ("collapse": a constant frame; "silence": nothing published), `trials`
// times. Healthy frames are N(0, 1) per element from a NumPy-compatible
// stream (seed 7). All times are CLOCK_MONOTONIC in this process. With
// aux_rate_hz > 0 it also publishes geometry_msgs/Twist on /phm_bench/aux at
// that rate, a stand-in high-rate sensor topic for the detectors to watch.
//
// Prints one JSON object. It measures, it does not decide pass/fail.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "phm_core/numpy_random.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"
#include "phm_msgs/msg/policy_health_status.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{

double mono()
{
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stats
{
  std::size_t n = 0;
  double p50 = NAN, p95 = NAN, p99 = NAN, max = NAN, mean = NAN;
};

Stats summarize(std::vector<double> v)
{
  Stats s;
  s.n = v.size();
  if (v.empty()) {
    return s;
  }
  std::sort(v.begin(), v.end());
  const auto pct = [&v](double p) {
      const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
      const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
      const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
      return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
    };
  s.p50 = pct(50);
  s.p95 = pct(95);
  s.p99 = pct(99);
  s.max = v.back();
  double total = 0;
  for (double x : v) {
    total += x;
  }
  s.mean = total / static_cast<double>(v.size());
  return s;
}

std::string json(const Stats & s, double scale)
{
  char buf[256];
  if (s.n == 0) {
    std::snprintf(buf, sizeof(buf), "{\"n\": 0}");
  } else {
    std::snprintf(
      buf, sizeof(buf),
      "{\"n\": %zu, \"p50\": %.3f, \"p95\": %.3f, \"p99\": %.3f, \"max\": %.3f, \"mean\": %.3f}",
      s.n, s.p50 * scale, s.p95 * scale, s.p99 * scale, s.max * scale, s.mean * scale);
  }
  return buf;
}

class ChainBench : public rclcpp::Node
{
public:
  ChainBench()
  : rclcpp::Node("phm_chain_bench"), rng_(7)
  {
    rate_hz_ = declare_parameter<double>("rate_hz", 50.0);
    dim_ = static_cast<std::size_t>(declare_parameter<int64_t>("dim", 384));
    healthy_sec_ = declare_parameter<double>("healthy_sec", 20.0);
    fault_sec_ = declare_parameter<double>("fault_sec", 5.0);
    trials_ = static_cast<int>(declare_parameter<int64_t>("trials", 3));
    fault_ = declare_parameter<std::string>("fault", "collapse");
    warmup_sec_ = declare_parameter<double>("warmup_sec", 3.0);
    verdict_source_ = declare_parameter<std::string>("verdict_source", "phm_ood_cpp");
    const double aux_rate_hz = declare_parameter<double>("aux_rate_hz", 0.0);

    msg_.embedding.resize(dim_);
    msg_.dim = static_cast<uint32_t>(dim_);
    frozen_.resize(dim_, 0.25f);

    pub_ = create_publisher<phm_msgs::msg::PolicyEmbedding>(
      "/policy/embedding", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
    verdict_sub_ = create_subscription<phm_msgs::msg::DetectorVerdict>(
      "/phm/verdicts", rclcpp::QoS(rclcpp::KeepLast(1000)).reliable(),
      [this](phm_msgs::msg::DetectorVerdict::ConstSharedPtr m) {on_verdict(*m);});
    health_sub_ = create_subscription<phm_msgs::msg::PolicyHealthStatus>(
      "/phm/health", rclcpp::QoS(rclcpp::KeepLast(100)).reliable().transient_local(),
      [this](phm_msgs::msg::PolicyHealthStatus::ConstSharedPtr m) {on_health(*m);});
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/phm/cmd_vel", rclcpp::QoS(rclcpp::KeepLast(100)).reliable(),
      [this](geometry_msgs::msg::Twist::ConstSharedPtr) {on_cmd();});
    timer_ = create_wall_timer(
      std::chrono::nanoseconds(static_cast<int64_t>(1e9 / rate_hz_)), [this]() {tick();});
    if (aux_rate_hz > 0.0) {
      aux_pub_ = create_publisher<geometry_msgs::msg::Twist>(
        "/phm_bench/aux", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort());
      aux_timer_ = create_wall_timer(
        std::chrono::nanoseconds(static_cast<int64_t>(1e9 / aux_rate_hz)), [this]() {
          aux_pub_->publish(geometry_msgs::msg::Twist());
          ++aux_sent_;
        });
    }
    t0_ = mono();
  }

  bool done() const {return finished_;}

  void report() const
  {
    const double run = last_tick_ - t0_;
    std::printf("{\n  \"config\": {\"rate_hz\": %.1f, \"dim\": %zu, \"fault\": \"%s\", "
      "\"healthy_sec\": %.1f, \"fault_sec\": %.1f, \"trials\": %d, \"verdict_source\": \"%s\"},\n",
      rate_hz_, dim_, fault_.c_str(), healthy_sec_, fault_sec_, trials_, verdict_source_.c_str());
    std::printf("  \"run_sec\": %.3f,\n", run);
    std::printf("  \"embeddings_sent\": %zu,\n", sent_);
    std::printf("  \"aux_messages_sent\": %zu,\n", aux_sent_);
    std::printf("  \"embeddings_sent_after_warmup\": %zu,\n", sent_after_warmup_);
    std::printf("  \"ood_verdicts_matched_after_warmup\": %zu,\n", latencies_.size());
    std::printf(
      "  \"ood_verdicts_missing_after_warmup\": %zu,\n",
      sent_after_warmup_ >= latencies_.size() ? sent_after_warmup_ - latencies_.size() : 0);
    std::printf("  \"verdicts_received_total\": %zu,\n", verdicts_total_);
    std::printf("  \"health_received_total\": %zu,\n", health_total_);
    std::printf("  \"cmd_vel_received_total\": %zu,\n", cmd_total_);
    std::printf(
      "  \"verdict_throughput_hz\": %.2f,\n",
      run > 0 ? static_cast<double>(verdicts_total_) / run : 0.0);
    const double deadline = 1.0 / rate_hz_;
    std::size_t missed = 0;
    for (double l : latencies_) {
      missed += l > deadline ? 1 : 0;
    }
    std::printf("  \"deadline_ms\": %.3f,\n", deadline * 1e3);
    std::printf("  \"verdict_latency_over_deadline\": %zu,\n", missed);
    std::printf(
      "  \"embedding_to_ood_verdict_ms\": %s,\n", json(summarize(latencies_), 1e3).c_str());
    std::printf("  \"fault_trials\": [\n");
    for (std::size_t i = 0; i < trials_log_.size(); ++i) {
      const auto & t = trials_log_[i];
      const auto f = [](const std::optional<double> & x) {
          char b[32];
          if (!x) {
            return std::string("null");
          }
          std::snprintf(b, sizeof(b), "%.4f", *x);
          return std::string(b);
        };
      std::printf(
        "    {\"first_violating_verdict_s\": %s, \"first_health_intervene_s\": %s, "
        "\"first_health_stop_s\": %s, \"first_cmd_vel_s\": %s}%s\n",
        f(t.verdict).c_str(), f(t.intervene).c_str(), f(t.stop).c_str(), f(t.cmd).c_str(),
        i + 1 < trials_log_.size() ? "," : "");
    }
    std::printf("  ],\n");
    std::vector<double> stop_lat;
    std::vector<double> cmd_lat;
    std::vector<double> verdict_lat;
    for (const auto & t : trials_log_) {
      if (t.stop) {
        stop_lat.push_back(*t.stop);
      }
      if (t.cmd) {
        cmd_lat.push_back(*t.cmd);
      }
      if (t.verdict) {
        verdict_lat.push_back(*t.verdict);
      }
    }
    std::printf("  \"fault_to_violating_verdict_ms\": %s,\n",
      json(summarize(verdict_lat), 1e3).c_str());
    std::printf("  \"fault_to_health_stop_ms\": %s,\n", json(summarize(stop_lat), 1e3).c_str());
    std::printf("  \"fault_to_cmd_vel_ms\": %s\n}\n", json(summarize(cmd_lat), 1e3).c_str());
  }

private:
  struct Sent
  {
    double time;
    bool counted;  // healthy phase after warm-up
  };
  struct Trial
  {
    double start = 0;
    std::optional<double> verdict, intervene, stop, cmd;
  };

  void tick()
  {
    const double now = mono();
    last_tick_ = now;
    const double cycle = healthy_sec_ + fault_sec_;
    const double t = now - t0_;
    const int k = static_cast<int>(t / cycle);
    if (k >= trials_) {
      finished_ = true;
      return;
    }
    const bool in_fault = (t - k * cycle) >= healthy_sec_;
    if (in_fault && (trials_log_.size() < static_cast<std::size_t>(k + 1))) {
      trials_log_.push_back(Trial{now, {}, {}, {}, {}});
    }
    active_ = in_fault;
    if (in_fault && fault_ == "silence") {
      return;
    }
    if (in_fault) {
      std::copy(frozen_.begin(), frozen_.end(), msg_.embedding.begin());
    } else {
      for (auto & x : msg_.embedding) {
        x = static_cast<float>(rng_.standard_normal());
      }
    }
    const uint64_t seq = ++seq_;
    msg_.policy_id = "seq" + std::to_string(seq);
    msg_.header.stamp = get_clock()->now();
    if (sent_times_.size() > 100000) {
      sent_times_.clear();
    }
    const bool counted = t >= warmup_sec_ && !in_fault;
    sent_times_[seq] = Sent{now, counted};
    pub_->publish(msg_);
    ++sent_;
    if (counted) {
      ++sent_after_warmup_;
    }
  }

  void on_verdict(const phm_msgs::msg::DetectorVerdict & v)
  {
    const double now = mono();
    ++verdicts_total_;
    if (v.source != verdict_source_) {
      return;
    }
    if (v.reason.rfind("[seq", 0) == 0) {
      const uint64_t seq = std::strtoull(v.reason.c_str() + 4, nullptr, 10);
      const auto it = sent_times_.find(seq);
      if (it != sent_times_.end()) {
        if (it->second.counted) {
          latencies_.push_back(now - it->second.time);
        }
        sent_times_.erase(it);
      }
    }
    if (v.violating && active_ && !trials_log_.empty() && !trials_log_.back().verdict) {
      trials_log_.back().verdict = now - trials_log_.back().start;
    }
  }

  void on_health(const phm_msgs::msg::PolicyHealthStatus & h)
  {
    const double now = mono();
    ++health_total_;
    if (!active_ || trials_log_.empty()) {
      return;
    }
    Trial & t = trials_log_.back();
    if (h.state >= 2 && !t.intervene) {
      t.intervene = now - t.start;
    }
    if (h.state >= 3 && !t.stop) {
      t.stop = now - t.start;
    }
  }

  void on_cmd()
  {
    const double now = mono();
    ++cmd_total_;
    if (active_ && !trials_log_.empty() && !trials_log_.back().cmd) {
      trials_log_.back().cmd = now - trials_log_.back().start;
    }
  }

  double rate_hz_;
  std::size_t dim_;
  double healthy_sec_, fault_sec_, warmup_sec_;
  int trials_;
  std::string fault_;
  std::string verdict_source_;
  phm_core::numpy_random::Generator rng_;
  std::vector<float> frozen_;
  phm_msgs::msg::PolicyEmbedding msg_;
  rclcpp::Publisher<phm_msgs::msg::PolicyEmbedding>::SharedPtr pub_;
  rclcpp::Subscription<phm_msgs::msg::DetectorVerdict>::SharedPtr verdict_sub_;
  rclcpp::Subscription<phm_msgs::msg::PolicyHealthStatus>::SharedPtr health_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr aux_pub_;
  rclcpp::TimerBase::SharedPtr aux_timer_;
  std::size_t aux_sent_ = 0;
  double t0_ = 0, last_tick_ = 0;
  bool finished_ = false;
  bool active_ = false;
  uint64_t seq_ = 0;
  std::unordered_map<uint64_t, Sent> sent_times_;
  std::vector<double> latencies_;
  std::vector<Trial> trials_log_;
  std::size_t sent_ = 0, sent_after_warmup_ = 0, verdicts_total_ = 0, health_total_ = 0,
    cmd_total_ = 0;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ChainBench>();
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  while (rclcpp::ok() && !node->done()) {
    exec.spin_once(std::chrono::milliseconds(10));
  }
  // Drain late messages briefly.
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < end) {
    exec.spin_once(std::chrono::milliseconds(10));
  }
  node->report();
  rclcpp::shutdown();
  return 0;
}
