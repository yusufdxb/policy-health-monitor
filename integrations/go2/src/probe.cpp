// Copyright 2026 Yusuf Guenena. MIT License.
// phm_go2_probe: calibrate PHM on live policy latents and record PHM's output.
//
//   phm_go2_probe calibrate [--seconds 60] [--window 30] [--percentile 1.0] --out calib.npz
//       Collect --seconds of /policy/embedding, compute the rolling-spread
//       threshold at --percentile (phm_core calibration, the NumPy-identical
//       math), and write the .npz the OOD node loads (key "threshold") with the
//       raw latents (float32) so the threshold can be recomputed offline. Prints
//       one JSON summary line.
//   phm_go2_probe record --seconds S [--window 30] --out session.jsonl
//       Write every /phm/health, /phm/verdicts and /policy/embedding arrival as
//       one JSON line (embeddings as stamp + rolling spread, not the vector).
//       "t" is the wall-clock receive time.
//
// Read-only on the ROS graph: this tool creates no publishers.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/format.hpp"
#include "phm_core/npy.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/spread_backend.hpp"
#include "phm_msgs/msg/detector_verdict.hpp"
#include "phm_msgs/msg/policy_embedding.hpp"
#include "phm_msgs/msg/policy_health_status.hpp"
#include "phm_tools/json.hpp"
#include "rclcpp/rclcpp.hpp"

using phm_tools::json::Value;

namespace
{

const auto kBestEffort = rclcpp::QoS(rclcpp::KeepLast(100)).best_effort().durability_volatile();
const auto kReliable = rclcpp::QoS(rclcpp::KeepLast(100)).reliable().durability_volatile();

double wall_now()
{
  return std::chrono::duration<double>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

void spin_for(const rclcpp::Node::SharedPtr & node, double seconds)
{
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  const auto end = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(seconds));
  while (rclcpp::ok() && std::chrono::steady_clock::now() < end) {
    exec.spin_once(std::chrono::milliseconds(50));
  }
}

template<typename Header>
double stamp_of(const Header & h)
{
  return static_cast<double>(h.stamp.sec) + static_cast<double>(h.stamp.nanosec) * 1e-9;
}

int calibrate(double seconds, int64_t window, double percentile, const std::string & out)
{
  auto node = std::make_shared<rclcpp::Node>("phm_go2_calibrate");
  std::vector<std::vector<float>> frames;
  auto sub = node->create_subscription<phm_msgs::msg::PolicyEmbedding>(
    "/policy/embedding", kBestEffort,
    [&frames](phm_msgs::msg::PolicyEmbedding::ConstSharedPtr m) {
      frames.push_back(m->embedding);
    });
  spin_for(node, seconds);
  sub.reset();
  if (static_cast<int64_t>(frames.size()) < 2 * window) {
    std::printf(
      "only %zu frames in %ss; need >= %" PRId64 "\n", frames.size(),
      phm_core::fmt::repr(seconds).c_str(), 2 * window);
    return 1;
  }
  const std::size_t dim = frames.front().size();
  phm_core::Matrix hidden(frames.size(), dim);
  std::vector<float> latents;
  latents.reserve(frames.size() * dim);
  for (std::size_t r = 0; r < frames.size(); ++r) {
    if (frames[r].size() != dim) {
      std::fprintf(stderr, "embedding dimension changed during calibration\n");
      return 1;
    }
    for (std::size_t c = 0; c < dim; ++c) {
      hidden(r, c) = static_cast<double>(frames[r][c]);
    }
    latents.insert(latents.end(), frames[r].begin(), frames[r].end());
  }
  std::vector<double> valid;
  for (double s : phm_core::rolling_spread(hidden, static_cast<std::size_t>(window))) {
    if (!std::isnan(s)) {
      valid.push_back(s);
    }
  }
  const double threshold = phm_core::calibrate_threshold(valid, percentile);
  std::vector<std::pair<std::string, phm_core::npy::Array>> arrays;
  arrays.emplace_back("threshold", phm_core::npy::Array::scalar_double(threshold));
  arrays.emplace_back("window", phm_core::npy::Array::scalar_int64(window));
  arrays.emplace_back("percentile", phm_core::npy::Array::scalar_double(percentile));
  arrays.emplace_back(
    "n_frames", phm_core::npy::Array::scalar_int64(static_cast<int64_t>(frames.size())));
  arrays.emplace_back("seconds", phm_core::npy::Array::scalar_double(seconds));
  arrays.emplace_back("latents", phm_core::npy::Array::from_floats(latents, {frames.size(), dim}));
  phm_core::npy::save_npz(out, arrays);

  // np.median: the middle element, or the mean of the two middle elements.
  std::vector<double> sorted = valid;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t n = sorted.size();
  const double median = n % 2 ? sorted[n / 2] : phm_core::numpy_mean(&sorted[n / 2 - 1], 2);
  // round(len / seconds, 2), correctly rounded as Python does.
  const double rate = std::strtod(
    phm_core::fmt::fixed(static_cast<double>(frames.size()) / seconds, 2).c_str(), nullptr);
  Value v = Value::object();
  v.set("out", out);
  v.set("n_frames", static_cast<int64_t>(frames.size()));
  v.set("rate_hz", rate);
  v.set("window", window);
  v.set("percentile", percentile);
  v.set("threshold", threshold);
  v.set("spread_min", sorted.front());
  v.set("spread_p50", median);
  v.set("spread_max", sorted.back());
  std::printf("%s\n", phm_tools::json::dumps(v).c_str());
  return 0;
}

int record(double seconds, int64_t window, const std::string & out_path)
{
  auto node = std::make_shared<rclcpp::Node>("phm_go2_record");
  std::ofstream out(out_path);
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  const auto write = [&out](const Value & v) {
      out << phm_tools::json::dumps(v) << '\n';
      out.flush();
    };
  // Rolling window of embeddings for the spread (NumPy-order kernel).
  auto backend = phm_core::make_plain_backend();
  std::vector<float> ring;
  std::size_t dim = 0;
  std::size_t filled = 0;
  std::size_t next = 0;
  const std::size_t w = static_cast<std::size_t>(window);

  auto health = node->create_subscription<phm_msgs::msg::PolicyHealthStatus>(
    "/phm/health", kReliable, [&write](phm_msgs::msg::PolicyHealthStatus::ConstSharedPtr m) {
      Value v = Value::object();
      v.set("t", wall_now());
      v.set("topic", "/phm/health");
      v.set("stamp", stamp_of(m->header));
      v.set("state", static_cast<int64_t>(m->state));
      v.set("score", static_cast<double>(m->score));
      v.set("source", m->source);
      v.set("reason", m->reason);
      v.set("action", static_cast<int64_t>(m->suggested_action));
      write(v);
    });
  auto verdicts = node->create_subscription<phm_msgs::msg::DetectorVerdict>(
    "/phm/verdicts", kReliable, [&write](phm_msgs::msg::DetectorVerdict::ConstSharedPtr m) {
      Value v = Value::object();
      v.set("t", wall_now());
      v.set("topic", "/phm/verdicts");
      v.set("stamp", stamp_of(m->header));
      v.set("source", m->source);
      v.set("score", static_cast<double>(m->score));
      v.set("violating", static_cast<bool>(m->violating));
      v.set("reason", m->reason);
      write(v);
    });
  auto embeddings = node->create_subscription<phm_msgs::msg::PolicyEmbedding>(
    "/policy/embedding", kBestEffort,
    [&](phm_msgs::msg::PolicyEmbedding::ConstSharedPtr m) {
      const double t = wall_now();
      const std::size_t n = m->embedding.size();
      if (n != dim || ring.size() != w * n) {
        dim = n;
        ring.assign(w * n, 0.0f);
        filled = 0;
        next = 0;
      }
      std::copy(m->embedding.begin(), m->embedding.end(), ring.begin() + next * dim);
      next = (next + 1) % w;
      filled = std::min(filled + 1, w);
      Value v = Value::object();
      v.set("t", t);
      v.set("topic", "/policy/embedding");
      v.set("stamp", stamp_of(m->header));
      v.set("dim", static_cast<int64_t>(m->dim));
      if (filled == w) {
        v.set("spread", backend->spread(phm_core::WindowView{ring.data(), w, dim, next}));
      } else {
        v.set("spread", nullptr);
      }
      write(v);
    });
  spin_for(node, seconds);
  out.close();
  std::printf("wrote %s\n", out_path.c_str());
  return 0;
}

void usage()
{
  std::fprintf(
    stderr,
    "usage: phm_go2_probe calibrate [--seconds S] [--window W] [--percentile P] --out FILE\n"
    "       phm_go2_probe record --seconds S [--window W] --out FILE\n");
}

}  // namespace

int main(int argc, char ** argv)
{
  // ROS arguments (--ros-args ...) are stripped before parsing ours.
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  if (args.size() < 2 || (args[1] != "calibrate" && args[1] != "record")) {
    usage();
    return 2;
  }
  const std::string cmd = args[1];
  std::map<std::string, std::string> opt;
  for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
    opt[args[i]] = args[i + 1];
  }
  if (!opt.count("--out") || (cmd == "record" && !opt.count("--seconds"))) {
    usage();
    return 2;
  }
  const double seconds =
    opt.count("--seconds") ? std::strtod(opt["--seconds"].c_str(), nullptr) : 60.0;
  const int64_t window =
    opt.count("--window") ? std::strtoll(opt["--window"].c_str(), nullptr, 10) : 30;
  const double percentile =
    opt.count("--percentile") ? std::strtod(opt["--percentile"].c_str(), nullptr) : 1.0;
  if (window < 1) {
    usage();
    return 2;
  }
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rc = cmd == "calibrate" ? calibrate(seconds, window, percentile, opt["--out"]) :
      record(seconds, window, opt["--out"]);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "phm_go2_probe: %s\n", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
