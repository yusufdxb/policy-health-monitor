// Copyright 2026 Yusuf Guenena. MIT License.
// Latency micro-benchmark for the rolling-spread OOD core.
//
// Feeds N synthetic embedding frames through phm_core::OodCore::update (the
// exact path the node runs: ring-buffer append, backend spread, threshold,
// hysteresis, verdict text) and reports per-frame latency percentiles.
//
// Usage: bench_latency [N=100000] [window=30] [dim=512]
//        (backend: env PHM_BACKEND=plain|eigen|libtorch)
//
// SCOPE: host-only CPU micro-benchmark of the detector core. It is not a
// target-platform or end-to-end ROS latency measurement (no DDS, no executor);
// record the host, load and configuration separately before citing a result.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "phm_core/numpy_random.hpp"
#include "phm_core/ood_core.hpp"
#include "phm_core/spread_backend.hpp"

int main(int argc, char ** argv)
{
  std::size_t n = 100000;
  std::size_t window = 30;
  std::size_t dim = 512;
  if (argc > 1) {n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));}
  if (argc > 2) {window = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));}
  if (argc > 3) {dim = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));}
  if (n == 0 || window < 2 || dim == 0) {
    std::fprintf(stderr, "usage: bench_latency [N>0] [window>=2] [dim>0]\n");
    return 2;
  }

  // compute_every = 1 so every frame does the full work (worst case).
  phm_core::OodConfig cfg;
  cfg.window = window;
  cfg.threshold = 1.0;
  cfg.min_consecutive = 2;
  cfg.compute_every = 1;
  cfg.source = "phm_ood_cpp";
  phm_core::OodCore core(cfg, phm_core::make_default_backend());

  // Pre-generate a pool of frames so RNG cost stays out of the timed region.
  const std::size_t pool = std::min<std::size_t>(n, 4096);
  phm_core::numpy_random::Generator rng(12345);
  std::vector<std::vector<float>> frames(pool, std::vector<float>(dim));
  for (auto & f : frames) {
    for (auto & x : f) {
      x = static_cast<float>(rng.standard_normal());
    }
  }

  std::vector<double> us;
  us.reserve(n);
  const std::string pid = "bench";
  volatile double sink = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto & frame = frames[i % pool];
    const auto t0 = std::chrono::steady_clock::now();
    const auto & v = core.update(frame.data(), frame.size(), pid);
    const auto t1 = std::chrono::steady_clock::now();
    sink = sink + v.score + core.last_spread();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  std::sort(us.begin(), us.end());
  const auto pct = [&us](double p) {
      const double idx = (p / 100.0) * static_cast<double>(us.size() - 1);
      const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
      const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
      const double frac = idx - static_cast<double>(lo);
      return us[lo] * (1.0 - frac) + us[hi] * frac;
    };
  double mean = 0.0;
  for (double x : us) {
    mean += x;
  }
  mean /= static_cast<double>(us.size());

  std::printf("=== phm_ood_cpp rolling-spread latency micro-benchmark ===\n");
  std::printf("SCOPE: host-only CPU micro-benchmark; not target-platform or ROS latency\n");
  std::printf(
    "backend          : %s (force with env PHM_BACKEND=plain|eigen|libtorch)\n",
    core.backend_name());
  std::printf("frames (N)       : %zu\n", n);
  std::printf("window           : %zu\n", window);
  std::printf("embedding dim    : %zu\n", dim);
  std::printf("compute_every    : 1 (every frame does full work)\n");
  std::printf("--- per-frame latency (microseconds) ---\n");
  std::printf("median (p50)     : %.3f us\n", pct(50.0));
  std::printf("mean             : %.3f us\n", mean);
  std::printf("p90              : %.3f us\n", pct(90.0));
  std::printf("p99              : %.3f us\n", pct(99.0));
  std::printf("max              : %.3f us\n", us.back());
  std::printf("(sink=%.6f, ignore)\n", static_cast<double>(sink));
  return 0;
}
