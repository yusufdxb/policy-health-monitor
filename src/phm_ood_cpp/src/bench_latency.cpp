// Copyright 2026 Yusuf Guenena. MIT License.
// Latency micro-benchmark for the C++ rolling-spread compute.
//
// Feeds N synthetic embedding frames through OodCore::update (the exact path the
// node runs: rolling-window maintenance + backend rolling_spread + threshold +
// hysteresis + verdict build) and reports median and p99 per-frame latency in
// microseconds.
//
// IMPORTANT LABEL: these numbers are measured on this DESKTOP WORKSTATION CPU
// (RTX 5070 build box), used here as a CPU proxy. They are NOT a Jetson Orin NX
// measurement. A Jetson number would require running this same binary on the
// Orin and will be slower per the Orin's lower single-core clock.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "phm_ood_cpp/backend.hpp"
#include "phm_ood_cpp/ood_core.hpp"

using phm_ood_cpp::OodCore;
using phm_ood_cpp::make_default_backend;

int main(int argc, char ** argv)
{
  // Defaults chosen to match the node: window 30, 512-D recurrent feature
  // (PolicyEmbedding.msg comment), N = 100000 frames.
  std::size_t n = 100000;
  std::size_t window = 30;
  std::size_t dim = 512;
  if (argc > 1) {n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));}
  if (argc > 2) {window = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));}
  if (argc > 3) {dim = static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10));}

  // compute_every = 1 so EVERY frame does real work (worst case / true per-frame).
  OodCore core(window, /*threshold=*/1.0, /*min_consecutive=*/2,
               /*compute_every=*/1, make_default_backend());

  // Pre-generate frames so RNG cost is excluded from the timed region.
  std::mt19937 rng(12345);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::vector<float>> frames(n, std::vector<float>(dim));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t d = 0; d < dim; ++d) {
      frames[i][d] = nd(rng);
    }
  }

  std::vector<double> us;
  us.reserve(n);
  const std::string pid = "bench";
  volatile double sink = 0.0;  // prevent the compiler eliding update()

  for (std::size_t i = 0; i < n; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto v = core.update(frames[i], pid);
    const auto t1 = std::chrono::steady_clock::now();
    sink += v.score + core.last_spread();
    const double micros =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
    us.push_back(micros);
  }

  std::sort(us.begin(), us.end());
  auto pct = [&us](double p) {
      const double idx = (p / 100.0) * static_cast<double>(us.size() - 1);
      const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
      const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
      if (lo == hi) {return us[lo];}
      const double frac = idx - static_cast<double>(lo);
      return us[lo] * (1.0 - frac) + us[hi] * frac;
    };

  double mean = 0.0;
  for (double x : us) {mean += x;}
  mean /= static_cast<double>(us.size());

  std::printf("=== phm_ood_cpp rolling-spread latency micro-benchmark ===\n");
  std::printf("HARDWARE: DESKTOP WORKSTATION CPU (AMD Ryzen 9 9900X, RTX 5070\n");
  std::printf("          build box) - CPU proxy ONLY. This is NOT a Jetson Orin NX\n");
  std::printf("          number. Re-run this same binary on the Orin for an\n");
  std::printf("          embedded figure (expect it slower per the Orin's clock).\n");
  std::printf("backend          : %s (force with env PHM_BACKEND=plain|eigen)\n",
              core.backend_name().c_str());
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
