// Copyright 2026 Yusuf Guenena. MIT License.
// The README's concrete example: a healthy stream, then the same policy
// collapsing at frame 200, with the threshold calibrated on the healthy phase
// at its 1st percentile.
#include <cmath>
#include <cstdio>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/sim.hpp"

int main()
{
  const auto batch = phm_core::generate_embeddings(
    /*dim=*/64, /*n_frames=*/200, /*in_dist_scale=*/1.0, /*ood_scale=*/0.01, /*seed=*/42);
  const double threshold =
    phm_core::calibrate_threshold(phm_core::rolling_spread(batch.in_dist, 20), 1.0);

  // The policy collapses at frame 200.
  const auto stream = phm_core::vstack({&batch.in_dist, &batch.ood});
  const std::vector<double> spread = phm_core::rolling_spread(stream, 20);

  std::vector<double> healthy;
  std::vector<double> collapsed;
  int first_alarm = -1;
  int false_alarms = 0;
  int valid = 0;
  for (std::size_t t = 0; t < spread.size(); ++t) {
    if (std::isnan(spread[t])) {
      continue;
    }
    if (t < 200) {
      healthy.push_back(spread[t]);
      ++valid;
      false_alarms += spread[t] < threshold ? 1 : 0;
      continue;
    }
    if (t >= 220) {
      collapsed.push_back(spread[t]);
    }
    if (first_alarm < 0 && spread[t] < threshold) {
      first_alarm = static_cast<int>(t);
    }
  }
  std::printf("calibrated threshold   %.2f\n", threshold);
  std::printf(
    "mean spread, healthy   %.2f\n", phm_core::numpy_mean(healthy.data(), healthy.size()));
  std::printf(
    "mean spread, collapsed %.2f\n", phm_core::numpy_mean(collapsed.data(), collapsed.size()));
  std::printf("first alarm at/after collapse   frame %d\n", first_alarm);
  std::printf("false alarms on healthy frames  %d / %d\n", false_alarms, valid);
  return 0;
}
