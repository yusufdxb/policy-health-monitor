// Copyright 2026 Yusuf Guenena. MIT License.
// Worst-wins arbitration of detector verdicts into one health status.
//
// For every source's latest verdict:
//   - A non-finite score is a poisoned detector: DEGRADED with the finite
//     sentinel score 0.5, reason "bad-score:<source>", action NONE. It can
//     neither force STOP nor make arbitration non-deterministic.
//   - Stale (now - timestamp > staleness_sec, strictly): never dropped and
//     never de-escalated. Reason "stale:<source>", action LOG_ONLY. A
//     non-violating stale verdict is exactly DEGRADED at score 0.25; a
//     violating stale verdict keeps max(DEGRADED, its own band) and
//     max(0.25, its own score).
//   - Fresh and violating: competes with state max(DEGRADED, band(score)) (a
//     violating verdict is never OK), its own score and reason, and its
//     suggested action clamped to the allowlist {NONE..REWIND} (else NONE).
//     REWIND passes through so the recovery layer can act on it.
//   - Fresh and not violating: contributes nothing.
// The winner is the candidate with the highest (state, score); among equals
// the first in input order wins. No candidate yields OK, score 0, reason
// "all detectors nominal", empty source, action NONE.
#ifndef PHM_CORE__ARBITER_HPP_
#define PHM_CORE__ARBITER_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace phm_core
{

// Score of a stale verdict's DEGRADED floor (the DEGRADED band lower edge).
constexpr double kStaleScore = 0.25;
// Finite sentinel score for a verdict whose score is NaN or infinite.
constexpr double kBadScoreSentinel = 0.5;

// One source's latest verdict as the arbiter sees it.
struct ArbiterInput
{
  std::string source;
  double score = 0.0;
  bool violating = false;
  std::string reason;
  uint8_t suggested_action = 0;
  double timestamp = 0.0;      // seconds, same clock as `now`
  bool has_timestamp = true;   // false: treated as fresh
};

// Mirror of phm_msgs/PolicyHealthStatus.msg minus the header.
struct HealthStatusData
{
  uint8_t state = 0;
  double score = 0.0;
  std::string reason;
  std::string source;
  uint8_t suggested_action = 0;
};

bool action_in_allowlist(uint8_t action);

HealthStatusData arbitrate(
  const std::vector<ArbiterInput> & verdicts, double now, double staleness_sec = 1.0);

}  // namespace phm_core

#endif  // PHM_CORE__ARBITER_HPP_
