// Copyright 2026 Yusuf Guenena. MIT License.
// Severity banding and action enum for the C++ OOD node.
//
// Mirrors the Python severity module so a C++ verdict carries the same score
// and suggested_action as the rclpy node. Source of truth:
//   phm_core/phm_core/severity.py:36-51, 63-89
// (state/action enums, DEGRADED/INTERVENE/STOP cut points, normalize()).
// The action enum also matches phm_msgs/DetectorVerdict.msg (uint8 suggested_action)
// and phm_msgs/PolicyHealthStatus.msg action numbering.
#ifndef PHM_OOD_CPP__SEVERITY_HPP_
#define PHM_OOD_CPP__SEVERITY_HPP_

#include <cstdint>

namespace phm_ood_cpp
{

// Suggested-action enum, mirrors severity.py:42-46 and PolicyHealthStatus.msg.
constexpr uint8_t ACTION_NONE = 0;
constexpr uint8_t ACTION_LOG_ONLY = 1;
constexpr uint8_t ACTION_HOLD = 2;
constexpr uint8_t ACTION_STOP_AND_HOLD = 3;
constexpr uint8_t ACTION_REWIND = 4;

// Score cut points, mirrors severity.py:49-51.
constexpr double DEGRADED_THRESHOLD = 0.25;
constexpr double INTERVENE_THRESHOLD = 0.50;
constexpr double STOP_THRESHOLD = 0.80;

// Linear map of a raw signal onto [0,1], clamped. Mirrors severity.py:63-89.
// Pass healthy > worst when a LOW raw value is the unhealthy one (the
// rolling-spread collapse case: healthy = threshold, worst = 0).
// Caller must ensure healthy != worst.
inline double normalize(double raw, double healthy, double worst)
{
  const double frac = (raw - healthy) / (worst - healthy);
  if (frac < 0.0) {
    return 0.0;
  }
  if (frac > 1.0) {
    return 1.0;
  }
  return frac;
}

}  // namespace phm_ood_cpp

#endif  // PHM_OOD_CPP__SEVERITY_HPP_
