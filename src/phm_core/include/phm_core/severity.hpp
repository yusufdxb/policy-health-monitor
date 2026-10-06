// Copyright 2026 Yusuf Guenena. MIT License.
// Severity mapping: raw detector signal -> (score, state, suggested_action).
//
// This is the single place that defines the OK / DEGRADED / INTERVENE / STOP
// thresholds for the whole monitor. The per-detector verdicts, the OOD core and
// the arbiter all read these bands so a score means the same thing everywhere.
//
// State and action enums match phm_msgs/PolicyHealthStatus.msg exactly:
//   state:  0 OK, 1 DEGRADED, 2 INTERVENE, 3 STOP
//   action: 0 NONE, 1 LOG_ONLY, 2 HOLD, 3 STOP_AND_HOLD, 4 REWIND
//
// Default banding on a normalized score in [0, 1] (0 healthy, 1 worst):
//   score <  0.25         -> OK,        action NONE
//   0.25 <= score < 0.50  -> DEGRADED,  action LOG_ONLY
//   0.50 <= score < 0.80  -> INTERVENE, action HOLD
//   score >= 0.80         -> STOP,      action STOP_AND_HOLD
//
// The cut points and the state-to-action pairing are a project choice (the
// message definitions fix only the enums). REWIND is never selected by score
// banding; it is a recovery-policy choice, exposed here only as a constant.
#ifndef PHM_CORE__SEVERITY_HPP_
#define PHM_CORE__SEVERITY_HPP_

#include <cstdint>

namespace phm_core
{

constexpr uint8_t STATE_OK = 0;
constexpr uint8_t STATE_DEGRADED = 1;
constexpr uint8_t STATE_INTERVENE = 2;
constexpr uint8_t STATE_STOP = 3;

constexpr uint8_t ACTION_NONE = 0;
constexpr uint8_t ACTION_LOG_ONLY = 1;
constexpr uint8_t ACTION_HOLD = 2;
constexpr uint8_t ACTION_STOP_AND_HOLD = 3;
constexpr uint8_t ACTION_REWIND = 4;

// A score >= a band's lower edge selects that band.
constexpr double DEGRADED_THRESHOLD = 0.25;
constexpr double INTERVENE_THRESHOLD = 0.50;
constexpr double STOP_THRESHOLD = 0.80;

// Score reported when a detector cannot produce a meaningful verdict because
// its input was malformed (dimension mismatch, non-finite embedding).
constexpr double BAD_INPUT_SCORE = 0.5;

struct Severity
{
  double score;
  uint8_t state;
  uint8_t suggested_action;
};

// Linear map of a raw signal onto [0, 1], clamped: healthy -> 0, worst -> 1.
// Pass healthy > worst when a LOW raw value is the unhealthy one (the
// rolling-spread collapse case). Throws std::invalid_argument when
// healthy == worst. A NaN raw value propagates as NaN.
double normalize(double raw, double healthy, double worst);

// Classify a normalized score (clamped to [0, 1] first) into a state and the
// suggested action of that band.
Severity classify(double score);

// The state of a score's band, the state field of classify(score).
uint8_t score_to_state(double score);

}  // namespace phm_core

#endif  // PHM_CORE__SEVERITY_HPP_
