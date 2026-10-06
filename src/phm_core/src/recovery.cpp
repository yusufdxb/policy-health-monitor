// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/recovery.hpp"

#include <string>
#include <utility>

#include "phm_core/format.hpp"
#include "phm_core/severity.hpp"

namespace phm_core
{

const char * to_string(EnvelopeStatus status)
{
  switch (status) {
    case EnvelopeStatus::kAccepted:
      return "ACCEPTED";
    case EnvelopeStatus::kSuppressedDisabled:
      return "SUPPRESSED_DISABLED";
    case EnvelopeStatus::kSuppressedAllowlist:
      return "SUPPRESSED_ALLOWLIST";
    case EnvelopeStatus::kSuppressedCooldown:
      return "SUPPRESSED_COOLDOWN";
  }
  return "UNKNOWN";
}

std::string action_name(int action)
{
  switch (action) {
    case ACTION_NONE:
      return "NONE";
    case ACTION_LOG_ONLY:
      return "LOG_ONLY";
    case ACTION_HOLD:
      return "HOLD";
    case ACTION_STOP_AND_HOLD:
      return "STOP_AND_HOLD";
    case ACTION_REWIND:
      return "REWIND";
    default:
      return std::to_string(action);
  }
}

namespace
{

bool in_allowlist(int action)
{
  return action >= ACTION_NONE && action <= ACTION_REWIND;
}

bool actuates(int action)
{
  return action == ACTION_HOLD || action == ACTION_STOP_AND_HOLD || action == ACTION_REWIND;
}

}  // namespace

SafetyEnvelope::SafetyEnvelope(bool enabled, double cooldown_seconds)
: enabled_(enabled), cooldown_seconds_(cooldown_seconds)
{
}

EnvelopeResult SafetyEnvelope::evaluate(
  int action, const std::string & fault_key, double now, bool hold_already_active)
{
  if (!enabled_) {
    return {EnvelopeStatus::kSuppressedDisabled, false, "recovery.enabled is false"};
  }
  const std::string name = action_name(action);
  if (!in_allowlist(action)) {
    return {EnvelopeStatus::kSuppressedAllowlist, false, "action " + name + " not in allowlist"};
  }
  // Cooldown damps NEW holds only. A continuation of an active hold is exempt
  // and must never be released by the cooldown gate (that would let the robot
  // move mid-fault); it does not touch the cooldown clock either.
  if (action == ACTION_HOLD || action == ACTION_STOP_AND_HOLD) {
    if (hold_already_active) {
      return {
        EnvelopeStatus::kAccepted, true, "continue active hold (" + name + ", cooldown-exempt)"};
    }
    const auto key = std::make_pair(action, fault_key);
    const auto it = last_action_time_.find(key);
    if (it != last_action_time_.end() && (now - it->second) < cooldown_seconds_) {
      const double elapsed = now - it->second;
      // publish=false means "do not START a new hold"; the caller must not
      // clear an existing hold on this result.
      return {
        EnvelopeStatus::kSuppressedCooldown, false,
        "cooldown active for " + fault_key + " (" + fmt::fixed(elapsed, 2) + "s < " +
        fmt::fixed(cooldown_seconds_, 2) + "s)"};
    }
    last_action_time_[key] = now;
  }
  return {EnvelopeStatus::kAccepted, actuates(action), "action " + name + " accepted"};
}

EnvelopeResult SafetyEnvelope::evaluate_resume(const std::string & fault_key, double) const
{
  if (!enabled_) {
    return {EnvelopeStatus::kSuppressedDisabled, false, "recovery.enabled is false"};
  }
  return {
    EnvelopeStatus::kAccepted, true, "RESUME accepted for " + fault_key + " (cooldown exempt)"};
}

HealthActionDecision HealthToActionMapper::map(
  int state, int suggested_action, const std::string & source, const std::string & reason)
{
  if (state == STATE_STOP) {
    hold_active_ = true;
    return {ACTION_STOP_AND_HOLD, true, "STATE_STOP from " + source + ": " + reason};
  }
  if (state == STATE_INTERVENE) {
    // INTERVENE always actuates at least a HOLD; the suggested action can only
    // escalate it (REWIND also invokes the rewind hook), never downgrade it.
    hold_active_ = true;
    if (suggested_action == ACTION_REWIND) {
      return {ACTION_REWIND, true, "ACTION_REWIND from " + source + ": " + reason};
    }
    return {ACTION_HOLD, true, "STATE_INTERVENE/HOLD from " + source + ": " + reason};
  }
  if (state == STATE_OK || state == STATE_DEGRADED) {
    if (hold_active_) {
      hold_active_ = false;
      return {
        ACTION_NONE, false,
        "hold cleared: state=" + std::to_string(state) + " from " + source};
    }
    const uint8_t action = suggested_action == ACTION_LOG_ONLY ? ACTION_LOG_ONLY : ACTION_NONE;
    return {
      action, false, "state " + std::to_string(state) + " from " + source + ": no actuation"};
  }
  // Unknown state: treat conservatively as STOP.
  hold_active_ = true;
  return {
    ACTION_STOP_AND_HOLD, true,
    "unknown state " + std::to_string(state) + " from " + source +
    ": conservative STOP_AND_HOLD"};
}

void RewindHook::trigger() const
{
  if (callback_) {
    callback_();
    return;
  }
  if (log_) {
    log_(
      LogLevel::kWarn,
      "ACTION_REWIND triggered but no rewind callback registered; holding and logging. "
      "Register a callback with RewindHook::register_callback() to implement "
      "return-to-last-safe-waypoint.");
  }
}

RecoveryController::RecoveryController(bool enabled, double cooldown_seconds)
: envelope_(enabled, cooldown_seconds)
{
}

RecoveryStep RecoveryController::on_health(
  int state, int suggested_action, const std::string & source, const std::string & reason,
  double now)
{
  RecoveryStep step;
  const std::string fault_key = source.empty() ? std::string("unknown") : source;
  step.decision = mapper_.map(state, suggested_action, source, reason);

  // The state recovered and the mapper released its hold: release the
  // actuation through a cooldown-exempt RESUME.
  if (!step.decision.hold_active && (state == STATE_OK || state == STATE_DEGRADED)) {
    step.resume_evaluated = true;
    step.result = envelope_.evaluate_resume(fault_key, now);
    if (step.result->publish) {
      hold_actuating_ = false;
    }
    step.hold_actuating = hold_actuating_;
    return step;
  }
  const uint8_t action = step.decision.action;
  if (action == ACTION_NONE) {
    step.hold_actuating = hold_actuating_;
    return step;
  }
  const bool continuing = hold_actuating_ &&
    (action == ACTION_HOLD || action == ACTION_STOP_AND_HOLD || action == ACTION_REWIND);
  step.result = envelope_.evaluate(action, fault_key, now, continuing);
  // A suppressed re-assert never clears an ongoing hold.
  if (step.result->publish) {
    if (action == ACTION_HOLD || action == ACTION_STOP_AND_HOLD) {
      hold_actuating_ = true;
    } else if (action == ACTION_REWIND) {
      hold_actuating_ = true;
      step.rewind_triggered = true;
      rewind_hook_.trigger();
    }
  }
  step.hold_actuating = hold_actuating_;
  return step;
}

}  // namespace phm_core
