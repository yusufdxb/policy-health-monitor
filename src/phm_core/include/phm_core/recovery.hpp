// Copyright 2026 Yusuf Guenena. MIT License.
// Recovery decision logic: SafetyEnvelope, HealthToActionMapper, RewindHook.
//
// SafetyEnvelope is ported from HELIX helix_recovery/recovery_node.py
// (per-action cooldown, allowlist, RESUME exempt from cooldown), extended to
// the PHM action set and decoupled from the HELIX message types.
#ifndef PHM_CORE__RECOVERY_HPP_
#define PHM_CORE__RECOVERY_HPP_

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "phm_core/detector.hpp"

namespace phm_core
{

enum class EnvelopeStatus
{
  kAccepted,
  kSuppressedDisabled,
  kSuppressedAllowlist,
  kSuppressedCooldown,
};

// "ACCEPTED", "SUPPRESSED_DISABLED", "SUPPRESSED_ALLOWLIST", "SUPPRESSED_COOLDOWN".
const char * to_string(EnvelopeStatus status);
// "NONE", "LOG_ONLY", "HOLD", "STOP_AND_HOLD", "REWIND", or the number.
std::string action_name(int action);

struct EnvelopeResult
{
  EnvelopeStatus status = EnvelopeStatus::kAccepted;
  bool publish = false;  // whether the caller should actuate
  std::string reason;
};

// Gate for requested actions. Disabled suppresses everything. Actions outside
// the allowlist are suppressed. HOLD and STOP_AND_HOLD are cooldown-damped per
// (action, fault_key): a NEW hold within cooldown_seconds of the last one for
// the same key is suppressed (strict <, so exactly cooldown_seconds later is
// accepted), but a continuation of an already-active hold is exempt and keeps
// publishing. REWIND actuates without cooldown; NONE and LOG_ONLY are accepted
// without actuation. RESUME (clearing a hold) is never rate-limited.
class SafetyEnvelope
{
public:
  SafetyEnvelope(bool enabled, double cooldown_seconds);

  EnvelopeResult evaluate(
    int action, const std::string & fault_key, double now, bool hold_already_active = false);
  EnvelopeResult evaluate_resume(const std::string & fault_key, double now) const;

  bool enabled() const {return enabled_;}
  double cooldown_seconds() const {return cooldown_seconds_;}

private:
  bool enabled_;
  double cooldown_seconds_;
  std::map<std::pair<int, std::string>, double> last_action_time_;
};

struct HealthActionDecision
{
  uint8_t action = 0;        // ACTION_* constant
  bool hold_active = false;  // the zero-velocity hold should be active
  std::string reason;
};

// Maps a health state + suggested action to an actuation decision:
//   STOP                      -> STOP_AND_HOLD, hold on (whatever the action)
//   INTERVENE + REWIND        -> REWIND, hold on
//   INTERVENE + anything else -> HOLD, hold on (never downgraded below a hold)
//   OK / DEGRADED, hold on    -> NONE, hold cleared
//   OK / DEGRADED, no hold    -> LOG_ONLY if suggested, else NONE; no actuation
//   unknown state             -> STOP_AND_HOLD, hold on (conservative)
class HealthToActionMapper
{
public:
  HealthActionDecision map(
    int state, int suggested_action, const std::string & source, const std::string & reason);
  bool hold_active() const {return hold_active_;}
  void clear_hold() {hold_active_ = false;}

private:
  bool hold_active_ = false;
};

// Pluggable callback for ACTION_REWIND. Without a registered callback, a
// trigger only logs a warning (the recovery node holds independently). A host
// stack registers its return-to-last-safe-waypoint behavior here.
class RewindHook
{
public:
  void register_callback(std::function<void()> callback) {callback_ = std::move(callback);}
  void set_logger(LogFn log) {log_ = std::move(log);}
  bool has_callback() const {return static_cast<bool>(callback_);}
  void trigger() const;

private:
  std::function<void()> callback_;
  LogFn log_;
};

// What one health message did, for logging and tests.
struct RecoveryStep
{
  HealthActionDecision decision;
  bool resume_evaluated = false;         // the RESUME path ran
  std::optional<EnvelopeResult> result;  // the envelope result (RESUME or action)
  bool rewind_triggered = false;
  bool hold_actuating = false;           // actuation state after this message
};

// The recovery node's per-message logic, ROS-free. A hold actuates (and the
// node's timer publishes zero velocity) only when the envelope says publish;
// cooldown damps NEW holds only, a continuation of an actuating hold is
// exempt, and OK / DEGRADED releases the hold through a cooldown-exempt
// RESUME. The fault key is the health source, or "unknown" when empty.
class RecoveryController
{
public:
  RecoveryController(bool enabled, double cooldown_seconds);

  RecoveryStep on_health(
    int state, int suggested_action, const std::string & source, const std::string & reason,
    double now);

  bool hold_actuating() const {return hold_actuating_;}
  RewindHook & rewind_hook() {return rewind_hook_;}
  const SafetyEnvelope & envelope() const {return envelope_;}

private:
  SafetyEnvelope envelope_;
  HealthToActionMapper mapper_;
  RewindHook rewind_hook_;
  bool hold_actuating_ = false;
};

}  // namespace phm_core

#endif  // PHM_CORE__RECOVERY_HPP_
