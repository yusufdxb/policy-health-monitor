// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/arbiter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "phm_core/severity.hpp"

namespace phm_core
{

bool action_in_allowlist(uint8_t action)
{
  return action <= ACTION_REWIND;
}

HealthStatusData arbitrate(
  const std::vector<ArbiterInput> & verdicts, double now, double staleness_sec)
{
  enum class Kind { kBadScore, kStale, kViolating };
  bool have = false;
  uint8_t best_state = 0;
  double best_score = 0.0;
  std::size_t best_index = 0;
  Kind best_kind = Kind::kViolating;
  uint8_t best_action = ACTION_NONE;

  for (std::size_t i = 0; i < verdicts.size(); ++i) {
    const ArbiterInput & v = verdicts[i];
    uint8_t state;
    double score;
    Kind kind;
    uint8_t action;
    if (!std::isfinite(v.score)) {
      state = STATE_DEGRADED;
      score = kBadScoreSentinel;
      kind = Kind::kBadScore;
      action = ACTION_NONE;
    } else {
      const double age = v.has_timestamp ? now - v.timestamp : 0.0;
      if (age > staleness_sec) {
        if (v.violating) {
          state = std::max(STATE_DEGRADED, score_to_state(v.score));
          score = std::max(kStaleScore, v.score);
        } else {
          state = STATE_DEGRADED;
          score = kStaleScore;
        }
        kind = Kind::kStale;
        action = ACTION_LOG_ONLY;
      } else if (v.violating) {
        state = std::max(STATE_DEGRADED, score_to_state(v.score));
        score = v.score;
        kind = Kind::kViolating;
        action = action_in_allowlist(v.suggested_action) ? v.suggested_action : ACTION_NONE;
      } else {
        continue;  // fresh and healthy: contributes nothing
      }
    }
    // Strictly greater (state, score) replaces the incumbent, so the first of
    // equal candidates wins, as Python's max() did.
    if (!have || state > best_state || (state == best_state && score > best_score)) {
      have = true;
      best_state = state;
      best_score = score;
      best_index = i;
      best_kind = kind;
      best_action = action;
    }
  }

  HealthStatusData out;
  if (!have) {
    out.state = STATE_OK;
    out.score = 0.0;
    out.reason = "all detectors nominal";
    out.suggested_action = ACTION_NONE;
    return out;
  }
  const ArbiterInput & w = verdicts[best_index];
  out.state = best_state;
  out.score = best_score;
  out.source = w.source;
  out.suggested_action = best_action;
  switch (best_kind) {
    case Kind::kBadScore:
      out.reason = "bad-score:" + w.source;
      break;
    case Kind::kStale:
      out.reason = "stale:" + w.source;
      break;
    case Kind::kViolating:
      out.reason = w.reason;
      break;
  }
  return out;
}

}  // namespace phm_core
