// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/severity.hpp"

#include <limits>
#include <stdexcept>

#include "phm_core/hysteresis.hpp"

namespace phm_core
{

double normalize(double raw, double healthy, double worst)
{
  if (healthy == worst) {
    throw std::invalid_argument("healthy and worst must differ to define a scale");
  }
  const double frac = (raw - healthy) / (worst - healthy);
  if (frac < 0.0) {
    return 0.0;
  }
  if (frac > 1.0) {
    return 1.0;
  }
  return frac;
}

Severity classify(double score)
{
  const double s = score < 0.0 ? 0.0 : (score > 1.0 ? 1.0 : score);
  if (s >= STOP_THRESHOLD) {
    return Severity{s, STATE_STOP, ACTION_STOP_AND_HOLD};
  }
  if (s >= INTERVENE_THRESHOLD) {
    return Severity{s, STATE_INTERVENE, ACTION_HOLD};
  }
  if (s >= DEGRADED_THRESHOLD) {
    return Severity{s, STATE_DEGRADED, ACTION_LOG_ONLY};
  }
  return Severity{s, STATE_OK, ACTION_NONE};
}

uint8_t score_to_state(double score)
{
  return classify(score).state;
}

Hysteresis::Hysteresis(int min_consecutive)
: min_consecutive_(min_consecutive)
{
  if (min_consecutive < 1) {
    throw std::invalid_argument("min_consecutive must be >= 1");
  }
}

bool Hysteresis::observe(bool violating)
{
  if (!violating) {
    count_ = 0;  // a healthy sample resets the run
    return false;
  }
  // A violating sample extends the run. The count saturates instead of
  // overflowing; it is far above any min_consecutive by then.
  if (count_ < std::numeric_limits<int>::max()) {
    ++count_;
  }
  return count_ >= min_consecutive_;
}

}  // namespace phm_core
