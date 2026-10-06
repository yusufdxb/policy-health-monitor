// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/adapters.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "phm_core/calibration.hpp"
#include "phm_core/format.hpp"
#include "phm_core/numerics.hpp"
#include "phm_core/severity.hpp"

namespace phm_core
{

// ---------------------------------------------------------------------------
// FrequencyDropAdapter (frequency.py:52-152)
// ---------------------------------------------------------------------------
FrequencyDropAdapter::FrequencyDropAdapter(
  const std::string & target_topic, double tolerance_percent, int min_consecutive)
: Detector("freq:" + target_topic, target_topic),
  tolerance_pct_(tolerance_percent), hysteresis_(min_consecutive)
{
  samples_.reserve(kFrequencyLearningSamples);
}

std::optional<VerdictData> FrequencyDropAdapter::update(const FrequencySample & sample)
{
  if (sample.topic != target_topic_) {
    return std::nullopt;
  }
  const double hz = sample.frequency_hz;

  // Learning phase: average the first samples into a baseline.
  if (!has_baseline_) {
    samples_.push_back(hz);
    if (static_cast<int>(samples_.size()) < kFrequencyLearningSamples) {
      return std::nullopt;
    }
    // Python's sum() over a list is a left-to-right running sum.
    double total = 0.0;
    for (double s : samples_) {
      total += s;
    }
    baseline_ = total / static_cast<double>(samples_.size());
    has_baseline_ = true;
    samples_.clear();
    if (has_logger()) {
      log(
        LogLevel::kInfo,
        "FrequencyDropAdapter baseline for " + target_topic_ + ": " + fmt::fixed(baseline_, 2) +
        " Hz");
    }
    return std::nullopt;
  }

  // Monitoring phase.
  const double floor_hz = baseline_ * (1.0 - tolerance_pct_ / 100.0);
  const bool raw_violating = hz < floor_hz;
  const bool post_hyst = hysteresis_.observe(raw_violating);

  // Score: 0 at the baseline rate, 1 at 0 Hz; an over-rate topic stays 0.
  double score = baseline_ > 0.0 ? normalize(hz, baseline_, 0.0) : 0.0;
  score = std::max(0.0, std::min(1.0, score));
  const Severity sev = classify(score);

  VerdictData v;
  v.source = name_;
  v.score = score;
  v.violating = post_hyst;
  v.suggested_action = sev.suggested_action;
  if (raw_violating) {
    v.reason = "freq:" + target_topic_ + " " + fmt::fixed(hz, 2) + " Hz below floor " +
      fmt::fixed(floor_hz, 2) + " Hz (baseline " + fmt::fixed(baseline_, 2) +
      " Hz, tolerance " + fmt::repr(tolerance_pct_) + "%)";
    if (has_logger()) {
      log(
        LogLevel::kWarn,
        "FrequencyDropAdapter " + target_topic_ + ": " + fmt::fixed(hz, 2) + " Hz < floor " +
        fmt::fixed(floor_hz, 2) + " Hz");
    }
  } else {
    v.reason = "freq:" + target_topic_ + " " + fmt::fixed(hz, 2) + " Hz healthy (baseline " +
      fmt::fixed(baseline_, 2) + " Hz)";
  }
  return v;
}

// ---------------------------------------------------------------------------
// StaticThresholdAdapter (threshold.py:75-154)
// ---------------------------------------------------------------------------
StaticThresholdAdapter::StaticThresholdAdapter(
  const std::string & target_topic, const std::string & metric, int min_consecutive)
: Detector("threshold:" + metric, target_topic), metric_(metric),
  hysteresis_(min_consecutive)
{
}

std::optional<VerdictData> StaticThresholdAdapter::update(const ThresholdSample & sample)
{
  if (sample.metric != metric_) {
    return std::nullopt;
  }
  const double value = sample.value;
  const double threshold = sample.threshold;
  const bool raw_violating = value > threshold;
  const bool post_hyst = hysteresis_.observe(raw_violating);

  // Score: 0 at value 0, 1 at twice the limit (0.5 at the limit itself).
  const double worst = threshold > 0.0 ? threshold * 2.0 : 1.0;
  const double score = normalize(value, 0.0, worst);
  const Severity sev = classify(score);

  VerdictData v;
  v.source = name_;
  v.score = score;
  v.violating = post_hyst;
  v.suggested_action = sev.suggested_action;
  if (raw_violating) {
    v.reason = "threshold:" + metric_ + " " + fmt::fixed(value, 2) + " exceeds limit " +
      fmt::fixed(threshold, 2);
    if (has_logger()) {
      log(
        LogLevel::kWarn,
        "StaticThresholdAdapter " + metric_ + ": " + fmt::fixed(value, 2) + " > " +
        fmt::fixed(threshold, 2));
    }
  } else {
    v.reason = "threshold:" + metric_ + " " + fmt::fixed(value, 2) + " within limit " +
      fmt::fixed(threshold, 2);
  }
  return v;
}

// ---------------------------------------------------------------------------
// DeadTopicAdapter (dead_topic.py:50-127)
// ---------------------------------------------------------------------------
DeadTopicAdapter::DeadTopicAdapter(const std::string & target_topic, double timeout_sec)
: Detector("dead:" + target_topic, target_topic), timeout_sec_(timeout_sec)
{
}

void DeadTopicAdapter::mark_alive(double now_sec)
{
  alerted_ = false;
  if (has_logger()) {
    log(
      LogLevel::kDebug,
      "DeadTopicAdapter " + target_topic_ + ": topic alive at " + fmt::fixed(now_sec, 3));
  }
}

std::optional<VerdictData> DeadTopicAdapter::update(const DeadTopicSample & sample)
{
  if (sample.topic != target_topic_) {
    return std::nullopt;
  }
  const double elapsed = sample.now_sec - sample.last_seen_sec;
  const bool raw_violating = elapsed > timeout_sec_;
  const double worst = timeout_sec_ > 0.0 ? timeout_sec_ * 2.0 : 1.0;
  const double score = normalize(elapsed, 0.0, worst);
  const Severity sev = classify(score);

  VerdictData v;
  v.source = name_;
  v.score = score;
  v.violating = raw_violating;
  if (raw_violating) {
    // Only log the first alert per silence window (dead_topic.py:94).
    if (!alerted_) {
      if (has_logger()) {
        log(
          LogLevel::kWarn,
          "DeadTopicAdapter " + target_topic_ + ": silent " + fmt::fixed(elapsed, 1) +
          "s > timeout " + fmt::fixed(timeout_sec_, 1) + "s");
      }
      alerted_ = true;
    }
    v.reason = "dead:" + target_topic_ + " silent " + fmt::fixed(elapsed, 1) + "s (timeout " +
      fmt::repr(timeout_sec_) + "s)";
    // A dead topic is unrecoverable until it resumes; the arbiter may
    // downgrade per its policy.
    v.suggested_action = ACTION_STOP_AND_HOLD;
  } else {
    alerted_ = false;
    v.reason = "dead:" + target_topic_ + " alive (last seen " + fmt::fixed(elapsed, 2) +
      "s ago)";
    v.suggested_action = sev.suggested_action;
  }
  return v;
}

// ---------------------------------------------------------------------------
// RecurrentTemporalSpreadAdapter (supercombo-blindspot e6_detector.py:16-31)
// ---------------------------------------------------------------------------
RecurrentTemporalSpreadAdapter::RecurrentTemporalSpreadAdapter(
  const std::string & target_topic, std::size_t window, double threshold, int min_consecutive)
: Detector("recurrent_temporal_spread:" + target_topic, target_topic),
  window_(window), threshold_(threshold), hysteresis_(min_consecutive)
{
  if (window < 2) {
    throw std::invalid_argument("window must be >= 2");
  }
}

double RecurrentTemporalSpreadAdapter::calibrate_from_data(
  const Matrix & in_dist_hidden, double percentile)
{
  threshold_ = calibrate_threshold(rolling_spread(in_dist_hidden, window_), percentile);
  return threshold_;
}

std::optional<VerdictData> RecurrentTemporalSpreadAdapter::update(
  const RecurrentSpreadSample & sample)
{
  if (sample.topic != target_topic_) {
    return std::nullopt;
  }
  buffer_.push_back(sample.embedding);
  if (buffer_.size() > window_) {
    buffer_.erase(buffer_.begin());
  }

  VerdictData v;
  v.source = name_;
  if (buffer_.size() < window_) {
    v.score = 0.0;
    v.violating = false;
    v.reason = "recurrent_temporal_spread:" + target_topic_ + " warming up: " +
      std::to_string(buffer_.size()) + "/" + std::to_string(window_) + " frames";
    v.suggested_action = ACTION_NONE;
    return v;
  }

  // Spread of the full window (rolling_spread(hidden, window)[-1]).
  const std::size_t dim = buffer_.front().size();
  Matrix hidden(window_, dim);
  for (std::size_t r = 0; r < window_; ++r) {
    if (buffer_[r].size() != dim) {
      throw std::invalid_argument("RecurrentTemporalSpreadAdapter: embedding dim changed");
    }
    std::copy(buffer_[r].begin(), buffer_[r].end(), hidden.row(r));
  }
  const double spread = block_spread(hidden);
  last_spread_ = spread;
  has_spread_ = true;

  const bool raw_violating = spread < threshold_;
  const bool post_hyst = hysteresis_.observe(raw_violating);
  double score;
  if (threshold_ > 0.0) {
    score = normalize(spread, threshold_, 0.0);
  } else {
    score = raw_violating ? 1.0 : 0.0;
  }
  const Severity sev = classify(score);

  v.score = score;
  v.violating = post_hyst;
  v.suggested_action = sev.suggested_action;
  const std::string tail = " (window " + std::to_string(window_) + ")";
  if (raw_violating) {
    v.reason = "recurrent_temporal_spread:" + target_topic_ + " spread " + fmt::fixed(spread, 6) +
      " < threshold " + fmt::fixed(threshold_, 6) + tail;
    if (has_logger()) {
      log(
        LogLevel::kWarn,
        "RecurrentTemporalSpreadAdapter " + target_topic_ + ": spread " + fmt::fixed(spread, 6) +
        " < threshold " + fmt::fixed(threshold_, 6));
    }
  } else {
    v.reason = "recurrent_temporal_spread:" + target_topic_ + " spread " + fmt::fixed(spread, 6) +
      " >= threshold " + fmt::fixed(threshold_, 6) + tail;
  }
  return v;
}

}  // namespace phm_core
