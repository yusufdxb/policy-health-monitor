// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/ood_core.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "phm_core/calibration.hpp"
#include "phm_core/format.hpp"
#include "phm_core/severity.hpp"

namespace phm_core
{

OodCore::OodCore(OodConfig config, std::unique_ptr<SpreadBackend> backend)
: config_(std::move(config)), backend_(std::move(backend)),
  hysteresis_(config_.min_consecutive)
{
  if (config_.window < 2) {
    throw std::invalid_argument("window must be >= 2");
  }
  if (config_.compute_every < 1) {
    throw std::invalid_argument("compute_every must be >= 1");
  }
  if (!backend_) {
    throw std::invalid_argument("backend must not be null");
  }
  if (config_.window > kMaxWindowFloats ||
    config_.embed_dim > kMaxWindowFloats / config_.window)
  {
    throw std::invalid_argument("window * embed_dim exceeds the window buffer limit");
  }
  if (config_.embed_dim > 0) {
    dim_ = config_.embed_dim;
    dim_pinned_ = true;
  }
}

void OodCore::clear_window()
{
  filled_ = 0;
  next_slot_ = 0;
  frame_count_ = 0;
  has_spread_ = false;
  last_spread_ = 0.0;
  has_last_ = false;
  hysteresis_.reset();
}

void OodCore::reset()
{
  clear_window();
  if (config_.embed_dim == 0) {
    dim_ = 0;
    dim_pinned_ = false;
  }
  last_status_ = FrameStatus::kOk;
}

double OodCore::calibrate_from_data(const Matrix & in_dist_hidden, double percentile)
{
  config_.threshold = calibrate_threshold(
    rolling_spread(in_dist_hidden, config_.window), percentile);
  return config_.threshold;
}

const VerdictData & OodCore::bad_input(const std::string & reason)
{
  scratch_.source = config_.source;
  scratch_.score = BAD_INPUT_SCORE;
  scratch_.violating = false;
  scratch_.reason = reason;
  scratch_.suggested_action = ACTION_NONE;
  return scratch_;
}

const VerdictData & OodCore::update(
  const float * embedding, std::size_t n, const std::string & policy_id)
{
  // 1. Dimension validation. With embed_dim set the expected size is fixed;
  // otherwise the first non-empty frame pins it. Empty frames before that are
  // scored as zero-dimension frames, as the NumPy core scored them (an
  // all-empty stream has zero spread and reads as a collapse), but they never
  // pin the dimension: the first non-empty frame restarts the window at its
  // own dimension instead of being rejected for the rest of the run.
  if (dim_pinned_ && n != dim_) {
    last_status_ = FrameStatus::kDimMismatch;
    return bad_input(
      "dim mismatch: expected " + std::to_string(dim_) + ", got " + std::to_string(n));
  }
  if (!dim_pinned_ && n > 0) {
    if (n > kMaxWindowFloats / config_.window) {
      last_status_ = FrameStatus::kDimMismatch;
      return bad_input(
        "embedding too large: dim " + std::to_string(n) + " x window " +
        std::to_string(config_.window) + " exceeds the window buffer limit");
    }
    if (filled_ > 0) {
      clear_window();
    }
    dim_ = n;
    dim_pinned_ = true;
  }
  last_status_ = FrameStatus::kOk;
  if (ring_.size() != config_.window * dim_) {
    ring_.assign(config_.window * dim_, 0.0f);
  }

  // 2. Append to the ring (overwriting the oldest frame once full).
  if (dim_ > 0) {
    std::memcpy(ring_.data() + next_slot_ * dim_, embedding, dim_ * sizeof(float));
  }
  next_slot_ = (next_slot_ + 1) % config_.window;
  if (filled_ < config_.window) {
    ++filled_;
  }
  ++frame_count_;

  // 3. Warm-up until the window is full.
  if (filled_ < config_.window) {
    scratch_.source = config_.source;
    scratch_.score = 0.0;
    scratch_.violating = false;
    scratch_.reason = "warming up: ";
    scratch_.reason += std::to_string(filled_);
    scratch_.reason += '/';
    scratch_.reason += std::to_string(config_.window);
    scratch_.reason += " frames";
    scratch_.suggested_action = ACTION_NONE;
    return scratch_;
  }

  // 4. Frequency gate: replay the last computed verdict on skipped frames.
  if (has_last_ && (frame_count_ % config_.compute_every) != 0) {
    return last_;
  }

  // 5. Spread of the full window; when full, next_slot_ is the oldest frame.
  WindowView view{ring_.data(), config_.window, dim_, next_slot_};
  const double spread = backend_->spread(view);
  last_spread_ = spread;
  has_spread_ = true;

  // Fail closed on a non-finite spread. `NaN < threshold` is false, so without
  // this a window containing NaN or Inf would be reported healthy: it would
  // fail open on precisely the input a broken upstream policy emits. One
  // non-finite element propagates through the variance and the sum, so testing
  // the spread costs nothing per element. The verdict is a detector-health
  // fault (score at the intervene boundary, not violating, no action) and is
  // cached so the frequency gate carries it forward instead of replaying the
  // last good verdict.
  has_last_ = true;
  if (!std::isfinite(spread)) {
    last_.source = config_.source;
    last_.score = BAD_INPUT_SCORE;
    last_.violating = false;
    last_.reason = "non-finite spread: embedding contains NaN or Inf";
    last_.suggested_action = ACTION_NONE;
    return last_;
  }

  // 6. Threshold, hysteresis, severity banding.
  const bool raw_violating = spread < config_.threshold;
  const bool fired = hysteresis_.observe(raw_violating);
  make_verdict(spread, raw_violating, fired, policy_id, last_);
  return last_;
}

void OodCore::make_verdict(
  double spread, bool raw_violating, bool fired, const std::string & policy_id,
  VerdictData & out) const
{
  out.source = config_.source;
  std::string & reason = out.reason;
  reason.clear();
  if (!policy_id.empty()) {
    reason += '[';
    reason += policy_id;
    reason += "] ";
  }
  reason += "ood: rolling-spread ";
  fmt::append_fixed(reason, spread, 4);

  if (!raw_violating) {
    reason += " >= thr ";
    fmt::append_fixed(reason, config_.threshold, 4);
    out.score = 0.0;
    out.violating = false;
    out.suggested_action = ACTION_NONE;
    return;
  }

  // OOD: normalized severity, spread == threshold -> 0, spread == 0 -> 1.
  const double score = config_.threshold > 0.0 ?
    normalize(spread, config_.threshold, 0.0) : 1.0;
  reason += " < thr ";
  fmt::append_fixed(reason, config_.threshold, 4);
  reason += " for ";
  reason += std::to_string(hysteresis_.count());
  reason += " frame(s)";
  out.score = score;

  if (!fired) {
    reason += " (pre-hysteresis)";
    out.violating = false;
    out.suggested_action = ACTION_NONE;
    return;
  }

  // A post-hysteresis score below the DEGRADED floor is a zero-severity
  // signal: reporting it as violating would hand the arbiter a violating
  // verdict whose score bands to OK. Report it as a non-violating pass.
  if (score < DEGRADED_THRESHOLD) {
    reason += " (below severity floor)";
    out.violating = false;
    out.suggested_action = ACTION_NONE;
    return;
  }

  if (score >= STOP_THRESHOLD) {
    out.suggested_action = ACTION_STOP_AND_HOLD;
    reason += " [stop]";
  } else if (score >= INTERVENE_THRESHOLD) {
    out.suggested_action = ACTION_HOLD;
    reason += " [intervene]";
  } else {
    out.suggested_action = ACTION_LOG_ONLY;
    reason += " [degraded]";
  }
  out.violating = true;
}

}  // namespace phm_core
