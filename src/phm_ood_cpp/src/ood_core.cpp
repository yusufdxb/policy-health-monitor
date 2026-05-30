// Copyright 2026 Yusuf Guenena. MIT License.
// ROS-free OOD core. See ood_core.hpp for the Python source mapping.
#include "phm_ood_cpp/ood_core.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

#include "phm_ood_cpp/severity.hpp"

namespace phm_ood_cpp
{

// ---------------------------------------------------------------------------
// Hysteresis (hysteresis.py:48-69)
// ---------------------------------------------------------------------------
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
    count_ = 0;  // healthy resets the run (threshold.py:108-109)
    return false;
  }
  ++count_;  // violating extends the run (threshold.py:112-114)
  return count_ >= min_consecutive_;
}

// ---------------------------------------------------------------------------
// OodCore
// ---------------------------------------------------------------------------
OodCore::OodCore(
  std::size_t window, double threshold, int min_consecutive,
  int compute_every, std::unique_ptr<Backend> backend)
: window_(window),
  threshold_(threshold),
  compute_every_(compute_every < 1 ? 1 : compute_every),
  backend_(std::move(backend)),
  hysteresis_(min_consecutive)
{
  if (window < 2) {
    throw std::invalid_argument("window must be >= 2");
  }
  if (!backend_) {
    throw std::invalid_argument("backend must not be null");
  }
  backend_name_ = backend_->name();
  buffer_.reserve(window_);
}

VerdictData OodCore::ok_verdict(const std::string & reason) const
{
  VerdictData v;
  v.source = SOURCE;
  v.score = 0.0;
  v.violating = false;
  v.reason = reason;
  v.suggested_action = ACTION_NONE;
  return v;
}

VerdictData OodCore::update(
  const std::vector<float> & embedding, const std::string & policy_id)
{
  // First frame fixes the dimension; later frames must match (mirrors the
  // Python node's dim check on PolicyEmbedding).
  if (dim_ == 0) {
    if (embedding.empty()) {
      throw std::invalid_argument("embedding must be non-empty");
    }
    dim_ = embedding.size();
  } else if (embedding.size() != dim_) {
    throw std::invalid_argument("embedding dim changed between frames");
  }

  // Append and trim to window (_core.py:180-182).
  buffer_.push_back(embedding);
  if (buffer_.size() > window_) {
    buffer_.erase(buffer_.begin());
  }
  ++frame_count_;

  // Buffer not yet full -> warming-up OK verdict (_core.py:187-190).
  if (buffer_.size() < window_) {
    char buf[64];
    std::snprintf(
      buf, sizeof(buf), "warming up: %zu/%zu frames", buffer_.size(), window_);
    return ok_verdict(buf);
  }

  // Frequency gate: re-use last verdict on non-compute frames (_core.py:192-196).
  if (have_last_ && (frame_count_ % compute_every_ != 0)) {
    return last_verdict_;
  }

  // Flatten the current window into a row-major (window x dim) block of doubles
  // and hand it to the backend (the rolling_spread contract).
  std::vector<double> flat(window_ * dim_);
  for (std::size_t f = 0; f < window_; ++f) {
    const std::vector<float> & row = buffer_[f];
    for (std::size_t d = 0; d < dim_; ++d) {
      flat[f * dim_ + d] = static_cast<double>(row[d]);
    }
  }
  const double spread = backend_->rolling_spread(flat, window_, dim_);
  last_spread_ = spread;

  const bool raw_violating = spread < threshold_;
  const bool fired = hysteresis_.observe(raw_violating);

  VerdictData v = make_verdict(spread, raw_violating, fired, policy_id);
  last_verdict_ = v;
  have_last_ = true;
  return v;
}

VerdictData OodCore::make_verdict(
  double spread, bool raw_violating, bool fired, const std::string & policy_id)
{
  // Mirrors _core.py:226-325 exactly.
  const std::string pid = policy_id.empty() ? std::string() : "[" + policy_id + "] ";
  const int n_consec = hysteresis_.count();
  char num[128];

  if (!raw_violating) {
    std::snprintf(
      num, sizeof(num), "ood: rolling-spread %.4f >= thr %.4f", spread, threshold_);
    VerdictData v;
    v.score = 0.0;
    v.violating = false;
    v.reason = pid + num;
    v.suggested_action = ACTION_NONE;
    return v;
  }

  // OOD: normalized severity (_core.py:266-271).
  double score;
  if (threshold_ > 0.0) {
    score = normalize(spread, /*healthy=*/threshold_, /*worst=*/0.0);
  } else {
    score = 1.0;
  }

  std::snprintf(
    num, sizeof(num), "ood: rolling-spread %.4f < thr %.4f for %d frame(s)",
    spread, threshold_, n_consec);
  const std::string base = pid + num;

  if (!fired) {
    // Pre-hysteresis (_core.py:279-287).
    VerdictData v;
    v.score = score;
    v.violating = false;
    v.reason = base + " (pre-hysteresis)";
    v.suggested_action = ACTION_NONE;
    return v;
  }

  // Below severity floor: report non-violating (_core.py:294-301).
  if (score < DEGRADED_THRESHOLD) {
    VerdictData v;
    v.score = score;
    v.violating = false;
    v.reason = base + " (below severity floor)";
    v.suggested_action = ACTION_NONE;
    return v;
  }

  // Post-hysteresis action banding (_core.py:307-325).
  uint8_t action;
  const char * band;
  if (score >= STOP_THRESHOLD) {
    action = ACTION_STOP_AND_HOLD;
    band = "stop";
  } else if (score >= INTERVENE_THRESHOLD) {
    action = ACTION_HOLD;
    band = "intervene";
  } else {
    action = ACTION_LOG_ONLY;
    band = "degraded";
  }

  VerdictData v;
  v.score = score;
  v.violating = true;
  v.reason = base + " [" + band + "]";
  v.suggested_action = action;
  return v;
}

}  // namespace phm_ood_cpp
