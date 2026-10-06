// Copyright 2026 Yusuf Guenena. MIT License.
// Rolling-spread OOD detector core, shared by the phm_ood and phm_ood_cpp
// nodes (they differ only in configuration and ROS interface).
//
// Each update() appends one embedding frame to a fixed ring buffer of `window`
// frames. Once the window is full it computes the rolling spread, compares it
// to the calibrated threshold (spread < threshold means OOD), applies
// consecutive-frame hysteresis and returns a verdict. The decision sequence is
// the one the original Python OodCore implemented:
//
//   1. dimension check      -> bad-input verdict (score 0.5, not violating);
//                              the frame does not enter the window
//   2. append, count frame
//   3. window not full      -> "warming up: k/W frames", score 0
//   4. frequency gate       -> replay the last computed verdict unless
//                              frame_count % compute_every == 0
//   5. non-finite spread    -> bad-input verdict, cached so the gate replays it
//   6. threshold + hysteresis + severity banding:
//        spread >= threshold                    -> score 0, not violating
//        below threshold, run < min_consecutive -> "(pre-hysteresis)"
//        fired, score < DEGRADED                -> "(below severity floor)"
//        fired, DEGRADED <= score < INTERVENE   -> violating, LOG_ONLY
//        fired, INTERVENE <= score < STOP       -> violating, HOLD
//        fired, score >= STOP                   -> violating, STOP_AND_HOLD
//      with score = normalize(spread, healthy=threshold, worst=0) (1.0 when
//      the threshold is not positive).
//
// The ring buffer and the reason strings are allocated once and reused, so a
// steady-state update allocates nothing in the plain backend.
#ifndef PHM_CORE__OOD_CORE_HPP_
#define PHM_CORE__OOD_CORE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "phm_core/detector.hpp"
#include "phm_core/hysteresis.hpp"
#include "phm_core/matrix.hpp"
#include "phm_core/spread_backend.hpp"

namespace phm_core
{

// Upper bound on window * embedding dimension, in floats (1 GiB of ring
// buffer). A configuration or first frame that would need more is refused
// instead of overflowing the size computation or exhausting memory.
constexpr std::size_t kMaxWindowFloats = std::size_t{1} << 28;

struct OodConfig
{
  std::size_t window = 30;     // frames in the rolling covariance, >= 2
  double threshold = 0.0;      // spread < threshold flags OOD
  int min_consecutive = 3;     // hysteresis run length, >= 1
  int compute_every = 1;       // frequency gate, >= 1
  std::size_t embed_dim = 0;   // expected dimension; 0 = pin to the first frame
  std::string source = "phm_ood";
};

enum class FrameStatus
{
  kOk,            // frame accepted (warm-up, computed or gated verdict)
  kDimMismatch,   // wrong, empty or oversized dimension; a bad-input verdict, frame not used
};

class OodCore
{
public:
  // Throws std::invalid_argument for window < 2, min_consecutive < 1,
  // compute_every < 1 or a null backend.
  OodCore(OodConfig config, std::unique_ptr<SpreadBackend> backend);

  // Feed one float32 frame (the PolicyEmbedding element type). The returned
  // reference stays valid until the next update() or reset().
  const VerdictData & update(const float * embedding, std::size_t n, const std::string & policy_id);
  const VerdictData & update(const std::vector<float> & embedding, const std::string & policy_id)
  {
    return update(embedding.data(), embedding.size(), policy_id);
  }

  FrameStatus last_status() const {return last_status_;}

  // Drop all rolling state: window, frame counter, hysteresis run, cached
  // verdict and the pinned dimension. Configuration is kept.
  void reset();

  // Calibrate the threshold on in-distribution frames (T x D, float64) with
  // calibrate_threshold(rolling_spread(frames, window), percentile).
  double calibrate_from_data(const Matrix & in_dist_hidden, double percentile = 1.0);
  void set_threshold(double threshold) {config_.threshold = threshold;}

  const OodConfig & config() const {return config_;}
  double threshold() const {return config_.threshold;}
  std::size_t window() const {return config_.window;}
  // Pinned embedding dimension, 0 until the first frame is accepted.
  std::size_t dim() const {return dim_;}
  bool has_spread() const {return has_spread_;}
  // Most recently computed spread (NaN-able); valid when has_spread().
  double last_spread() const {return last_spread_;}
  int hysteresis_count() const {return hysteresis_.count();}
  const char * backend_name() const {return backend_->name();}

  // Severity banding of one post-threshold decision, exposed for tests:
  // fills `out` exactly as step 6 above does.
  void make_verdict(
    double spread, bool raw_violating, bool fired, const std::string & policy_id,
    VerdictData & out) const;

private:
  const VerdictData & bad_input(const std::string & reason);
  void clear_window();

  OodConfig config_;
  std::unique_ptr<SpreadBackend> backend_;
  Hysteresis hysteresis_;

  std::vector<float> ring_;     // window * dim floats, allocated on first frame
  std::size_t dim_ = 0;         // 0 while unpinned (empty frames only so far)
  bool dim_pinned_ = false;     // pinned by embed_dim or the first non-empty frame
  std::size_t filled_ = 0;      // frames in the ring, <= window
  std::size_t next_slot_ = 0;   // ring slot written next
  int64_t frame_count_ = 0;     // every accepted frame, warm-up included

  bool has_spread_ = false;
  double last_spread_ = 0.0;
  bool has_last_ = false;
  VerdictData last_;            // last computed verdict (frequency-gate cache)
  VerdictData scratch_;         // warm-up and bad-input verdicts
  FrameStatus last_status_ = FrameStatus::kOk;
};

}  // namespace phm_core

#endif  // PHM_CORE__OOD_CORE_HPP_
