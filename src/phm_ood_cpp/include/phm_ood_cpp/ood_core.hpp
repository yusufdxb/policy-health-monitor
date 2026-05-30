// Copyright 2026 Yusuf Guenena. MIT License.
// ROS-free core of the C++ rolling-spread OOD detector.
//
// Mirrors the decision logic in the Python rclpy detector so the C++ node emits
// the same verdicts:
//   phm_ood/phm_ood/_core.py:180-326  (buffer, frequency gate, threshold,
//                                       hysteresis, score, action banding)
//   phm_core/phm_core/hysteresis.py:48-69  (consecutive-violation debounce)
//
// Kept free of rclcpp so it can be unit-tested and latency-benchmarked without a
// ROS graph. The node (ood_node.cpp) is a thin adapter over OodCore.
#ifndef PHM_OOD_CPP__OOD_CORE_HPP_
#define PHM_OOD_CPP__OOD_CORE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "phm_ood_cpp/backend.hpp"
#include "phm_ood_cpp/severity.hpp"

namespace phm_ood_cpp
{

constexpr char SOURCE[] = "phm_ood_cpp";

// Plain-old-data verdict, field-for-field the phm_msgs/DetectorVerdict payload
// minus the Header (the node stamps that). Matches DetectorVerdictData in
// phm_core/phm_core/detector.py and _core.py verdict construction.
struct VerdictData
{
  std::string source = SOURCE;
  double score = 0.0;          // normalized severity [0,1], published as float32
  bool violating = false;      // post-hysteresis boolean
  std::string reason;
  uint8_t suggested_action = ACTION_NONE;
};

// Consecutive-violation hysteresis. Ported from hysteresis.py:48-69:
// reset on healthy, increment on violating, fire at count >= min_consecutive.
class Hysteresis
{
public:
  explicit Hysteresis(int min_consecutive);
  // Returns true once there is an unbroken run of >= min_consecutive violations.
  bool observe(bool violating);
  void reset() { count_ = 0; }
  int count() const { return count_; }

private:
  int min_consecutive_;
  int count_ = 0;
};

class OodCore
{
public:
  // window:          frames in the rolling covariance (>= 2).
  // threshold:       calibrated spread; spread < threshold -> OOD.
  // min_consecutive: hysteresis run length to confirm a violation (>= 1).
  // compute_every:   frequency gate; only recompute every Nth full-buffer frame
  //                  (re-uses the last verdict otherwise), mirrors _core.py:192.
  OodCore(
    std::size_t window, double threshold, int min_consecutive,
    int compute_every, std::unique_ptr<Backend> backend);

  // Feed one embedding frame; returns the verdict for this frame.
  VerdictData update(const std::vector<float> & embedding, const std::string & policy_id);

  double last_spread() const { return last_spread_; }
  const std::string & backend_name() const { return backend_name_; }
  std::size_t window() const { return window_; }

private:
  VerdictData ok_verdict(const std::string & reason) const;
  VerdictData make_verdict(
    double spread, bool raw_violating, bool fired, const std::string & policy_id);

  std::size_t window_;
  double threshold_;
  int compute_every_;
  std::unique_ptr<Backend> backend_;
  std::string backend_name_;
  Hysteresis hysteresis_;

  std::vector<std::vector<float>> buffer_;  // rolling window of embedding frames
  std::size_t dim_ = 0;
  int64_t frame_count_ = 0;
  double last_spread_ = 0.0;
  bool have_last_ = false;
  VerdictData last_verdict_;
};

}  // namespace phm_ood_cpp

#endif  // PHM_OOD_CPP__OOD_CORE_HPP_
