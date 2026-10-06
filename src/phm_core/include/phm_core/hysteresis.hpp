// Copyright 2026 Yusuf Guenena. MIT License.
// Consecutive-violation hysteresis counter.
//
// The shared debounce for every detector: count consecutive violating
// observations, fire once the run reaches min_consecutive, keep firing while
// the run continues, and reset on any healthy observation. Extracted from the
// identical patterns in BlackBoxRS (anomaly_engine/detectors/threshold.py) and
// HELIX (helix_core/anomaly_detector.py).
#ifndef PHM_CORE__HYSTERESIS_HPP_
#define PHM_CORE__HYSTERESIS_HPP_

namespace phm_core
{

class Hysteresis
{
public:
  // min_consecutive >= 1; 1 fires on the first violating observation. Throws
  // std::invalid_argument otherwise.
  explicit Hysteresis(int min_consecutive);

  // Record one observation; true once there is an unbroken run of at least
  // min_consecutive violating observations. count() saturates at INT_MAX.
  bool observe(bool violating);

  void reset() {count_ = 0;}
  int count() const {return count_;}
  int min_consecutive() const {return min_consecutive_;}

private:
  int min_consecutive_;
  int count_ = 0;
};

}  // namespace phm_core

#endif  // PHM_CORE__HYSTERESIS_HPP_
