// Copyright 2026 Yusuf Guenena. MIT License.
// Health detectors that bridge BlackBoxRS detector patterns into PHM verdicts.
// Each holds its own Hysteresis and returns VerdictData; the phm_detectors
// node feeds them from the ROS graph and publishes the verdicts.
//
//   FrequencyDropAdapter            topic rate below a learned baseline
//                                   (BlackBoxRS anomaly_engine/detectors/frequency.py)
//   StaticThresholdAdapter          host metric above a static limit
//                                   (BlackBoxRS anomaly_engine/detectors/threshold.py)
//   DeadTopicAdapter                topic silent for more than timeout_sec
//                                   (BlackBoxRS anomaly_engine/detectors/dead_topic.py)
//   RecurrentTemporalSpreadAdapter  policy recurrent feature freezes (rolling
//                                   spread collapses), supercombo-blindspot E6
#ifndef PHM_CORE__ADAPTERS_HPP_
#define PHM_CORE__ADAPTERS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "phm_core/detector.hpp"
#include "phm_core/hysteresis.hpp"
#include "phm_core/matrix.hpp"

namespace phm_core
{

struct FrequencySample
{
  std::string topic;
  double frequency_hz = 0.0;
};

struct ThresholdSample
{
  std::string metric;
  double value = 0.0;
  double threshold = 0.0;
};

struct DeadTopicSample
{
  std::string topic;
  double last_seen_sec = 0.0;
  double now_sec = 0.0;
};

struct RecurrentSpreadSample
{
  std::string topic;
  std::vector<double> embedding;
};

// Learning samples averaged into the frequency baseline (frequency.py:20).
constexpr int kFrequencyLearningSamples = 10;

// Fires when a topic's rate stays below baseline * (1 - tolerance/100) for
// min_consecutive samples. The first kFrequencyLearningSamples samples set the
// baseline and return nullopt. Score 0 at the baseline rate, 1 at 0 Hz.
class FrequencyDropAdapter : public Detector<FrequencySample>
{
public:
  FrequencyDropAdapter(
    const std::string & target_topic, double tolerance_percent = 20.0, int min_consecutive = 2);
  std::optional<VerdictData> update(const FrequencySample & sample) override;
  bool learned() const {return has_baseline_;}
  double baseline() const {return baseline_;}

private:
  double tolerance_pct_;
  Hysteresis hysteresis_;
  std::vector<double> samples_;
  bool has_baseline_ = false;
  double baseline_ = 0.0;
};

// Fires when one named metric exceeds its limit for min_consecutive samples.
// Score 0 at value 0, 0.5 at the limit, 1 at twice the limit.
class StaticThresholdAdapter : public Detector<ThresholdSample>
{
public:
  StaticThresholdAdapter(
    const std::string & target_topic, const std::string & metric, int min_consecutive = 2);
  std::optional<VerdictData> update(const ThresholdSample & sample) override;
  const std::string & metric() const {return metric_;}

private:
  std::string metric_;
  Hysteresis hysteresis_;
};

// Fires when the watched topic has been silent for more than timeout_sec
// (strictly greater). Driven by clock ticks, so it detects silence even when
// nothing else arrives. A dead verdict suggests STOP_AND_HOLD. Score 0 when
// just seen, 1 at twice the timeout. No hysteresis.
class DeadTopicAdapter : public Detector<DeadTopicSample>
{
public:
  explicit DeadTopicAdapter(const std::string & target_topic, double timeout_sec = 5.0);
  // A message just arrived on the topic: clear the alert flag.
  void mark_alive(double now_sec);
  std::optional<VerdictData> update(const DeadTopicSample & sample) override;
  bool alerted() const {return alerted_;}

private:
  double timeout_sec_;
  bool alerted_ = false;
};

// Fires when a policy's recurrent feature freezes: the trace of the rolling
// covariance over `window` frames drops below the calibrated threshold for
// min_consecutive frames. Low spread is the unhealthy direction. Returns a
// healthy warm-up verdict until the window fills. Score
// normalize(spread, healthy=threshold, worst=0), or 1.0 for any breach of a
// non-positive (uncalibrated) threshold.
class RecurrentTemporalSpreadAdapter : public Detector<RecurrentSpreadSample>
{
public:
  RecurrentTemporalSpreadAdapter(
    const std::string & target_topic, std::size_t window = 30, double threshold = 0.0,
    int min_consecutive = 2);
  std::optional<VerdictData> update(const RecurrentSpreadSample & sample) override;
  double calibrate_from_data(const Matrix & in_dist_hidden, double percentile = 1.0);
  void set_threshold(double threshold) {threshold_ = threshold;}
  double threshold() const {return threshold_;}
  std::size_t window() const {return window_;}
  bool has_spread() const {return has_spread_;}
  double last_spread() const {return last_spread_;}

private:
  std::size_t window_;
  double threshold_;
  Hysteresis hysteresis_;
  std::vector<std::vector<double>> buffer_;
  bool has_spread_ = false;
  double last_spread_ = 0.0;
};

}  // namespace phm_core

#endif  // PHM_CORE__ADAPTERS_HPP_
