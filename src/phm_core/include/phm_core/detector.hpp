// Copyright 2026 Yusuf Guenena. MIT License.
// Detector interface and the plain verdict struct that mirrors
// phm_msgs/DetectorVerdict.msg.
//
// Detectors return a VerdictData (no ROS types) and the ROS node wrappers copy
// it into phm_msgs/DetectorVerdict when publishing, so all detector logic is
// testable without a ROS graph. The interface follows BlackBoxRS's
// BaseDetector (a name plus one check method that returns an event or
// nothing); here the method is update() and the event is a VerdictData.
#ifndef PHM_CORE__DETECTOR_HPP_
#define PHM_CORE__DETECTOR_HPP_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "phm_core/severity.hpp"

namespace phm_core
{

// Field-for-field the DetectorVerdict.msg payload minus the std_msgs/Header,
// which the node stamps at publish time.
struct VerdictData
{
  std::string source;        // which detector produced this verdict
  double score = 0.0;        // normalized severity in [0, 1], 0 healthy
  bool violating = false;    // post-hysteresis boolean
  std::string reason;        // human-readable explanation
  uint8_t suggested_action = ACTION_NONE;
};

enum class LogLevel { kDebug, kInfo, kWarn };

// Optional diagnostic sink for detector-internal events (baseline learned,
// first dead-topic alert). Empty by default: the core never prints on its own.
using LogFn = std::function<void (LogLevel, const std::string &)>;

// A detector watches one logical input (a topic, a metric, an embedding
// stream) and turns each sample into an optional partial verdict; the arbiter
// fuses the verdicts of every detector into one health signal.
template<typename SampleT>
class Detector
{
public:
  virtual ~Detector() = default;

  // Unique identifier, also used as DetectorVerdict.source.
  const std::string & name() const {return name_;}
  // The logical input this detector watches, e.g. "/policy/embedding".
  const std::string & target_topic() const {return target_topic_;}

  // Process one sample; a verdict when there is one to report, else nullopt.
  virtual std::optional<VerdictData> update(const SampleT & sample) = 0;

  void set_logger(LogFn log) {log_ = std::move(log);}

protected:
  Detector(std::string name, std::string target_topic)
  : name_(std::move(name)), target_topic_(std::move(target_topic)) {}

  void log(LogLevel level, const std::string & msg) const
  {
    if (log_) {
      log_(level, msg);
    }
  }
  bool has_logger() const {return static_cast<bool>(log_);}

  std::string name_;
  std::string target_topic_;

private:
  LogFn log_;
};

}  // namespace phm_core

#endif  // PHM_CORE__DETECTOR_HPP_
