// Copyright 2026 Yusuf Guenena. MIT License.
// Loading and reduction of an onboard session evidence directory.
//
// A session directory holds JSONL recordings (one object per line, written by
// the recorder subscribing to /phm/health, /phm/verdicts and /policy/embedding),
// the embedder logs (one per fault trial) and calibrate.json. This module reads
// those files and reduces them to the numbers they support: health state
// counts, OOD verdict statistics and per-trial fault detection latencies.
//
// All times are one host's system clock: the recorder's receive time `t` and the
// embedder's log stamps. Detection latencies are measured from the fault log
// stamp to the receive time `t` of the first matching row, not to the header
// stamp, so they include delivery to and compute in the detector.
//
// The reductions reproduce the original Python implementation exactly: dict
// and Counter ordering, round(x, n), statistics.median and f-string formats.
#ifndef INTEGRATIONS__GO2__TOOLS__SESSION_HPP_
#define INTEGRATIONS__GO2__TOOLS__SESSION_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "phm_tools/json.hpp"

namespace phm_go2
{

// Health state names by numeric state (0..3).
constexpr char kStateNames[4][10] = {"OK", "DEGRADED", "INTERVENE", "STOP"};
// Pre-fault rows in the first kSettleSec of a trial are excluded from its
// false-alarm count: a trial starts right after the previous trial's induced
// fault, whose STOP verdicts are still clearing.
constexpr double kSettleSec = 5.0;
constexpr char kTopicHealth[] = "/phm/health";
constexpr char kTopicVerdicts[] = "/phm/verdicts";
constexpr char kTopicEmbedding[] = "/policy/embedding";
constexpr char kOodSource[] = "phm_ood_cpp";

// Name of a health state, or nullptr when `state` is not 0..3.
const char * state_name(int64_t state);

enum class Topic : uint8_t { kOther, kHealth, kEmbedding, kVerdicts };

// The fields of one recorded message that the reductions use.
struct Row
{
  double t = 0.0;       // recorder receive time
  double stamp = 0.0;   // message header stamp (NaN when absent)
  double spread = 0.0;  // rolling latent spread, valid when has_spread
  int64_t state = -1;   // health state, valid when has_state
  int64_t violating = 0;
  Topic topic = Topic::kOther;
  bool has_state = false;
  bool has_spread = false;
  bool has_source = false;  // key present with a string value
  bool has_reason = false;
  std::string source;
  std::string reason;
};

// Python's round(x, ndigits): correctly rounded, half to even on the exact value.
double py_round(double x, int ndigits);

// Contents of `path`, or of `path + ".gz"` when only that exists; nullopt when
// neither does.
std::optional<std::string> read_text(const std::string & path);
// Python's str.splitlines (line terminators are not included).
std::vector<std::string> splitlines(const std::string & text);

// Rows of a JSONL file (or its .gz copy), streamed line by line. Missing file
// gives no rows. The reason text is kept only when keep_reason is set.
std::vector<Row> load_rows(const std::string & path, bool keep_reason = false);

// Time of the first "FAULT INJECTED" line of an embedder log, if any.
std::optional<double> fault_time(const std::string & log_path);

// Entries ordered by descending count; ties keep first-insertion order, as
// collections.Counter.most_common does.
using Counts = std::vector<std::pair<std::string, int64_t>>;
Counts most_common(const Counts & counts, std::size_t n);

struct HealthSummary
{
  int64_t health_msgs = 0;
  Counts state_counts;
  std::optional<double> non_ok_fraction;
  Counts non_ok_by_source;

  int64_t count_of(const std::string & name) const;
};

struct OodSummary
{
  int64_t ood_verdicts = 0;
  int64_t ood_violating = 0;
  std::optional<double> ood_violating_fraction;
  int64_t embeddings = 0;
  std::optional<double> embedding_rate_hz;
  std::optional<double> spread_min;
  std::optional<double> spread_median;
};

HealthSummary health_summary(const std::vector<Row> & rows);
OodSummary ood_summary(const std::vector<Row> & rows);

// First row received at or after t0 that matched the predicate.
struct FirstHit
{
  double latency_s = 0.0;  // round(t - t0, 3)
  std::optional<std::string> source;
  std::optional<std::string> reason;
  std::optional<std::string> state;
};

struct FaultSummary
{
  std::optional<double> fault_time;
  bool missing_fault_line = false;
  HealthSummary before_health;
  OodSummary before_ood;
  std::optional<FirstHit> first_violating_verdict;
  std::optional<FirstHit> first_health_degraded_or_worse;
  std::optional<FirstHit> first_health_intervene_or_worse;
  std::optional<FirstHit> first_health_stop;
  HealthSummary after_health;
};

// Verdict sources that count as a detection for each injected fault kind
// ("freeze", "stop", "sensors").
const std::vector<std::string> & fault_sources(const std::string & kind);

FaultSummary fault_summary(
  const std::vector<Row> & rows, std::optional<double> t_fault,
  const std::vector<std::string> & verdict_sources);

struct LatencyStat
{
  int64_t detected = 0;
  int64_t total = 0;
  std::optional<double> min_s;
  std::optional<double> median_s;  // sorted latencies at index len // 2
  std::optional<double> max_s;
};

struct Aggregate
{
  int64_t n_trials = 0;
  LatencyStat first_violating_verdict;
  LatencyStat first_health_degraded_or_worse;
  LatencyStat first_health_intervene_or_worse;
  LatencyStat first_health_stop;
  int64_t pre_fault_health_msgs = 0;
  int64_t pre_fault_non_ok_msgs = 0;
};

Aggregate aggregate(const std::vector<FaultSummary> & trials);

// Trial log names of one fault kind ("embedder_<kind>_*.log*" with a trailing
// ".gz" removed), sorted as strings and de-duplicated.
std::vector<std::string> embedder_logs(const std::string & dir, const std::string & kind);
// The trial number text of a log name: the part after the last "_" of its stem.
std::string trial_text(const std::string & log_name);
// int(text); throws std::runtime_error when it is not an integer.
int64_t parse_trial_number(const std::string & text);
// Final component of a directory path, as pathlib.Path(...).name.
std::string path_name(const std::string & path);

// JSON forms, with the key order of the original output.
phm_tools::json::Value to_json(const HealthSummary & s);
phm_tools::json::Value to_json(const FaultSummary & s);
phm_tools::json::Value to_json(const Aggregate & a);

// The complete summary object of a session directory.
phm_tools::json::Value summarize_session(const std::string & dir);

// Last line of calibrate.json parsed, or nullopt when the file is absent.
std::optional<phm_tools::json::Value> load_calibration(const std::string & dir);

}  // namespace phm_go2

#endif  // INTEGRATIONS__GO2__TOOLS__SESSION_HPP_
