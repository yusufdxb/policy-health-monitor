// Copyright 2026 Yusuf Guenena. MIT License.
// Loading and reduction of an onboard session evidence directory. See
// session.hpp for the file layout and the time conventions.
#include "session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "phm_core/npy.hpp"

namespace phm_go2
{

namespace fs = std::filesystem;
namespace json = phm_tools::json;

namespace
{

bool is_blank(const std::string & line)
{
  // Characters str.isspace() accepts in the ASCII range.
  return std::all_of(
    line.begin(), line.end(), [](char c) {
      return c == ' ' || (c >= '\t' && c <= '\r') || (c >= '\x1c' && c <= '\x1f');
    });
}

// True when the three bytes at i are the UTF-8 form of U+2028 or U+2029.
bool is_line_or_paragraph_separator(const std::string & text, std::size_t i)
{
  const auto byte = [&](std::size_t k) {return static_cast<unsigned char>(text[k]);};
  return byte(i) == 0xe2 && byte(i + 1) == 0x80 && (byte(i + 2) == 0xa8 || byte(i + 2) == 0xa9);
}

bool file_exists(const std::string & path)
{
  std::error_code ec;
  return fs::exists(path, ec);
}

std::string join(const std::string & dir, const std::string & name)
{
  return (fs::path(dir) / name).string();
}

bool ends_with(const std::string & s, const std::string & suffix)
{
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string & s, const std::string & prefix)
{
  return s.compare(0, prefix.size(), prefix) == 0;
}

// Calls fn(line) for every non-blank line of `text` split on '\n'.
template<typename Fn>
void for_each_text_line(const std::string & text, Fn && fn)
{
  std::size_t start = 0;
  while (start < text.size()) {
    std::size_t nl = text.find('\n', start);
    if (nl == std::string::npos) {
      nl = text.size();
    }
    std::string line = text.substr(start, nl - start);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!is_blank(line)) {
      fn(line);
    }
    start = nl + 1;
  }
}

// Streams a plain file line by line; a missing file falls back to its .gz copy,
// which is decompressed in one piece. Returns false when neither exists.
template<typename Fn>
bool for_each_line(const std::string & path, Fn && fn)
{
  if (file_exists(path)) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (!is_blank(line)) {
        fn(line);
      }
    }
    return true;
  }
  const std::string gz = path + ".gz";
  if (file_exists(gz)) {
    for_each_text_line(phm_core::npy::gunzip(phm_core::npy::read_file_bytes(gz)), fn);
    return true;
  }
  return false;
}

Row to_row(const json::Value & v, bool keep_reason)
{
  if (!v.is_object()) {
    throw std::runtime_error("recorded line is not a JSON object");
  }
  Row r;
  r.t = v.at("t").as_double();
  const std::string & topic = v.at("topic").as_string();
  if (topic == kTopicHealth) {
    r.topic = Topic::kHealth;
  } else if (topic == kTopicEmbedding) {
    r.topic = Topic::kEmbedding;
  } else if (topic == kTopicVerdicts) {
    r.topic = Topic::kVerdicts;
  }
  r.stamp = std::nan("");
  if (const json::Value * p = v.find("stamp"); p != nullptr && p->is_number()) {
    r.stamp = p->as_double();
  }
  if (const json::Value * p = v.find("state"); p != nullptr) {
    r.has_state = true;
    r.state = p->is_number() ? p->as_int() : -1;
  }
  if (const json::Value * p = v.find("spread"); p != nullptr && p->is_number()) {
    r.has_spread = true;
    r.spread = p->as_double();
  }
  if (const json::Value * p = v.find("violating"); p != nullptr && !p->is_null()) {
    r.violating = p->as_int();
  }
  if (const json::Value * p = v.find("source"); p != nullptr && p->is_string()) {
    r.has_source = true;
    r.source = p->as_string();
  }
  if (keep_reason) {
    if (const json::Value * p = v.find("reason"); p != nullptr && p->is_string()) {
      r.has_reason = true;
      r.reason = p->as_string();
    }
  }
  return r;
}

void bump(Counts & counts, const std::string & key)
{
  for (auto & kv : counts) {
    if (kv.first == key) {
      ++kv.second;
      return;
    }
  }
  counts.emplace_back(key, 1);
}

json::Value opt(const std::optional<double> & v)
{
  return v ? json::Value(*v) : json::Value(nullptr);
}

json::Value opt(const std::optional<std::string> & v)
{
  return v ? json::Value(*v) : json::Value(nullptr);
}

json::Value counts_json(const Counts & counts)
{
  json::Value o = json::Value::object();
  for (const auto & kv : counts) {
    o.set(kv.first, kv.second);
  }
  return o;
}

void append_ood(json::Value & o, const OodSummary & s)
{
  o.set("ood_verdicts", s.ood_verdicts);
  o.set("ood_violating", s.ood_violating);
  o.set("ood_violating_fraction", opt(s.ood_violating_fraction));
  o.set("embeddings", s.embeddings);
  o.set("embedding_rate_hz", opt(s.embedding_rate_hz));
  o.set("spread_min", opt(s.spread_min));
  o.set("spread_median", opt(s.spread_median));
}

json::Value hit_json(const std::optional<FirstHit> & h)
{
  if (!h) {
    return json::Value(nullptr);
  }
  json::Value o = json::Value::object();
  o.set("latency_s", h->latency_s);
  o.set("source", opt(h->source));
  o.set("reason", opt(h->reason));
  o.set("state", opt(h->state));
  return o;
}

json::Value stat_json(const LatencyStat & s)
{
  json::Value o = json::Value::object();
  o.set("detected", std::to_string(s.detected) + "/" + std::to_string(s.total));
  o.set("min_s", opt(s.min_s));
  o.set("median_s", opt(s.median_s));
  o.set("max_s", opt(s.max_s));
  return o;
}

template<typename Pred>
std::optional<FirstHit> first_after(
  const std::vector<const Row *> & by_time, double t0, Pred pred)
{
  for (const Row * r : by_time) {
    if (r->t >= t0 && pred(*r)) {
      FirstHit h;
      h.latency_s = py_round(r->t - t0, 3);
      if (r->has_source) {
        h.source = r->source;
      }
      if (r->has_reason) {
        h.reason = r->reason;
      }
      if (r->has_state && state_name(r->state) != nullptr) {
        h.state = std::string(state_name(r->state));
      }
      return h;
    }
  }
  return std::nullopt;
}

LatencyStat latency_stat(
  const std::vector<FaultSummary> & trials, const std::optional<FirstHit> FaultSummary::* key)
{
  std::vector<double> lat;
  for (const auto & t : trials) {
    if ((t.*key).has_value()) {
      lat.push_back((t.*key)->latency_s);
    }
  }
  std::sort(lat.begin(), lat.end());
  LatencyStat s;
  s.detected = static_cast<int64_t>(lat.size());
  s.total = static_cast<int64_t>(trials.size());
  if (!lat.empty()) {
    s.min_s = lat.front();
    s.median_s = lat[lat.size() / 2];
    s.max_s = lat.back();
  }
  return s;
}

}  // namespace

const char * state_name(int64_t state)
{
  return (state >= 0 && state < 4) ? kStateNames[state] : nullptr;
}

double py_round(double x, int ndigits)
{
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%.*f", ndigits, x);
  return std::strtod(buf, nullptr);
}

std::optional<std::string> read_text(const std::string & path)
{
  if (file_exists(path)) {
    const std::vector<uint8_t> b = phm_core::npy::read_file_bytes(path);
    return std::string(b.begin(), b.end());
  }
  const std::string gz = path + ".gz";
  if (file_exists(gz)) {
    return phm_core::npy::gunzip(phm_core::npy::read_file_bytes(gz));
  }
  return std::nullopt;
}

std::vector<std::string> splitlines(const std::string & text)
{
  std::vector<std::string> out;
  std::size_t start = 0;
  std::size_t i = 0;
  const std::size_t n = text.size();
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    std::size_t eol = 0;  // length of the terminator at i, 0 when none
    if (c == '\r') {
      eol = (i + 1 < n && text[i + 1] == '\n') ? 2 : 1;
    } else if (c == '\n' || c == '\x0b' || c == '\x0c' || (c >= 0x1c && c <= 0x1e)) {
      eol = 1;
    } else if (c == 0xc2 && i + 1 < n && static_cast<unsigned char>(text[i + 1]) == 0x85) {
      eol = 2;  // U+0085
    } else if (c == 0xe2 && i + 2 < n && is_line_or_paragraph_separator(text, i)) {
      eol = 3;  // U+2028, U+2029
    }
    if (eol > 0) {
      out.push_back(text.substr(start, i - start));
      i += eol;
      start = i;
    } else {
      ++i;
    }
  }
  if (start < n) {
    out.push_back(text.substr(start));
  }
  return out;
}

std::vector<Row> load_rows(const std::string & path, bool keep_reason)
{
  std::vector<Row> rows;
  for_each_line(
    path, [&](const std::string & line) {
      rows.push_back(to_row(json::parse(line), keep_reason));
    });
  return rows;
}

std::optional<double> fault_time(const std::string & log_path)
{
  const auto text = read_text(log_path);
  if (!text) {
    return std::nullopt;
  }
  // \[(\d+\.\d+)\] \[phoenix_shadow_embedder\]: FAULT INJECTED, leftmost match.
  static const std::string kNeedle = "] [phoenix_shadow_embedder]: FAULT INJECTED";
  const auto is_digit = [](char c) {return c >= '0' && c <= '9';};
  std::size_t pos = 0;
  while ((pos = text->find(kNeedle, pos)) != std::string::npos) {
    std::size_t i = pos;
    std::size_t frac_end = i;
    while (i > 0 && is_digit((*text)[i - 1])) {
      --i;
    }
    const std::size_t frac_begin = i;
    if (frac_begin < frac_end && i > 0 && (*text)[i - 1] == '.') {
      --i;
      const std::size_t int_end = i;
      while (i > 0 && is_digit((*text)[i - 1])) {
        --i;
      }
      if (i < int_end && i > 0 && (*text)[i - 1] == '[') {
        return std::strtod(text->substr(i, frac_end - i).c_str(), nullptr);
      }
    }
    pos += 1;
  }
  return std::nullopt;
}

Counts most_common(const Counts & counts, std::size_t n)
{
  Counts sorted = counts;
  std::stable_sort(
    sorted.begin(), sorted.end(),
    [](const auto & a, const auto & b) {return a.second > b.second;});
  if (sorted.size() > n) {
    sorted.resize(n);
  }
  return sorted;
}

int64_t HealthSummary::count_of(const std::string & name) const
{
  for (const auto & kv : state_counts) {
    if (kv.first == name) {
      return kv.second;
    }
  }
  return 0;
}

HealthSummary health_summary(const std::vector<Row> & rows)
{
  HealthSummary s;
  Counts reasons;
  for (const Row & r : rows) {
    if (r.topic != Topic::kHealth) {
      continue;
    }
    const char * name = state_name(r.state);
    if (name == nullptr) {
      throw std::runtime_error("unknown health state " + std::to_string(r.state));
    }
    ++s.health_msgs;
    bump(s.state_counts, name);
    if (r.state != 0) {
      bump(reasons, std::string(name) + " " + (r.has_source ? r.source : std::string("None")));
    }
  }
  if (s.health_msgs > 0) {
    s.non_ok_fraction = py_round(
      1.0 - static_cast<double>(s.count_of("OK")) / static_cast<double>(s.health_msgs), 4);
  }
  s.non_ok_by_source = most_common(reasons, 10);
  return s;
}

OodSummary ood_summary(const std::vector<Row> & rows)
{
  OodSummary s;
  std::vector<double> spreads;
  const Row * first_emb = nullptr;
  const Row * last_emb = nullptr;
  for (const Row & r : rows) {
    if (r.topic == Topic::kVerdicts && r.has_source && r.source == kOodSource) {
      ++s.ood_verdicts;
      s.ood_violating += r.violating;
    } else if (r.topic == Topic::kEmbedding) {
      ++s.embeddings;
      if (first_emb == nullptr) {
        first_emb = &r;
      }
      last_emb = &r;
      if (r.has_spread) {
        spreads.push_back(r.spread);
      }
    }
  }
  std::sort(spreads.begin(), spreads.end());
  const double span = s.embeddings > 1 ? last_emb->stamp - first_emb->stamp : 0.0;
  if (s.ood_verdicts > 0) {
    s.ood_violating_fraction = py_round(
      static_cast<double>(s.ood_violating) / static_cast<double>(s.ood_verdicts), 4);
  }
  if (span > 0) {
    s.embedding_rate_hz = py_round(static_cast<double>(s.embeddings - 1) / span, 2);
  }
  if (!spreads.empty()) {
    s.spread_min = spreads.front();
    s.spread_median = spreads[spreads.size() / 2];
  }
  return s;
}

const std::vector<std::string> & fault_sources(const std::string & kind)
{
  static const std::vector<std::string> kFreeze = {"phm_ood_cpp"};
  static const std::vector<std::string> kStop = {
    "dead:/policy/embedding", "freq:/policy/embedding", "phm_ood_cpp"};
  static const std::vector<std::string> kNone;
  if (kind == "freeze") {
    return kFreeze;
  }
  if (kind == "stop") {
    return kStop;
  }
  return kNone;
}

FaultSummary fault_summary(
  const std::vector<Row> & rows, std::optional<double> t_fault,
  const std::vector<std::string> & verdict_sources)
{
  FaultSummary out;
  out.fault_time = t_fault;
  if (!t_fault) {
    out.missing_fault_line = true;
    return out;
  }
  const double tf = *t_fault;
  bool have_emb = false;
  double min_emb = 0.0;
  for (const Row & r : rows) {
    if (r.topic == Topic::kEmbedding && (!have_emb || r.stamp < min_emb)) {
      min_emb = r.stamp;
      have_emb = true;
    }
  }
  const double settle = (have_emb ? min_emb : tf) + kSettleSec;
  std::vector<Row> before;
  std::vector<Row> after;
  for (const Row & r : rows) {
    if (settle <= r.stamp && r.stamp < tf) {
      before.push_back(r);
    }
    if (r.stamp >= tf) {
      after.push_back(r);
    }
  }
  out.before_health = health_summary(before);
  out.before_ood = ood_summary(before);

  std::vector<const Row *> by_time;
  by_time.reserve(rows.size());
  for (const Row & r : rows) {
    by_time.push_back(&r);
  }
  std::stable_sort(
    by_time.begin(), by_time.end(), [](const Row * a, const Row * b) {return a->t < b->t;});

  out.first_violating_verdict = first_after(
    by_time, tf, [&](const Row & r) {
      if (r.topic != Topic::kVerdicts || r.violating == 0 || !r.has_source) {
        return false;
      }
      return std::any_of(
        verdict_sources.begin(), verdict_sources.end(),
        [&](const std::string & p) {return starts_with(r.source, p);});
    });
  const auto health_at_least = [&](int64_t level) {
      return first_after(
        by_time, tf, [level](const Row & r) {
          return r.topic == Topic::kHealth && r.state >= level;
        });
    };
  out.first_health_degraded_or_worse = health_at_least(1);
  out.first_health_intervene_or_worse = health_at_least(2);
  out.first_health_stop = health_at_least(3);
  out.after_health = health_summary(after);
  return out;
}

Aggregate aggregate(const std::vector<FaultSummary> & trials)
{
  Aggregate a;
  a.n_trials = static_cast<int64_t>(trials.size());
  a.first_violating_verdict = latency_stat(trials, &FaultSummary::first_violating_verdict);
  a.first_health_degraded_or_worse =
    latency_stat(trials, &FaultSummary::first_health_degraded_or_worse);
  a.first_health_intervene_or_worse =
    latency_stat(trials, &FaultSummary::first_health_intervene_or_worse);
  a.first_health_stop = latency_stat(trials, &FaultSummary::first_health_stop);
  for (const auto & t : trials) {
    if (!t.missing_fault_line) {
      a.pre_fault_health_msgs += t.before_health.health_msgs;
      a.pre_fault_non_ok_msgs += t.before_health.health_msgs - t.before_health.count_of("OK");
    }
  }
  return a;
}

std::vector<std::string> embedder_logs(const std::string & dir, const std::string & kind)
{
  const std::string prefix = "embedder_" + kind + "_";
  std::vector<std::string> names;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    if (starts_with(name, prefix) && name.find(".log", prefix.size()) != std::string::npos) {
      names.push_back(ends_with(name, ".gz") ? name.substr(0, name.size() - 3) : name);
    }
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  return names;
}

std::string trial_text(const std::string & log_name)
{
  std::string stem = log_name;
  const std::size_t dot = stem.rfind('.');
  if (dot != std::string::npos && dot > 0 && dot + 1 < stem.size()) {
    stem.resize(dot);
  }
  const std::size_t us = stem.rfind('_');
  return us == std::string::npos ? stem : stem.substr(us + 1);
}

int64_t parse_trial_number(const std::string & text)
{
  std::size_t b = 0;
  std::size_t e = text.size();
  while (b < e && (text[b] == ' ' || (text[b] >= '\t' && text[b] <= '\r'))) {
    ++b;
  }
  while (e > b && (text[e - 1] == ' ' || (text[e - 1] >= '\t' && text[e - 1] <= '\r'))) {
    --e;
  }
  std::size_t i = b;
  bool neg = false;
  if (i < e && (text[i] == '+' || text[i] == '-')) {
    neg = text[i] == '-';
    ++i;
  }
  if (i == e) {
    throw std::runtime_error("invalid literal for int(): '" + text + "'");
  }
  int64_t v = 0;
  for (; i < e; ++i) {
    if (text[i] < '0' || text[i] > '9' || v > (INT64_MAX - 9) / 10) {
      throw std::runtime_error("invalid literal for int(): '" + text + "'");
    }
    v = v * 10 + (text[i] - '0');
  }
  return neg ? -v : v;
}

std::string path_name(const std::string & path)
{
  std::string last;
  std::size_t start = 0;
  while (start <= path.size()) {
    std::size_t slash = path.find('/', start);
    if (slash == std::string::npos) {
      slash = path.size();
    }
    const std::string part = path.substr(start, slash - start);
    if (!part.empty() && part != ".") {
      last = part;
    }
    start = slash + 1;
  }
  return last;
}

json::Value to_json(const HealthSummary & s)
{
  json::Value o = json::Value::object();
  o.set("health_msgs", s.health_msgs);
  o.set("state_counts", counts_json(s.state_counts));
  o.set("non_ok_fraction", opt(s.non_ok_fraction));
  o.set("non_ok_by_source", counts_json(s.non_ok_by_source));
  return o;
}

json::Value to_json(const FaultSummary & s)
{
  json::Value o = json::Value::object();
  o.set("fault_time", opt(s.fault_time));
  if (s.missing_fault_line) {
    o.set("error", "no FAULT INJECTED line in the embedder log");
    return o;
  }
  json::Value before = to_json(s.before_health);
  append_ood(before, s.before_ood);
  o.set("before_fault", before);
  o.set("first_violating_verdict", hit_json(s.first_violating_verdict));
  o.set("first_health_degraded_or_worse", hit_json(s.first_health_degraded_or_worse));
  o.set("first_health_intervene_or_worse", hit_json(s.first_health_intervene_or_worse));
  o.set("first_health_stop", hit_json(s.first_health_stop));
  o.set("after_fault", to_json(s.after_health));
  return o;
}

json::Value to_json(const Aggregate & a)
{
  json::Value o = json::Value::object();
  o.set("n_trials", a.n_trials);
  o.set("first_violating_verdict", stat_json(a.first_violating_verdict));
  o.set("first_health_degraded_or_worse", stat_json(a.first_health_degraded_or_worse));
  o.set("first_health_intervene_or_worse", stat_json(a.first_health_intervene_or_worse));
  o.set("first_health_stop", stat_json(a.first_health_stop));
  o.set("pre_fault_health_msgs", a.pre_fault_health_msgs);
  o.set("pre_fault_non_ok_msgs", a.pre_fault_non_ok_msgs);
  return o;
}

std::optional<json::Value> load_calibration(const std::string & dir)
{
  const std::string path = join(dir, "calibrate.json");
  if (!file_exists(path)) {
    return std::nullopt;
  }
  const auto text = read_text(path);
  const std::vector<std::string> lines = splitlines(*text);
  if (lines.empty()) {
    throw std::runtime_error(path + " is empty");
  }
  return json::parse(lines.back());
}

json::Value summarize_session(const std::string & dir)
{
  json::Value summary = json::Value::object();
  summary.set("session", path_name(dir));
  const std::string session_txt = join(dir, "session.txt");
  if (file_exists(session_txt)) {
    json::Value lines = json::Value::array();
    for (const auto & l : splitlines(*read_text(session_txt))) {
      lines.push_back(l);
    }
    summary.set("session_txt", lines);
  }
  if (auto cal = load_calibration(dir)) {
    summary.set("calibration", *cal);
  }
  const std::string cpu = join(dir, "cpu_nominal.csv");
  if (file_exists(cpu)) {
    json::Value lines = json::Value::array();
    for (const auto & l : splitlines(*read_text(cpu))) {
      if (!starts_with(l, "#")) {
        lines.push_back(l);
      }
    }
    summary.set("cpu_nominal", lines);
  }
  {
    const std::vector<Row> nominal = load_rows(join(dir, "nominal.jsonl"));
    json::Value n = to_json(health_summary(nominal));
    append_ood(n, ood_summary(nominal));
    summary.set("nominal", n);
  }
  for (const char * kind : {"freeze", "stop"}) {
    json::Value trials = json::Value::array();
    std::vector<FaultSummary> summaries;
    for (const std::string & name : embedder_logs(dir, kind)) {
      const std::string n = trial_text(name);
      const int64_t number = parse_trial_number(n);
      const std::vector<Row> rows =
        load_rows(join(dir, std::string("fault_") + kind + "_" + n + ".jsonl"), true);
      summaries.push_back(fault_summary(rows, fault_time(join(dir, name)), fault_sources(kind)));
      json::Value t = json::Value::object();
      t.set("trial", number);
      const json::Value fields = to_json(summaries.back());
      for (const auto & kv : fields.as_object()) {
        t.set(kv.first, kv.second);
      }
      trials.push_back(t);
    }
    json::Value f = json::Value::object();
    f.set("aggregate", to_json(aggregate(summaries)));
    f.set("trials", trials);
    summary.set(std::string("fault_") + kind, f);
  }
  return summary;
}

}  // namespace phm_go2
