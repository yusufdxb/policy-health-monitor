// Copyright 2026 Yusuf Guenena. MIT License.
// phm_go2_plot <session_dir> --out <dir>: render the README figures of an
// onboard session as SVG and print the numbers they show. With fault trials it
// writes go2_freeze_fault.svg and go2_stop_fault.svg; it always ends with
// go2_nominal.svg, except that a session without fault trials whose nominal
// recording holds a stand-up burst gets go2_standup.svg instead. Every figure
// number is computed from the recorded files.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/format.hpp"
#include "phm_tools/json.hpp"
#include "phm_tools/svg.hpp"
#include "session.hpp"

namespace
{

namespace fs = std::filesystem;
using phm_core::fmt::fixed;
using phm_go2::FaultSummary;
using phm_go2::FirstHit;
using phm_go2::Row;
using phm_go2::Topic;
using phm_tools::svg::Axes;
using phm_tools::svg::Figure;
using phm_tools::svg::Style;

constexpr char kBlue[] = "#2a78d6";
constexpr char kInk[] = "#0b0b0b";
constexpr char kMuted[] = "#52514e";
constexpr char kRed[] = "#d64545";
constexpr const char * kStateColor[4] = {"#1baf7a", "#e8b931", "#e8821e", "#d64545"};
constexpr const char * kTrialColors[8] = {
  "#2a78d6", "#52514e", "#8a5fd1", "#1baf7a", "#c9578f", "#898781", "#2aa6c9", "#b8863b"};

// Python f"{x}" for an optional float: repr or "None".
std::string repr_opt(const std::optional<double> & v)
{
  return v ? phm_core::fmt::repr(*v) : "None";
}

// Thousands separators in the integer part of a formatted number.
std::string group(const std::string & number)
{
  const std::size_t sign = (!number.empty() && number[0] == '-') ? 1 : 0;
  std::size_t end = number.find('.');
  if (end == std::string::npos) {
    end = number.size();
  }
  std::string out = number.substr(0, sign);
  for (std::size_t i = sign; i < end; ++i) {
    if (i > sign && (end - i) % 3 == 0) {
      out += ',';
    }
    out += number[i];
  }
  return out + number.substr(end);
}

std::string group(int64_t v)
{
  return group(std::to_string(v));
}

std::string percent2(double ratio)
{
  return fixed(ratio * 100.0, 2) + "%";
}

double median(std::vector<double> v)
{
  if (v.empty()) {
    throw std::runtime_error("no median for empty data");
  }
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

const char * color_of(int64_t state)
{
  if (phm_go2::state_name(state) == nullptr) {
    throw std::runtime_error("unknown health state " + std::to_string(state));
  }
  return kStateColor[state];
}

struct Trial
{
  int64_t n = 0;
  double t0 = 0.0;
  std::vector<Row> health;  // /phm/health rows ordered by receive time
  std::vector<Row> emb;     // embedding rows that carry a spread
  FaultSummary summ;
};

std::vector<Trial> load_trials(const std::string & dir, const std::string & kind)
{
  std::vector<std::pair<int64_t, std::string>> logs;
  for (const std::string & name : phm_go2::embedder_logs(dir, kind)) {
    logs.emplace_back(phm_go2::parse_trial_number(phm_go2::trial_text(name)), name);
  }
  std::stable_sort(
    logs.begin(), logs.end(), [](const auto & a, const auto & b) {return a.first < b.first;});
  std::vector<Trial> trials;
  for (const auto & [n, name] : logs) {
    const std::optional<double> t0 = phm_go2::fault_time((fs::path(dir) / name).string());
    const std::string rows_path =
      (fs::path(dir) / ("fault_" + kind + "_" + std::to_string(n) + ".jsonl")).string();
    const std::vector<Row> rows = phm_go2::load_rows(rows_path, true);
    if (!t0 || rows.empty()) {
      continue;
    }
    Trial t;
    t.n = n;
    t.t0 = *t0;
    for (const Row & r : rows) {
      if (r.topic == Topic::kHealth) {
        t.health.push_back(r);
      } else if (r.topic == Topic::kEmbedding && r.has_spread) {
        t.emb.push_back(r);
      }
    }
    std::stable_sort(
      t.health.begin(), t.health.end(), [](const Row & a, const Row & b) {return a.t < b.t;});
    t.summ = phm_go2::fault_summary(rows, t0, phm_go2::fault_sources(kind));
    trials.push_back(std::move(t));
  }
  return trials;
}

struct Latency
{
  std::optional<double> median;
  int64_t hits = 0;
  std::string top_source = "n/a";
};

// Median latency (receive time) of the first health row at the given level
// (1 = non-OK, 3 = STOP) across trials, and the most common source among hits.
Latency med_latency(const std::vector<Trial> & trials, int level)
{
  std::vector<FaultSummary> sums;
  for (const Trial & t : trials) {
    sums.push_back(t.summ);
  }
  const phm_go2::Aggregate agg = phm_go2::aggregate(sums);
  const std::optional<FirstHit> FaultSummary::* key =
    level == 1 ? &FaultSummary::first_health_degraded_or_worse : &FaultSummary::first_health_stop;
  Latency out;
  out.median = (level == 1 ? agg.first_health_degraded_or_worse : agg.first_health_stop).median_s;
  phm_go2::Counts sources;
  for (const FaultSummary & s : sums) {
    if ((s.*key).has_value()) {
      ++out.hits;
      const std::string src = (s.*key)->source.value_or("None");
      auto it = std::find_if(
        sources.begin(), sources.end(), [&](const auto & kv) {return kv.first == src;});
      if (it == sources.end()) {
        sources.emplace_back(src, 1);
      } else {
        ++it->second;
      }
    }
  }
  if (!sources.empty()) {
    out.top_source = phm_go2::most_common(sources, 1)[0].first;
  }
  return out;
}

// Colored runs of the health state of consecutive messages, drawn as one
// rectangle per run on row y of `ax`; the last run extends to xmax.
void state_strip(Axes & ax, const std::vector<Row> & health, double t0, double y, double xmax)
{
  std::size_t i = 0;
  while (i < health.size()) {
    std::size_t j = i + 1;
    while (j < health.size() && health[j].state == health[i].state) {
      ++j;
    }
    const double a = health[i].t - t0;
    const double b = j < health.size() ? health[j].t - t0 : xmax;
    ax.rect(a, y - 0.4, b - a, 0.8, color_of(health[i].state));
    i = j;
  }
}

void state_legend(Axes & ax)
{
  for (int s = 0; s < 4; ++s) {
    ax.legend_patch(kStateColor[s], std::to_string(s) + " " + phm_go2::kStateNames[s]);
  }
  ax.set_legend("above_row");
}

Style line_style(const char * color, double width, double alpha, const std::string & label)
{
  Style s;
  s.color = color;
  s.width = width;
  s.alpha = alpha;
  s.label = label;
  return s;
}

Style threshold_style(double width, double thr)
{
  Style s = line_style(kRed, width, 1.0, "calibrated threshold " + fixed(thr, 5));
  s.dash = "6,4";
  return s;
}

std::string freeze_figure(const std::vector<Trial> & trials, double thr, const std::string & out)
{
  const std::size_t n = trials.size();
  constexpr double kXlo = -10.0;
  constexpr double kXhi = 5.0;
  Figure fig(1000, 680);
  Axes & a1 = fig.add_axes(80, 55, 890, 395);
  Axes & a2 = fig.add_axes(80, 520, 890, 90);
  a1.set_log_y(true);
  double lo = thr;
  for (std::size_t i = 0; i < n; ++i) {
    const Trial & t = trials[i];
    std::vector<double> xs;
    std::vector<double> ys;
    for (const Row & r : t.emb) {
      const double x = r.t - t.t0;
      if (kXlo <= x && x <= kXhi) {
        xs.push_back(x);
        ys.push_back(r.spread);
      }
      if (r.spread > 0) {
        lo = std::min(lo, r.spread);
      }
    }
    a1.line(xs, ys, line_style(kTrialColors[i % 8], 1.3, 0.85, "trial " + std::to_string(t.n)));
    state_strip(a2, t.health, t.t0, static_cast<double>(n - 1 - i), kXhi);
  }
  a1.hline(thr, threshold_style(1.5, thr));
  a1.vline(0, line_style(kInk, 1.4, 1.0, ""));
  a2.vline(0, line_style(kInk, 1.4, 1.0, ""));
  a1.set_ylabel("rolling spread (30 frames)");
  a1.set_legend("center left");
  const Latency stop = med_latency(trials, 3);
  const Latency deg = med_latency(trials, 1);
  const std::string count = std::to_string(n);
  std::string txt = stop.median ?
    "median time to STOP " + fixed(*stop.median, 2) + " s (" + std::to_string(stop.hits) + "/" +
    count + " trials, source " + stop.top_source + ")" :
    "no STOP in " + count + " trials";
  if (deg.median) {
    txt += "\nmedian time to first non-OK " + fixed(*deg.median, 2) + " s (" +
      std::to_string(deg.hits) + "/" + count + ")";
  }
  a1.set_title("Frozen policy input: latent spread collapses, PHM escalates to STOP");
  a1.text(0.01, 0.03, txt, "start", 10, true);
  a1.set_xlim(kXlo, kXhi);
  a1.set_ylim_low(lo * 0.6);
  a1.set_xticks_visible(false);
  a2.set_ylim(-0.6, static_cast<double>(n) - 0.4);
  std::vector<double> ticks;
  std::vector<std::string> labels;
  for (std::size_t i = 0; i < n; ++i) {
    ticks.push_back(static_cast<double>(i));
    labels.push_back("trial " + std::to_string(trials[n - 1 - i].n));
  }
  a2.set_yticks(ticks, labels);
  a2.set_grid(false);
  a2.set_xlim(kXlo, kXhi);
  a2.set_xlabel("time relative to fault injection (s)");
  a2.set_ylabel("/phm/health");
  state_legend(a2);
  fig.save((fs::path(out) / "go2_freeze_fault.svg").string());
  return "freeze: n=" + count + " median_stop=" + repr_opt(stop.median) + " (" +
         std::to_string(stop.hits) + "/" + count + ") median_non_ok=" + repr_opt(deg.median) +
         " (" + std::to_string(deg.hits) + "/" + count + ") stop_source=" + stop.top_source;
}

std::string stop_figure(const std::vector<Trial> & trials, const std::string & out)
{
  const std::size_t n = trials.size();
  Figure fig(1000, 450);
  Axes & ax = fig.add_axes(80, 50, 890, 350);
  double xlo = -10.0;
  double xhi = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const Trial & t = trials[i];
    std::vector<double> xs;
    std::vector<double> ys;
    // A step plot only changes where the state does, so repeats are dropped.
    for (std::size_t k = 0; k < t.health.size(); ++k) {
      if (k == 0 || k + 1 == t.health.size() || t.health[k].state != t.health[k - 1].state) {
        xs.push_back(t.health[k].t - t.t0);
        ys.push_back(static_cast<double>(t.health[k].state));
      }
    }
    if (!t.health.empty()) {
      xhi = std::max(xhi, t.health.back().t - t.t0);
    }
    ax.step(xs, ys, line_style(kTrialColors[i % 8], 1.8, 0.85, "trial " + std::to_string(t.n)));
  }
  xhi = std::min(xhi, 15.0) + 0.5;
  ax.vline(0, line_style(kInk, 1.4, 1.0, ""));
  const Latency non = med_latency(trials, 1);
  const Latency stop = med_latency(trials, 3);
  const std::string count = std::to_string(n);
  std::string txt;
  if (non.median) {
    txt += "median time to first non-OK " + fixed(*non.median, 2) + " s (" +
      std::to_string(non.hits) + "/" + count + "), source " + non.top_source;
  }
  if (stop.median) {
    txt += std::string(txt.empty() ? "" : "\n") + "median time to STOP " +
      fixed(*stop.median, 2) + " s (" + std::to_string(stop.hits) + "/" + count + "), source " +
      stop.top_source;
  }
  ax.text(0.01, 0.97, txt.empty() ? "no detection" : txt, "start", 10, true, kInk, true);
  std::vector<std::string> labels;
  for (int s = 0; s < 4; ++s) {
    labels.push_back(std::to_string(s) + " " + phm_go2::kStateNames[s]);
  }
  ax.set_yticks({0, 1, 2, 3}, labels);
  ax.set_ylim(-0.2, 3.5);
  ax.set_xlim(xlo, xhi);
  ax.set_xlabel("time relative to fault injection (s)");
  // No y label: the state tick labels name the axis, and a rotated label at the
  // fixed offset would overlap the widest of them ("2 INTERVENE").
  ax.set_title("Policy process goes silent: /phm/health state after the fault");
  ax.set_legend("center left");
  fig.save((fs::path(out) / "go2_stop_fault.svg").string());
  return "stop: n=" + count + " median_non_ok=" + repr_opt(non.median) + " (" +
         std::to_string(non.hits) + "/" + count + ", " + non.top_source + ") median_stop=" +
         repr_opt(stop.median) + " (" + std::to_string(stop.hits) + "/" + count + ", " +
         stop.top_source + ")";
}

// Figures over the nominal recording.
struct Nominal
{
  std::vector<Row> rows;
  phm_go2::HealthSummary health;
  int64_t ok = 0;
  int64_t ood = 0;
  int64_t violating = 0;  // verdicts with a truthy "violating" flag
};

Nominal load_nominal(const std::string & dir)
{
  Nominal nom;
  nom.rows = phm_go2::load_rows((fs::path(dir) / "nominal.jsonl").string());
  nom.health = phm_go2::health_summary(nom.rows);
  nom.ok = nom.health.count_of("OK");
  for (const Row & r : nom.rows) {
    if (r.topic == Topic::kVerdicts && r.has_source && r.source == phm_go2::kOodSource) {
      ++nom.ood;
      nom.violating += r.violating != 0 ? 1 : 0;
    }
  }
  return nom;
}

std::string nominal_figure(const Nominal & nom, double thr, const std::string & out)
{
  std::vector<double> sp;
  for (const Row & r : nom.rows) {
    if (r.topic == Topic::kEmbedding && r.has_spread) {
      sp.push_back(r.spread);
    }
  }
  if (sp.empty() || nom.health.health_msgs == 0) {
    throw std::runtime_error("nominal recording has no spread samples or health messages");
  }
  const auto mm = std::minmax_element(sp.begin(), sp.end());
  const double lo = std::min(*mm.first, thr);
  const double hi = std::max(*mm.second, thr);
  Figure fig(1000, 460);
  Axes & ax = fig.add_axes(80, 80, 890, 320);
  ax.hist(sp, 60, kBlue, 0.85);
  ax.vline(thr, threshold_style(1.8, thr));
  ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo));
  ax.set_xlabel("rolling spread (30 frames), nominal phase");
  ax.set_ylabel("frames");
  // A 1st-percentile threshold puts about 1% of nominal samples below it; the
  // OOD node's hysteresis and severity floor decide whether a dip is a
  // violation. The title states what the data shows, not a fixed claim.
  const auto below = static_cast<int64_t>(
    std::count_if(sp.begin(), sp.end(), [thr](double s) {return s < thr;}));
  std::string title = "Nominal operation: spread stays above the collapse threshold";
  if (nom.violating > 0) {
    title = "Nominal operation: " + group(nom.violating) + " violating OOD verdicts";
  } else if (below > 0) {
    title = "Nominal operation: no violating OOD verdict";
  }
  fig.text(80, 46, title, 13, kInk);
  const double ok_ratio =
    static_cast<double>(nom.ok) / static_cast<double>(nom.health.health_msgs);
  ax.text(
    0, 1.02,
    group(nom.health.health_msgs) + " health messages, " + percent2(ok_ratio) + " OK; " +
    group(nom.ood) + " OOD verdicts, " + std::to_string(nom.violating) + " violating; " +
    group(static_cast<int64_t>(sp.size())) + " spread samples, min " + fixed(*mm.first, 5) +
    ", " + group(below) + " below threshold (" +
    percent2(static_cast<double>(below) / static_cast<double>(sp.size())) + ")",
    "start", 10, false, kMuted);
  ax.set_legend("upper right");
  fig.save((fs::path(out) / "go2_nominal.svg").string());
  return "nominal: health=" + std::to_string(nom.health.health_msgs) + " ok=" +
         std::to_string(nom.ok) + " ood=" + std::to_string(nom.ood) + " violating=" +
         std::to_string(nom.violating) + " spreads=" + std::to_string(sp.size()) + " min=" +
         fixed(*mm.first, 6) + " thr=" + fixed(thr, 6);
}

// Stand-up figure when the nominal recording holds a spread burst far above
// the threshold; nullopt otherwise.
std::optional<std::string> standup_figure(const Nominal & nom, double thr, const std::string & out)
{
  if (nom.rows.empty()) {
    throw std::runtime_error("nominal recording is empty");
  }
  double t_start = nom.rows.front().t;
  for (const Row & r : nom.rows) {
    t_start = std::min(t_start, r.t);
  }
  std::vector<std::pair<double, double>> sp;
  for (const Row & r : nom.rows) {
    if (r.topic == Topic::kEmbedding && r.has_spread) {
      sp.emplace_back(r.t - t_start, r.spread);
    }
  }
  std::sort(sp.begin(), sp.end());
  double peak = 0.0;
  for (const auto & p : sp) {
    peak = std::max(peak, p.second);
  }
  if (sp.empty() || peak <= 100 * thr) {
    return std::nullopt;
  }
  std::vector<double> early;
  for (const auto & p : sp) {
    if (p.first < 20.0) {
      early.push_back(p.second);
    }
  }
  const double base0 = median(early);
  std::vector<double> burst;
  for (const auto & p : sp) {
    if (p.second > 10 * base0) {
      burst.push_back(p.first);
    }
  }
  if (burst.empty()) {
    throw std::runtime_error("no burst above 10x the early median");
  }
  const double b0 = burst.front();
  const double b1 = burst.back();
  std::vector<double> before;
  std::vector<double> after;
  for (const auto & p : sp) {
    if (p.first < b0) {
      before.push_back(p.second);
    }
    if (p.first > b1) {
      after.push_back(p.second);
    }
  }
  const double pre = median(before);
  const double post = median(after);
  const double xhi = sp.back().first;
  Figure fig(1000, 560);
  Axes & a1 = fig.add_axes(80, 75, 890, 340);
  Axes & a2 = fig.add_axes(80, 460, 890, 50);
  a1.set_log_y(true);
  std::vector<double> xs;
  std::vector<double> ys;
  for (const auto & p : sp) {
    xs.push_back(p.first);
    ys.push_back(p.second);
  }
  a1.line(xs, ys, line_style(kBlue, 1.0, 1.0, ""));
  a1.hline(thr, threshold_style(1.5, thr));
  a1.axvspan(b0, b1, "#e8b931", 0.25);
  a1.set_xlim(0, xhi);
  a1.text(
    ((b0 + b1) / 2) / xhi, 0.97, "robot stands up (operator, remote)", "middle", 10, true, kInk,
    true);
  a1.set_ylabel("rolling spread (30 frames)");
  fig.text(80, 45, "Operator stands the robot up: spread bursts, PHM stays OK", 13, kInk);
  a1.text(
    0, 1.02,
    group(nom.health.health_msgs) + " health messages, " + group(nom.ok) + " OK; " +
    group(nom.ood) + " OOD verdicts, " + std::to_string(nom.violating) + " violating; " +
    "peak / pre-burst median " + group(fixed(peak / pre, 0)) + "x; " +
    "post-burst / pre-burst median " + fixed(post / pre, 2) + "x",
    "start", 10, false, kMuted);
  a1.set_legend("upper left");
  a1.set_xticks_visible(false);
  std::vector<Row> health;
  for (const Row & r : nom.rows) {
    if (r.topic == Topic::kHealth) {
      health.push_back(r);
    }
  }
  std::stable_sort(
    health.begin(), health.end(), [](const Row & a, const Row & b) {return a.t < b.t;});
  state_strip(a2, health, t_start, 0, xhi);
  a2.set_ylim(-0.6, 0.6);
  a2.set_yticks({0}, {"/phm/health"});
  a2.set_grid(false);
  a2.set_xlim(0, xhi);
  a2.set_xlabel("time since nominal start (s)");
  state_legend(a2);
  fig.save((fs::path(out) / "go2_standup.svg").string());
  return "standup: burst " + fixed(b0, 1) + "-" + fixed(b1, 1) + " s of " + fixed(xhi, 1) +
         " s, health=" + std::to_string(nom.health.health_msgs) + " ok=" +
         std::to_string(nom.ok) + " ood=" + std::to_string(nom.ood) + " violating=" +
         std::to_string(nom.violating) + " pre=" + fixed(pre, 5) + " post=" + fixed(post, 5) +
         " peak=" + fixed(peak, 3) + " peak/pre=" + fixed(peak / pre, 1) + " post/pre=" +
         fixed(post / pre, 2);
}

int run(const std::string & dir, const std::string & out)
{
  fs::create_directories(out);
  const auto cal = phm_go2::load_calibration(dir);
  if (!cal) {
    throw std::runtime_error("missing calibrate.json in " + dir);
  }
  const double thr = cal->at("threshold").as_double();
  std::vector<std::string> lines;
  const std::vector<Trial> freeze = load_trials(dir, "freeze");
  const std::vector<Trial> stop = load_trials(dir, "stop");
  const bool has_faults = !freeze.empty() || !stop.empty();
  if (has_faults) {
    lines.push_back(freeze_figure(freeze, thr, out));
    lines.push_back(stop_figure(stop, out));
  }
  const Nominal nom = load_nominal(dir);
  // The burst swamps a nominal histogram, so the stand-up figure replaces it.
  std::optional<std::string> standup;
  if (!has_faults) {
    standup = standup_figure(nom, thr, out);
  }
  lines.push_back(standup ? *standup : nominal_figure(nom, thr, out));
  std::string text;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    text += (i > 0 ? "\n" : "") + lines[i];
  }
  text += "\n";
  std::fwrite(text.data(), 1, text.size(), stdout);
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string dir;
  std::string out;
  bool have_out = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
      have_out = true;
    } else if (a.rfind("--out=", 0) == 0) {
      out = a.substr(6);
      have_out = true;
    } else if (dir.empty() && a.rfind("--", 0) != 0) {
      dir = a;
    } else {
      dir.clear();
      break;
    }
  }
  if (dir.empty() || !have_out) {
    std::fprintf(stderr, "usage: %s <session_dir> --out <dir>\n", argv[0]);
    return 2;
  }
  try {
    return run(dir, out);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
