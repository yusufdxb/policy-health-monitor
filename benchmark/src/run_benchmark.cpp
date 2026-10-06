// Copyright 2026 Yusuf Guenena. MIT License.
// phm_benchmark: PHM rolling-spread OOD score vs canonical baselines on two
// synthetic failure families, written to RESULTS.md and results_<mode>.csv.
//
//   phm_benchmark [--dim 64] [--n 600] [--window 20] [--n-boot 1000] [--seed 42]
//                 [--latency-repeats 30] [--mlp-latency-repeats 3] [--out-dir .]
//
// For each scenario the calibration ID stream uses --seed and the evaluation
// ID/OOD streams use --seed + 1, so no detector scores its own fit set. ID and
// OOD streams are scored separately so the rolling window never spans the
// boundary. Latency is the median, over repeats, of (fit + score of the whole
// evaluation stream) / frames, in microseconds per frame on this host.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "phm_bench/bench.hpp"
#include "phm_core/format.hpp"
#include "phm_core/numerics.hpp"

using phm_bench::Matrix;
using phm_core::fmt::fixed;

namespace
{

using Scorer = std::function<std::vector<double>(const Matrix &, const Matrix &)>;

struct Result
{
  double auroc, auroc_lo, auroc_hi;
  double aupr, aupr_lo, aupr_hi;
  double fpr95, fpr95_lo, fpr95_hi;
  double latency_us;
};

struct Report
{
  std::string title;
  phm_bench::StreamSpec spec;
  std::vector<std::pair<std::string, Result>> results;
  double spread_id, spread_ood;
  // Collapse geometry, for the narrative.
  double anchor_dist, id_radius, maha_id_median, maha_ood_median;
};

double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

Result metrics_for(
  const std::vector<double> & scores, const std::vector<int> & labels, double latency_us,
  int n_boot)
{
  Result r{};
  const auto au = phm_bench::bootstrap_ci(phm_bench::auroc, scores, labels, n_boot);
  const auto ap = phm_bench::bootstrap_ci(phm_bench::aupr, scores, labels, n_boot);
  const auto fp = phm_bench::bootstrap_ci(
    [](const std::vector<double> & s, const std::vector<int> & y) {
      return phm_bench::fpr_at_tpr(s, y, 0.95);
    }, scores, labels, n_boot);
  r.auroc = phm_bench::auroc(scores, labels);
  r.auroc_lo = au.lo;
  r.auroc_hi = au.hi;
  r.aupr = phm_bench::aupr(scores, labels);
  r.aupr_lo = ap.lo;
  r.aupr_hi = ap.hi;
  r.fpr95 = phm_bench::fpr_at_tpr(scores, labels, 0.95);
  r.fpr95_lo = fp.lo;
  r.fpr95_hi = fp.hi;
  r.latency_us = latency_us;
  return r;
}

double latency_us(const Scorer & scorer, const Matrix & fid, const Matrix & stream, int repeats)
{
  std::vector<double> t;
  for (int i = 0; i < repeats; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto s = scorer(fid, stream);
    const auto t1 = std::chrono::steady_clock::now();
    if (s.empty()) {
      std::abort();
    }
    t.push_back(
      std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(stream.rows) * 1e6);
  }
  return median(t);
}

Report run(
  const std::string & title, const phm_bench::StreamSpec & spec, std::size_t window, int n_boot,
  int repeats, int mlp_repeats)
{
  Report rep;
  rep.title = title;
  rep.spec = spec;
  const Matrix fid = phm_bench::generate_stream(spec).id;
  phm_bench::StreamSpec eval_spec = spec;
  eval_spec.seed = spec.seed + 1;
  const auto eval = phm_bench::generate_stream(eval_spec);
  const Matrix both = phm_core::vstack({&eval.id, &eval.ood});
  std::vector<int> labels(eval.id.rows, 0);
  labels.resize(eval.id.rows + eval.ood.rows, 1);

  const std::vector<std::pair<std::string, Scorer>> detectors = {
    {"PHM rolling-spread", [window](const Matrix & f, const Matrix & t) {
        return phm_bench::phm_scores(f, t, window);
      }},
    {"Mahalanobis", [](const Matrix & f, const Matrix & t) {
        return phm_bench::mahalanobis(f, t);
      }},
    {"Relative Mahalanobis", [](const Matrix & f, const Matrix & t) {
        return phm_bench::relative_mahalanobis(f, t);
      }},
    {"KNN (k=50, L2-normalized)", [](const Matrix & f, const Matrix & t) {
        return phm_bench::knn_distance(f, t, 50, true);
      }},
    {"KNN (k=50, unnormalized)", [](const Matrix & f, const Matrix & t) {
        return phm_bench::knn_distance(f, t, 50, false);
      }},
    {"RND (closed-form)", [](const Matrix & f, const Matrix & t) {
        return phm_bench::rnd_closed_form(f, t);
      }},
    {"RND (MLP, gradient-trained)", [](const Matrix & f, const Matrix & t) {
        return phm_bench::rnd_mlp(f, t).scores;
      }},
  };
  for (const auto & [name, scorer] : detectors) {
    std::vector<double> scores = scorer(fid, eval.id);
    const std::vector<double> ood = scorer(fid, eval.ood);
    scores.insert(scores.end(), ood.begin(), ood.end());
    const bool is_mlp = name.rfind("RND (MLP", 0) == 0;
    const double lat = latency_us(scorer, fid, both, is_mlp ? mlp_repeats : repeats);
    rep.results.emplace_back(name, metrics_for(scores, labels, lat, n_boot));
    std::fprintf(stderr, "[%s] %s done\n", spec.ood_mode.c_str(), name.c_str());
  }
  rep.spread_id = phm_bench::rolling_spread_trace(eval.id, window);
  rep.spread_ood = phm_bench::rolling_spread_trace(eval.ood, window);

  // Collapse geometry on the evaluation stream: distance of the collapse
  // anchor (the last ID frame) from the ID mean vs the median ID radius, and
  // the median Mahalanobis score of ID vs OOD frames.
  std::vector<double> mean(eval.id.cols, 0.0);
  for (std::size_t r = 0; r < eval.id.rows; ++r) {
    for (std::size_t c = 0; c < eval.id.cols; ++c) {
      mean[c] += eval.id(r, c) / static_cast<double>(eval.id.rows);
    }
  }
  std::vector<double> radii;
  for (std::size_t r = 0; r < eval.id.rows; ++r) {
    double s = 0.0;
    for (std::size_t c = 0; c < eval.id.cols; ++c) {
      s += (eval.id(r, c) - mean[c]) * (eval.id(r, c) - mean[c]);
    }
    radii.push_back(std::sqrt(s));
  }
  rep.anchor_dist = radii.back();
  rep.id_radius = median(radii);
  rep.maha_id_median = median(phm_bench::mahalanobis(fid, eval.id));
  rep.maha_ood_median = median(phm_bench::mahalanobis(fid, eval.ood));
  return rep;
}

std::string csv_field(const std::string & s)
{
  return s.find(',') == std::string::npos ? s : "\"" + s + "\"";
}

void write_csv(const std::string & path, const Report & rep)
{
  std::ofstream f(path, std::ios::binary);
  // Python's csv.writer: minimal quoting, CRLF line endings.
  f << "detector,auroc,auroc_ci_lo,auroc_ci_hi,aupr,aupr_ci_lo,aupr_ci_hi,fpr95,fpr95_ci_lo,"
    "fpr95_ci_hi,latency_us_median\r\n";
  for (const auto & [name, r] : rep.results) {
    f << csv_field(name) << ',' << fixed(r.auroc, 4) << ',' << fixed(r.auroc_lo, 4) << ',' <<
      fixed(r.auroc_hi, 4) << ',' << fixed(r.aupr, 4) << ',' << fixed(r.aupr_lo, 4) << ',' <<
      fixed(r.aupr_hi, 4) << ',' << fixed(r.fpr95, 4) << ',' << fixed(r.fpr95_lo, 4) << ',' <<
      fixed(r.fpr95_hi, 4) << ',' << fixed(r.latency_us, 2) << "\r\n";
  }
}

std::string table(const Report & rep)
{
  std::ostringstream o;
  o << "| Detector | AUROC (95% CI) | AUPR (95% CI) | FPR@95 (95% CI) | "
    "Latency (us/frame, median) |\n|---|---|---|---|---|\n";
  for (const auto & [name, r] : rep.results) {
    o << "| " << name << " | " << fixed(r.auroc, 3) << " [" << fixed(r.auroc_lo, 3) << ", " <<
      fixed(r.auroc_hi, 3) << "] | " << fixed(r.aupr, 3) << " [" << fixed(r.aupr_lo, 3) << ", " <<
      fixed(r.aupr_hi, 3) << "] | " << fixed(r.fpr95, 3) << " [" << fixed(r.fpr95_lo, 3) <<
      ", " << fixed(r.fpr95_hi, 3) << "] | " << fixed(r.latency_us, 2) << " |\n";
  }
  return o.str();
}

void write_md(const std::string & path, const std::vector<Report> & reps, int n_boot, int repeats,
  int mlp_repeats, std::size_t window)
{
  std::ostringstream o;
  o << "# Reliability Benchmark: PHM OOD detector vs canonical baselines\n\n";
  o << "Detector under test: the PHM internal-feature OOD score (`phm_core::rolling_spread` + "
    "`phm_core::calibrate_threshold`), the windowed trace of the policy hidden-state "
    "covariance. Lower spread = more OOD; the harness negates it to the common higher-is-OOD "
    "convention before scoring (`phm_bench::phm_scores`).\n\n";
  o << "Baselines (reimplemented from supercombo-blindspot `src/baselines.py`): Mahalanobis "
    "(Lee et al. 2018), Relative Mahalanobis (Ren et al. 2021), KNN k=50 (Sun et al. 2022). "
    "RND (Burda et al. 2019) is the 4th method, in two forms: a closed-form ridge predictor and "
    "a gradient-trained MLP predictor (Adam, 300 full-batch steps).\n\n";
  o << "## Not-applicable baselines (regression / embedding setting)\n\n";
  o << "| Baseline | Applies? | Reason |\n|---|---|---|\n";
  o << "| MSP | No | No softmax classification head over a closed label set; the stream is raw "
    "policy embeddings. |\n";
  o << "| Energy | No | No logits to logsumexp; there is no classifier head on the embedding. |\n";
  o << "| ViM | No | Requires a classifier weight matrix + logits; neither exists for an "
    "embedding stream. |\n\n";
  o << "These three are N/A here for the same structural reason as in supercombo-blindspot "
    "(`src/baselines.py:60-107`): there are no classifier logits, only an internal feature "
    "vector.\n\n";
  o << "Metrics are threshold-free (AUROC, AUPR, FPR@95TPR) with stratified-bootstrap 95% CIs ("
    << n_boot << " resamples). Latency is the median per-frame wall-clock cost (fit + score over "
    "the full evaluation stream, divided by frame count) over " << repeats << " repeats (" <<
    mlp_repeats << " for the gradient-trained RND), single-threaded C++ on the machine that "
    "generated this file. AUROC, AUPR and FPR@95 follow the sklearn definitions and are "
    "unit-tested against hand-computed fixtures (`benchmark/test/test_metrics.cpp`).\n\n";
  for (const Report & rep : reps) {
    const auto & s = rep.spec;
    o << "## Scenario: " << rep.title << "\n\n";
    o << "- OOD mode: `" << s.ood_mode << "`  dim=" << s.dim << ", n_id=" << s.n_id <<
      ", n_ood=" << s.n_ood << ", window=" << window << ", in_dist_scale=" <<
      phm_core::fmt::repr(s.in_dist_scale) << ", ood_scale=" << phm_core::fmt::repr(s.ood_scale) <<
      ", ood_shift=" << phm_core::fmt::repr(s.ood_shift) << ", ar_rho=" <<
      phm_core::fmt::repr(s.ar_rho) << ", seed=" << s.seed << "\n";
    o << "- Mean rolling-spread (window=" << window << "): ID=" << fixed(rep.spread_id, 4) <<
      ", OOD=" << fixed(rep.spread_ood, 4) << "\n\n";
    o << table(rep) << "\n";
  }
  // Narrative numbers come from this run.
  const Report & c = reps.front();
  double lo = 1.0;
  double hi = 0.0;
  for (const auto & [name, r] : c.results) {
    if (name != "PHM rolling-spread") {
      lo = std::min(lo, r.auroc);
      hi = std::max(hi, r.auroc);
    }
  }
  double lat_lo = 1e300;
  double lat_hi = 0.0;
  double mlp_lat = 0.0;
  for (const Report & rep : reps) {
    for (const auto & [name, r] : rep.results) {
      if (name.rfind("RND (MLP", 0) == 0) {
        mlp_lat = std::max(mlp_lat, r.latency_us);
      } else {
        lat_lo = std::min(lat_lo, r.latency_us);
        lat_hi = std::max(lat_hi, r.latency_us);
      }
    }
  }
  o << "## Headline\n\n";
  o << "The two scenarios separate the two failure families. On the **collapse** scenario the "
    "PHM rolling-spread detector scores AUROC " << fixed(c.results.front().second.auroc, 3) <<
    " (FPR@95 " << fixed(c.results.front().second.fpr95, 3) << ") while every location-based "
    "baseline scores AUROC " << fixed(lo, 2) << " to " << fixed(hi, 2) << ". This is not a "
    "harness bug: the collapsed cluster freezes around a real ID anchor point that sits INSIDE "
    "the ID cloud (anchor-to-ID-mean distance " << fixed(c.anchor_dist, 2) << " vs median ID "
    "radius " << fixed(c.id_radius, 2) << ", evaluation stream seed " << c.spec.seed + 1 <<
    "), so a distance / density score assigns it a LOWER (more-in-distribution) value than "
    "typical spread-out ID frames (Mahalanobis median ID " << fixed(c.maha_id_median, 0) <<
    " vs collapsed-OOD " << fixed(c.maha_ood_median, 0) << "). A collapse is a SECOND-order "
    "anomaly (variance drops, location does not move), and first-order location detectors are "
    "structurally blind to it. On the **shift** scenario the location detectors "
    "(Mahalanobis, RMD, unnormalized KNN, both RND forms) and the PHM detector all separate "
    "the streams, because the shifted region's within-window spread also differs.\n\n";
  o << "## Notes\n\n";
  o << "- The PHM rolling-spread detector targets the collapse / frozen-embedding failure "
    "(second-order: variance drops). It is location-invariant by construction (watches the "
    "trace of the windowed covariance, not absolute position).\n";
  o << "- Mahalanobis / RMD / KNN target location shift (first-order: the embedding moves to a "
    "new region). On a pure collapse with no mean shift they are at-or-below chance; on a shift "
    "they are strong. The two scenarios make this contrast explicit.\n";
  o << "- L2-normalized KNN (Sun et al. 2022 default, the supercombo-blindspot default) projects "
    "onto the unit sphere and discards the radial magnitude. It is at-or-below chance on BOTH "
    "scenarios here because the synthetic shift lives in magnitude; the unnormalized variant "
    "recovers it. Both are reported so this is visible rather than buried.\n";
  o << "- RND captures the shift (closed-form ridge predictor and gradient-trained MLP predictor "
    "agree) but NOT the collapse, for the same in-cloud-anchor reason as the distance "
    "baselines: the predictor reproduces the target well on the in-distribution anchor.\n";
  o << "- Latency: the closed-form detectors cost " << fixed(lat_lo, 2) << " to " <<
    fixed(lat_hi, 2) << " us/frame (fit + score amortised over the stream). The "
    "gradient-trained RND costs " << fixed(mlp_lat, 0) << " us/frame because each fit trains an "
    "MLP for 300 Adam steps; it is reported for correctness corroboration, not as a latency "
    "contender.\n";
  o << "- Both scenarios are synthetic: these numbers bound the detectors' structural coverage, "
    "not their accuracy on a real policy.\n";
  std::ofstream(path) << o.str();
}

}  // namespace

int main(int argc, char ** argv)
{
  std::map<std::string, std::string> opt = {
    {"--dim", "64"}, {"--n", "600"}, {"--window", "20"}, {"--n-boot", "1000"}, {"--seed", "42"},
    {"--latency-repeats", "30"}, {"--mlp-latency-repeats", "3"}, {"--out-dir", "."}};
  for (int i = 1; i < argc; i += 2) {
    const std::string k = argv[i];
    if (!opt.count(k) || i + 1 >= argc) {
      std::fprintf(
        stderr,
        "usage: phm_benchmark [--dim 64] [--n 600] [--window 20] [--n-boot 1000] [--seed 42] "
        "[--latency-repeats 30] [--mlp-latency-repeats 3] [--out-dir .]\n");
      return 2;
    }
    opt[k] = argv[i + 1];
  }
  const int dim = std::atoi(opt["--dim"].c_str());
  const int n = std::atoi(opt["--n"].c_str());
  const std::size_t window = static_cast<std::size_t>(std::atoi(opt["--window"].c_str()));
  const int n_boot = std::atoi(opt["--n-boot"].c_str());
  const uint64_t seed = std::strtoull(opt["--seed"].c_str(), nullptr, 10);
  const int repeats = std::atoi(opt["--latency-repeats"].c_str());
  const int mlp_repeats = std::atoi(opt["--mlp-latency-repeats"].c_str());
  const std::string out = opt["--out-dir"];

  std::vector<Report> reps;
  for (const auto & [title, mode] : std::vector<std::pair<std::string, std::string>>{
      {"collapse OOD (frozen low-variance embedding)", "collapse"},
      {"shift OOD (mean-shifted embedding region)", "shift"}})
  {
    phm_bench::StreamSpec spec;
    spec.dim = dim;
    spec.n_id = n;
    spec.n_ood = n;
    spec.ood_mode = mode;
    spec.seed = seed;
    reps.push_back(run(title, spec, window, n_boot, repeats, mlp_repeats));
  }
  write_md(out + "/RESULTS.md", reps, n_boot, repeats, mlp_repeats, window);
  for (const Report & rep : reps) {
    write_csv(out + "/results_" + rep.spec.ood_mode + ".csv", rep);
    std::printf("\n=== %s ===\n", rep.title.c_str());
    std::printf("spread ID=%s  OOD=%s\n", fixed(rep.spread_id, 4).c_str(),
      fixed(rep.spread_ood, 4).c_str());
    std::printf("%s", table(rep).c_str());
  }
  return 0;
}
