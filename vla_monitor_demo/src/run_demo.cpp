// Copyright 2026 Yusuf Guenena. MIT License.
// vla_monitor_demo: train the stand-in policy, sweep alpha, write
// alpha_sweep.csv and alpha_sweep.svg into --out-dir (default "."), and print
// the measured lead time.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "phm_core/format.hpp"
#include "phm_tools/svg.hpp"
#include "vla_demo/harness.hpp"

using phm_core::fmt::fixed;

namespace
{

void write_csv(const vla_demo::SweepResult & r, const std::string & path)
{
  std::ofstream f(path, std::ios::binary);
  f << "alpha,normalized_output_error,ood_score_mean_rolling_spread,monitor_fired_fraction,"
    "raw_output_error\r\n";
  for (std::size_t i = 0; i < r.alphas.size(); ++i) {
    f << fixed(r.alphas[i], 4) << ',' << fixed(r.output_error[i], 6) << ',' <<
      fixed(r.ood_score[i], 6) << ',' << fixed(r.fired_fraction[i], 6) << ',' <<
      fixed(r.raw_output_error[i], 6) << "\r\n";
  }
}

void plot(const vla_demo::SweepResult & r, const std::string & path)
{
  namespace svg = phm_tools::svg;
  svg::Figure fig(960, 600);
  svg::Axes & ax = fig.add_axes(90, 80, 760, 440);
  svg::Style err;
  err.color = "#b0472b";
  err.marker = 'o';
  err.label = "normalized output error";
  ax.line(r.alphas, r.output_error, err);
  svg::Style fired;
  fired.color = "#2a78d6";
  fired.marker = 's';
  fired.label = "monitor fired fraction (frames below threshold)";
  ax.line(r.alphas, r.fired_fraction, fired);
  svg::Style lvl;
  lvl.color = "#b0472b";
  lvl.width = 0.8;
  lvl.dash = "2,3";
  lvl.label = "output degradation level (" + fixed(r.output_degradation_level, 2) + ")";
  ax.hline(r.output_degradation_level, lvl);
  svg::Style trip;
  trip.color = "#2a78d6";
  trip.width = 0.8;
  trip.dash = "2,3";
  trip.label = "monitor tripwire (" + fixed(r.monitor_fire_fraction, 2) + ", clean FPR=" +
    fixed(r.clean_fpr, 3) + ")";
  ax.hline(r.monitor_fire_fraction, trip);
  if (!std::isnan(r.monitor_fires_at)) {
    svg::Style m;
    m.color = "#2a78d6";
    m.width = 1.6;
    m.label = "MONITOR fires @ alpha=" + fixed(r.monitor_fires_at, 2);
    ax.vline(r.monitor_fires_at, m);
  }
  if (!std::isnan(r.output_collapses_at)) {
    svg::Style o;
    o.color = "#b0472b";
    o.width = 1.6;
    o.label = "OUTPUT collapses @ alpha=" + fixed(r.output_collapses_at, 2);
    ax.vline(r.output_collapses_at, o);
  }
  ax.set_xlim(-0.05, 1.05);
  // Headroom above the data keeps the legend clear of the curves.
  ax.set_ylim(-0.03, 1.55);
  ax.set_xlabel("alpha (0 = clean input, 1 = full distribution shift)");
  ax.set_ylabel("normalized output error / monitor fired fraction (0..1)");
  svg::Axes & right = ax.twinx();
  svg::Style ood;
  ood.color = "#1baf7a";
  ood.marker = '^';
  ood.dash = "6,4";
  ood.alpha = 0.8;
  ood.label = "OOD score (mean rolling spread of hidden features)";
  right.line(r.alphas, r.ood_score, ood);
  svg::Style thr;
  thr.color = "#1baf7a";
  thr.width = 1.0;
  thr.dash = "6,4";
  thr.label = "calibrated OOD threshold (" + fixed(r.threshold, 4) + ")";
  right.hline(r.threshold, thr);
  double lo = r.threshold;
  double hi = r.threshold;
  for (double v : r.ood_score) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  right.set_ylim(lo - 0.05 * (hi - lo), hi + 0.7 * (hi - lo));
  right.set_ylabel("OOD score: mean rolling spread of policy embedding");
  ax.set_title("PHM internal-feature monitor vs stand-in policy output collapse");
  fig.text(
    90, 52, "measured lead-time = " + std::string(r.lead_time >= 0 ? "+" : "") +
    fixed(r.lead_time, 2) + " alpha (positive = monitor fires earlier)", 11, "#52514e");
  ax.set_legend("upper left");
  fig.save(path);
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string out = ".";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::string(argv[i]) == "--out-dir") {
      out = argv[i + 1];
    }
  }
  const auto policy = vla_demo::train_policy();
  std::printf("[train] stand-in policy trained, final MSE = %.6f\n", policy.final_mse);
  const auto res = vla_demo::run_sweep(policy.mlp);
  write_csv(res, out + "/alpha_sweep.csv");
  plot(res, out + "/alpha_sweep.svg");
  std::printf("[plot] wrote %s/alpha_sweep.svg and %s/alpha_sweep.csv\n", out.c_str(),
    out.c_str());
  std::printf("---\n");
  std::printf("calibrated OOD threshold (p=1.0): %.6f\n", res.threshold);
  std::printf("clean false-positive rate (frame-flag on clean batch): %.4f\n", res.clean_fpr);
  std::printf("monitor tripwire (frame-flag fraction): %.3f\n", res.monitor_fire_fraction);
  std::printf("output degradation level (norm error): %.3f\n", res.output_degradation_level);
  std::printf("monitor fires at alpha    = %s\n", phm_core::fmt::repr(res.monitor_fires_at).c_str());
  std::printf(
    "output collapses at alpha = %s\n", phm_core::fmt::repr(res.output_collapses_at).c_str());
  std::printf("MEASURED LEAD-TIME (alpha) = %+.4f\n", res.lead_time);
  if (res.lead_time > 0) {
    std::printf("HEADLINE: monitor fires BEFORE output collapse (positive lead-time).\n");
  } else if (res.lead_time == 0) {
    std::printf("HEADLINE: monitor and output cross at the SAME alpha (zero lead-time).\n");
  } else {
    std::printf("HEADLINE: monitor fires AFTER output collapse, or a level was never "
      "crossed.\n");
  }
  return 0;
}
