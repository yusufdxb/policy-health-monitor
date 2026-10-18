// Copyright 2026 Yusuf Guenena. MIT License.
// phm_go2_ood_replay <in.npz> <key> <out.npz> --threshold T [--window W]
//                    [--min-consecutive K]
// Feed every row of the [T, D] array <key> of in.npz, in order, through
// phm_core::OodCore (the detector the OOD node runs) and write one row per
// frame to out.npz: spread (NaN while warming up), violating (0/1), score and
// suggested_action, all float64 [T]. No ROS needed; this is how recorded
// embeddings are scored offline with the deployed detector.
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/npy.hpp"
#include "phm_core/ood_core.hpp"
#include "phm_core/spread_backend.hpp"

namespace npy = phm_core::npy;

int main(int argc, char ** argv)
{
  if (argc < 6) {
    std::fprintf(
      stderr,
      "usage: %s <in.npz> <key> <out.npz> --threshold T [--window W] "
      "[--min-consecutive K]\n",
      argv[0]);
    return 2;
  }
  try {
    phm_core::OodConfig cfg;
    bool have_threshold = false;
    for (int i = 4; i < argc; ++i) {
      const std::string flag = argv[i];
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
      }
      const std::string value = argv[++i];
      if (flag == "--threshold") {
        cfg.threshold = std::stod(value);
        have_threshold = true;
      } else if (flag == "--window") {
        cfg.window = static_cast<std::size_t>(std::stoul(value));
      } else if (flag == "--min-consecutive") {
        cfg.min_consecutive = std::stoi(value);
      } else {
        throw std::invalid_argument("unknown option " + flag);
      }
    }
    if (!have_threshold) {
      throw std::invalid_argument("--threshold is required");
    }
    const auto in = npy::load_npz(argv[1]);
    const auto it = in.find(argv[2]);
    if (it == in.end()) {
      throw std::runtime_error(std::string("missing array '") + argv[2] + "'");
    }
    const npy::Array & a = it->second;
    if (a.shape.size() != 2 || a.shape[0] == 0 || a.shape[1] == 0) {
      throw std::runtime_error(std::string("array '") + argv[2] + "' must be [T, D]");
    }
    const std::size_t rows = a.shape[0];
    const std::size_t dim = a.shape[1];
    const auto values = a.as_doubles();

    phm_core::OodCore core(cfg, phm_core::make_plain_backend());
    std::vector<double> spread(rows), violating(rows), score(rows), action(rows);
    std::vector<float> frame(dim);
    for (std::size_t t = 0; t < rows; ++t) {
      for (std::size_t j = 0; j < dim; ++j) {
        frame[j] = static_cast<float>(values[t * dim + j]);
      }
      const phm_core::VerdictData & v = core.update(frame, "replay");
      spread[t] = core.has_spread() ? core.last_spread() :
        std::numeric_limits<double>::quiet_NaN();
      violating[t] = v.violating ? 1.0 : 0.0;
      score[t] = v.score;
      action[t] = static_cast<double>(v.suggested_action);
    }
    std::vector<std::pair<std::string, npy::Array>> out;
    out.emplace_back("spread", npy::Array::from_doubles(spread, {rows}));
    out.emplace_back("violating", npy::Array::from_doubles(violating, {rows}));
    out.emplace_back("score", npy::Array::from_doubles(score, {rows}));
    out.emplace_back("suggested_action", npy::Array::from_doubles(action, {rows}));
    npy::save_npz(argv[3], out);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
