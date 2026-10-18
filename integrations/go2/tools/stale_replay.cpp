// Copyright 2026 Yusuf Guenena. MIT License.
// phm_go2_stale_replay <in.npz> <out.npz> [--gyro-max N] [--joint-max N]
// Feed recorded readings, in order, through phm_core::StaleSignalAdapter with
// the grouping phoenix_shadow_embedder uses under stale_check (gyro axes
// together; q with dq of each motor) and write one row per sample to out.npz:
// violating (0/1), score, suggested_action and worst_run, float64 [T]. in.npz
// holds gyro [T, 3], q [T, 12] and dq [T, 12] in Unitree motor order. Defaults
// match the node (5 and 12). No ROS needed.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/adapters.hpp"
#include "phm_core/npy.hpp"

namespace npy = phm_core::npy;

namespace
{

std::vector<double> column_block(
  const std::map<std::string, npy::Array> & in, const std::string & key, std::size_t dim,
  std::size_t & rows)
{
  const auto it = in.find(key);
  if (it == in.end()) {
    throw std::runtime_error("missing array '" + key + "'");
  }
  const npy::Array & a = it->second;
  if (a.shape.size() != 2 || a.shape[1] != dim) {
    throw std::runtime_error("array '" + key + "' must be [T, " + std::to_string(dim) + "]");
  }
  if (rows != 0 && a.shape[0] != rows) {
    throw std::runtime_error("array '" + key + "' has a different row count");
  }
  rows = a.shape[0];
  return a.as_doubles();
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(
      stderr, "usage: %s <in.npz> <out.npz> [--gyro-max N] [--joint-max N]\n", argv[0]);
    return 2;
  }
  try {
    int gyro_max = 5;
    int joint_max = 12;
    for (int i = 3; i < argc; ++i) {
      const std::string flag = argv[i];
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
      }
      const int value = std::stoi(argv[++i]);
      if (flag == "--gyro-max") {
        gyro_max = value;
      } else if (flag == "--joint-max") {
        joint_max = value;
      } else {
        throw std::invalid_argument("unknown option " + flag);
      }
    }
    const auto in = npy::load_npz(argv[1]);
    std::size_t rows = 0;
    const auto gyro = column_block(in, "gyro", 3, rows);
    const auto q = column_block(in, "q", 12, rows);
    const auto dq = column_block(in, "dq", 12, rows);

    std::vector<phm_core::StaleGroup> groups{{"gyro", {0, 1, 2}, gyro_max}};
    for (std::size_t m = 0; m < 12; ++m) {
      groups.push_back({"motor" + std::to_string(m), {3 + m, 15 + m}, joint_max});
    }
    phm_core::StaleSignalAdapter det("/lowstate", groups);
    phm_core::StaleSignalSample s{"/lowstate", std::vector<double>(27)};
    std::vector<double> violating(rows), score(rows), action(rows), worst(rows);
    for (std::size_t t = 0; t < rows; ++t) {
      for (std::size_t j = 0; j < 3; ++j) {
        // Through float, as the node reads the float32 message fields.
        s.values[j] = static_cast<float>(gyro[t * 3 + j]);
      }
      for (std::size_t m = 0; m < 12; ++m) {
        s.values[3 + m] = static_cast<float>(q[t * 12 + m]);
        s.values[15 + m] = static_cast<float>(dq[t * 12 + m]);
      }
      const auto v = det.update(s);
      violating[t] = v->violating ? 1.0 : 0.0;
      score[t] = v->score;
      action[t] = static_cast<double>(v->suggested_action);
      int w = 0;
      for (std::size_t g = 0; g < groups.size(); ++g) {
        w = std::max(w, det.run(g));
      }
      worst[t] = static_cast<double>(w);
    }
    std::vector<std::pair<std::string, npy::Array>> out;
    out.emplace_back("violating", npy::Array::from_doubles(violating, {rows}));
    out.emplace_back("score", npy::Array::from_doubles(score, {rows}));
    out.emplace_back("suggested_action", npy::Array::from_doubles(action, {rows}));
    out.emplace_back("worst_run", npy::Array::from_doubles(worst, {rows}));
    npy::save_npz(argv[2], out);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
