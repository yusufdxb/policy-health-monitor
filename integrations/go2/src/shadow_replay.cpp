// Copyright 2026 Yusuf Guenena. MIT License.
// phoenix_shadow_replay: run the shadow policy offline on a recorded or
// synthetic /lowstate sequence and save every observation, action and latent.
//
//   phoenix_shadow_replay <model.onnx> <lowstate.npz> <out.npz>
//                         [--fault none|freeze_obs|freeze_sensors|stop] [--fault-at TICK]
//
// lowstate.npz holds float32 arrays q [T, 12], dq [T, 12], quat_wxyz [T, 4]
// and gyro [T, 3] in Unitree motor order, one row per policy tick. The replay
// uses the same ShadowStepper as the live node (last-action feedback, the
// freeze_obs, freeze_sensors and stop faults), so its output is what the node would have
// published for that input. out.npz holds obs [N, 48], action [N, 12],
// latent [N, D] (float32) and tick [N] (int64, the input row of each output),
// the inputs for a parity check against another implementation.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/npy.hpp"
#include "phm_go2/shadow_policy.hpp"

namespace npy = phm_core::npy;

int main(int argc, char ** argv)
{
  if (argc < 4) {
    std::fprintf(
      stderr,
      "usage: phoenix_shadow_replay <model.onnx> <lowstate.npz> <out.npz> "
      "[--fault none|freeze_obs|freeze_sensors|stop] [--fault-at TICK]\n");
    return 2;
  }
  try {
    std::string fault_name = "none";
    int64_t fault_at = -1;
    for (int i = 4; i + 1 < argc; i += 2) {
      const std::string flag = argv[i];
      if (flag == "--fault") {
        fault_name = argv[i + 1];
      } else if (flag == "--fault-at") {
        fault_at = std::strtoll(argv[i + 1], nullptr, 10);
      } else {
        std::fprintf(stderr, "unknown option %s\n", flag.c_str());
        return 2;
      }
    }
    const auto in = npy::load_npz(argv[2]);
    // Every input must be a [T, width] array with the same T, so the loop
    // below never indexes past one of them.
    std::size_t ticks = 0;
    const auto column = [&](const char * key, std::size_t width) {
        const auto it = in.find(key);
        if (it == in.end()) {
          throw std::runtime_error(std::string("missing array '") + key + "'");
        }
        const npy::Array & a = it->second;
        if (a.shape.size() != 2 || a.shape[1] != width) {
          throw std::runtime_error(
            std::string("array '") + key + "' must have shape [T, " + std::to_string(width) +
            "]");
        }
        if (key == std::string("q")) {
          ticks = a.shape[0];
        } else if (a.shape[0] != ticks) {
          throw std::runtime_error(
            std::string("array '") + key + "' has " + std::to_string(a.shape[0]) +
            " rows, but q has " + std::to_string(ticks));
        }
        return a.as_doubles();
      };
    const auto q = column("q", phm_go2::kJoints);
    const auto dq = column("dq", phm_go2::kJoints);
    const auto quat = column("quat_wxyz", 4);
    const auto gyro = column("gyro", 3);

    phm_go2::PolicyRunner runner(argv[1], 1);
    phm_go2::ShadowStepper stepper(runner, phm_go2::parse_fault(fault_name));
    const std::size_t d = runner.latent_dim();
    std::vector<float> obs;
    std::vector<float> action;
    std::vector<float> latent;
    std::vector<double> tick_index;
    phm_go2::LowStateSample s;
    for (std::size_t t = 0; t < ticks; ++t) {
      for (std::size_t j = 0; j < phm_go2::kJoints; ++j) {
        s.q[j] = static_cast<float>(q[t * phm_go2::kJoints + j]);
        s.dq[j] = static_cast<float>(dq[t * phm_go2::kJoints + j]);
      }
      for (std::size_t j = 0; j < 4; ++j) {
        s.quat_wxyz[j] = static_cast<float>(quat[t * 4 + j]);
      }
      for (std::size_t j = 0; j < 3; ++j) {
        s.gyro[j] = static_cast<float>(gyro[t * 3 + j]);
      }
      const bool fault_active = fault_at >= 0 && static_cast<int64_t>(t) >= fault_at;
      if (stepper.step(&s, fault_active) != phm_go2::StepOutcome::kPublished) {
        continue;
      }
      obs.insert(obs.end(), stepper.obs().begin(), stepper.obs().end());
      action.insert(action.end(), stepper.last_action().begin(), stepper.last_action().end());
      latent.insert(latent.end(), stepper.latent(), stepper.latent() + d);
      tick_index.push_back(static_cast<double>(t));
    }
    const std::size_t n = tick_index.size();
    std::vector<std::pair<std::string, npy::Array>> out;
    out.emplace_back("obs", npy::Array::from_floats(obs, {n, phm_go2::kObsDim}));
    out.emplace_back("action", npy::Array::from_floats(action, {n, phm_go2::kJoints}));
    out.emplace_back("latent", npy::Array::from_floats(latent, {n, d}));
    npy::Array ticks_arr;
    ticks_arr.descr = "<i8";
    ticks_arr.shape = {n};
    ticks_arr.bytes.resize(n * 8);
    for (std::size_t i = 0; i < n; ++i) {
      const int64_t v = static_cast<int64_t>(tick_index[i]);
      for (int b = 0; b < 8; ++b) {
        ticks_arr.bytes[i * 8 + static_cast<std::size_t>(b)] =
          static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xff);
      }
    }
    out.emplace_back("tick", ticks_arr);
    npy::save_npz(argv[3], out);
    std::printf("replayed %zu ticks, %zu outputs, latent_dim %zu\n", ticks, n, d);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "phoenix_shadow_replay: %s\n", e.what());
    return 1;
  }
  return 0;
}
