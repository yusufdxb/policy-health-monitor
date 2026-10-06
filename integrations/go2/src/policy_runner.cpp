// Copyright 2026 Yusuf Guenena. MIT License.
// ONNX Runtime inference for the shadow policy (C++ API, CPU provider).
#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "phm_go2/shadow_policy.hpp"

namespace phm_go2
{

struct PolicyRunner::Impl
{
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "phm_go2"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<float, kObsDim> obs{};
  std::vector<float> action;
  std::vector<float> latent;
  std::vector<Ort::Value> inputs;
  std::vector<Ort::Value> outputs;
  std::array<const char *, 1> in_names{{"obs"}};
  std::array<const char *, 2> out_names{{"action", "latent"}};
  Ort::RunOptions run_options;
  double last_run_ms = 0.0;
};

PolicyRunner::PolicyRunner(const std::string & onnx_path, int intra_op_threads)
: impl_(std::make_unique<Impl>())
{
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(intra_op_threads);
  impl_->session = std::make_unique<Ort::Session>(impl_->env, onnx_path.c_str(), opts);

  Ort::AllocatorWithDefaultOptions alloc;
  std::vector<std::string> outs;
  // The outputs must exist by name; a dimension of -1 is a dynamic size in
  // the model, which is accepted for 'action' (it is bound as 12 below) but
  // not for 'latent', whose size the node must know before the first tick.
  bool has_latent = false;
  bool has_action = false;
  int64_t latent_dim = -1;
  int64_t action_dim = -1;
  for (std::size_t i = 0; i < impl_->session->GetOutputCount(); ++i) {
    outs.emplace_back(impl_->session->GetOutputNameAllocated(i, alloc).get());
    const auto shape =
      impl_->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    if (outs.back() == "latent") {
      has_latent = true;
      latent_dim = shape.empty() ? -1 : shape.back();
    } else if (outs.back() == "action") {
      has_action = true;
      action_dim = shape.empty() ? -1 : shape.back();
    }
  }
  bool has_obs = false;
  for (std::size_t i = 0; i < impl_->session->GetInputCount(); ++i) {
    has_obs |= std::string(impl_->session->GetInputNameAllocated(i, alloc).get()) == "obs";
  }
  if (!has_latent || latent_dim <= 0 || !has_action ||
    (action_dim != static_cast<int64_t>(kJoints) && action_dim != -1) || !has_obs)
  {
    std::string names;
    for (const auto & n : outs) {
      names += (names.empty() ? "" : ", ") + n;
    }
    throw std::runtime_error(
      onnx_path + " has outputs [" + names + "]; need input 'obs' and outputs 'action' (12) "
      "and 'latent' (export with phoenix.sim2real.export --emit-latent)");
  }

  impl_->action.assign(kJoints, 0.0f);
  impl_->latent.assign(static_cast<std::size_t>(latent_dim), 0.0f);
  const std::array<int64_t, 2> obs_shape{1, static_cast<int64_t>(kObsDim)};
  const std::array<int64_t, 2> act_shape{1, static_cast<int64_t>(kJoints)};
  const std::array<int64_t, 2> lat_shape{1, latent_dim};
  impl_->inputs.push_back(
    Ort::Value::CreateTensor<float>(
      impl_->mem, impl_->obs.data(), impl_->obs.size(), obs_shape.data(), obs_shape.size()));
  impl_->outputs.push_back(
    Ort::Value::CreateTensor<float>(
      impl_->mem, impl_->action.data(), impl_->action.size(), act_shape.data(), act_shape.size()));
  impl_->outputs.push_back(
    Ort::Value::CreateTensor<float>(
      impl_->mem, impl_->latent.data(), impl_->latent.size(), lat_shape.data(), lat_shape.size()));
}

PolicyRunner::~PolicyRunner() = default;

std::size_t PolicyRunner::latent_dim() const
{
  return impl_->latent.size();
}

void PolicyRunner::run(const float * obs)
{
  for (std::size_t i = 0; i < kObsDim; ++i) {
    impl_->obs[i] = obs[i];
  }
  const auto t0 = std::chrono::steady_clock::now();
  impl_->session->Run(
    impl_->run_options, impl_->in_names.data(), impl_->inputs.data(), impl_->inputs.size(),
    impl_->out_names.data(), impl_->outputs.data(), impl_->outputs.size());
  impl_->last_run_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double PolicyRunner::last_run_ms() const
{
  return impl_->last_run_ms;
}

const float * PolicyRunner::action() const
{
  return impl_->action.data();
}

const float * PolicyRunner::latent() const
{
  return impl_->latent.data();
}

}  // namespace phm_go2
