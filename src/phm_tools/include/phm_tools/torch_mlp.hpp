// Copyright 2026 Yusuf Guenena. MIT License.
// A small float32 multilayer perceptron trained full-batch with Adam, matching
// what the original PyTorch research scripts did (torch.nn.Linear layers,
// MSELoss with mean reduction, torch.optim.Adam defaults), for the offline
// benchmark and demo only.
//
// Initialization reproduces PyTorch exactly: torch.manual_seed(seed) seeds an
// MT19937 stream and each nn.Linear draws its weight (kaiming_uniform_ with
// a = sqrt(5), i.e. U(-1/sqrt(fan_in), 1/sqrt(fan_in))) and then its bias from
// it, in construction order, through torch's float32 uniform_ kernel. Training
// is the same algorithm in float32, but matrix products accumulate in a
// different order than PyTorch's BLAS, so trained weights agree with PyTorch's
// only approximately (the benchmark and demo report the measured difference).
//
// Header-only; requires Eigen 3 in the including target.
#ifndef PHM_TOOLS__TORCH_MLP_HPP_
#define PHM_TOOLS__TORCH_MLP_HPP_

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace phm_tools
{
namespace torch_mlp
{

using MatrixF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVectorF = Eigen::Matrix<float, 1, Eigen::Dynamic>;

// torch.manual_seed(seed): the CPU generator is an MT19937 seeded with the
// low 32 bits of the seed.
inline std::mt19937 manual_seed(uint64_t seed)
{
  return std::mt19937(static_cast<uint32_t>(seed & 0xffffffffULL));
}

// tensor.uniform_(from, to) on a float32 CPU tensor: 24 random bits of one
// 32-bit draw, then x * (to - from) + from in float32 (torch's vectorized
// kernel fuses the multiply-add, so std::fma reproduces its rounding).
inline void uniform_(std::mt19937 & gen, float * out, std::size_t n, double from_d, double to_d)
{
  const float from = static_cast<float>(from_d);
  const float to = static_cast<float>(to_d);
  for (std::size_t i = 0; i < n; ++i) {
    const uint32_t r = gen();
    const float x = static_cast<float>(r & ((1u << 24) - 1)) * (1.0f / 16777216.0f);
    out[i] = std::fma(x, to - from, from);
  }
}

enum class Activation { kReLU, kTanh };

struct Linear
{
  MatrixF weight;      // out x in (torch layout)
  RowVectorF bias;     // out
  // Adam state.
  MatrixF m_w, v_w;
  RowVectorF m_b, v_b;

  Linear(int in, int out, std::mt19937 & gen)
  : weight(out, in), bias(out)
  {
    // kaiming_uniform_(a=sqrt(5)): gain = sqrt(2 / (1 + a^2)),
    // bound = sqrt(3) * gain / sqrt(fan_in) (= 1 / sqrt(fan_in)).
    const double a = std::sqrt(5.0);
    const double gain = std::sqrt(2.0 / (1.0 + std::pow(a, 2)));
    const double bound = std::sqrt(3.0) * (gain / std::sqrt(static_cast<double>(in)));
    uniform_(gen, weight.data(), static_cast<std::size_t>(weight.size()), -bound, bound);
    const double bb = 1.0 / std::sqrt(static_cast<double>(in));
    uniform_(gen, bias.data(), static_cast<std::size_t>(bias.size()), -bb, bb);
    m_w = MatrixF::Zero(out, in);
    v_w = MatrixF::Zero(out, in);
    m_b = RowVectorF::Zero(out);
    v_b = RowVectorF::Zero(out);
  }
};

class Mlp
{
public:
  // sizes = {in, hidden..., out}; `act` between layers, none after the last.
  Mlp(const std::vector<int> & sizes, Activation act, std::mt19937 & gen)
  : act_(act)
  {
    if (sizes.size() < 2) {
      throw std::invalid_argument("Mlp needs at least an input and an output size");
    }
    for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
      layers_.emplace_back(sizes[i], sizes[i + 1], gen);
    }
  }

  // Output for a batch (rows are samples). `hidden_out`, if given, receives
  // the activation feeding the last layer (the penultimate representation).
  MatrixF forward(const MatrixF & x, MatrixF * hidden_out = nullptr) const
  {
    MatrixF h = x;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      MatrixF z = h * layers_[i].weight.transpose();
      z.rowwise() += layers_[i].bias;
      if (i + 1 < layers_.size()) {
        h = activate(z);
      } else {
        h = z;
      }
      if (hidden_out != nullptr && i + 2 == layers_.size()) {
        *hidden_out = h;
      }
    }
    return h;
  }

  // One full-batch Adam step on MSELoss(mean); returns the loss before the step.
  float train_step(
    const MatrixF & x, const MatrixF & y, double lr, double beta1 = 0.9, double beta2 = 0.999,
    double eps = 1e-8)
  {
    // Forward, keeping pre- and post-activations.
    std::vector<MatrixF> acts{ x };
    std::vector<MatrixF> pre;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      MatrixF z = acts.back() * layers_[i].weight.transpose();
      z.rowwise() += layers_[i].bias;
      pre.push_back(z);
      acts.push_back(i + 1 < layers_.size() ? activate(z) : z);
    }
    const MatrixF diff = acts.back() - y;
    const double numel = static_cast<double>(diff.size());
    const float loss = static_cast<float>(diff.cast<double>().array().square().sum() / numel);
    // dL/dout = (out - y) * 2 / numel.
    MatrixF grad = diff * static_cast<float>(2.0 / numel);
    ++step_;
    // Python evaluates these scalars in double before torch casts them.
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step_));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step_));
    const float step_size = static_cast<float>(lr / bc1);
    const float bc2_sqrt = static_cast<float>(std::sqrt(bc2));
    const float one_minus_b1 = static_cast<float>(1.0 - beta1);
    const float one_minus_b2 = static_cast<float>(1.0 - beta2);
    const float b2 = static_cast<float>(beta2);
    const float e = static_cast<float>(eps);
    for (std::size_t li = layers_.size(); li-- > 0; ) {
      Linear & L = layers_[li];
      const MatrixF gw = grad.transpose() * acts[li];
      const RowVectorF gb = grad.colwise().sum();
      if (li > 0) {
        MatrixF gh = grad * L.weight;
        grad = activation_backward(pre[li - 1], acts[li], gh);
      }
      adam(L.weight, L.m_w, L.v_w, gw, one_minus_b1, b2, one_minus_b2, step_size, bc2_sqrt, e);
      adam(L.bias, L.m_b, L.v_b, gb, one_minus_b1, b2, one_minus_b2, step_size, bc2_sqrt, e);
    }
    return loss;
  }

  const std::vector<Linear> & layers() const {return layers_;}

private:
  MatrixF activate(const MatrixF & z) const
  {
    if (act_ == Activation::kReLU) {
      return z.cwiseMax(0.0f);
    }
    return z.array().tanh().matrix();
  }

  MatrixF activation_backward(const MatrixF & z, const MatrixF & a, const MatrixF & g) const
  {
    if (act_ == Activation::kReLU) {
      return (z.array() > 0.0f).select(g, MatrixF::Zero(g.rows(), g.cols()));
    }
    return (g.array() * (1.0f - a.array().square())).matrix();
  }

  // torch.optim.Adam single-tensor update (no weight decay, no amsgrad).
  template<typename P, typename G>
  static void adam(
    P & param, P & m, P & v, const G & grad, float one_minus_b1, float b2, float one_minus_b2,
    float step_size, float bc2_sqrt, float eps)
  {
    m.array() += one_minus_b1 * (grad.array() - m.array());
    v.array() = v.array() * b2 + one_minus_b2 * grad.array() * grad.array();
    const auto denom = v.array().sqrt() / bc2_sqrt + eps;
    param.array() -= step_size * (m.array() / denom);
  }

  Activation act_;
  std::vector<Linear> layers_;
  int64_t step_ = 0;
};

}  // namespace torch_mlp
}  // namespace phm_tools

#endif  // PHM_TOOLS__TORCH_MLP_HPP_
