// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/sim.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace phm_core
{

EmbeddingBatch generate_embeddings(
  std::size_t dim, std::size_t n_frames, double in_dist_scale, double ood_scale, uint64_t seed)
{
  numpy_random::Generator rng(seed);
  EmbeddingBatch batch{Matrix(n_frames, dim), Matrix(n_frames, dim)};
  rng.normal(0.0, in_dist_scale, batch.in_dist.data.data(), batch.in_dist.data.size());
  rng.normal(0.0, ood_scale, batch.ood.data.data(), batch.ood.data.size());
  return batch;
}

EmbeddingStream::EmbeddingStream(
  std::size_t dim, std::size_t n_in_dist, double in_dist_scale, double ood_scale,
  std::string policy_id, uint64_t seed)
: dim_(dim), n_in_dist_(n_in_dist), in_dist_scale_(in_dist_scale), ood_scale_(ood_scale),
  policy_id_(std::move(policy_id)), rng_(seed)
{
  if (n_in_dist < 1) {
    throw std::invalid_argument("n_in_dist must be >= 1");
  }
}

void EmbeddingStream::next_frame(double * out)
{
  const double scale = frame_index_ < n_in_dist_ ? in_dist_scale_ : ood_scale_;
  rng_.normal(0.0, scale, out, dim_);
  ++frame_index_;
}

std::vector<double> EmbeddingStream::next_frame()
{
  std::vector<double> out(dim_);
  next_frame(out.data());
  return out;
}

void EmbeddingStream::trigger_ood()
{
  if (frame_index_ < n_in_dist_) {
    frame_index_ = n_in_dist_;
  }
}

}  // namespace phm_core
