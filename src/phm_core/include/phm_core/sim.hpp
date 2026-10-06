// Copyright 2026 Yusuf Guenena. MIT License.
// Synthetic policy-embedding streams for end-to-end tests without a policy.
//
// In-distribution frames are iid N(0, in_dist_scale^2 I); OOD (collapse)
// frames are iid N(0, ood_scale^2 I), so the rolling spread drops by
// (ood_scale / in_dist_scale)^2. Draws come from a NumPy-compatible
// Generator, so a seed yields the same frames the original NumPy generator
// produced (numpy.random.default_rng(seed).normal, C order).
#ifndef PHM_CORE__SIM_HPP_
#define PHM_CORE__SIM_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/matrix.hpp"
#include "phm_core/numpy_random.hpp"

namespace phm_core
{

struct EmbeddingBatch
{
  Matrix in_dist;  // n_frames x dim
  Matrix ood;      // n_frames x dim
};

// One generator draws the in-distribution batch, then the OOD batch.
EmbeddingBatch generate_embeddings(
  std::size_t dim = 64, std::size_t n_frames = 200, double in_dist_scale = 1.0,
  double ood_scale = 0.01, uint64_t seed = 42);

// Frame-by-frame stream: n_in_dist healthy frames, then OOD frames forever
// (or from trigger_ood()).
class EmbeddingStream
{
public:
  // Throws std::invalid_argument when n_in_dist < 1.
  EmbeddingStream(
    std::size_t dim = 64, std::size_t n_in_dist = 100, double in_dist_scale = 1.0,
    double ood_scale = 0.01, std::string policy_id = "phm_sim", uint64_t seed = 42);

  // Fill out[0..dim) with the next frame and advance the frame index.
  void next_frame(double * out);
  std::vector<double> next_frame();

  // Rewind to frame 0. The random stream continues; it is not re-seeded.
  void reset() {frame_index_ = 0;}
  // Jump to the OOD phase now; no-op if already there.
  void trigger_ood();

  std::size_t dim() const {return dim_;}
  const std::string & policy_id() const {return policy_id_;}
  std::size_t frame_index() const {return frame_index_;}
  bool is_ood_phase() const {return frame_index_ >= n_in_dist_;}

private:
  std::size_t dim_;
  std::size_t n_in_dist_;
  double in_dist_scale_;
  double ood_scale_;
  std::string policy_id_;
  numpy_random::Generator rng_;
  std::size_t frame_index_ = 0;
};

}  // namespace phm_core

#endif  // PHM_CORE__SIM_HPP_
