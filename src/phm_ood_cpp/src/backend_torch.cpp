// Copyright 2026 Yusuf Guenena. MIT License.
// LibTorch backend, compiled only when -DPHM_WITH_LIBTORCH=ON and find_package(Torch)
// succeeds (CMake sets PHM_WITH_LIBTORCH). Same population-variance math as the
// other backends (phm_core/phm_core/calibration.py:32-35); torch.var with
// unbiased=false matches numpy.var's default ddof=0.
#ifdef PHM_WITH_LIBTORCH

#include <cstdint>
#include <stdexcept>
#include <string>

#include <torch/torch.h>  // NOLINT(build/include_order)

#include "phm_ood_cpp/backend.hpp"

namespace phm_ood_cpp
{
namespace
{

class TorchBackend : public Backend
{
public:
  double rolling_spread(
    const std::vector<double> & block, std::size_t window, std::size_t dim) const override
  {
    if (window == 0 || dim == 0 || block.size() != window * dim) {
      throw std::invalid_argument("TorchBackend: block.size() must equal window*dim");
    }
    // from_blob does not own the buffer; clone so the tensor is safe past return.
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor t = torch::from_blob(
      const_cast<double *>(block.data()),
      {static_cast<int64_t>(window), static_cast<int64_t>(dim)}, opts).clone();
    // Population variance per column (dim 0), unbiased=false == ddof=0, then sum.
    torch::Tensor var = t.var(/*dim=*/0, /*unbiased=*/false);
    return var.sum().item<double>();
  }

  std::string name() const override { return "libtorch"; }
};

}  // namespace

std::unique_ptr<Backend> make_torch_backend()
{
  return std::make_unique<TorchBackend>();
}

}  // namespace phm_ood_cpp

#endif  // PHM_WITH_LIBTORCH
