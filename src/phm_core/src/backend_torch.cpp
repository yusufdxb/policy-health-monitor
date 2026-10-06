// Copyright 2026 Yusuf Guenena. MIT License.
// LibTorch rolling-spread backend, compiled only with -DPHM_WITH_LIBTORCH=ON
// and a found Torch package. torch.var(unbiased=false) matches numpy.var's
// ddof=0. Opt-in only: LibTorch is a large dependency the runtime never needs.
#ifdef PHM_WITH_LIBTORCH

#include <cstdint>
#include <memory>

#include <torch/torch.h>  // NOLINT(build/include_order)

#include "phm_core/spread_backend.hpp"

namespace phm_core
{
namespace
{

class TorchBackend : public SpreadBackend
{
public:
  double spread(const WindowView & view) override
  {
    // The ring holds the window frames contiguously; frame order does not
    // change the variance, so the ring is wrapped without reordering.
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor t = torch::from_blob(
      const_cast<float *>(view.data),
      {static_cast<int64_t>(view.window), static_cast<int64_t>(view.dim)}, opts);
    return t.to(torch::kFloat64).var(/*dim=*/0, /*unbiased=*/false).sum().item<double>();
  }

  const char * name() const override {return "libtorch";}
};

}  // namespace

std::unique_ptr<SpreadBackend> make_torch_backend_impl()
{
  return std::make_unique<TorchBackend>();
}

}  // namespace phm_core

#endif  // PHM_WITH_LIBTORCH
