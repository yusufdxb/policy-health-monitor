// Copyright 2026 Yusuf Guenena. MIT License.
// Dependency-free std::vector backend. This is the default and guarantees
// colcon always builds even with no Eigen and no LibTorch present.
//
// rolling_spread = sum over dims of population (ddof=0) variance of each column,
// byte-faithful to phm_core/phm_core/calibration.py:32-35 and its source
// phantom-braking/src/e6_detector.py:19-22 (numpy.var default is biased).
#include "phm_ood_cpp/backend.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace phm_ood_cpp
{
namespace
{

class PlainBackend : public Backend
{
public:
  double rolling_spread(
    const std::vector<double> & block, std::size_t window, std::size_t dim) const override
  {
    if (window == 0 || dim == 0 || block.size() != window * dim) {
      throw std::invalid_argument("PlainBackend: block.size() must equal window*dim");
    }
    double total = 0.0;
    const double inv_w = 1.0 / static_cast<double>(window);
    for (std::size_t d = 0; d < dim; ++d) {
      // First pass: column mean.
      double mean = 0.0;
      for (std::size_t f = 0; f < window; ++f) {
        mean += block[f * dim + d];
      }
      mean *= inv_w;
      // Second pass: sum of squared deviations, divided by window (ddof=0).
      double ss = 0.0;
      for (std::size_t f = 0; f < window; ++f) {
        const double x = block[f * dim + d] - mean;
        ss += x * x;
      }
      total += ss * inv_w;
    }
    return total;
  }

  std::string name() const override { return "plain"; }
};

}  // namespace

std::unique_ptr<Backend> make_plain_backend()
{
  return std::make_unique<PlainBackend>();
}

std::unique_ptr<Backend> make_default_backend()
{
  const char * forced = std::getenv("PHM_BACKEND");
  const std::string want = forced ? std::string(forced) : std::string();

#ifdef PHM_WITH_LIBTORCH
  if (want.empty() || want == "libtorch") {
    return make_torch_backend();
  }
#endif
#ifdef PHM_HAVE_EIGEN
  if (want.empty() || want == "eigen") {
    return make_eigen_backend();
  }
#endif
  // "plain" requested, or requested backend not compiled in: fall back to plain.
  return make_plain_backend();
}

}  // namespace phm_ood_cpp
