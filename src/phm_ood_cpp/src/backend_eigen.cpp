// Copyright 2026 Yusuf Guenena. MIT License.
// Eigen backend, compiled only when Eigen headers were found at configure time
// (CMake sets PHM_HAVE_EIGEN). Same population-variance math as the plain and
// Python paths (phm_core/phm_core/calibration.py:32-35), expressed as column
// reductions on an Eigen::Map over the node's row-major (window x dim) block.
#ifdef PHM_HAVE_EIGEN

#include <stdexcept>
#include <string>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include "phm_ood_cpp/backend.hpp"

namespace phm_ood_cpp
{
namespace
{

class EigenBackend : public Backend
{
public:
  double rolling_spread(
    const std::vector<double> & block, std::size_t window, std::size_t dim) const override
  {
    if (window == 0 || dim == 0 || block.size() != window * dim) {
      throw std::invalid_argument("EigenBackend: block.size() must equal window*dim");
    }
    // Map the row-major buffer as (window x dim): row = frame, col = dim.
    using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMajor> m(
      block.data(), static_cast<Eigen::Index>(window), static_cast<Eigen::Index>(dim));
    // Column means, then centered matrix, then population variance per column.
    const Eigen::RowVectorXd mean = m.colwise().mean();
    const RowMajor centered = m.rowwise() - mean;
    const Eigen::RowVectorXd var =
      centered.array().square().colwise().sum() / static_cast<double>(window);
    return var.sum();
  }

  std::string name() const override { return "eigen"; }
};

}  // namespace

std::unique_ptr<Backend> make_eigen_backend()
{
  return std::make_unique<EigenBackend>();
}

}  // namespace phm_ood_cpp

#endif  // PHM_HAVE_EIGEN
