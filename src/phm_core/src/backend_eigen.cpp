// Copyright 2026 Yusuf Guenena. MIT License.
// Eigen rolling-spread backend, compiled only when Eigen 3 was found
// (PHM_HAVE_EIGEN). Same population-variance math as the plain backend,
// expressed as Eigen column reductions over the ring buffer in place (frame
// order does not change the variance; Eigen's vectorized reduction order makes
// the result agree with plain to ~1e-12 relative rather than bit-for-bit).
#ifdef PHM_HAVE_EIGEN

#include <memory>

#include <Eigen/Core>  // NOLINT(build/include_order)

#include "phm_core/spread_backend.hpp"

namespace phm_core
{
namespace
{

class EigenBackend : public SpreadBackend
{
public:
  double spread(const WindowView & view) override
  {
    using RowMajorF =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Index w = static_cast<Eigen::Index>(view.window);
    const Eigen::Index d = static_cast<Eigen::Index>(view.dim);
    Eigen::Map<const RowMajorF> m(view.data, w, d);
    if (mean_.size() != d) {
      mean_.resize(d);
      ss_.resize(d);
    }
    mean_.noalias() = m.cast<double>().colwise().sum();
    mean_ /= static_cast<double>(view.window);
    ss_.setZero();
    for (Eigen::Index r = 0; r < w; ++r) {
      ss_.array() += (m.row(r).cast<double>() - mean_).array().square();
    }
    return ss_.sum() / static_cast<double>(view.window);
  }

  const char * name() const override {return "eigen";}

private:
  Eigen::RowVectorXd mean_;
  Eigen::RowVectorXd ss_;
};

}  // namespace

std::unique_ptr<SpreadBackend> make_eigen_backend_impl()
{
  return std::make_unique<EigenBackend>();
}

}  // namespace phm_core

#endif  // PHM_HAVE_EIGEN
