// Copyright 2026 Yusuf Guenena. MIT License.
// Pluggable kernel for the rolling spread of one window.
//
// Math contract (identical for every backend):
//   spread(window of W frames x D dims) = sum_d Var_pop(frames[:, d])
// where Var_pop is the population (ddof=0) variance, numpy.var's default. This
// is the per-frame trace of the rolling covariance of the policy embedding, as
// in supercombo-blindspot src/e6_detector.py (rolling_spread).
//
// Backends:
//   plain     dependency-free, always built. Reproduces NumPy's evaluation
//             order bit-for-bit (frame-order column sums, pairwise final sum),
//             so it scores exactly what calibrate_threshold() calibrated.
//   eigen     built when Eigen 3 is found (option PHM_WITH_EIGEN). Agrees with
//             plain to ~1e-12 relative, not bit-for-bit.
//   libtorch  opt-in (option PHM_WITH_LIBTORCH, default OFF).
//
// The window is read in place from the OOD core's ring buffer, so scoring a
// frame copies and allocates nothing in the plain backend.
#ifndef PHM_CORE__SPREAD_BACKEND_HPP_
#define PHM_CORE__SPREAD_BACKEND_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace phm_core
{

// A rolling window of `window` frames of `dim` float32 values held in a ring:
// frame k (0 = oldest, window - 1 = newest) starts at
// data + ((oldest + k) % window) * dim.
struct WindowView
{
  const float * data = nullptr;
  std::size_t window = 0;
  std::size_t dim = 0;
  std::size_t oldest = 0;

  const float * frame(std::size_t k) const
  {
    return data + ((oldest + k) % window) * dim;
  }
};

class SpreadBackend
{
public:
  virtual ~SpreadBackend() = default;
  // Rolling spread of the window; non-finite input yields a non-finite result.
  virtual double spread(const WindowView & view) = 0;
  // "plain", "eigen" or "libtorch".
  virtual const char * name() const = 0;
};

std::unique_ptr<SpreadBackend> make_plain_backend();

// The named backend, or nullptr if it was not compiled in.
std::unique_ptr<SpreadBackend> make_backend(const std::string & name);

// Names of the backends compiled into this build.
std::vector<std::string> available_backends();

// The env var PHM_BACKEND ("plain" | "eigen" | "libtorch") selects a backend;
// unset, the preference is libtorch (if built) > eigen (if built) > plain. A
// requested backend that was not built falls back to plain.
std::unique_ptr<SpreadBackend> make_default_backend();

}  // namespace phm_core

#endif  // PHM_CORE__SPREAD_BACKEND_HPP_
