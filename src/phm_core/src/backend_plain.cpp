// Copyright 2026 Yusuf Guenena. MIT License.
// Dependency-free rolling-spread backend (always built) and backend selection.
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "phm_core/spread_backend.hpp"
#include "spread_kernel.hpp"

namespace phm_core
{

#ifdef PHM_HAVE_EIGEN
std::unique_ptr<SpreadBackend> make_eigen_backend_impl();
#endif
#ifdef PHM_WITH_LIBTORCH
std::unique_ptr<SpreadBackend> make_torch_backend_impl();
#endif

namespace
{

class PlainBackend : public SpreadBackend
{
public:
  double spread(const WindowView & view) override
  {
    return detail::numpy_window_spread(
      [&view](std::size_t k) {return view.frame(k);}, view.window, view.dim, scratch_);
  }

  const char * name() const override {return "plain";}

private:
  std::vector<double> scratch_;
};

}  // namespace

std::unique_ptr<SpreadBackend> make_plain_backend()
{
  return std::make_unique<PlainBackend>();
}

std::unique_ptr<SpreadBackend> make_backend(const std::string & name)
{
  if (name == "plain") {
    return make_plain_backend();
  }
#ifdef PHM_HAVE_EIGEN
  if (name == "eigen") {
    return make_eigen_backend_impl();
  }
#endif
#ifdef PHM_WITH_LIBTORCH
  if (name == "libtorch") {
    return make_torch_backend_impl();
  }
#endif
  return nullptr;
}

std::vector<std::string> available_backends()
{
  std::vector<std::string> out{"plain"};
#ifdef PHM_HAVE_EIGEN
  out.emplace_back("eigen");
#endif
#ifdef PHM_WITH_LIBTORCH
  out.emplace_back("libtorch");
#endif
  return out;
}

std::unique_ptr<SpreadBackend> make_default_backend()
{
  const char * forced = std::getenv("PHM_BACKEND");
  const std::string want = forced ? std::string(forced) : std::string();
  if (!want.empty()) {
    auto backend = make_backend(want);
    return backend ? std::move(backend) : make_plain_backend();
  }
#ifdef PHM_WITH_LIBTORCH
  return make_torch_backend_impl();
#elif defined(PHM_HAVE_EIGEN)
  return make_eigen_backend_impl();
#else
  return make_plain_backend();
#endif
}

}  // namespace phm_core
