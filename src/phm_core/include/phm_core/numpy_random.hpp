// Copyright 2026 Yusuf Guenena. MIT License.
// NumPy-compatible random streams, so a seed means the same thing in C++ as it
// did in the original NumPy implementation of the generators and benchmark.
//
// Two NumPy APIs are reproduced bit-for-bit (pinned by test_numpy_random.cpp):
//
//   Generator    == numpy.random.default_rng(seed): SeedSequence -> PCG64
//                   (XSL-RR 128/64) bit generator, 256-layer ziggurat normal.
//   RandomState  == numpy.random.RandomState(seed): legacy MT19937 stream,
//                   polar Box-Muller gauss (randn), masked bounded integers
//                   (randint, choice with replacement) and the legacy
//                   Fisher-Yates shuffle (permutation, choice without
//                   replacement).
//
// Only the draws this project uses are implemented. Seeds are non-negative
// integers below 2**64 (Generator) or 2**32 (RandomState), as in NumPy.
#ifndef PHM_CORE__NUMPY_RANDOM_HPP_
#define PHM_CORE__NUMPY_RANDOM_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace phm_core
{
namespace numpy_random
{

// 128-bit unsigned arithmetic for PCG64 (GCC/Clang extension).
__extension__ typedef unsigned __int128 uint128_t;

// numpy.random.SeedSequence(entropy) with the default pool size of 4 words and
// no spawn key.
class SeedSequence
{
public:
  explicit SeedSequence(uint64_t entropy);
  std::vector<uint32_t> generate_state_u32(std::size_t n_words) const;
  std::vector<uint64_t> generate_state_u64(std::size_t n_words) const;

private:
  uint32_t pool_[4];
};

// numpy.random.PCG64 seeded through SeedSequence.
class Pcg64
{
public:
  explicit Pcg64(uint64_t seed);
  uint64_t next_u64();
  uint32_t next_u32();
  // (next_u64() >> 11) * 2**-53, numpy's next_double for PCG64.
  double next_double();

private:
  uint128_t state_;
  uint128_t inc_;
  bool has_u32_ = false;
  uint32_t u32_ = 0;
};

// numpy.random.Generator(PCG64(seed)), i.e. numpy.random.default_rng(seed).
class Generator
{
public:
  explicit Generator(uint64_t seed);
  double standard_normal();
  // loc + scale * standard_normal(), the element formula of Generator.normal.
  double normal(double loc, double scale);
  // Fill out[0..n) in C order, as Generator.normal(loc, scale, size=n) does.
  void normal(double loc, double scale, double * out, std::size_t n);
  void standard_normal(double * out, std::size_t n);
  // Generator.random(): uniform [0, 1).
  double random();

private:
  Pcg64 bitgen_;
};

// numpy.random.RandomState(seed), the legacy MT19937 API.
class RandomState
{
public:
  explicit RandomState(uint32_t seed);
  uint32_t next_u32();
  // random_sample(): 53-bit uniform in [0, 1) from two 32-bit draws.
  double random_sample();
  // One standard normal from randn() / standard_normal() (legacy_gauss).
  double gauss();
  // randint(low, high) for one int64 in [low, high), legacy masked rejection.
  int64_t randint(int64_t low, int64_t high);
  // randint(low, high, size=n) in C order.
  void randint(int64_t low, int64_t high, int64_t * out, std::size_t n);
  // permutation(n): legacy shuffle of arange(n).
  std::vector<int64_t> permutation(std::size_t n);
  // choice(n, size, replace=False) == permutation(n)[:size].
  std::vector<int64_t> choice_without_replacement(std::size_t n, std::size_t size);

private:
  uint64_t interval(uint64_t max);
  std::mt19937 mt_;
  bool has_gauss_ = false;
  double gauss_ = 0.0;
};

}  // namespace numpy_random
}  // namespace phm_core

#endif  // PHM_CORE__NUMPY_RANDOM_HPP_
