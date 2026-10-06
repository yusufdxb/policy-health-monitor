// Copyright 2026 Yusuf Guenena. MIT License.
// NumPy-compatible random streams. The algorithms follow NumPy's
// numpy/random/bit_generator.pyx (SeedSequence), src/pcg64/pcg64.h (PCG64),
// src/distributions/distributions.c (random_standard_normal,
// random_bounded_uint64_fill, random_interval) and src/legacy/
// legacy-distributions.c (legacy_gauss). test_numpy_random.cpp pins every
// stream against values produced by NumPy 1.26.
#include "phm_core/numpy_random.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace phm_core
{
namespace numpy_random
{
namespace detail
{
extern const uint64_t kZigKi[256];
extern const double kZigWi[256];
extern const double kZigFi[256];
}  // namespace detail

namespace
{

// SeedSequence constants (bit_generator.pyx).
constexpr uint32_t kInitA = 0x43b0d7e5u;
constexpr uint32_t kMultA = 0x931e8875u;
constexpr uint32_t kInitB = 0x8b51f9ddu;
constexpr uint32_t kMultB = 0x58f38dedu;
constexpr uint32_t kMixMultL = 0xca01f9ddu;
constexpr uint32_t kMixMultR = 0x4973f715u;
constexpr int kXShift = 16;

uint32_t hashmix(uint32_t value, uint32_t & hash_const)
{
  value ^= hash_const;
  hash_const *= kMultA;
  value *= hash_const;
  value ^= value >> kXShift;
  return value;
}

uint32_t mix(uint32_t x, uint32_t y)
{
  uint32_t result = kMixMultL * x - kMixMultR * y;
  result ^= result >> kXShift;
  return result;
}

// PCG64 multiplier 2549297995355413924 * 2**64 + 4865540595714422341.
const uint128_t kPcgMult =
  (static_cast<uint128_t>(2549297995355413924ULL) << 64) + 4865540595714422341ULL;

inline uint64_t rotr64(uint64_t value, unsigned int rot)
{
  return (value >> rot) | (value << ((-rot) & 63u));
}

// ziggurat_nor_r and ziggurat_nor_inv_r from ziggurat_constants.h.
constexpr double kZigNorR = 3.6541528853610087963519472518;
constexpr double kZigNorInvR = 0.27366123732975827203338247596;

}  // namespace

// ---------------------------------------------------------------------------
// SeedSequence
// ---------------------------------------------------------------------------
SeedSequence::SeedSequence(uint64_t entropy)
{
  // _coerce_to_uint32_array(int): little-endian 32-bit words, [0] for zero.
  std::vector<uint32_t> words;
  if (entropy == 0) {
    words.push_back(0u);
  }
  while (entropy != 0) {
    words.push_back(static_cast<uint32_t>(entropy & 0xffffffffULL));
    entropy >>= 32;
  }
  uint32_t hash_const = kInitA;
  for (std::size_t i = 0; i < 4; ++i) {
    pool_[i] = hashmix(i < words.size() ? words[i] : 0u, hash_const);
  }
  for (std::size_t src = 0; src < 4; ++src) {
    for (std::size_t dst = 0; dst < 4; ++dst) {
      if (src != dst) {
        pool_[dst] = mix(pool_[dst], hashmix(pool_[src], hash_const));
      }
    }
  }
  for (std::size_t src = 4; src < words.size(); ++src) {
    for (std::size_t dst = 0; dst < 4; ++dst) {
      pool_[dst] = mix(pool_[dst], hashmix(words[src], hash_const));
    }
  }
}

std::vector<uint32_t> SeedSequence::generate_state_u32(std::size_t n_words) const
{
  std::vector<uint32_t> out(n_words);
  uint32_t hash_const = kInitB;
  for (std::size_t i = 0; i < n_words; ++i) {
    uint32_t value = pool_[i % 4];
    value ^= hash_const;
    hash_const *= kMultB;
    value *= hash_const;
    value ^= value >> kXShift;
    out[i] = value;
  }
  return out;
}

std::vector<uint64_t> SeedSequence::generate_state_u64(std::size_t n_words) const
{
  const std::vector<uint32_t> words = generate_state_u32(2 * n_words);
  std::vector<uint64_t> out(n_words);
  for (std::size_t i = 0; i < n_words; ++i) {
    out[i] = static_cast<uint64_t>(words[2 * i]) |
      (static_cast<uint64_t>(words[2 * i + 1]) << 32);
  }
  return out;
}

// ---------------------------------------------------------------------------
// PCG64
// ---------------------------------------------------------------------------
Pcg64::Pcg64(uint64_t seed)
{
  const std::vector<uint64_t> val = SeedSequence(seed).generate_state_u64(4);
  // pcg64_set_seed: PCG_128BIT_CONSTANT(high, low) with high = val[0].
  const uint128_t initstate = (static_cast<uint128_t>(val[0]) << 64) | val[1];
  const uint128_t initseq = (static_cast<uint128_t>(val[2]) << 64) | val[3];
  state_ = 0;
  inc_ = (initseq << 1u) | 1u;
  state_ = state_ * kPcgMult + inc_;
  state_ += initstate;
  state_ = state_ * kPcgMult + inc_;
}

uint64_t Pcg64::next_u64()
{
  state_ = state_ * kPcgMult + inc_;
  const uint64_t hi = static_cast<uint64_t>(state_ >> 64);
  const uint64_t lo = static_cast<uint64_t>(state_);
  return rotr64(hi ^ lo, static_cast<unsigned int>(state_ >> 122u));
}

uint32_t Pcg64::next_u32()
{
  if (has_u32_) {
    has_u32_ = false;
    return u32_;
  }
  const uint64_t next = next_u64();
  has_u32_ = true;
  u32_ = static_cast<uint32_t>(next >> 32);
  return static_cast<uint32_t>(next & 0xffffffffULL);
}

double Pcg64::next_double()
{
  return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------
Generator::Generator(uint64_t seed)
: bitgen_(seed)
{
}

double Generator::standard_normal()
{
  using detail::kZigFi;
  using detail::kZigKi;
  using detail::kZigWi;
  for (;;) {
    uint64_t r = bitgen_.next_u64();
    const int idx = static_cast<int>(r & 0xff);
    r >>= 8;
    const int sign = static_cast<int>(r & 0x1);
    const uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
    double x = static_cast<double>(rabs) * kZigWi[idx];
    if (sign & 0x1) {
      x = -x;
    }
    if (rabs < kZigKi[idx]) {
      return x;  // 99.3% of draws
    }
    if (idx == 0) {
      for (;;) {
        // 1 - U avoids log(0), as in NumPy (GH 13361).
        const double xx = -kZigNorInvR * std::log1p(-bitgen_.next_double());
        const double yy = -std::log1p(-bitgen_.next_double());
        if (yy + yy > xx * xx) {
          return ((rabs >> 8) & 0x1) ? -(kZigNorR + xx) : kZigNorR + xx;
        }
      }
    } else {
      if (((kZigFi[idx - 1] - kZigFi[idx]) * bitgen_.next_double() + kZigFi[idx]) <
        std::exp(-0.5 * x * x))
      {
        return x;
      }
    }
  }
}

double Generator::normal(double loc, double scale)
{
  return loc + scale * standard_normal();
}

void Generator::normal(double loc, double scale, double * out, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = loc + scale * standard_normal();
  }
}

void Generator::standard_normal(double * out, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = standard_normal();
  }
}

double Generator::random()
{
  return bitgen_.next_double();
}

// ---------------------------------------------------------------------------
// RandomState (legacy MT19937)
// ---------------------------------------------------------------------------
RandomState::RandomState(uint32_t seed)
: mt_(seed)
{
}

uint32_t RandomState::next_u32()
{
  return static_cast<uint32_t>(mt_());
}

double RandomState::random_sample()
{
  const int32_t a = static_cast<int32_t>(next_u32() >> 5);
  const int32_t b = static_cast<int32_t>(next_u32() >> 6);
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

double RandomState::gauss()
{
  if (has_gauss_) {
    has_gauss_ = false;
    const double tmp = gauss_;
    gauss_ = 0.0;
    return tmp;
  }
  double x1;
  double x2;
  double r2;
  do {
    x1 = 2.0 * random_sample() - 1.0;
    x2 = 2.0 * random_sample() - 1.0;
    r2 = x1 * x1 + x2 * x2;
  } while (r2 >= 1.0 || r2 == 0.0);
  // Polar method, a more efficient version of Box-Muller.
  const double f = std::sqrt(-2.0 * std::log(r2) / r2);
  gauss_ = f * x1;
  has_gauss_ = true;
  return f * x2;
}

uint64_t RandomState::interval(uint64_t max)
{
  if (max == 0) {
    return 0;
  }
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  uint64_t value;
  if (max <= 0xffffffffULL) {
    while ((value = (next_u32() & mask)) > max) {
    }
  } else {
    for (;;) {
      // MT19937 next_uint64: high word first, then low word.
      const uint64_t hi = next_u32();
      const uint64_t lo = next_u32();
      value = ((hi << 32) | lo) & mask;
      if (value <= max) {
        break;
      }
    }
  }
  return value;
}

int64_t RandomState::randint(int64_t low, int64_t high)
{
  int64_t out = 0;
  randint(low, high, &out, 1);
  return out;
}

void RandomState::randint(int64_t low, int64_t high, int64_t * out, std::size_t n)
{
  if (high <= low) {
    throw std::invalid_argument("randint: high must be > low");
  }
  // _rand_int64 with the legacy masked method on the closed interval
  // [low, high - 1]; ranges that fit in 32 bits draw 32-bit words.
  const uint64_t rng = static_cast<uint64_t>(high - 1) - static_cast<uint64_t>(low);
  if (rng > 0xffffffffULL) {
    throw std::invalid_argument("randint: ranges wider than 2**32 are not implemented");
  }
  const uint64_t off = static_cast<uint64_t>(low);
  if (rng == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = low;
    }
    return;
  }
  if (rng == 0xffffffffULL) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = static_cast<int64_t>(off + next_u32());
    }
    return;
  }
  uint32_t mask = static_cast<uint32_t>(rng);
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  for (std::size_t i = 0; i < n; ++i) {
    uint32_t value;
    while ((value = (next_u32() & mask)) > rng) {
    }
    out[i] = static_cast<int64_t>(off + value);
  }
}

std::vector<int64_t> RandomState::permutation(std::size_t n)
{
  std::vector<int64_t> arr(n);
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = static_cast<int64_t>(i);
  }
  // Legacy _shuffle_raw: for i in reversed(range(1, n)): swap(i, interval(i)).
  for (std::size_t i = n; i-- > 1; ) {
    const std::size_t j = static_cast<std::size_t>(interval(i));
    const int64_t tmp = arr[j];
    arr[j] = arr[i];
    arr[i] = tmp;
  }
  return arr;
}

std::vector<int64_t> RandomState::choice_without_replacement(std::size_t n, std::size_t size)
{
  if (size > n) {
    throw std::invalid_argument("choice: cannot take a larger sample than population");
  }
  std::vector<int64_t> perm = permutation(n);
  perm.resize(size);
  return perm;
}

}  // namespace numpy_random
}  // namespace phm_core
