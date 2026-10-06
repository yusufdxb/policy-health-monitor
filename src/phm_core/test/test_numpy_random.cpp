// Copyright 2026 Yusuf Guenena. MIT License.
// Pins the NumPy-compatible streams to values produced by NumPy 1.26.4
// (hex floats, so the comparison is bit-exact).
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "phm_core/numpy_random.hpp"

using phm_core::numpy_random::Generator;
using phm_core::numpy_random::RandomState;
using phm_core::numpy_random::SeedSequence;

TEST(Generator, StandardNormalMatchesNumpyDefaultRng)
{
  // np.random.default_rng(42).standard_normal(6)
  const double expected[] = {
    0x1.3807c1104fc6bp-2, -0x1.0a3c65fca9a7ep+0, 0x1.803b239e77350p-1,
    0x1.e191b2d157f36p-1, -0x1.f3770ac89d08fp+0, -0x1.4d5ba2db7ebc8p+0};
  Generator g(42);
  for (double e : expected) {
    EXPECT_EQ(g.standard_normal(), e);
  }
}

TEST(Generator, NormalLocScaleMatchesNumpy)
{
  // np.random.default_rng(0).normal(0, 0.01, 4)
  const double expected[] = {
    0x1.49981f82338a5p-10, -0x1.5a4e12b2fb580p-10, 0x1.a3b5176488fe8p-8, 0x1.12fd46e697e0ap-10};
  Generator g(0);
  std::vector<double> out(4);
  g.normal(0.0, 0.01, out.data(), out.size());
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(out[i], expected[i]);
  }
}

TEST(Generator, SeedsWiderThan32BitsUseSeveralEntropyWords)
{
  // np.random.default_rng(2**63 + 11).standard_normal(3)
  const double expected[] = {0x1.d125fc33bf8e9p-3, -0x1.f4044a73cafc8p-2, 0x1.e19718ac5458ap-3};
  Generator g((1ULL << 63) + 11);
  for (double e : expected) {
    EXPECT_EQ(g.standard_normal(), e);
  }
}

TEST(Generator, SameSeedSameStreamDifferentSeedDifferentStream)
{
  Generator a(7);
  Generator b(7);
  Generator c(8);
  bool any_diff = false;
  for (int i = 0; i < 100; ++i) {
    const double x = a.standard_normal();
    EXPECT_EQ(x, b.standard_normal());
    any_diff |= (x != c.standard_normal());
  }
  EXPECT_TRUE(any_diff);
}

TEST(SeedSequence, StateIsDeterministic)
{
  EXPECT_EQ(SeedSequence(5).generate_state_u64(4), SeedSequence(5).generate_state_u64(4));
  EXPECT_NE(SeedSequence(5).generate_state_u64(4), SeedSequence(6).generate_state_u64(4));
}

TEST(RandomState, RandnMatchesLegacyGauss)
{
  // np.random.RandomState(42).randn(6)
  const double expected[] = {
    0x1.fca2a28a9307cp-2, -0x1.1b2a505de052ap-3, 0x1.4b9dd50245e68p-1,
    0x1.85e548e01aa2bp+0, -0x1.df8bcdf57a640p-3, -0x1.df8332670db3bp-3};
  RandomState r(42);
  for (double e : expected) {
    EXPECT_EQ(r.gauss(), e);
  }
}

TEST(RandomState, ChoiceWithReplacementMatchesNumpy)
{
  // np.random.RandomState(42).choice(600, 8, replace=True)
  RandomState r(42);
  std::vector<int64_t> out(8);
  r.randint(0, 600, out.data(), out.size());
  EXPECT_EQ(out, (std::vector<int64_t>{102, 435, 270, 106, 71, 20, 121, 466}));
}

TEST(RandomState, ChoiceWithoutReplacementAndPermutation)
{
  // np.random.RandomState(0).choice(600, 2, replace=False)
  EXPECT_EQ(RandomState(0).choice_without_replacement(600, 2), (std::vector<int64_t>{434, 122}));
  // np.random.RandomState(0).permutation(10)
  EXPECT_EQ(
    RandomState(0).permutation(10), (std::vector<int64_t>{2, 8, 4, 9, 1, 6, 7, 3, 0, 5}));
  EXPECT_THROW(RandomState(0).choice_without_replacement(3, 4), std::invalid_argument);
}

TEST(RandomState, RandomSampleMatchesNumpy)
{
  // np.random.RandomState(7).random_sample(3)
  const double expected[] = {0x1.388f0a7465c98p-4, 0x1.8f5184146a0a7p-1, 0x1.c0ee597d36686p-2};
  RandomState r(7);
  for (double e : expected) {
    EXPECT_EQ(r.random_sample(), e);
  }
}
