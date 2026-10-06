// Copyright 2026 Yusuf Guenena. MIT License.
// fixed() and repr() reproduce Python's f"{x:.Nf}" and repr(float(x)).
#include <gtest/gtest.h>

#include <limits>
#include <string>

#include "phm_core/format.hpp"

using phm_core::fmt::fixed;
using phm_core::fmt::repr;

TEST(Format, FixedMatchesPythonFormatSpec)
{
  EXPECT_EQ(fixed(0.012345, 4), "0.0123");
  EXPECT_EQ(fixed(2.675, 2), "2.67");  // binary 2.67499999... rounds down, as in Python
  EXPECT_EQ(fixed(0.125, 2), "0.12");  // exact half rounds to even
  EXPECT_EQ(fixed(-1.5, 0), "-2");
  EXPECT_EQ(fixed(std::numeric_limits<double>::quiet_NaN(), 4), "nan");
  EXPECT_EQ(fixed(-std::numeric_limits<double>::quiet_NaN(), 4), "nan");
  EXPECT_EQ(fixed(std::numeric_limits<double>::infinity(), 2), "inf");
  EXPECT_EQ(fixed(-std::numeric_limits<double>::infinity(), 2), "-inf");
}

TEST(Format, ReprMatchesPythonFloatRepr)
{
  EXPECT_EQ(repr(20.0), "20.0");
  EXPECT_EQ(repr(5.0), "5.0");
  EXPECT_EQ(repr(0.1), "0.1");
  EXPECT_EQ(repr(-2.5), "-2.5");
  EXPECT_EQ(repr(1e16), "1e+16");
  EXPECT_EQ(repr(1234567890123456.0), "1234567890123456.0");
  EXPECT_EQ(repr(0.0001), "0.0001");
  EXPECT_EQ(repr(0.00001), "1e-05");
  EXPECT_EQ(repr(1.5e-7), "1.5e-07");
  EXPECT_EQ(repr(56.86232380290468), "56.86232380290468");
  EXPECT_EQ(repr(0.0), "0.0");
  EXPECT_EQ(repr(-0.0), "-0.0");
  EXPECT_EQ(repr(123456789012345678.0), "1.2345678901234568e+17");
  EXPECT_EQ(repr(std::numeric_limits<double>::quiet_NaN()), "nan");
}
