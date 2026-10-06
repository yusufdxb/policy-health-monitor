// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/format.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <system_error>

namespace phm_core
{
namespace fmt
{
namespace
{

bool append_non_finite(std::string & out, double value)
{
  if (std::isnan(value)) {
    out += "nan";
    return true;
  }
  if (std::isinf(value)) {
    out += value > 0 ? "inf" : "-inf";
    return true;
  }
  return false;
}

}  // namespace

void append_fixed(std::string & out, double value, int precision)
{
  if (append_non_finite(out, value)) {
    return;
  }
  // glibc printf rounds the exact binary value half-to-even, as Python does.
  char buf[64];
  int n = std::snprintf(buf, sizeof(buf), "%.*f", precision, value);
  if (n < 0) {
    return;
  }
  if (static_cast<std::size_t>(n) < sizeof(buf)) {
    out.append(buf, static_cast<std::size_t>(n));
    return;
  }
  std::string big(static_cast<std::size_t>(n) + 1, '\0');
  std::snprintf(&big[0], big.size(), "%.*f", precision, value);
  big.resize(static_cast<std::size_t>(n));
  out += big;
}

std::string fixed(double value, int precision)
{
  std::string out;
  append_fixed(out, value, precision);
  return out;
}

void append_repr(std::string & out, double value)
{
  if (append_non_finite(out, value)) {
    return;
  }
  // Shortest round-trip digits in scientific form: [-]d[.ddd]e[+-]XX.
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
  if (res.ec != std::errc()) {
    return;
  }
  const std::string sci(buf, res.ptr);
  std::size_t pos = 0;
  std::string sign;
  if (sci[0] == '-') {
    sign = "-";
    pos = 1;
  }
  const std::size_t e_pos = sci.find('e');
  std::string digits;
  for (std::size_t i = pos; i < e_pos; ++i) {
    if (sci[i] != '.') {
      digits += sci[i];
    }
  }
  const int exp10 = std::atoi(sci.c_str() + e_pos + 1);
  // Python's float_repr_style 'short': decpt is the position of the decimal
  // point relative to the digit string (value = 0.digits * 10**decpt).
  const int decpt = exp10 + 1;
  out += sign;
  if (decpt <= -4 || decpt > 16) {
    out += digits[0];
    if (digits.size() > 1) {
      out += '.';
      out.append(digits, 1, std::string::npos);
    }
    char ebuf[16];
    std::snprintf(ebuf, sizeof(ebuf), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    out += ebuf;
    return;
  }
  if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-decpt), '0');
    out += digits;
    return;
  }
  const std::size_t point = static_cast<std::size_t>(decpt);
  if (digits.size() <= point) {
    out += digits;
    out.append(point - digits.size(), '0');
    out += ".0";
    return;
  }
  out.append(digits, 0, point);
  out += '.';
  out.append(digits, point, std::string::npos);
}

std::string repr(double value)
{
  std::string out;
  append_repr(out, value);
  return out;
}

}  // namespace fmt
}  // namespace phm_core
