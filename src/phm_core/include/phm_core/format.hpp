// Copyright 2026 Yusuf Guenena. MIT License.
// Number formatting that matches the strings the original Python
// implementation produced, so verdict reasons and report files keep their
// exact text:
//
//   fixed(x, p)  == f"{x:.{p}f}"  (nan / inf / -inf spelled as Python does)
//   repr(x)      == repr(float(x)) (shortest round-trip, "20.0", "1e-05")
#ifndef PHM_CORE__FORMAT_HPP_
#define PHM_CORE__FORMAT_HPP_

#include <string>

namespace phm_core
{
namespace fmt
{

std::string fixed(double value, int precision);
std::string repr(double value);
// Appending variants that reuse the destination's capacity.
void append_fixed(std::string & out, double value, int precision);
void append_repr(std::string & out, double value);

}  // namespace fmt
}  // namespace phm_core

#endif  // PHM_CORE__FORMAT_HPP_
