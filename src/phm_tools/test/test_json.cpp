// Copyright 2026 Yusuf Guenena. MIT License.
// The JSON writer matches Python's json.dumps output (expected strings were
// produced by Python 3.10), and the parser round-trips it.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "phm_tools/json.hpp"

using phm_tools::json::dumps;
using phm_tools::json::parse;
using phm_tools::json::Value;

TEST(Json, PrettyOutputMatchesPythonIndent2)
{
  Value v = Value::object();
  v.set("a", 1);
  v.set("b", 1.0);
  Value c = Value::array();
  c.push_back(1);
  c.push_back(2.5);
  c.push_back(nullptr);
  c.push_back(true);
  v.set("c", c);
  v.set("d", Value::object());
  v.set("e", Value::array());
  v.set("s", std::string("x\"\\\n\xc3\xa9"));
  v.set("n", std::numeric_limits<double>::quiet_NaN());
  v.set("big", 1e16);
  v.set("small", 1e-5);
  v.set("neg", -0.0);
  const std::string expected =
    "{\n  \"a\": 1,\n  \"b\": 1.0,\n  \"c\": [\n    1,\n    2.5,\n    null,\n    true\n  ],\n"
    "  \"d\": {},\n  \"e\": [],\n  \"s\": \"x\\\"\\\\\\n\\u00e9\",\n  \"n\": NaN,\n"
    "  \"big\": 1e+16,\n  \"small\": 1e-05,\n  \"neg\": -0.0\n}";
  EXPECT_EQ(dumps(v, 2), expected);
}

TEST(Json, CompactOutputMatchesPythonDefault)
{
  const Value v = parse("{\"a\": [1, {\"b\": 2}], \"c\": \"d\"}");
  EXPECT_EQ(dumps(v), "{\"a\": [1, {\"b\": 2}], \"c\": \"d\"}");
  EXPECT_EQ(dumps(parse("[[]]"), 2), "[\n  []\n]");
}

TEST(Json, ParserKeepsIntFloatDistinctionAndPythonSpecials)
{
  const Value v = parse(
    "{\"t\": 1727712345.1234567, \"state\": 3, \"spread\": null, \"x\": NaN, "
    "\"ok\": true, \"u\": \"\\u00e9\"}");
  EXPECT_EQ(v.at("state").type(), Value::Type::kInt);
  EXPECT_EQ(v.at("t").type(), Value::Type::kDouble);
  EXPECT_DOUBLE_EQ(v.at("t").as_double(), 1727712345.1234567);
  EXPECT_TRUE(v.at("spread").is_null());
  EXPECT_TRUE(std::isnan(v.at("x").as_double()));
  EXPECT_TRUE(v.at("ok").as_bool());
  EXPECT_EQ(v.at("u").as_string(), "\xc3\xa9");
  EXPECT_EQ(dumps(v), "{\"t\": 1727712345.1234567, \"state\": 3, \"spread\": null, "
    "\"x\": NaN, \"ok\": true, \"u\": \"\\u00e9\"}");
  EXPECT_THROW(parse("{\"a\": }"), std::runtime_error);
}
