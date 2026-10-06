// Copyright 2026 Yusuf Guenena. MIT License.
// Tests for the onboard session tools (session.cpp): file loading with the .gz
// fallback, fault-time parsing, latency and settle-window semantics, the
// aggregate median, Counter and round() behaviour, and the full summary JSON of
// a small synthetic session. The expected JSON was produced by the original
// Python implementation on the same fixture.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "phm_tools/json.hpp"
#include "session.hpp"

namespace fs = std::filesystem;
namespace json = phm_tools::json;
using phm_go2::FaultSummary;
using phm_go2::FirstHit;
using phm_go2::Row;
using phm_go2::Topic;

namespace
{

class TempDir
{
public:
  TempDir()
  {
    std::string pattern = (fs::temp_directory_path() / "phm_go2_test_XXXXXX").string();
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = pattern;
  }
  ~TempDir()
  {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir &) = delete;
  TempDir & operator=(const TempDir &) = delete;
  const fs::path & path() const {return path_;}

private:
  fs::path path_;
};

uint32_t crc32(const std::string & s)
{
  uint32_t crc = 0xffffffffu;
  for (unsigned char c : s) {
    crc ^= c;
    for (int k = 0; k < 8; ++k) {
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

void put_le(std::string & out, uint32_t v, int bytes)
{
  for (int i = 0; i < bytes; ++i) {
    out += static_cast<char>((v >> (8 * i)) & 0xffu);
  }
}

// A valid gzip stream made of stored (uncompressed) deflate blocks.
std::string gzip_stored(const std::string & text)
{
  std::string out("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff", 10);
  std::size_t pos = 0;
  do {
    const std::size_t len = std::min<std::size_t>(text.size() - pos, 65535);
    const bool last = pos + len == text.size();
    out += static_cast<char>(last ? 1 : 0);
    put_le(out, static_cast<uint32_t>(len), 2);
    put_le(out, static_cast<uint32_t>(~len & 0xffffu), 2);
    out += text.substr(pos, len);
    pos += len;
  } while (pos < text.size());
  put_le(out, crc32(text), 4);
  put_le(out, static_cast<uint32_t>(text.size()), 4);
  return out;
}

void write_file(const fs::path & p, const std::string & content)
{
  std::ofstream f(p, std::ios::binary);
  f << content;
}

void put_text(const fs::path & dir, const std::string & name, bool gz, const std::string & text)
{
  if (gz) {
    write_file(dir / (name + ".gz"), gzip_stored(text));
  } else {
    write_file(dir / name, text);
  }
}

void put(
  const fs::path & dir, const std::string & name, bool gz, const std::vector<std::string> & rows)
{
  std::string text;
  for (const auto & r : rows) {
    text += r + "\n";
  }
  put_text(dir, name, gz, text);
}

std::string H(
  double t, double stamp, int state, const std::string & source, const std::string & reason)
{
  json::Value v = json::Value::object();
  v.set("t", t);
  v.set("topic", "/phm/health");
  v.set("stamp", stamp);
  v.set("state", state);
  v.set("source", source);
  v.set("reason", reason);
  return json::dumps(v);
}

std::string E(double t, double stamp, std::optional<double> spread)
{
  json::Value v = json::Value::object();
  v.set("t", t);
  v.set("topic", "/policy/embedding");
  v.set("stamp", stamp);
  v.set("dim", 384);
  v.set("spread", spread ? json::Value(*spread) : json::Value(nullptr));
  return json::dumps(v);
}

std::string V(
  double t, double stamp, const std::string & source, bool violating, const char * reason)
{
  json::Value v = json::Value::object();
  v.set("t", t);
  v.set("topic", "/phm/verdicts");
  v.set("stamp", stamp);
  v.set("source", source);
  v.set("violating", violating);
  if (reason != nullptr) {
    v.set("reason", reason);
  }
  return json::dumps(v);
}

// Writes the synthetic session. With `mixed`, some files are stored as .gz only.
void write_fixture(const fs::path & dir, bool mixed)
{
  put(dir, "nominal.jsonl", mixed, {
      E(1000.01, 1000.0, std::nullopt),
      V(1000.011, 1000.0, "phm_ood_cpp", false, "nominal"),
      E(1000.51, 1000.5, std::nullopt),
      V(1000.511, 1000.5, "phm_ood_cpp", false, "nominal"),
      E(1001.01, 1001.0, std::nullopt),
      V(1001.011, 1001.0, "phm_ood_cpp", false, "nominal"),
      E(1001.51, 1001.5, 0.02),
      V(1001.511, 1001.5, "phm_ood_cpp", false, "nominal"),
      E(1002.01, 1002.0, 0.03),
      V(1002.011, 1002.0, "phm_ood_cpp", false, "nominal"),
      E(1002.51, 1002.5, 0.01),
      V(1002.511, 1002.5, "phm_ood_cpp", false, "nominal"),
      E(1003.01, 1003.0, 0.05),
      V(1003.011, 1003.0, "phm_ood_cpp", false, "nominal"),
      E(1003.51, 1003.5, 0.04),
      V(1003.511, 1003.5, "phm_ood_cpp", true, "nominal"),
      E(1004.01, 1004.0, 0.06),
      V(1004.011, 1004.0, "phm_ood_cpp", false, "nominal"),
      E(1004.51, 1004.5, 0.07),
      V(1004.511, 1004.5, "phm_ood_cpp", false, "nominal"),
      V(1001.0, 1001.0, "other", true, "not ood"),
      H(1000.02, 1000.019, 0, "", "all nominal"),
      H(1000.52, 1000.519, 0, "", "all nominal"),
      H(1001.02, 1001.019, 0, "", "all nominal"),
      H(1001.52, 1001.519, 1, "zz", "detector"),
      H(1002.02, 1002.019, 1, "aa", "detector"),
      H(1002.52, 1002.519, 2, "aa", "detector"),
      H(1003.02, 1003.019, 0, "", "all nominal"),
      H(1003.52, 1003.519, 1, "aa", "detector"),
      H(1004.02, 1004.019, 2, "zz", "detector"),
      H(1004.52, 1004.519, 0, "", "all nominal"),
    });
  put(dir, "fault_freeze_1.jsonl", false, {
      E(1992.02, 1992.0, std::nullopt),
      E(1993.02, 1993.0, 0.011),
      E(1994.02, 1994.0, 0.012),
      E(1995.02, 1995.0, 0.013),
      E(1996.02, 1996.0, 0.014),
      E(1997.02, 1997.0, std::nullopt),
      E(1998.02, 1998.0, 0.016),
      E(1999.02, 1999.0, 0.017),
      E(2000.02, 2000.0, 0.018),
      E(2001.02, 2001.0, 0.019),
      E(2002.02, 2002.0, std::nullopt),
      E(2003.02, 2003.0, 0.021),
      H(1992.03, 1992.0, 3, "old", "stale stop"),
      H(1996.03, 1996.0, 1, "early", "settling"),
      H(1997.03, 1997.0, 1, "x", "edge of window"),
      H(1999.03, 1999.0, 0, "", "ok"),
      V(1999.02, 1999.0, "phm_ood_cpp", true, "early violating"),
      V(2000.01, 2000.01, "phm_ood_cpp", false, "not violating"),
      V(2000.02, 2000.02, "other_src", true, "wrong source"),
      V(2000.0625, 1999.5, "phm_ood_cpp", true, "spread collapsed"),
      V(2000.9, 2000.9, "phm_ood_cpp_extra", true, nullptr),
      H(2000.7123, 2000.7113, 3, "freq:/policy/embedding", "no embedding"),
      H(2000.5, 2000.499, 2, "phm_ood_cpp", "intervene"),
      H(2000.1875, 2000.1865, 1, "phm_ood_cpp", "degraded"),
    });
  put(dir, "fault_freeze_2.jsonl", mixed, {
      E(2092.02, 2092.0, std::nullopt),
      E(2093.02, 2093.0, 0.011),
      E(2094.02, 2094.0, 0.012),
      E(2095.02, 2095.0, 0.013),
      E(2096.02, 2096.0, 0.014),
      E(2097.02, 2097.0, std::nullopt),
      E(2098.02, 2098.0, 0.016),
      E(2099.02, 2099.0, 0.017),
      E(2100.02, 2100.0, 0.018),
      E(2101.02, 2101.0, 0.019),
      E(2102.02, 2102.0, std::nullopt),
      E(2103.02, 2103.0, 0.021),
      H(2092.03, 2092.0, 3, "old", "stale stop"),
      H(2096.03, 2096.0, 1, "early", "settling"),
      H(2097.03, 2097.0, 1, "x", "edge of window"),
      H(2099.03, 2099.0, 0, "", "ok"),
      V(2099.02, 2099.0, "phm_ood_cpp", true, "early violating"),
      V(2100.01, 2100.01, "phm_ood_cpp", false, "not violating"),
      V(2100.02, 2100.02, "other_src", true, "wrong source"),
      V(2100.4123, 2099.5, "phm_ood_cpp", true, "spread collapsed"),
      V(2100.9, 2100.9, "phm_ood_cpp_extra", true, nullptr),
      H(2100.9, 2100.899, 3, "dead:/policy/embedding", "no embedding"),
      H(2100.6, 2100.5989999999997, 2, "phm_ood_cpp", "intervene"),
      H(2100.3, 2100.299, 1, "phm_ood_cpp", "degraded"),
    });
  put(dir, "fault_freeze_10.jsonl", false, {
      E(2192.02, 2192.0, std::nullopt),
      E(2193.02, 2193.0, 0.011),
      E(2194.02, 2194.0, 0.012),
      E(2195.02, 2195.0, 0.013),
      E(2196.02, 2196.0, 0.014),
      E(2197.02, 2197.0, std::nullopt),
      E(2198.02, 2198.0, 0.016),
      E(2199.02, 2199.0, 0.017),
      E(2200.02, 2200.0, 0.018),
      E(2201.02, 2201.0, 0.019),
      E(2202.02, 2202.0, std::nullopt),
      E(2203.02, 2203.0, 0.021),
      H(2192.03, 2192.0, 3, "old", "stale stop"),
      H(2196.03, 2196.0, 1, "early", "settling"),
      H(2197.03, 2197.0, 1, "x", "edge of window"),
      H(2199.03, 2199.0, 0, "", "ok"),
      V(2199.02, 2199.0, "phm_ood_cpp", true, "early violating"),
      V(2200.01, 2200.01, "phm_ood_cpp", false, "not violating"),
      V(2200.02, 2200.02, "other_src", true, "wrong source"),
      V(2200.25, 2199.5, "phm_ood_cpp", true, "spread collapsed"),
      V(2200.9, 2200.9, "phm_ood_cpp_extra", true, nullptr),
      H(2200.8, 2200.799, 3, "dead:/policy/embedding", "no embedding"),
      H(2200.35, 2200.3489999999997, 2, "phm_ood_cpp", "intervene"),
      H(2200.0005, 2199.9995, 1, "phm_ood_cpp", "degraded"),
    });
  put(dir, "fault_stop_3.jsonl", false, {
      E(2992.02, 2992.0, std::nullopt),
      E(2993.02, 2993.0, 0.011),
      E(2994.02, 2994.0, 0.012),
      E(2995.02, 2995.0, 0.013),
      E(2996.02, 2996.0, 0.014),
      E(2997.02, 2997.0, std::nullopt),
      E(2998.02, 2998.0, 0.016),
      E(2999.02, 2999.0, 0.017),
      E(3000.02, 3000.0, 0.018),
      E(3001.02, 3001.0, 0.019),
      E(3002.02, 3002.0, std::nullopt),
      E(3003.02, 3003.0, 0.021),
      H(2992.03, 2992.0, 3, "old", "stale stop"),
      H(2996.03, 2996.0, 1, "early", "settling"),
      H(2997.03, 2997.0, 1, "x", "edge of window"),
      H(2999.03, 2999.0, 0, "", "ok"),
      V(2999.02, 2999.0, "phm_ood_cpp", true, "early violating"),
      V(3000.01, 3000.01, "dead:/policy/embedding", false, "not violating"),
      V(3000.02, 3000.02, "other_src", true, "wrong source"),
      V(3001.5, 2999.5, "dead:/policy/embedding", true, "spread collapsed"),
      V(3000.9, 3000.9, "dead:/policy/embedding2", true, nullptr),
      H(3002.0, 3001.999, 3, "freq:/policy/embedding", "no embedding"),
      H(3001.2, 3001.1989999999996, 2, "phm_ood_cpp", "intervene"),
      H(3000.98, 3000.979, 1, "phm_ood_cpp", "degraded"),
    });
  put(dir, "fault_stop_4.jsonl", false, {
      E(3092.02, 3092.0, std::nullopt),
      E(3093.02, 3093.0, 0.011),
      E(3094.02, 3094.0, 0.012),
      E(3095.02, 3095.0, 0.013),
      E(3096.02, 3096.0, 0.014),
      E(3097.02, 3097.0, std::nullopt),
      E(3098.02, 3098.0, 0.016),
      E(3099.02, 3099.0, 0.017),
      E(3100.02, 3100.0, 0.018),
      E(3101.02, 3101.0, 0.019),
      E(3102.02, 3102.0, std::nullopt),
      E(3103.02, 3103.0, 0.021),
      H(3092.03, 3092.0, 3, "old", "stale stop"),
      H(3096.03, 3096.0, 1, "early", "settling"),
      H(3097.03, 3097.0, 1, "x", "edge of window"),
      H(3099.03, 3099.0, 0, "", "ok"),
      V(3099.02, 3099.0, "phm_ood_cpp", true, "early violating"),
      V(3100.01, 3100.01, "freq:/policy/embedding", false, "not violating"),
      V(3100.02, 3100.02, "other_src", true, "wrong source"),
      V(3101.75, 3099.5, "freq:/policy/embedding", true, "spread collapsed"),
      V(3100.9, 3100.9, "phm_ood_cpp", true, nullptr),
      H(3101.25, 3101.249, 2, "phm_ood_cpp", "intervene"),
      H(3101.0, 3100.999, 1, "phm_ood_cpp", "degraded"),
    });
  put_text(dir, "session.txt", false,
      "start test\n"
      "threshold 0.00544\n"
      "unicode caf\xc3\xa9\n");
  put_text(dir, "calibrate.json", false,
      "{\"old\": 1}\n"
      "{\"threshold\": 0.00544, \"n_frames\": 3001, \"window\": 30, \"rate_hz\": 50.02}\n");
  put_text(dir, "cpu_nominal.csv", false,
      "# name, cpu\n"
      "detectors, 153, 300.7, 50.9\n"
      "arbiter, 16, 300.7, 5.3\n");
  put_text(dir, "embedder_freeze_1.log", false,
      "[INFO] [1979.5] [phoenix_shadow_embedder]: started\n"
      "[INFO] [1.5] [other_node]: FAULT INJECTED\n"
      "[WARN] [2000.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: freeze_obs\n"
      "[WARN] [2005.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: again\n");
  put_text(dir, "embedder_freeze_2.log", false,
      "[INFO] [2079.5] [phoenix_shadow_embedder]: started\n"
      "[INFO] [1.5] [other_node]: FAULT INJECTED\n"
      "[WARN] [2100.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: freeze_obs\n"
      "[WARN] [2105.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: again\n");
  put_text(dir, "embedder_freeze_10.log", mixed,
      "[INFO] [2179.5] [phoenix_shadow_embedder]: started\n"
      "[INFO] [1.5] [other_node]: FAULT INJECTED\n"
      "[WARN] [2200.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: freeze_obs\n"
      "[WARN] [2205.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: again\n");
  put_text(dir, "embedder_stop_1.log", false,
      "[INFO] [10.0] [phoenix_shadow_embedder]: started\n");
  put_text(dir, "embedder_stop_3.log", false,
      "[INFO] [2979.5] [phoenix_shadow_embedder]: started\n"
      "[INFO] [1.5] [other_node]: FAULT INJECTED\n"
      "[WARN] [3000.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: stop\n"
      "[WARN] [3005.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: again\n");
  put_text(dir, "embedder_stop_4.log", false,
      "[INFO] [3079.5] [phoenix_shadow_embedder]: started\n"
      "[INFO] [1.5] [other_node]: FAULT INJECTED\n"
      "[WARN] [3100.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: stop\n"
      "[WARN] [3105.000000000] [phoenix_shadow_embedder]: FAULT INJECTED: again\n");
}

const char kExpectedSummary[] =
    "{\n"
    "  \"session\": \"session_fixture\",\n"
    "  \"session_txt\": [\n"
    "    \"start test\",\n"
    "    \"threshold 0.00544\",\n"
    "    \"unicode caf\\u00e9\"\n"
    "  ],\n"
    "  \"calibration\": {\n"
    "    \"threshold\": 0.00544,\n"
    "    \"n_frames\": 3001,\n"
    "    \"window\": 30,\n"
    "    \"rate_hz\": 50.02\n"
    "  },\n"
    "  \"cpu_nominal\": [\n"
    "    \"detectors, 153, 300.7, 50.9\",\n"
    "    \"arbiter, 16, 300.7, 5.3\"\n"
    "  ],\n"
    "  \"nominal\": {\n"
    "    \"health_msgs\": 10,\n"
    "    \"state_counts\": {\n"
    "      \"OK\": 5,\n"
    "      \"DEGRADED\": 3,\n"
    "      \"INTERVENE\": 2\n"
    "    },\n"
    "    \"non_ok_fraction\": 0.5,\n"
    "    \"non_ok_by_source\": {\n"
    "      \"DEGRADED aa\": 2,\n"
    "      \"DEGRADED zz\": 1,\n"
    "      \"INTERVENE aa\": 1,\n"
    "      \"INTERVENE zz\": 1\n"
    "    },\n"
    "    \"ood_verdicts\": 10,\n"
    "    \"ood_violating\": 1,\n"
    "    \"ood_violating_fraction\": 0.1,\n"
    "    \"embeddings\": 10,\n"
    "    \"embedding_rate_hz\": 2.0,\n"
    "    \"spread_min\": 0.01,\n"
    "    \"spread_median\": 0.04\n"
    "  },\n"
    "  \"fault_freeze\": {\n"
    "    \"aggregate\": {\n"
    "      \"n_trials\": 3,\n"
    "      \"first_violating_verdict\": {\n"
    "        \"detected\": \"3/3\",\n"
    "        \"min_s\": 0.062,\n"
    "        \"median_s\": 0.25,\n"
    "        \"max_s\": 0.412\n"
    "      },\n"
    "      \"first_health_degraded_or_worse\": {\n"
    "        \"detected\": \"3/3\",\n"
    "        \"min_s\": 0.001,\n"
    "        \"median_s\": 0.188,\n"
    "        \"max_s\": 0.3\n"
    "      },\n"
    "      \"first_health_intervene_or_worse\": {\n"
    "        \"detected\": \"3/3\",\n"
    "        \"min_s\": 0.35,\n"
    "        \"median_s\": 0.5,\n"
    "        \"max_s\": 0.6\n"
    "      },\n"
    "      \"first_health_stop\": {\n"
    "        \"detected\": \"3/3\",\n"
    "        \"min_s\": 0.712,\n"
    "        \"median_s\": 0.8,\n"
    "        \"max_s\": 0.9\n"
    "      },\n"
    "      \"pre_fault_health_msgs\": 7,\n"
    "      \"pre_fault_non_ok_msgs\": 4\n"
    "    },\n"
    "    \"trials\": [\n"
    "      {\n"
    "        \"trial\": 1,\n"
    "        \"fault_time\": 2000.0,\n"
    "        \"before_fault\": {\n"
    "          \"health_msgs\": 2,\n"
    "          \"state_counts\": {\n"
    "            \"DEGRADED\": 1,\n"
    "            \"OK\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 0.5,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"DEGRADED x\": 1\n"
    "          },\n"
    "          \"ood_verdicts\": 2,\n"
    "          \"ood_violating\": 2,\n"
    "          \"ood_violating_fraction\": 1.0,\n"
    "          \"embeddings\": 3,\n"
    "          \"embedding_rate_hz\": 1.0,\n"
    "          \"spread_min\": 0.016,\n"
    "          \"spread_median\": 0.017\n"
    "        },\n"
    "        \"first_violating_verdict\": {\n"
    "          \"latency_s\": 0.062,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"spread collapsed\",\n"
    "          \"state\": null\n"
    "        },\n"
    "        \"first_health_degraded_or_worse\": {\n"
    "          \"latency_s\": 0.188,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"degraded\",\n"
    "          \"state\": \"DEGRADED\"\n"
    "        },\n"
    "        \"first_health_intervene_or_worse\": {\n"
    "          \"latency_s\": 0.5,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"intervene\",\n"
    "          \"state\": \"INTERVENE\"\n"
    "        },\n"
    "        \"first_health_stop\": {\n"
    "          \"latency_s\": 0.712,\n"
    "          \"source\": \"freq:/policy/embedding\",\n"
    "          \"reason\": \"no embedding\",\n"
    "          \"state\": \"STOP\"\n"
    "        },\n"
    "        \"after_fault\": {\n"
    "          \"health_msgs\": 3,\n"
    "          \"state_counts\": {\n"
    "            \"STOP\": 1,\n"
    "            \"INTERVENE\": 1,\n"
    "            \"DEGRADED\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 1.0,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"STOP freq:/policy/embedding\": 1,\n"
    "            \"INTERVENE phm_ood_cpp\": 1,\n"
    "            \"DEGRADED phm_ood_cpp\": 1\n"
    "          }\n"
    "        }\n"
    "      },\n"
    "      {\n"
    "        \"trial\": 10,\n"
    "        \"fault_time\": 2200.0,\n"
    "        \"before_fault\": {\n"
    "          \"health_msgs\": 3,\n"
    "          \"state_counts\": {\n"
    "            \"DEGRADED\": 2,\n"
    "            \"OK\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 0.6667,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"DEGRADED x\": 1,\n"
    "            \"DEGRADED phm_ood_cpp\": 1\n"
    "          },\n"
    "          \"ood_verdicts\": 2,\n"
    "          \"ood_violating\": 2,\n"
    "          \"ood_violating_fraction\": 1.0,\n"
    "          \"embeddings\": 3,\n"
    "          \"embedding_rate_hz\": 1.0,\n"
    "          \"spread_min\": 0.016,\n"
    "          \"spread_median\": 0.017\n"
    "        },\n"
    "        \"first_violating_verdict\": {\n"
    "          \"latency_s\": 0.25,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"spread collapsed\",\n"
    "          \"state\": null\n"
    "        },\n"
    "        \"first_health_degraded_or_worse\": {\n"
    "          \"latency_s\": 0.001,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"degraded\",\n"
    "          \"state\": \"DEGRADED\"\n"
    "        },\n"
    "        \"first_health_intervene_or_worse\": {\n"
    "          \"latency_s\": 0.35,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"intervene\",\n"
    "          \"state\": \"INTERVENE\"\n"
    "        },\n"
    "        \"first_health_stop\": {\n"
    "          \"latency_s\": 0.8,\n"
    "          \"source\": \"dead:/policy/embedding\",\n"
    "          \"reason\": \"no embedding\",\n"
    "          \"state\": \"STOP\"\n"
    "        },\n"
    "        \"after_fault\": {\n"
    "          \"health_msgs\": 2,\n"
    "          \"state_counts\": {\n"
    "            \"STOP\": 1,\n"
    "            \"INTERVENE\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 1.0,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"STOP dead:/policy/embedding\": 1,\n"
    "            \"INTERVENE phm_ood_cpp\": 1\n"
    "          }\n"
    "        }\n"
    "      },\n"
    "      {\n"
    "        \"trial\": 2,\n"
    "        \"fault_time\": 2100.0,\n"
    "        \"before_fault\": {\n"
    "          \"health_msgs\": 2,\n"
    "          \"state_counts\": {\n"
    "            \"DEGRADED\": 1,\n"
    "            \"OK\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 0.5,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"DEGRADED x\": 1\n"
    "          },\n"
    "          \"ood_verdicts\": 2,\n"
    "          \"ood_violating\": 2,\n"
    "          \"ood_violating_fraction\": 1.0,\n"
    "          \"embeddings\": 3,\n"
    "          \"embedding_rate_hz\": 1.0,\n"
    "          \"spread_min\": 0.016,\n"
    "          \"spread_median\": 0.017\n"
    "        },\n"
    "        \"first_violating_verdict\": {\n"
    "          \"latency_s\": 0.412,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"spread collapsed\",\n"
    "          \"state\": null\n"
    "        },\n"
    "        \"first_health_degraded_or_worse\": {\n"
    "          \"latency_s\": 0.3,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"degraded\",\n"
    "          \"state\": \"DEGRADED\"\n"
    "        },\n"
    "        \"first_health_intervene_or_worse\": {\n"
    "          \"latency_s\": 0.6,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"intervene\",\n"
    "          \"state\": \"INTERVENE\"\n"
    "        },\n"
    "        \"first_health_stop\": {\n"
    "          \"latency_s\": 0.9,\n"
    "          \"source\": \"dead:/policy/embedding\",\n"
    "          \"reason\": \"no embedding\",\n"
    "          \"state\": \"STOP\"\n"
    "        },\n"
    "        \"after_fault\": {\n"
    "          \"health_msgs\": 3,\n"
    "          \"state_counts\": {\n"
    "            \"STOP\": 1,\n"
    "            \"INTERVENE\": 1,\n"
    "            \"DEGRADED\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 1.0,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"STOP dead:/policy/embedding\": 1,\n"
    "            \"INTERVENE phm_ood_cpp\": 1,\n"
    "            \"DEGRADED phm_ood_cpp\": 1\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    ]\n"
    "  },\n"
    "  \"fault_stop\": {\n"
    "    \"aggregate\": {\n"
    "      \"n_trials\": 3,\n"
    "      \"first_violating_verdict\": {\n"
    "        \"detected\": \"2/3\",\n"
    "        \"min_s\": 0.9,\n"
    "        \"median_s\": 0.9,\n"
    "        \"max_s\": 0.9\n"
    "      },\n"
    "      \"first_health_degraded_or_worse\": {\n"
    "        \"detected\": \"2/3\",\n"
    "        \"min_s\": 0.98,\n"
    "        \"median_s\": 1.0,\n"
    "        \"max_s\": 1.0\n"
    "      },\n"
    "      \"first_health_intervene_or_worse\": {\n"
    "        \"detected\": \"2/3\",\n"
    "        \"min_s\": 1.2,\n"
    "        \"median_s\": 1.25,\n"
    "        \"max_s\": 1.25\n"
    "      },\n"
    "      \"first_health_stop\": {\n"
    "        \"detected\": \"1/3\",\n"
    "        \"min_s\": 2.0,\n"
    "        \"median_s\": 2.0,\n"
    "        \"max_s\": 2.0\n"
    "      },\n"
    "      \"pre_fault_health_msgs\": 4,\n"
    "      \"pre_fault_non_ok_msgs\": 2\n"
    "    },\n"
    "    \"trials\": [\n"
    "      {\n"
    "        \"trial\": 1,\n"
    "        \"fault_time\": null,\n"
    "        \"error\": \"no FAULT INJECTED line in the embedder log\"\n"
    "      },\n"
    "      {\n"
    "        \"trial\": 3,\n"
    "        \"fault_time\": 3000.0,\n"
    "        \"before_fault\": {\n"
    "          \"health_msgs\": 2,\n"
    "          \"state_counts\": {\n"
    "            \"DEGRADED\": 1,\n"
    "            \"OK\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 0.5,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"DEGRADED x\": 1\n"
    "          },\n"
    "          \"ood_verdicts\": 1,\n"
    "          \"ood_violating\": 1,\n"
    "          \"ood_violating_fraction\": 1.0,\n"
    "          \"embeddings\": 3,\n"
    "          \"embedding_rate_hz\": 1.0,\n"
    "          \"spread_min\": 0.016,\n"
    "          \"spread_median\": 0.017\n"
    "        },\n"
    "        \"first_violating_verdict\": {\n"
    "          \"latency_s\": 0.9,\n"
    "          \"source\": \"dead:/policy/embedding2\",\n"
    "          \"reason\": null,\n"
    "          \"state\": null\n"
    "        },\n"
    "        \"first_health_degraded_or_worse\": {\n"
    "          \"latency_s\": 0.98,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"degraded\",\n"
    "          \"state\": \"DEGRADED\"\n"
    "        },\n"
    "        \"first_health_intervene_or_worse\": {\n"
    "          \"latency_s\": 1.2,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"intervene\",\n"
    "          \"state\": \"INTERVENE\"\n"
    "        },\n"
    "        \"first_health_stop\": {\n"
    "          \"latency_s\": 2.0,\n"
    "          \"source\": \"freq:/policy/embedding\",\n"
    "          \"reason\": \"no embedding\",\n"
    "          \"state\": \"STOP\"\n"
    "        },\n"
    "        \"after_fault\": {\n"
    "          \"health_msgs\": 3,\n"
    "          \"state_counts\": {\n"
    "            \"STOP\": 1,\n"
    "            \"INTERVENE\": 1,\n"
    "            \"DEGRADED\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 1.0,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"STOP freq:/policy/embedding\": 1,\n"
    "            \"INTERVENE phm_ood_cpp\": 1,\n"
    "            \"DEGRADED phm_ood_cpp\": 1\n"
    "          }\n"
    "        }\n"
    "      },\n"
    "      {\n"
    "        \"trial\": 4,\n"
    "        \"fault_time\": 3100.0,\n"
    "        \"before_fault\": {\n"
    "          \"health_msgs\": 2,\n"
    "          \"state_counts\": {\n"
    "            \"DEGRADED\": 1,\n"
    "            \"OK\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 0.5,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"DEGRADED x\": 1\n"
    "          },\n"
    "          \"ood_verdicts\": 1,\n"
    "          \"ood_violating\": 1,\n"
    "          \"ood_violating_fraction\": 1.0,\n"
    "          \"embeddings\": 3,\n"
    "          \"embedding_rate_hz\": 1.0,\n"
    "          \"spread_min\": 0.016,\n"
    "          \"spread_median\": 0.017\n"
    "        },\n"
    "        \"first_violating_verdict\": {\n"
    "          \"latency_s\": 0.9,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": null,\n"
    "          \"state\": null\n"
    "        },\n"
    "        \"first_health_degraded_or_worse\": {\n"
    "          \"latency_s\": 1.0,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"degraded\",\n"
    "          \"state\": \"DEGRADED\"\n"
    "        },\n"
    "        \"first_health_intervene_or_worse\": {\n"
    "          \"latency_s\": 1.25,\n"
    "          \"source\": \"phm_ood_cpp\",\n"
    "          \"reason\": \"intervene\",\n"
    "          \"state\": \"INTERVENE\"\n"
    "        },\n"
    "        \"first_health_stop\": null,\n"
    "        \"after_fault\": {\n"
    "          \"health_msgs\": 2,\n"
    "          \"state_counts\": {\n"
    "            \"INTERVENE\": 1,\n"
    "            \"DEGRADED\": 1\n"
    "          },\n"
    "          \"non_ok_fraction\": 1.0,\n"
    "          \"non_ok_by_source\": {\n"
    "            \"INTERVENE phm_ood_cpp\": 1,\n"
    "            \"DEGRADED phm_ood_cpp\": 1\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    ]\n"
    "  }\n"
    "}\n";

Row row(Topic topic, double t, double stamp)
{
  Row r;
  r.topic = topic;
  r.t = t;
  r.stamp = stamp;
  return r;
}

Row health_row(double t, double stamp, int64_t state)
{
  Row r = row(Topic::kHealth, t, stamp);
  r.has_state = true;
  r.state = state;
  r.has_source = true;
  r.source = "s";
  return r;
}

Row verdict_row(double t, double stamp, const std::string & source, bool violating)
{
  Row r = row(Topic::kVerdicts, t, stamp);
  r.has_source = true;
  r.source = source;
  r.violating = violating ? 1 : 0;
  return r;
}

FaultSummary summary_with_latency(std::optional<double> stop_latency)
{
  FaultSummary s;
  s.fault_time = 0.0;
  if (stop_latency) {
    FirstHit h;
    h.latency_s = *stop_latency;
    s.first_health_stop = h;
  }
  return s;
}

}  // namespace

TEST(SessionTools, SummaryJsonMatchesPythonReference)
{
  for (const bool mixed : {false, true}) {
    TempDir tmp;
    const fs::path dir = tmp.path() / "session_fixture";
    fs::create_directories(dir);
    write_fixture(dir, mixed);
    const std::string got = json::dumps(phm_go2::summarize_session(dir.string()), 2) + "\n";
    EXPECT_EQ(got, kExpectedSummary) << (mixed ? "mixed .gz layout" : "plain layout");
  }
}

TEST(SessionTools, TrialsFollowStringOrderOfLogNames)
{
  TempDir tmp;
  const fs::path dir = tmp.path() / "session_fixture";
  fs::create_directories(dir);
  write_fixture(dir, true);
  const auto names = phm_go2::embedder_logs(dir.string(), "freeze");
  // "10" sorts before "2"; the .gz copy of trial 10 appears once, without the suffix.
  const std::vector<std::string> expected = {
    "embedder_freeze_1.log", "embedder_freeze_10.log", "embedder_freeze_2.log"};
  EXPECT_EQ(names, expected);
  const json::Value summary = phm_go2::summarize_session(dir.string());
  const auto & trials = summary.at("fault_freeze").at("trials").as_array();
  ASSERT_EQ(trials.size(), 3u);
  EXPECT_EQ(trials[0].at("trial").as_int(), 1);
  EXPECT_EQ(trials[1].at("trial").as_int(), 10);
  EXPECT_EQ(trials[2].at("trial").as_int(), 2);
  EXPECT_EQ(trials[1].at("trial").type(), json::Value::Type::kInt);
}

TEST(SessionTools, ReadTextFallsBackToGzipCopy)
{
  TempDir tmp;
  const std::string plain = (tmp.path() / "a.log").string();
  EXPECT_FALSE(phm_go2::read_text(plain).has_value());
  write_file(tmp.path() / "a.log.gz", gzip_stored("from gzip\nsecond\n"));
  EXPECT_EQ(phm_go2::read_text(plain).value(), "from gzip\nsecond\n");
  write_file(tmp.path() / "a.log", "plain wins\n");
  EXPECT_EQ(phm_go2::read_text(plain).value(), "plain wins\n");
}

TEST(SessionTools, LoadRowsReadsGzipAndSkipsBlankLines)
{
  TempDir tmp;
  write_file(
    tmp.path() / "r.jsonl.gz",
    gzip_stored(E(1.5, 1.25, 0.5) + "\n\n   \n" + H(2.5, 2.0, 1, "src", "why") + "\r\n"));
  const std::vector<Row> rows = phm_go2::load_rows((tmp.path() / "r.jsonl").string(), true);
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[0].topic, Topic::kEmbedding);
  EXPECT_TRUE(rows[0].has_spread);
  EXPECT_EQ(rows[1].topic, Topic::kHealth);
  EXPECT_EQ(rows[1].state, 1);
  EXPECT_EQ(rows[1].reason, "why");
  EXPECT_TRUE(phm_go2::load_rows((tmp.path() / "missing.jsonl").string()).empty());
}

TEST(SessionTools, FaultTimeRegex)
{
  TempDir tmp;
  const auto log_with = [&](const std::string & text) {
      write_file(tmp.path() / "e.log", text);
      return phm_go2::fault_time((tmp.path() / "e.log").string());
    };
  // First match wins; other nodes and malformed numbers are ignored.
  EXPECT_EQ(
    log_with(
      "[1.5] [other]: FAULT INJECTED\n[abc.def] [phoenix_shadow_embedder]: FAULT INJECTED\n"
      "[WARN] [1791250913.976144080] [phoenix_shadow_embedder]: FAULT INJECTED: x\n"
      "[WARN] [7.5] [phoenix_shadow_embedder]: FAULT INJECTED\n").value(),
    1791250913.976144080);
  EXPECT_FALSE(log_with("[12] [phoenix_shadow_embedder]: FAULT INJECTED\n").has_value());
  EXPECT_FALSE(log_with("[1.5] [phoenix_shadow_embedder]: fault injected\n").has_value());
  EXPECT_EQ(log_with("[0.25] [phoenix_shadow_embedder]: FAULT INJECTED").value(), 0.25);
  EXPECT_FALSE(phm_go2::fault_time((tmp.path() / "absent.log").string()).has_value());
}

TEST(SessionTools, LatencyIsMeasuredFromReceiveTimeNotStamp)
{
  // The verdict carries the embedding's stamp, which precedes the fault; only
  // the receive time t says when the detection arrived.
  std::vector<Row> rows = {
    row(Topic::kEmbedding, 91.0, 90.0),
    verdict_row(110.4, 99.0, "phm_ood_cpp", true),
    health_row(101.25, 50.0, 3),
  };
  const FaultSummary s = phm_go2::fault_summary(rows, 100.0, phm_go2::fault_sources("freeze"));
  ASSERT_TRUE(s.first_violating_verdict.has_value());
  EXPECT_EQ(s.first_violating_verdict->latency_s, 10.4);
  ASSERT_TRUE(s.first_health_stop.has_value());
  EXPECT_EQ(s.first_health_stop->latency_s, 1.25);
  EXPECT_EQ(s.first_health_stop->state.value(), "STOP");
  EXPECT_FALSE(s.first_health_stop->reason.has_value());
}

TEST(SessionTools, RowsBeforeFaultTimeAreNotDetections)
{
  std::vector<Row> rows = {
    row(Topic::kEmbedding, 91.0, 90.0),
    verdict_row(99.5, 99.5, "phm_ood_cpp", true),
    health_row(99.9, 99.9, 3),
    verdict_row(100.0, 100.0, "phm_ood_cpp", true),
  };
  const FaultSummary s = phm_go2::fault_summary(rows, 100.0, phm_go2::fault_sources("freeze"));
  EXPECT_EQ(s.first_violating_verdict->latency_s, 0.0);
  EXPECT_FALSE(s.first_health_stop.has_value());
}

TEST(SessionTools, SettleWindowUsesStampFromFirstEmbedding)
{
  // The first embedding stamp is 50, so pre-fault rows count from stamp 55.
  std::vector<Row> rows = {
    row(Topic::kEmbedding, 500.0, 50.0),
    health_row(1.0, 54.999, 1),    // inside the settle window
    health_row(2.0, 55.0, 0),      // first counted stamp (inclusive)
    health_row(3.0, 99.999, 1),
    health_row(4.0, 100.0, 3),     // at the fault: after, not before
    health_row(200.0, 80.0, 2),    // stamp decides, not t
  };
  const FaultSummary s = phm_go2::fault_summary(rows, 100.0, phm_go2::fault_sources("freeze"));
  EXPECT_EQ(s.before_health.health_msgs, 3);
  EXPECT_EQ(s.before_health.count_of("OK"), 1);
  EXPECT_EQ(s.after_health.health_msgs, 1);
  EXPECT_EQ(s.after_health.count_of("STOP"), 1);
}

TEST(SessionTools, MissingFaultLineGivesErrorEntry)
{
  const FaultSummary s = phm_go2::fault_summary({}, std::nullopt, phm_go2::fault_sources("stop"));
  EXPECT_EQ(
    json::dumps(phm_go2::to_json(s)),
    "{\"fault_time\": null, \"error\": \"no FAULT INJECTED line in the embedder log\"}");
}

TEST(SessionTools, AggregateMedianIsElementAtHalfTheCount)
{
  // len // 2 of the sorted latencies: the upper middle for an even count.
  const auto stop_of = [](std::vector<std::optional<double>> latencies) {
      std::vector<FaultSummary> trials;
      for (const auto & l : latencies) {
        trials.push_back(summary_with_latency(l));
      }
      return phm_go2::aggregate(trials).first_health_stop;
    };
  const auto two = stop_of({0.3, 0.1});
  EXPECT_EQ(two.median_s.value(), 0.3);
  EXPECT_EQ(two.min_s.value(), 0.1);
  EXPECT_EQ(two.max_s.value(), 0.3);
  EXPECT_EQ(stop_of({0.4, 0.1, 0.3, 0.2}).median_s.value(), 0.3);
  EXPECT_EQ(stop_of({0.4, 0.1, 0.2}).median_s.value(), 0.2);
  const auto none = stop_of({std::nullopt, std::nullopt});
  EXPECT_FALSE(none.median_s.has_value());
  EXPECT_EQ(none.detected, 0);
  EXPECT_EQ(none.total, 2);
  const auto partial = stop_of({0.5, std::nullopt, 0.25});
  EXPECT_EQ(partial.detected, 2);
  EXPECT_EQ(partial.total, 3);
  const json::Value j = phm_go2::to_json(phm_go2::aggregate({summary_with_latency(0.5)}));
  EXPECT_EQ(j.at("first_health_stop").at("detected").as_string(), "1/1");
  EXPECT_EQ(j.at("first_violating_verdict").at("detected").as_string(), "0/1");
}

TEST(SessionTools, MostCommonKeepsInsertionOrderOnTies)
{
  phm_go2::Counts counts;
  for (int i = 0; i < 12; ++i) {
    counts.emplace_back("k" + std::to_string(i), i % 3 == 0 ? 2 : 1);
  }
  const phm_go2::Counts top = phm_go2::most_common(counts, 10);
  ASSERT_EQ(top.size(), 10u);
  // Count-2 entries first (k0, k3, k6, k9), then count-1 entries in insertion order.
  const std::vector<std::string> expected = {
    "k0", "k3", "k6", "k9", "k1", "k2", "k4", "k5", "k7", "k8"};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(top[i].first, expected[i]);
  }
}

TEST(SessionTools, HealthSummaryCountsByStateAndSource)
{
  std::vector<Row> rows;
  const std::vector<std::pair<int64_t, std::string>> spec = {
    {0, ""}, {1, "zz"}, {1, "aa"}, {2, "aa"}, {1, "aa"}, {2, "zz"}};
  for (const auto & s : spec) {
    Row r = health_row(1.0, 1.0, s.first);
    r.source = s.second;
    rows.push_back(r);
  }
  const phm_go2::HealthSummary h = phm_go2::health_summary(rows);
  EXPECT_EQ(h.health_msgs, 6);
  EXPECT_EQ(h.non_ok_fraction.value(), 0.8333);
  EXPECT_EQ(
    json::dumps(phm_go2::to_json(h)),
    "{\"health_msgs\": 6, \"state_counts\": {\"OK\": 1, \"DEGRADED\": 3, \"INTERVENE\": 2}, "
    "\"non_ok_fraction\": 0.8333, \"non_ok_by_source\": {\"DEGRADED aa\": 2, \"DEGRADED zz\": 1, "
    "\"INTERVENE aa\": 1, \"INTERVENE zz\": 1}}");
  EXPECT_FALSE(phm_go2::health_summary({}).non_ok_fraction.has_value());
}

TEST(SessionTools, PyRoundIsCorrectlyRoundedHalfEven)
{
  EXPECT_EQ(phm_go2::py_round(0.0625, 3), 0.062);   // exact tie goes to the even digit
  EXPECT_EQ(phm_go2::py_round(0.1875, 3), 0.188);
  EXPECT_EQ(phm_go2::py_round(2.675, 2), 2.67);     // the binary value is below the tie
  EXPECT_EQ(phm_go2::py_round(1.0005, 3), 1.0);     // the binary value is below the tie
  EXPECT_EQ(phm_go2::py_round(0.5, 0), 0.0);
  EXPECT_EQ(phm_go2::py_round(1.0 - 3.0 / 7.0, 4), 0.5714);
}

TEST(SessionTools, PythonStringHelpers)
{
  const std::vector<std::string> lines = {"a", "", "b", "c", "d"};
  EXPECT_EQ(phm_go2::splitlines("a\n\nb\r\nc\rd\n"), lines);
  EXPECT_EQ(phm_go2::splitlines("x\x0b" "y\xe2\x80\xa8" "z").size(), 3u);
  EXPECT_TRUE(phm_go2::splitlines("").empty());
  EXPECT_EQ(phm_go2::trial_text("embedder_freeze_07.log"), "07");
  EXPECT_EQ(phm_go2::trial_text("embedder_stop_3.log"), "3");
  EXPECT_EQ(phm_go2::parse_trial_number("07"), 7);
  EXPECT_THROW(phm_go2::parse_trial_number("1.log"), std::runtime_error);
  EXPECT_EQ(phm_go2::path_name("a/b/session_x/"), "session_x");
  EXPECT_EQ(phm_go2::path_name("."), "");
}

TEST(SessionTools, EmptySessionDirectoryGivesNullFractions)
{
  TempDir tmp;
  const json::Value s = phm_go2::summarize_session(tmp.path().string());
  EXPECT_EQ(s.at("nominal").at("health_msgs").as_int(), 0);
  EXPECT_TRUE(s.at("nominal").at("non_ok_fraction").is_null());
  EXPECT_TRUE(s.at("nominal").at("spread_median").is_null());
  EXPECT_TRUE(s.at("fault_freeze").at("trials").as_array().empty());
  EXPECT_FALSE(s.has("session_txt"));
  EXPECT_FALSE(s.has("calibration"));
}
