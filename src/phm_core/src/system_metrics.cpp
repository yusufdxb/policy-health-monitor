// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/system_metrics.hpp"

#include <dirent.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace phm_core
{
namespace
{

bool read_file(const std::string & path, std::string & out)
{
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

std::string strip(const std::string & s)
{
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
    ++b;
  }
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
    --e;
  }
  return s.substr(b, e - b);
}

bool parse_double(const std::string & text, double & out)
{
  const std::string s = strip(text);
  if (s.empty()) {
    return false;
  }
  char * end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end == s.c_str() + s.size();
}

}  // namespace

SystemMetricsReader::SystemMetricsReader(std::string proc_root, std::string thermal_root)
: proc_(std::move(proc_root)), thermal_(std::move(thermal_root))
{
}

SystemMetrics SystemMetricsReader::read()
{
  SystemMetrics m;
  m.cpu_percent = cpu_percent();
  m.memory_percent = memory_percent();
  m.gpu_temp_c = gpu_temp_c();
  return m;
}

std::optional<double> SystemMetricsReader::cpu_percent()
{
  std::string text;
  if (!read_file(proc_ + "/stat", text)) {
    return std::nullopt;
  }
  std::istringstream lines(text);
  std::string first;
  if (!std::getline(lines, first)) {
    return std::nullopt;
  }
  std::istringstream fields(first);
  std::string label;
  fields >> label;
  std::vector<uint64_t> ticks;
  uint64_t v;
  while (fields >> v) {
    ticks.push_back(v);
  }
  if (ticks.size() < 4) {
    return std::nullopt;
  }
  const uint64_t idle = ticks[3] + (ticks.size() > 4 ? ticks[4] : 0);  // idle + iowait
  uint64_t total = 0;
  for (uint64_t t : ticks) {
    total += t;
  }
  const bool had_prev = has_prev_;
  const uint64_t prev_total = prev_total_;
  const uint64_t prev_idle = prev_idle_;
  has_prev_ = true;
  prev_total_ = total;
  prev_idle_ = idle;
  if (!had_prev || total <= prev_total) {
    return std::nullopt;
  }
  const double dt = static_cast<double>(total - prev_total);
  const double didle = static_cast<double>(idle) - static_cast<double>(prev_idle);
  const double busy = dt - didle;
  return 100.0 * busy / dt;
}

std::optional<double> SystemMetricsReader::memory_percent() const
{
  std::string text;
  if (!read_file(proc_ + "/meminfo", text)) {
    return std::nullopt;
  }
  std::optional<double> total;
  std::optional<double> avail;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, colon);
    std::istringstream rest(line.substr(colon + 1));
    std::string number;
    rest >> number;
    double value;
    if (!parse_double(number, value)) {
      continue;
    }
    if (key == "MemTotal") {
      total = value;
    } else if (key == "MemAvailable") {
      avail = value;
    }
  }
  if (!total || !avail || *total <= 0.0) {
    return std::nullopt;
  }
  return 100.0 * (1.0 - *avail / *total);
}

std::optional<double> SystemMetricsReader::gpu_temp_c() const
{
  DIR * dir = opendir(thermal_.c_str());
  if (dir == nullptr) {
    return std::nullopt;
  }
  std::vector<std::string> zones;
  while (struct dirent * entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name.rfind("thermal_zone", 0) == 0) {
      zones.push_back(name);
    }
  }
  closedir(dir);
  std::sort(zones.begin(), zones.end());
  std::optional<double> hottest;
  for (const std::string & zone : zones) {
    std::string type;
    std::string temp;
    if (!read_file(thermal_ + "/" + zone + "/type", type)) {
      continue;
    }
    std::string lowered = strip(type);
    std::transform(
      lowered.begin(), lowered.end(), lowered.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});
    if (lowered.find("gpu") == std::string::npos) {
      continue;
    }
    double millideg;
    if (!read_file(thermal_ + "/" + zone + "/temp", temp) || !parse_double(temp, millideg)) {
      continue;
    }
    const double c = millideg / 1000.0;
    if (!hottest || c > *hottest) {
      hottest = c;
    }
  }
  return hottest;
}

}  // namespace phm_core
