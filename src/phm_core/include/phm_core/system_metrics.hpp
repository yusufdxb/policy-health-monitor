// Copyright 2026 Yusuf Guenena. MIT License.
// Host metrics judged by the StaticThresholdAdapters, read from procfs/sysfs.
//
//   cpu_percent     busy share of all CPUs since the previous read
//                   (/proc/stat; iowait counts as idle), so the first read
//                   reports no CPU value
//   memory_percent  1 - MemAvailable / MemTotal (/proc/meminfo)
//   gpu_temp_c      hottest thermal zone whose type names the GPU
//                   (gpu-thermal on Jetson-class boards); hosts without one
//                   report no GPU value rather than a made-up number
//
// A metric that cannot be read is omitted from the result.
#ifndef PHM_CORE__SYSTEM_METRICS_HPP_
#define PHM_CORE__SYSTEM_METRICS_HPP_

#include <cstdint>
#include <optional>
#include <string>

namespace phm_core
{

struct SystemMetrics
{
  std::optional<double> cpu_percent;
  std::optional<double> memory_percent;
  std::optional<double> gpu_temp_c;
};

class SystemMetricsReader
{
public:
  explicit SystemMetricsReader(
    std::string proc_root = "/proc", std::string thermal_root = "/sys/class/thermal");
  SystemMetrics read();

private:
  std::optional<double> cpu_percent();
  std::optional<double> memory_percent() const;
  std::optional<double> gpu_temp_c() const;

  std::string proc_;
  std::string thermal_;
  bool has_prev_ = false;
  uint64_t prev_total_ = 0;
  uint64_t prev_idle_ = 0;
};

}  // namespace phm_core

#endif  // PHM_CORE__SYSTEM_METRICS_HPP_
