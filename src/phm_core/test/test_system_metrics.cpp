// Copyright 2026 Yusuf Guenena. MIT License.
// SystemMetricsReader against fake /proc and /sys/class/thermal trees
// (ported from phm_detectors/tests/test_system_metrics.py).
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "phm_core/adapters.hpp"
#include "phm_core/system_metrics.hpp"

using phm_core::SystemMetricsReader;

namespace
{

class FakeRoots : public ::testing::Test
{
protected:
  void SetUp() override
  {
    char tmpl[] = "/tmp/phm_metrics_XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    root_ = tmpl;
    proc_ = root_ + "/proc";
    thermal_ = root_ + "/thermal";
    mkdir(proc_.c_str(), 0700);
    mkdir(thermal_.c_str(), 0700);
    write(proc_ + "/meminfo",
      "MemTotal:       16000000 kB\nMemFree:  1000000 kB\nMemAvailable:   12000000 kB\n");
  }
  void TearDown() override
  {
    const std::string cmd = "rm -rf '" + root_ + "'";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
  }
  static void write(const std::string & path, const std::string & text)
  {
    std::ofstream(path) << text;
  }
  void stat_line(int user, int idle, int iowait = 0)
  {
    write(
      proc_ + "/stat", "cpu  " + std::to_string(user) + " 0 0 " + std::to_string(idle) + " " +
      std::to_string(iowait) + " 0 0 0\ncpu0 1 0 0 1 0 0 0 0\n");
  }
  void zone(int n, const std::string & kind, int millideg)
  {
    const std::string z = thermal_ + "/thermal_zone" + std::to_string(n);
    mkdir(z.c_str(), 0700);
    write(z + "/type", kind + "\n");
    write(z + "/temp", std::to_string(millideg) + "\n");
  }
  std::string root_;
  std::string proc_;
  std::string thermal_;
};

}  // namespace

TEST_F(FakeRoots, CpuNeedsTwoSamplesThenReportsBusyShare)
{
  SystemMetricsReader reader(proc_, thermal_);
  stat_line(100, 100);
  EXPECT_FALSE(reader.read().cpu_percent.has_value());
  stat_line(175, 125);
  EXPECT_DOUBLE_EQ(*reader.read().cpu_percent, 75.0);
}

TEST_F(FakeRoots, IowaitCountsAsIdle)
{
  SystemMetricsReader reader(proc_, thermal_);
  stat_line(0, 0, 0);
  reader.read();
  stat_line(10, 0, 90);
  EXPECT_DOUBLE_EQ(*reader.read().cpu_percent, 10.0);
}

TEST_F(FakeRoots, MemoryUsesMemAvailable)
{
  EXPECT_DOUBLE_EQ(*SystemMetricsReader(proc_, thermal_).read().memory_percent, 25.0);
}

TEST_F(FakeRoots, GpuTempTakesHottestGpuZoneOnly)
{
  zone(0, "cpu-thermal", 90000);
  zone(1, "gpu-thermal", 51500);
  zone(2, "GPU-therm", 53000);
  EXPECT_DOUBLE_EQ(*SystemMetricsReader(proc_, thermal_).read().gpu_temp_c, 53.0);
}

TEST_F(FakeRoots, HostWithoutGpuZoneReportsNoGpuMetric)
{
  zone(0, "x86_pkg_temp", 60000);
  EXPECT_FALSE(SystemMetricsReader(proc_, thermal_).read().gpu_temp_c.has_value());
}

TEST_F(FakeRoots, UnreadableRootsOmitMetrics)
{
  SystemMetricsReader reader(root_ + "/missing", root_ + "/missing");
  const auto m = reader.read();
  EXPECT_FALSE(m.cpu_percent || m.memory_percent || m.gpu_temp_c);
}

TEST_F(FakeRoots, ReaderOutputDrivesThresholdAdapter)
{
  zone(0, "gpu-thermal", 95000);
  SystemMetricsReader reader(proc_, thermal_);
  phm_core::StaticThresholdAdapter adapter("system:gpu", "gpu_temp_c", 2);
  std::vector<bool> violating;
  for (int i = 0; i < 2; ++i) {
    violating.push_back(
      adapter.update({"gpu_temp_c", *reader.read().gpu_temp_c, 85.0})->violating);
  }
  EXPECT_EQ(violating, (std::vector<bool>{false, true}));
}
