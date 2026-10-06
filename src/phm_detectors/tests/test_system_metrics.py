"""SystemMetricsReader against fake /proc and /sys/class/thermal trees.

The detectors node built CPU / memory / GPU-temperature threshold adapters but
never fed them, so host health never reached the arbiter. These tests pin the
reader that now feeds them, including the Jetson ``gpu-thermal`` zone and the
"no GPU zone on this host" case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phm_detectors import StaticThresholdAdapter, SystemMetricsReader, ThresholdSample

_MEMINFO = "MemTotal:       16000000 kB\nMemFree:  1000000 kB\nMemAvailable:   12000000 kB\n"


def _proc(root: Path, user: int, idle: int, iowait: int = 0) -> None:
    # cpu  user nice system idle iowait irq softirq steal
    (root / "stat").write_text(f"cpu  {user} 0 0 {idle} {iowait} 0 0 0\ncpu0 1 0 0 1 0 0 0 0\n")


def _zone(root: Path, n: int, kind: str, millideg: int) -> None:
    z = root / f"thermal_zone{n}"
    z.mkdir()
    (z / "type").write_text(kind + "\n")
    (z / "temp").write_text(f"{millideg}\n")


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    proc, thermal = tmp_path / "proc", tmp_path / "thermal"
    proc.mkdir()
    thermal.mkdir()
    (proc / "meminfo").write_text(_MEMINFO)
    return proc, thermal


def test_cpu_needs_two_samples_then_reports_busy_share(roots: tuple[Path, Path]) -> None:
    proc, thermal = roots
    reader = SystemMetricsReader(str(proc), str(thermal))
    _proc(proc, user=100, idle=100)
    assert "cpu_percent" not in reader.read()
    _proc(proc, user=175, idle=125)  # +75 busy, +25 idle
    assert reader.read()["cpu_percent"] == pytest.approx(75.0)


def test_iowait_counts_as_idle(roots: tuple[Path, Path]) -> None:
    proc, thermal = roots
    reader = SystemMetricsReader(str(proc), str(thermal))
    _proc(proc, user=0, idle=0, iowait=0)
    reader.read()
    _proc(proc, user=10, idle=0, iowait=90)
    assert reader.read()["cpu_percent"] == pytest.approx(10.0)


def test_memory_percent_uses_memavailable(roots: tuple[Path, Path]) -> None:
    proc, thermal = roots
    assert SystemMetricsReader(str(proc), str(thermal)).read()["memory_percent"] == (
        pytest.approx(25.0)
    )


def test_gpu_temp_takes_hottest_gpu_zone_only(roots: tuple[Path, Path]) -> None:
    proc, thermal = roots
    _zone(thermal, 0, "cpu-thermal", 90000)
    _zone(thermal, 1, "gpu-thermal", 51500)
    _zone(thermal, 2, "GPU-therm", 53000)
    assert SystemMetricsReader(str(proc), str(thermal)).read()["gpu_temp_c"] == (
        pytest.approx(53.0)
    )


def test_host_without_gpu_zone_reports_no_gpu_metric(roots: tuple[Path, Path]) -> None:
    proc, thermal = roots
    _zone(thermal, 0, "x86_pkg_temp", 60000)
    assert "gpu_temp_c" not in SystemMetricsReader(str(proc), str(thermal)).read()


def test_unreadable_proc_omits_metrics(tmp_path: Path) -> None:
    reader = SystemMetricsReader(str(tmp_path / "missing"), str(tmp_path / "missing"))
    assert reader.read() == {}


def test_reader_output_drives_threshold_adapter(roots: tuple[Path, Path]) -> None:
    proc, thermal = roots
    _zone(thermal, 0, "gpu-thermal", 95000)
    reader = SystemMetricsReader(str(proc), str(thermal))
    adapter = StaticThresholdAdapter("system:gpu", "gpu_temp_c", min_consecutive=2)
    verdicts = [
        adapter.update(ThresholdSample("gpu_temp_c", reader.read()["gpu_temp_c"], 85.0))
        for _ in range(2)
    ]
    assert [v.violating for v in verdicts] == [False, True]
    assert verdicts[-1].source == "threshold:gpu_temp_c"
