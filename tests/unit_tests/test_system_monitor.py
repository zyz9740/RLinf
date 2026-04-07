"""Unit tests for the standalone system monitor."""

from __future__ import annotations

from datetime import datetime

from yunzhe.monitor.monitor import (
    HostSample,
    compute_cpu_utilization,
    compute_cpu_utilizations,
    generate_report_html,
    parse_meminfo,
    parse_nvidia_smi_output,
    parse_proc_stat,
)


def test_parse_nvidia_smi_output() -> None:
    raw = "0, NVIDIA A100, 45, 1024, 40960\n1, NVIDIA A100, 70, 2048, 40960\n"

    samples = parse_nvidia_smi_output(raw, datetime(2026, 4, 1, 12, 0, 0), 3.0)

    assert len(samples) == 2
    assert samples[0].gpu_index == 0
    assert samples[0].gpu_name == "NVIDIA A100"
    assert samples[0].gpu_util_percent == 45.0
    assert samples[0].memory_util_percent == 2.5
    assert samples[1].gpu_index == 1
    assert samples[1].memory_used_mb == 2048.0


def test_parse_proc_stat_and_compute_cpu_utils() -> None:
    prev = parse_proc_stat(
        "\n".join(
            [
                "cpu  100 0 100 700 0 0 0 0 0 0",
                "cpu0 50 0 50 350 0 0 0 0 0 0",
                "cpu1 50 0 50 350 0 0 0 0 0 0",
            ]
        )
    )
    curr = parse_proc_stat(
        "\n".join(
            [
                "cpu  150 0 150 800 0 0 0 0 0 0",
                "cpu0 70 0 80 400 0 0 0 0 0 0",
                "cpu1 80 0 70 400 0 0 0 0 0 0",
            ]
        )
    )

    cpu_avg = compute_cpu_utilization(prev["cpu"], curr["cpu"])
    cpu_utils = compute_cpu_utilizations(prev, curr)

    assert round(cpu_avg, 2) == 50.0
    assert round(cpu_utils["cpu0"], 2) == 50.0
    assert round(cpu_utils["cpu1"], 2) == 50.0


def test_parse_meminfo() -> None:
    meminfo = "\n".join(
        [
            "MemTotal:       32768000 kB",
            "MemFree:         2048000 kB",
            "MemAvailable:   12288000 kB",
            "Buffers:          512000 kB",
            "Cached:          4096000 kB",
        ]
    )

    stats = parse_meminfo(meminfo)

    assert stats.total_mb == 32000.0
    assert stats.available_mb == 12000.0
    assert stats.used_mb == 20000.0
    assert stats.util_percent == 62.5


def test_generate_report_html_contains_expected_sections() -> None:
    gpu_samples = parse_nvidia_smi_output(
        "0, NVIDIA A100, 45, 1024, 40960\n0, NVIDIA A100, 55, 2048, 40960\n",
        datetime(2026, 4, 1, 12, 0, 0),
        1.0,
    )
    gpu_samples[1].elapsed_seconds = 2.0

    host_prev = parse_proc_stat(
        "\n".join(
            [
                "cpu  100 0 100 700 0 0 0 0 0 0",
                "cpu0 50 0 50 350 0 0 0 0 0 0",
                "cpu1 50 0 50 350 0 0 0 0 0 0",
            ]
        )
    )
    host_curr = parse_proc_stat(
        "\n".join(
            [
                "cpu  150 0 150 800 0 0 0 0 0 0",
                "cpu0 70 0 80 400 0 0 0 0 0 0",
                "cpu1 80 0 70 400 0 0 0 0 0 0",
            ]
        )
    )
    cpu_utils = compute_cpu_utilizations(host_prev, host_curr)
    host_html = generate_report_html(
        gpu_samples,
        [
            HostSample(
                timestamp=datetime(2026, 4, 1, 12, 0, 0),
                elapsed_seconds=1.0,
                mem_used_mb=20000.0,
                mem_total_mb=32000.0,
                mem_available_mb=12000.0,
                mem_util_percent=62.5,
                cpu_avg_util_percent=50.0,
                cpu_util_percent=cpu_utils,
            ),
            HostSample(
                timestamp=datetime(2026, 4, 1, 12, 0, 1),
                elapsed_seconds=2.0,
                mem_used_mb=21000.0,
                mem_total_mb=32000.0,
                mem_available_mb=11000.0,
                mem_util_percent=65.625,
                cpu_avg_util_percent=60.0,
                cpu_util_percent={"cpu0": 60.0, "cpu1": 60.0},
            ),
        ],
    )

    assert "System Monitor Report" in host_html
    assert "GPU Utilization" in host_html
    assert "CPU Heatmap" in host_html
    assert "GPU 0" in host_html
    assert "cpu0" in host_html