"""Sample GPU, CPU, and memory metrics once per second.

Example:
    python -m yunzhe.monitor --duration 60
    python -m yunzhe.monitor --interval 1 --name train-a --output-dir yunzhe/monitor/runs

Artifacts:
    - gpu_metrics.csv
    - host_metrics.csv
    - runtime.log
    - report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import logging
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_GPU_QUERY_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
]
_GPU_CSV_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "gpu_index",
    "gpu_name",
    "gpu_util_percent",
    "memory_used_mb",
    "memory_total_mb",
    "memory_util_percent",
]
_HOST_CSV_PREFIX = [
    "timestamp",
    "elapsed_seconds",
    "mem_used_mb",
    "mem_total_mb",
    "mem_available_mb",
    "mem_util_percent",
    "cpu_avg_util_percent",
]
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]


@dataclass(slots=True)
class GpuSample:
    """One GPU datapoint."""

    timestamp: datetime
    elapsed_seconds: float
    gpu_index: int
    gpu_name: str
    gpu_util_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_util_percent: float


@dataclass(slots=True)
class HostSample:
    """One host datapoint."""

    timestamp: datetime
    elapsed_seconds: float
    mem_used_mb: float
    mem_total_mb: float
    mem_available_mb: float
    mem_util_percent: float
    cpu_avg_util_percent: float
    cpu_util_percent: dict[str, float]


@dataclass(slots=True)
class MemoryStats:
    """System memory snapshot."""

    total_mb: float
    available_mb: float
    used_mb: float
    util_percent: float


CpuSnapshot = dict[str, tuple[int, ...]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample GPU/CPU/memory metrics.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds. Default: 1.0.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional total runtime in seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory that will contain per-run artifacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name. Default: timestamp-based.",
    )
    return parser.parse_args(argv)


def parse_nvidia_smi_output(
    raw_output: str,
    timestamp: datetime,
    elapsed_seconds: float,
) -> list[GpuSample]:
    """Parse `nvidia-smi --query-gpu ... --format=csv,noheader,nounits` output."""

    samples: list[GpuSample] = []
    reader = csv.reader(io.StringIO(raw_output.strip()))
    for row in reader:
        if not row:
            continue
        values = [column.strip() for column in row]
        if len(values) != len(_GPU_QUERY_FIELDS):
            raise ValueError(f"Unexpected nvidia-smi row: {row!r}")
        memory_total = float(values[4])
        memory_used = float(values[3])
        memory_util = 0.0 if memory_total <= 0 else memory_used / memory_total * 100.0
        samples.append(
            GpuSample(
                timestamp=timestamp,
                elapsed_seconds=elapsed_seconds,
                gpu_index=int(values[0]),
                gpu_name=values[1],
                gpu_util_percent=float(values[2]),
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                memory_util_percent=memory_util,
            )
        )
    return samples


def _query_gpu_samples(timestamp: datetime, elapsed_seconds: float) -> list[GpuSample]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown error"
        raise RuntimeError(f"nvidia-smi failed: {stderr}") from exc
    return parse_nvidia_smi_output(result.stdout, timestamp, elapsed_seconds)


def parse_proc_stat(raw_text: str) -> CpuSnapshot:
    """Parse `/proc/stat` CPU counters."""

    snapshot: CpuSnapshot = {}
    for line in raw_text.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        name = parts[0]
        if not (name == "cpu" or name.startswith("cpu")):
            continue
        snapshot[name] = tuple(int(value) for value in parts[1:])
    if "cpu" not in snapshot:
        raise ValueError("Missing aggregate CPU counters in /proc/stat")
    return snapshot


def _read_proc_stat() -> CpuSnapshot:
    return parse_proc_stat(Path("/proc/stat").read_text(encoding="utf-8"))


def _cpu_idle_and_total(counters: tuple[int, ...]) -> tuple[int, int]:
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    total = sum(counters)
    return idle, total


def compute_cpu_utilization(prev: tuple[int, ...], curr: tuple[int, ...]) -> float:
    """Compute utilization between two CPU snapshots."""

    prev_idle, prev_total = _cpu_idle_and_total(prev)
    curr_idle, curr_total = _cpu_idle_and_total(curr)
    delta_total = curr_total - prev_total
    delta_idle = curr_idle - prev_idle
    if delta_total <= 0:
        return 0.0
    busy = delta_total - delta_idle
    return max(0.0, min(100.0, busy / delta_total * 100.0))


def compute_cpu_utilizations(
    prev_snapshot: CpuSnapshot,
    curr_snapshot: CpuSnapshot,
) -> dict[str, float]:
    """Compute per-CPU utilizations for all CPU entries except the aggregate row."""

    cpu_utils: dict[str, float] = {}
    for name in sorted(curr_snapshot, key=_cpu_sort_key):
        if name == "cpu" or name not in prev_snapshot:
            continue
        cpu_utils[name] = compute_cpu_utilization(prev_snapshot[name], curr_snapshot[name])
    return cpu_utils


def _cpu_sort_key(name: str) -> tuple[int, str]:
    suffix = name[3:]
    if suffix.isdigit():
        return int(suffix), name
    return math.inf, name


def parse_meminfo(raw_text: str) -> MemoryStats:
    """Parse `/proc/meminfo` into MB-scale memory stats."""

    values_kb: dict[str, int] = {}
    for line in raw_text.splitlines():
        key, raw_value = line.split(":", maxsplit=1)
        fields = raw_value.strip().split()
        values_kb[key] = int(fields[0])

    total_kb = values_kb["MemTotal"]
    available_kb = values_kb.get("MemAvailable")
    if available_kb is None:
        free_kb = values_kb.get("MemFree", 0)
        buffers_kb = values_kb.get("Buffers", 0)
        cached_kb = values_kb.get("Cached", 0)
        available_kb = free_kb + buffers_kb + cached_kb

    used_kb = max(0, total_kb - available_kb)
    total_mb = total_kb / 1024.0
    available_mb = available_kb / 1024.0
    used_mb = used_kb / 1024.0
    util_percent = 0.0 if total_mb <= 0 else used_mb / total_mb * 100.0
    return MemoryStats(
        total_mb=total_mb,
        available_mb=available_mb,
        used_mb=used_mb,
        util_percent=util_percent,
    )


def _read_meminfo() -> MemoryStats:
    return parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))


def _build_run_dir(base_dir: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = run_name.strip().replace(" ", "-") if run_name else f"monitor-{timestamp}"
    run_dir = base_dir / safe_name
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("yunzhe.monitor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_gpu_rows(writer: csv.DictWriter, samples: list[GpuSample]) -> None:
    for sample in samples:
        writer.writerow(
            {
                "timestamp": sample.timestamp.isoformat(timespec="seconds"),
                "elapsed_seconds": f"{sample.elapsed_seconds:.3f}",
                "gpu_index": sample.gpu_index,
                "gpu_name": sample.gpu_name,
                "gpu_util_percent": f"{sample.gpu_util_percent:.2f}",
                "memory_used_mb": f"{sample.memory_used_mb:.2f}",
                "memory_total_mb": f"{sample.memory_total_mb:.2f}",
                "memory_util_percent": f"{sample.memory_util_percent:.2f}",
            }
        )


def _write_host_row(
    writer: csv.DictWriter,
    sample: HostSample,
    cpu_names: list[str],
) -> None:
    row = {
        "timestamp": sample.timestamp.isoformat(timespec="seconds"),
        "elapsed_seconds": f"{sample.elapsed_seconds:.3f}",
        "mem_used_mb": f"{sample.mem_used_mb:.2f}",
        "mem_total_mb": f"{sample.mem_total_mb:.2f}",
        "mem_available_mb": f"{sample.mem_available_mb:.2f}",
        "mem_util_percent": f"{sample.mem_util_percent:.2f}",
        "cpu_avg_util_percent": f"{sample.cpu_avg_util_percent:.2f}",
    }
    for cpu_name in cpu_names:
        row[f"{cpu_name}_util_percent"] = f"{sample.cpu_util_percent.get(cpu_name, 0.0):.2f}"
    writer.writerow(row)


def _line_chart_svg(
    title: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    max_y: float,
    width: int = 960,
    height: int = 220,
) -> str:
    if not series:
        return f"<section><h3>{html.escape(title)}</h3><p>No data.</p></section>"

    margin_left = 56
    margin_right = 16
    margin_top = 24
    margin_bottom = 28
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [point[0] for points in series.values() for point in points]
    min_x = min(x_values)
    max_x = max(x_values)
    if math.isclose(max_x, min_x):
        max_x = min_x + 1.0

    def scale_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + (1.0 - value / max_y) * plot_height

    grid_lines: list[str] = []
    for tick in range(5):
        ratio = tick / 4
        y_value = max_y * (1.0 - ratio)
        y_pos = margin_top + ratio * plot_height
        grid_lines.append(
            f"<line x1='{margin_left}' y1='{y_pos:.1f}' x2='{width - margin_right}' y2='{y_pos:.1f}' class='grid' />"
            f"<text x='{margin_left - 8}' y='{y_pos + 4:.1f}' class='axis-label'>{y_value:.0f}</text>"
        )

    x_axis = (
        f"<text x='{margin_left}' y='{height - 8}' class='axis-label'>{min_x:.0f}s</text>"
        f"<text x='{width - margin_right}' y='{height - 8}' text-anchor='end' class='axis-label'>{max_x:.0f}s</text>"
        f"<text x='{12}' y='{margin_top - 6}' class='axis-label'>{html.escape(y_label)}</text>"
    )

    polylines: list[str] = []
    legends: list[str] = []
    for index, (label, points) in enumerate(sorted(series.items())):
        color = _PALETTE[index % len(_PALETTE)]
        coordinates = " ".join(
            f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in sorted(points)
        )
        polylines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{coordinates}' />"
        )
        legends.append(
            f"<span class='legend-item'><span class='legend-chip' style='background:{color}'></span>{html.escape(label)}</span>"
        )

    return (
        "<section class='chart'>"
        f"<div class='chart-head'><h3>{html.escape(title)}</h3><div class='legend'>{''.join(legends)}</div></div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"{''.join(grid_lines)}"
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' class='axis' />"
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' class='axis' />"
        f"{''.join(polylines)}{x_axis}</svg></section>"
    )


def _cpu_heatmap_svg(host_samples: list[HostSample], width: int = 960) -> str:
    if not host_samples:
        return "<section><h3>CPU Heatmap</h3><p>No data.</p></section>"

    cpu_names = sorted(host_samples[0].cpu_util_percent, key=_cpu_sort_key)
    if not cpu_names:
        return "<section><h3>CPU Heatmap</h3><p>No per-CPU data.</p></section>"

    cell_width = max(6, min(18, (width - 88) // max(1, len(host_samples))))
    cell_height = 10
    margin_left = 56
    margin_top = 28
    height = margin_top + len(cpu_names) * cell_height + 28

    def fill_color(utilization: float) -> str:
        level = max(0.0, min(1.0, utilization / 100.0))
        red = int(20 + level * 220)
        green = int(158 - level * 90)
        blue = int(70 + level * 20)
        return f"rgb({red},{green},{blue})"

    labels: list[str] = []
    cells: list[str] = []
    label_step = max(1, len(cpu_names) // 16)
    for row_index, cpu_name in enumerate(cpu_names):
        y = margin_top + row_index * cell_height
        if row_index % label_step == 0 or row_index == len(cpu_names) - 1:
            labels.append(
                f"<text x='4' y='{y + 8}' class='axis-label'>{html.escape(cpu_name)}</text>"
            )
        for col_index, sample in enumerate(host_samples):
            x = margin_left + col_index * cell_width
            value = sample.cpu_util_percent.get(cpu_name, 0.0)
            cells.append(
                f"<rect x='{x}' y='{y}' width='{cell_width}' height='{cell_height}' fill='{fill_color(value)}'>"
                f"<title>{cpu_name} @ {sample.elapsed_seconds:.0f}s: {value:.1f}%</title></rect>"
            )

    return (
        "<section class='chart'>"
        "<div class='chart-head'><h3>CPU Heatmap</h3>"
        "<div class='legend'><span class='legend-item'>0%</span><span class='legend-gradient'></span><span class='legend-item'>100%</span></div></div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='CPU usage heatmap'>"
        f"{''.join(labels)}{''.join(cells)}"
        f"<text x='{margin_left}' y='{height - 8}' class='axis-label'>time</text>"
        f"<text x='{width - 12}' y='{height - 8}' text-anchor='end' class='axis-label'>{host_samples[-1].elapsed_seconds:.0f}s</text>"
        "</svg></section>"
    )


def _summary_cards(gpu_samples: list[GpuSample], host_samples: list[HostSample]) -> str:
    gpu_count = len({sample.gpu_index for sample in gpu_samples})
    cpu_count = len(host_samples[0].cpu_util_percent) if host_samples else 0
    duration = host_samples[-1].elapsed_seconds if host_samples else 0.0
    peak_gpu_util = max((sample.gpu_util_percent for sample in gpu_samples), default=0.0)
    peak_gpu_mem = max((sample.memory_used_mb for sample in gpu_samples), default=0.0)
    peak_mem = max((sample.mem_used_mb for sample in host_samples), default=0.0)
    peak_cpu_avg = max((sample.cpu_avg_util_percent for sample in host_samples), default=0.0)
    cards = [
        ("Duration", f"{duration:.0f}s"),
        ("Samples", str(len(host_samples))),
        ("GPUs", str(gpu_count)),
        ("CPUs", str(cpu_count)),
        ("Peak GPU Util", f"{peak_gpu_util:.1f}%"),
        ("Peak GPU Mem", f"{peak_gpu_mem:.0f} MB"),
        ("Peak CPU Avg", f"{peak_cpu_avg:.1f}%"),
        ("Peak Host Mem", f"{peak_mem:.0f} MB"),
    ]
    return "".join(
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div><div class='card-value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )


def generate_report_html(gpu_samples: list[GpuSample], host_samples: list[HostSample]) -> str:
    """Build a self-contained static HTML report."""

    gpu_util_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    gpu_mem_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample in gpu_samples:
        label = f"GPU {sample.gpu_index}"
        gpu_util_series[label].append((sample.elapsed_seconds, sample.gpu_util_percent))
        gpu_mem_series[label].append((sample.elapsed_seconds, sample.memory_util_percent))

    cpu_avg_series = {
        "CPU Avg": [(sample.elapsed_seconds, sample.cpu_avg_util_percent) for sample in host_samples]
    }
    mem_series = {
        "System Memory": [(sample.elapsed_seconds, sample.mem_util_percent) for sample in host_samples]
    }

    charts = [
        _line_chart_svg("GPU Utilization", "util %", gpu_util_series, max_y=100.0),
        _line_chart_svg("GPU Memory", "util %", gpu_mem_series, max_y=100.0),
        _line_chart_svg("CPU Average", "util %", cpu_avg_series, max_y=100.0),
        _line_chart_svg("System Memory", "util %", mem_series, max_y=100.0),
        _cpu_heatmap_svg(host_samples),
    ]

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>System Monitor Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --panel: #fffaf2;
      --ink: #1f2a35;
      --muted: #66717c;
      --line: #d7cfbe;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: linear-gradient(180deg, #f0eadf 0%, var(--bg) 100%); color: var(--ink); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-family: "IBM Plex Mono", monospace; font-size: 28px; }}
    p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 20px 0 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: 0 8px 20px rgba(31, 42, 53, 0.05); }}
    .card-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .card-value {{ margin-top: 6px; font-size: 24px; font-weight: 700; }}
    .chart {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 14px 16px; margin-bottom: 16px; box-shadow: 0 10px 24px rgba(31, 42, 53, 0.05); overflow-x: auto; }}
    .chart-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
    .chart-head h3 {{ margin: 0; font-size: 16px; }}
    .legend {{ display: flex; align-items: center; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 12px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend-chip {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
    .legend-gradient {{ width: 96px; height: 10px; border-radius: 999px; background: linear-gradient(90deg, rgb(20,158,70), rgb(240,68,90)); }}
    svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ stroke: #e6dece; stroke-width: 1; }}
    .axis {{ stroke: #918979; stroke-width: 1.2; }}
    .axis-label {{ fill: #716857; font-size: 11px; }}
    @media (max-width: 768px) {{
      main {{ padding: 16px; }}
      .card-value {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>System Monitor Report</h1>
      <p>Compact post-run summary for GPU, CPU, and memory utilization.</p>
    </header>
    <section class='cards'>{_summary_cards(gpu_samples, host_samples)}</section>
    {''.join(charts)}
  </main>
</body>
</html>
"""


def _write_report(report_path: Path, gpu_samples: list[GpuSample], host_samples: list[HostSample]) -> None:
    report_path.write_text(generate_report_html(gpu_samples, host_samples), encoding="utf-8")


def _sample_host(
    timestamp: datetime,
    elapsed_seconds: float,
    prev_cpu_snapshot: CpuSnapshot,
    curr_cpu_snapshot: CpuSnapshot,
    first_sample: bool,
) -> HostSample:
    mem = _read_meminfo()
    cpu_util_percent = (
        {name: 0.0 for name in sorted(curr_cpu_snapshot) if name.startswith("cpu") and name != "cpu"}
        if first_sample
        else compute_cpu_utilizations(prev_cpu_snapshot, curr_cpu_snapshot)
    )
    cpu_avg_util_percent = 0.0
    if not first_sample:
        cpu_avg_util_percent = compute_cpu_utilization(
            prev_cpu_snapshot["cpu"], curr_cpu_snapshot["cpu"]
        )
    return HostSample(
        timestamp=timestamp,
        elapsed_seconds=elapsed_seconds,
        mem_used_mb=mem.used_mb,
        mem_total_mb=mem.total_mb,
        mem_available_mb=mem.available_mb,
        mem_util_percent=mem.util_percent,
        cpu_avg_util_percent=cpu_avg_util_percent,
        cpu_util_percent=cpu_util_percent,
    )


def _should_stop(
    elapsed_seconds: float,
    sample_count: int,
    duration: float | None,
    max_samples: int | None,
) -> bool:
    if duration is not None and elapsed_seconds >= duration:
        return True
    if max_samples is not None and sample_count >= max_samples:
        return True
    return False


def run_monitor(args: argparse.Namespace) -> int:
    """Run sampling and emit run artifacts."""

    if args.interval <= 0:
        raise ValueError("interval must be positive")
    if args.duration is None and args.max_samples is None:
        args.max_samples = 60

    run_dir = _build_run_dir(args.output_dir, args.name)
    logger = _setup_logger(run_dir / "runtime.log")

    gpu_csv_path = run_dir / "gpu_metrics.csv"
    host_csv_path = run_dir / "host_metrics.csv"
    report_path = run_dir / "report.html"

    start_monotonic = time.monotonic()
    prev_cpu_snapshot = _read_proc_stat()
    cpu_names = [name for name in sorted(prev_cpu_snapshot, key=_cpu_sort_key) if name != "cpu"]

    gpu_samples_all: list[GpuSample] = []
    host_samples: list[HostSample] = []
    interrupted = False

    logger.info(f"Sampling to {run_dir}")

    with gpu_csv_path.open("w", newline="", encoding="utf-8") as gpu_file, host_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as host_file:
        gpu_writer = csv.DictWriter(gpu_file, fieldnames=_GPU_CSV_FIELDS)
        host_fieldnames = _HOST_CSV_PREFIX + [f"{cpu_name}_util_percent" for cpu_name in cpu_names]
        host_writer = csv.DictWriter(host_file, fieldnames=host_fieldnames)
        gpu_writer.writeheader()
        host_writer.writeheader()

        sample_count = 0
        while True:
            try:
                now = datetime.now()
                elapsed_seconds = time.monotonic() - start_monotonic
                curr_cpu_snapshot = _read_proc_stat()
                host_sample = _sample_host(
                    timestamp=now,
                    elapsed_seconds=elapsed_seconds,
                    prev_cpu_snapshot=prev_cpu_snapshot,
                    curr_cpu_snapshot=curr_cpu_snapshot,
                    first_sample=sample_count == 0,
                )
                gpu_samples = _query_gpu_samples(now, elapsed_seconds)

                host_samples.append(host_sample)
                gpu_samples_all.extend(gpu_samples)
                _write_host_row(host_writer, host_sample, cpu_names)
                _write_gpu_rows(gpu_writer, gpu_samples)
                gpu_file.flush()
                host_file.flush()

                prev_cpu_snapshot = curr_cpu_snapshot
                sample_count += 1

                if _should_stop(elapsed_seconds, sample_count, args.duration, args.max_samples):
                    break

                target = start_monotonic + sample_count * args.interval
                sleep_seconds = max(0.0, target - time.monotonic())
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                interrupted = True
                logger.info("Interrupted, finalizing artifacts")
                break

    if host_samples:
        _write_report(report_path, gpu_samples_all, host_samples)
    else:
        report_path.write_text("<html><body><p>No samples collected.</p></body></html>", encoding="utf-8")

    logger.info(
        "Done: %s host samples, %s GPU rows, report %s",
        len(host_samples),
        len(gpu_samples_all),
        report_path,
    )
    return 130 if interrupted else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())