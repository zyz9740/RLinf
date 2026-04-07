"""Rebuild missing monitor report.html files from existing CSV artifacts.

Usage:
    PYTHONPATH=/path/to/RLinf /path/to/venv/bin/python yunzhe/monitor/rebuild_reports.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from yunzhe.monitor.monitor import GpuSample, HostSample, generate_report_html


def _load_gpu_samples(path: Path) -> list[GpuSample]:
    samples: list[GpuSample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                GpuSample(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    gpu_index=int(row["gpu_index"]),
                    gpu_name=row["gpu_name"],
                    gpu_util_percent=float(row["gpu_util_percent"]),
                    memory_used_mb=float(row["memory_used_mb"]),
                    memory_total_mb=float(row["memory_total_mb"]),
                    memory_util_percent=float(row["memory_util_percent"]),
                )
            )
    return samples


def _load_host_samples(path: Path) -> list[HostSample]:
    samples: list[HostSample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        cpu_cols = [
            c
            for c in fieldnames
            if c.startswith("cpu")
            and c.endswith("_util_percent")
            and c != "cpu_avg_util_percent"
        ]
        for row in reader:
            cpu_map = {
                c.removesuffix("_util_percent"): float(row[c])
                for c in cpu_cols
                if row.get(c) not in (None, "")
            }
            samples.append(
                HostSample(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    mem_used_mb=float(row["mem_used_mb"]),
                    mem_total_mb=float(row["mem_total_mb"]),
                    mem_available_mb=float(row["mem_available_mb"]),
                    mem_util_percent=float(row["mem_util_percent"]),
                    cpu_avg_util_percent=float(row["cpu_avg_util_percent"]),
                    cpu_util_percent=cpu_map,
                )
            )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild missing monitor report.html files")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory containing monitor run folders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate report.html even if it already exists",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        print(f"runs dir not found: {runs_dir}")
        return 1

    created = 0
    skipped = 0
    failed = 0

    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue

        report = run / "report.html"
        gpu_csv = run / "gpu_metrics.csv"
        host_csv = run / "host_metrics.csv"

        if report.exists() and not args.force:
            skipped += 1
            continue
        if not gpu_csv.exists() or not host_csv.exists():
            skipped += 1
            continue

        try:
            gpu_samples = _load_gpu_samples(gpu_csv)
            host_samples = _load_host_samples(host_csv)
            if not host_samples:
                print(f"[skip-empty] {run}")
                skipped += 1
                continue

            report.write_text(
                generate_report_html(gpu_samples, host_samples), encoding="utf-8"
            )
            print(f"[created] {report}")
            created += 1
        except Exception as exc:  # pragma: no cover - defensive for mixed run dirs
            print(f"[failed] {run}: {exc}")
            failed += 1

    print(f"created={created} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
