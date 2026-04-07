#!/usr/bin/env python3
"""Analyze training logs and monitoring data across different parameter configurations.

Usage:
    python yunzhe/analyze_experiments.py --output analysis_report.html
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PerformanceMetrics:
    """Performance metrics extracted from training logs."""

    run_id: str
    total_num_envs: int
    micro_batch_size: int
    placement_mode: str
    timestamp: str

    # Rollout performance
    rollout_avg_time: float | None = None  # seconds per epoch
    rollout_total_epochs: int | None = None

    # Training performance
    step_time: float | None = None  # total step time in seconds
    generate_rollouts_time: float | None = None  # seconds
    actor_run_training_time: float | None = None  # seconds

    # Additional timing
    env_interact_step_time: float | None = None
    env_interact_time: float | None = None
    rollout_generate_one_epoch_time: float | None = None
    cal_adv_and_returns_time: float | None = None


@dataclass
class ResourceMetrics:
    """Resource utilization metrics from monitoring data."""

    run_id: str
    monitor_duration: float | None = None  # total monitoring time in seconds

    # GPU 0-1 metrics (rollout/env in split mode)
    gpu01_mem_avg_mb: float | None = None
    gpu01_mem_max_mb: float | None = None
    gpu01_mem_stable_mb: float | None = None  # avg of last 50% samples
    gpu01_util_avg_pct: float | None = None
    gpu01_util_max_pct: float | None = None

    # GPU 2-3 metrics (actor in split mode)
    gpu23_mem_avg_mb: float | None = None
    gpu23_mem_max_mb: float | None = None
    gpu23_mem_stable_mb: float | None = None
    gpu23_util_avg_pct: float | None = None
    gpu23_util_max_pct: float | None = None

    # Rollout phase metrics (first segment)
    rollout_phase_gpu01_mem_avg_mb: float | None = None
    rollout_phase_gpu23_mem_avg_mb: float | None = None
    rollout_phase_duration: float | None = None

    # Training phase metrics (second segment)
    training_phase_gpu01_mem_avg_mb: float | None = None
    training_phase_gpu23_mem_avg_mb: float | None = None
    training_phase_duration: float | None = None

    # Host metrics
    cpu_avg_util_pct: float | None = None
    cpu_max_util_pct: float | None = None
    mem_avg_util_pct: float | None = None
    mem_max_util_pct: float | None = None


def parse_train_log(log_path: Path) -> PerformanceMetrics:
    """Extract performance metrics from a training log file."""

    # Parse run_id and params from filename
    filename = log_path.stem  # e.g., "env64_mbs32-20260401-182308"
    match = re.match(r"env(\d+)_mbs(\d+)-(\d{8}-\d{6})", filename)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    total_num_envs = int(match.group(1))
    micro_batch_size = int(match.group(2))
    timestamp = match.group(3)
    run_id = f"env{total_num_envs}_mbs{micro_batch_size}"

    # Determine placement mode from log content
    log_content = log_path.read_text(encoding="utf-8")
    if "cluster.component_placement.actor=2-3" in log_content:
        placement_mode = "split"
    elif "cluster.component_placement.actor=all" in log_content:
        placement_mode = "all"
    else:
        placement_mode = "unknown"

    metrics = PerformanceMetrics(
        run_id=run_id,
        total_num_envs=total_num_envs,
        micro_batch_size=micro_batch_size,
        placement_mode=placement_mode,
        timestamp=timestamp,
    )

    # Extract rollout timing: "Generating Rollout Epochs: 100%|██████████| 8/8 [30:20<00:00, 227.56s/it]"
    rollout_pattern = r"Generating Rollout Epochs:\s+100%.*?\|\s+(\d+)/(\d+)\s+\[.*?,\s+([\d.]+)s/it\]"
    rollout_matches = list(re.finditer(rollout_pattern, log_content))
    if rollout_matches:
        last_match = rollout_matches[-1]
        metrics.rollout_total_epochs = int(last_match.group(2))
        metrics.rollout_avg_time = float(last_match.group(3))

    # Extract step time: "│ Elapsed: 59:15 │ ETA: 00:00 │ Step Time: 3555.125s"
    step_time_pattern = r"Step Time:\s+([\d.]+)s"
    step_time_matches = list(re.finditer(step_time_pattern, log_content))
    if step_time_matches:
        metrics.step_time = float(step_time_matches[-1].group(1))

    # Extract timing breakdown: "│generate_rollouts=1839.6               │"
    timing_patterns = {
        "generate_rollouts": r"generate_rollouts=([\d.]+)",
        "actor_run_training": r"actor/run_training=([\d.]+)",
        "env_interact_step": r"env/env_interact_step=([\d.]+)",
        "env_interact": r"env/interact=([\d.]+)",
        "rollout_generate_one_epoch": r"rollout/generate_one_epoch=([\d.]+)",
        "cal_adv_and_returns": r"cal_adv_and_returns=([\d.]+)",
    }

    for key, pattern in timing_patterns.items():
        matches = list(re.finditer(pattern, log_content))
        if matches:
            value = float(matches[-1].group(1))
            if key == "generate_rollouts":
                metrics.generate_rollouts_time = value
            elif key == "actor_run_training":
                metrics.actor_run_training_time = value
            elif key == "env_interact_step":
                metrics.env_interact_step_time = value
            elif key == "env_interact":
                metrics.env_interact_time = value
            elif key == "rollout_generate_one_epoch":
                metrics.rollout_generate_one_epoch_time = value
            elif key == "cal_adv_and_returns":
                metrics.cal_adv_and_returns_time = value

    return metrics


def parse_monitor_data(
    monitor_dir: Path, run_id: str, placement_mode: str, perf_metrics: PerformanceMetrics | None = None
) -> ResourceMetrics:
    """Extract resource metrics from monitoring CSV files."""

    gpu_csv = monitor_dir / "gpu_metrics.csv"
    host_csv = monitor_dir / "host_metrics.csv"

    metrics = ResourceMetrics(run_id=run_id)

    if not gpu_csv.exists() or not host_csv.exists():
        return metrics

    # Load GPU data
    gpu_df = pd.read_csv(gpu_csv)
    if len(gpu_df) == 0:
        return metrics

    metrics.monitor_duration = gpu_df["elapsed_seconds"].max()

    # Separate GPU groups based on placement mode
    if placement_mode == "split":
        gpu01_df = gpu_df[gpu_df["gpu_index"].isin([0, 1])]
        gpu23_df = gpu_df[gpu_df["gpu_index"].isin([2, 3])]
    else:
        # For "all" mode, all GPUs are used together
        gpu01_df = gpu_df  # Treat all GPUs as one group
        gpu23_df = pd.DataFrame()  # Empty

    # GPU 0-1 metrics
    if len(gpu01_df) > 0:
        metrics.gpu01_mem_avg_mb = gpu01_df["memory_used_mb"].mean()
        metrics.gpu01_mem_max_mb = gpu01_df["memory_used_mb"].max()
        # Stable value: average of last 50% samples
        stable_start = len(gpu01_df) // 2
        metrics.gpu01_mem_stable_mb = gpu01_df.iloc[stable_start:]["memory_used_mb"].mean()
        metrics.gpu01_util_avg_pct = gpu01_df["gpu_util_percent"].mean()
        metrics.gpu01_util_max_pct = gpu01_df["gpu_util_percent"].max()

    # GPU 2-3 metrics
    if len(gpu23_df) > 0:
        metrics.gpu23_mem_avg_mb = gpu23_df["memory_used_mb"].mean()
        metrics.gpu23_mem_max_mb = gpu23_df["memory_used_mb"].max()
        stable_start = len(gpu23_df) // 2
        metrics.gpu23_mem_stable_mb = gpu23_df.iloc[stable_start:]["memory_used_mb"].mean()
        metrics.gpu23_util_avg_pct = gpu23_df["gpu_util_percent"].mean()
        metrics.gpu23_util_max_pct = gpu23_df["gpu_util_percent"].max()

    # Detect rollout and training phases
    # Strategy 1: Use actual rollout timing from performance metrics if available
    # Strategy 2: Detect transition based on GPU memory usage pattern
    if placement_mode == "split" and len(gpu23_df) > 0:
        transition_time = None

        # Strategy 1: Calculate from training log timing
        if perf_metrics and perf_metrics.rollout_avg_time and perf_metrics.rollout_total_epochs:
            # Rollout phase duration = avg_time_per_epoch * num_epochs
            estimated_rollout_duration = perf_metrics.rollout_avg_time * perf_metrics.rollout_total_epochs
            # Add some buffer time for initialization
            transition_time = estimated_rollout_duration + 60.0  # 60s buffer

        # Strategy 2: Fallback to GPU memory detection
        if transition_time is None or transition_time > metrics.monitor_duration * 0.9:
            # Group by elapsed_seconds and get max memory per timestamp
            gpu23_timeline = gpu23_df.groupby("elapsed_seconds")["memory_used_mb"].max()

            # Filter out initial anomalies (values < 1000 MB are likely monitoring startup issues)
            valid_timeline = gpu23_timeline[gpu23_timeline > 1000.0]

            if len(valid_timeline) > 20:
                # Strategy: detect the transition from rollout to training phase
                # Training phase typically shows sustained high memory usage

                # Compute moving average to smooth noise
                window_size = min(10, len(valid_timeline) // 10)
                if window_size < 3:
                    window_size = 3

                moving_avg = valid_timeline.rolling(window=window_size, center=True).mean()

                # Find baseline (median of first 20% of samples)
                baseline_size = max(5, len(valid_timeline) // 5)
                baseline = valid_timeline.iloc[:baseline_size].median()

                # Look for sustained increase above baseline
                # Use a more sensitive threshold (20% increase from baseline)
                threshold = baseline * 0.2  # 20% increase from baseline
                target_memory = baseline + threshold

                # Find first point where moving average exceeds target and stays high
                transition_idx = None
                for idx in range(len(moving_avg)):
                    if pd.notna(moving_avg.iloc[idx]) and moving_avg.iloc[idx] > target_memory:
                        # Check if it stays high for at least 5 consecutive samples
                        end_check = min(idx + 5, len(moving_avg))
                        if all(moving_avg.iloc[idx:end_check] > target_memory):
                            transition_idx = idx
                            break

                # Alternative: use gradient detection if the above fails
                if transition_idx is None:
                    # Compute gradient (rate of change)
                    gradient = valid_timeline.diff()
                    # Look for large positive jumps (> 2000 MB increase)
                    large_jumps = gradient > 2000.0
                    if large_jumps.any():
                        transition_idx = large_jumps.idxmax()
                        # Convert Series index to positional index
                        transition_idx = valid_timeline.index.get_loc(transition_idx)

                if transition_idx is not None and transition_idx > 0:
                    transition_time = valid_timeline.index[transition_idx]

        # Apply the detected transition time
        if transition_time is not None and transition_time > 0:
            # Cap transition time at monitor duration
            if transition_time > metrics.monitor_duration:
                transition_time = metrics.monitor_duration * 0.5  # Fallback to midpoint

            metrics.rollout_phase_duration = transition_time
            metrics.training_phase_duration = metrics.monitor_duration - transition_time

            # Rollout phase metrics
            rollout_gpu01 = gpu01_df[gpu01_df["elapsed_seconds"] <= transition_time]
            rollout_gpu23 = gpu23_df[gpu23_df["elapsed_seconds"] <= transition_time]
            if len(rollout_gpu01) > 0:
                # Filter out anomalies
                rollout_gpu01_valid = rollout_gpu01[rollout_gpu01["memory_used_mb"] > 1000.0]
                if len(rollout_gpu01_valid) > 0:
                    metrics.rollout_phase_gpu01_mem_avg_mb = rollout_gpu01_valid["memory_used_mb"].mean()
            if len(rollout_gpu23) > 0:
                rollout_gpu23_valid = rollout_gpu23[rollout_gpu23["memory_used_mb"] > 1000.0]
                if len(rollout_gpu23_valid) > 0:
                    metrics.rollout_phase_gpu23_mem_avg_mb = rollout_gpu23_valid["memory_used_mb"].mean()

            # Training phase metrics
            training_gpu01 = gpu01_df[gpu01_df["elapsed_seconds"] > transition_time]
            training_gpu23 = gpu23_df[gpu23_df["elapsed_seconds"] > transition_time]
            if len(training_gpu01) > 0:
                training_gpu01_valid = training_gpu01[training_gpu01["memory_used_mb"] > 1000.0]
                if len(training_gpu01_valid) > 0:
                    metrics.training_phase_gpu01_mem_avg_mb = training_gpu01_valid["memory_used_mb"].mean()
            if len(training_gpu23) > 0:
                training_gpu23_valid = training_gpu23[training_gpu23["memory_used_mb"] > 1000.0]
                if len(training_gpu23_valid) > 0:
                    metrics.training_phase_gpu23_mem_avg_mb = training_gpu23_valid["memory_used_mb"].mean()

    # Load host data
    host_df = pd.read_csv(host_csv)
    if len(host_df) > 0:
        metrics.cpu_avg_util_pct = host_df["cpu_avg_util_percent"].mean()
        metrics.cpu_max_util_pct = host_df["cpu_avg_util_percent"].max()
        metrics.mem_avg_util_pct = host_df["mem_util_percent"].mean()
        metrics.mem_max_util_pct = host_df["mem_util_percent"].max()

    return metrics


def find_matching_monitor_dir(
    monitor_runs_dir: Path, run_id: str, timestamp: str, placement_mode: str
) -> Path | None:
    """Find the monitor directory matching the training run."""

    pattern = f"monitor-{placement_mode}-*-{run_id}-{timestamp}"
    matches = list(monitor_runs_dir.glob(pattern))
    return matches[0] if matches else None


def analyze_all_experiments(
    trainlog_dir: Path, monitor_runs_dir: Path
) -> list[dict[str, Any]]:
    """Analyze all experiments and return combined metrics."""

    results = []

    for log_file in sorted(trainlog_dir.glob("env*_mbs*.log")):
        # Skip monitor logs
        if log_file.stem.endswith(".monitor"):
            continue

        try:
            # Parse training log
            perf_metrics = parse_train_log(log_file)

            # Find matching monitor data
            monitor_dir = find_matching_monitor_dir(
                monitor_runs_dir,
                perf_metrics.run_id,
                perf_metrics.timestamp,
                perf_metrics.placement_mode,
            )

            # Parse monitor data if found
            if monitor_dir:
                resource_metrics = parse_monitor_data(
                    monitor_dir, perf_metrics.run_id, perf_metrics.placement_mode, perf_metrics
                )
            else:
                resource_metrics = ResourceMetrics(run_id=perf_metrics.run_id)
                print(f"Warning: No monitor data found for {perf_metrics.run_id}")

            # Combine metrics
            combined = {
                "run_id": perf_metrics.run_id,
                "placement_mode": perf_metrics.placement_mode,
                "total_num_envs": perf_metrics.total_num_envs,
                "micro_batch_size": perf_metrics.micro_batch_size,
                "timestamp": perf_metrics.timestamp,
                # Performance metrics
                "rollout_avg_time_s": perf_metrics.rollout_avg_time,
                "rollout_total_epochs": perf_metrics.rollout_total_epochs,
                "step_time_s": perf_metrics.step_time,
                "generate_rollouts_time_s": perf_metrics.generate_rollouts_time,
                "actor_run_training_time_s": perf_metrics.actor_run_training_time,
                "env_interact_time_s": perf_metrics.env_interact_time,
                "rollout_generate_one_epoch_time_s": perf_metrics.rollout_generate_one_epoch_time,
                # Resource metrics
                "monitor_duration_s": resource_metrics.monitor_duration,
                "gpu01_mem_avg_mb": resource_metrics.gpu01_mem_avg_mb,
                "gpu01_mem_max_mb": resource_metrics.gpu01_mem_max_mb,
                "gpu01_mem_stable_mb": resource_metrics.gpu01_mem_stable_mb,
                "gpu01_util_avg_pct": resource_metrics.gpu01_util_avg_pct,
                "gpu01_util_max_pct": resource_metrics.gpu01_util_max_pct,
                "gpu23_mem_avg_mb": resource_metrics.gpu23_mem_avg_mb,
                "gpu23_mem_max_mb": resource_metrics.gpu23_mem_max_mb,
                "gpu23_mem_stable_mb": resource_metrics.gpu23_mem_stable_mb,
                "gpu23_util_avg_pct": resource_metrics.gpu23_util_avg_pct,
                "gpu23_util_max_pct": resource_metrics.gpu23_util_max_pct,
                "rollout_phase_gpu01_mem_avg_mb": resource_metrics.rollout_phase_gpu01_mem_avg_mb,
                "rollout_phase_gpu23_mem_avg_mb": resource_metrics.rollout_phase_gpu23_mem_avg_mb,
                "rollout_phase_duration_s": resource_metrics.rollout_phase_duration,
                "training_phase_gpu01_mem_avg_mb": resource_metrics.training_phase_gpu01_mem_avg_mb,
                "training_phase_gpu23_mem_avg_mb": resource_metrics.training_phase_gpu23_mem_avg_mb,
                "training_phase_duration_s": resource_metrics.training_phase_duration,
                "cpu_avg_util_pct": resource_metrics.cpu_avg_util_pct,
                "cpu_max_util_pct": resource_metrics.cpu_max_util_pct,
                "mem_avg_util_pct": resource_metrics.mem_avg_util_pct,
                "mem_max_util_pct": resource_metrics.mem_max_util_pct,
            }
            results.append(combined)

        except Exception as e:
            print(f"Error processing {log_file.name}: {e}")
            continue

    return results


def generate_html_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Generate an interactive HTML report with sortable tables and charts."""

    df = pd.DataFrame(results)

    # Sort by placement_mode, total_num_envs, micro_batch_size
    df = df.sort_values(["placement_mode", "total_num_envs", "micro_batch_size"])

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Analysis Report</title>
    <style>
        :root {
            --bg: #f5f5f5;
            --panel: #ffffff;
            --border: #ddd;
            --header-bg: #2c3e50;
            --header-text: #ecf0f1;
            --accent: #3498db;
            --text: #333;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg);
            color: var(--text);
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        h1 {
            color: var(--header-bg);
            margin-bottom: 10px;
        }
        .summary {
            background: var(--panel);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary h2 {
            margin-top: 0;
            color: var(--accent);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--panel);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 30px;
        }
        th {
            background: var(--header-bg);
            color: var(--header-text);
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            position: sticky;
            top: 0;
            cursor: pointer;
            user-select: none;
        }
        th:hover {
            background: #34495e;
        }
        td {
            padding: 10px 8px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .numeric {
            text-align: right;
            font-family: 'Monaco', 'Courier New', monospace;
        }
        .section {
            background: var(--panel);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0;
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 10px;
        }
        .filter-group {
            margin-bottom: 15px;
        }
        .filter-group label {
            font-weight: 600;
            margin-right: 10px;
        }
        select {
            padding: 6px 10px;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 14px;
        }
        .highlight-best {
            background: #d4edda !important;
            font-weight: 600;
        }
        .highlight-worst {
            background: #f8d7da !important;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Training Experiment Analysis Report</h1>
        <p>Comprehensive analysis of performance and resource utilization across different parameter configurations</p>

        <div class="summary">
            <h2>📊 Summary</h2>
            <p><strong>Total Experiments:</strong> """ + str(len(df)) + """</p>
            <p><strong>Placement Modes:</strong> """ + ", ".join(df["placement_mode"].unique()) + """</p>
            <p><strong>Environment Counts:</strong> """ + ", ".join(map(str, sorted(df["total_num_envs"].unique()))) + """</p>
            <p><strong>Micro Batch Sizes:</strong> """ + ", ".join(map(str, sorted(df["micro_batch_size"].unique()))) + """</p>
        </div>

        <div class="section">
            <h2>⚡ Performance Metrics</h2>
            <table id="perfTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('perfTable', 0)">Run ID</th>
                        <th onclick="sortTable('perfTable', 1)">Mode</th>
                        <th onclick="sortTable('perfTable', 2)" class="numeric">Envs</th>
                        <th onclick="sortTable('perfTable', 3)" class="numeric">MBS</th>
                        <th onclick="sortTable('perfTable', 4)" class="numeric">Rollout Avg (s/epoch)</th>
                        <th onclick="sortTable('perfTable', 5)" class="numeric">Step Time (s)</th>
                        <th onclick="sortTable('perfTable', 6)" class="numeric">Generate Rollouts (s)</th>
                        <th onclick="sortTable('perfTable', 7)" class="numeric">Actor Training (s)</th>
                        <th onclick="sortTable('perfTable', 8)" class="numeric">Env Interact (s)</th>
                    </tr>
                </thead>
                <tbody>
"""

    for _, row in df.iterrows():
        rollout_avg = f"{row['rollout_avg_time_s']:.2f}" if pd.notna(row['rollout_avg_time_s']) else 'N/A'
        step_time = f"{row['step_time_s']:.2f}" if pd.notna(row['step_time_s']) else 'N/A'
        gen_rollouts = f"{row['generate_rollouts_time_s']:.2f}" if pd.notna(row['generate_rollouts_time_s']) else 'N/A'
        actor_train = f"{row['actor_run_training_time_s']:.2f}" if pd.notna(row['actor_run_training_time_s']) else 'N/A'
        env_interact = f"{row['env_interact_time_s']:.2f}" if pd.notna(row['env_interact_time_s']) else 'N/A'

        html_content += f"""                    <tr>
                        <td>{row['run_id']}</td>
                        <td>{row['placement_mode']}</td>
                        <td class="numeric">{row['total_num_envs']}</td>
                        <td class="numeric">{row['micro_batch_size']}</td>
                        <td class="numeric">{rollout_avg}</td>
                        <td class="numeric">{step_time}</td>
                        <td class="numeric">{gen_rollouts}</td>
                        <td class="numeric">{actor_train}</td>
                        <td class="numeric">{env_interact}</td>
                    </tr>
"""

    html_content += """                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>💾 GPU Memory Usage (MB)</h2>
            <table id="gpuMemTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('gpuMemTable', 0)">Run ID</th>
                        <th onclick="sortTable('gpuMemTable', 1)">Mode</th>
                        <th onclick="sortTable('gpuMemTable', 2)" class="numeric">Envs</th>
                        <th onclick="sortTable('gpuMemTable', 3)" class="numeric">MBS</th>
                        <th onclick="sortTable('gpuMemTable', 4)" class="numeric">GPU 0-1 Avg</th>
                        <th onclick="sortTable('gpuMemTable', 5)" class="numeric">GPU 0-1 Max</th>
                        <th onclick="sortTable('gpuMemTable', 6)" class="numeric">GPU 0-1 Stable</th>
                        <th onclick="sortTable('gpuMemTable', 7)" class="numeric">GPU 2-3 Avg</th>
                        <th onclick="sortTable('gpuMemTable', 8)" class="numeric">GPU 2-3 Max</th>
                        <th onclick="sortTable('gpuMemTable', 9)" class="numeric">GPU 2-3 Stable</th>
                    </tr>
                </thead>
                <tbody>
"""

    for _, row in df.iterrows():
        gpu01_avg = f"{row['gpu01_mem_avg_mb']:.0f}" if pd.notna(row['gpu01_mem_avg_mb']) else 'N/A'
        gpu01_max = f"{row['gpu01_mem_max_mb']:.0f}" if pd.notna(row['gpu01_mem_max_mb']) else 'N/A'
        gpu01_stable = f"{row['gpu01_mem_stable_mb']:.0f}" if pd.notna(row['gpu01_mem_stable_mb']) else 'N/A'
        gpu23_avg = f"{row['gpu23_mem_avg_mb']:.0f}" if pd.notna(row['gpu23_mem_avg_mb']) else 'N/A'
        gpu23_max = f"{row['gpu23_mem_max_mb']:.0f}" if pd.notna(row['gpu23_mem_max_mb']) else 'N/A'
        gpu23_stable = f"{row['gpu23_mem_stable_mb']:.0f}" if pd.notna(row['gpu23_mem_stable_mb']) else 'N/A'

        html_content += f"""                    <tr>
                        <td>{row['run_id']}</td>
                        <td>{row['placement_mode']}</td>
                        <td class="numeric">{row['total_num_envs']}</td>
                        <td class="numeric">{row['micro_batch_size']}</td>
                        <td class="numeric">{gpu01_avg}</td>
                        <td class="numeric">{gpu01_max}</td>
                        <td class="numeric">{gpu01_stable}</td>
                        <td class="numeric">{gpu23_avg}</td>
                        <td class="numeric">{gpu23_max}</td>
                        <td class="numeric">{gpu23_stable}</td>
                    </tr>
"""

    html_content += """                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>⚙️ GPU Utilization & Host Metrics</h2>
            <table id="utilTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('utilTable', 0)">Run ID</th>
                        <th onclick="sortTable('utilTable', 1)">Mode</th>
                        <th onclick="sortTable('utilTable', 2)" class="numeric">Envs</th>
                        <th onclick="sortTable('utilTable', 3)" class="numeric">MBS</th>
                        <th onclick="sortTable('utilTable', 4)" class="numeric">GPU 0-1 Util Avg (%)</th>
                        <th onclick="sortTable('utilTable', 5)" class="numeric">GPU 0-1 Util Max (%)</th>
                        <th onclick="sortTable('utilTable', 6)" class="numeric">GPU 2-3 Util Avg (%)</th>
                        <th onclick="sortTable('utilTable', 7)" class="numeric">GPU 2-3 Util Max (%)</th>
                        <th onclick="sortTable('utilTable', 8)" class="numeric">CPU Avg (%)</th>
                        <th onclick="sortTable('utilTable', 9)" class="numeric">Host Mem Avg (%)</th>
                        <th onclick="sortTable('utilTable', 10)" class="numeric">Monitor Duration (s)</th>
                    </tr>
                </thead>
                <tbody>
"""

    for _, row in df.iterrows():
        gpu01_util_avg = f"{row['gpu01_util_avg_pct']:.1f}" if pd.notna(row['gpu01_util_avg_pct']) else 'N/A'
        gpu01_util_max = f"{row['gpu01_util_max_pct']:.1f}" if pd.notna(row['gpu01_util_max_pct']) else 'N/A'
        gpu23_util_avg = f"{row['gpu23_util_avg_pct']:.1f}" if pd.notna(row['gpu23_util_avg_pct']) else 'N/A'
        gpu23_util_max = f"{row['gpu23_util_max_pct']:.1f}" if pd.notna(row['gpu23_util_max_pct']) else 'N/A'
        cpu_avg = f"{row['cpu_avg_util_pct']:.1f}" if pd.notna(row['cpu_avg_util_pct']) else 'N/A'
        mem_avg = f"{row['mem_avg_util_pct']:.1f}" if pd.notna(row['mem_avg_util_pct']) else 'N/A'
        monitor_dur = f"{row['monitor_duration_s']:.0f}" if pd.notna(row['monitor_duration_s']) else 'N/A'

        html_content += f"""                    <tr>
                        <td>{row['run_id']}</td>
                        <td>{row['placement_mode']}</td>
                        <td class="numeric">{row['total_num_envs']}</td>
                        <td class="numeric">{row['micro_batch_size']}</td>
                        <td class="numeric">{gpu01_util_avg}</td>
                        <td class="numeric">{gpu01_util_max}</td>
                        <td class="numeric">{gpu23_util_avg}</td>
                        <td class="numeric">{gpu23_util_max}</td>
                        <td class="numeric">{cpu_avg}</td>
                        <td class="numeric">{mem_avg}</td>
                        <td class="numeric">{monitor_dur}</td>
                    </tr>
"""

    html_content += """                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>🔄 Phase-wise Analysis (Rollout vs Training)</h2>
            <table id="phaseTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('phaseTable', 0)">Run ID</th>
                        <th onclick="sortTable('phaseTable', 1)">Mode</th>
                        <th onclick="sortTable('phaseTable', 2)" class="numeric">Envs</th>
                        <th onclick="sortTable('phaseTable', 3)" class="numeric">MBS</th>
                        <th onclick="sortTable('phaseTable', 4)" class="numeric">Rollout Duration (s)</th>
                        <th onclick="sortTable('phaseTable', 5)" class="numeric">Rollout GPU 0-1 Mem (MB)</th>
                        <th onclick="sortTable('phaseTable', 6)" class="numeric">Rollout GPU 2-3 Mem (MB)</th>
                        <th onclick="sortTable('phaseTable', 7)" class="numeric">Training Duration (s)</th>
                        <th onclick="sortTable('phaseTable', 8)" class="numeric">Training GPU 0-1 Mem (MB)</th>
                        <th onclick="sortTable('phaseTable', 9)" class="numeric">Training GPU 2-3 Mem (MB)</th>
                    </tr>
                </thead>
                <tbody>
"""

    for _, row in df.iterrows():
        rollout_dur = f"{row['rollout_phase_duration_s']:.0f}" if pd.notna(row['rollout_phase_duration_s']) else 'N/A'
        rollout_gpu01 = f"{row['rollout_phase_gpu01_mem_avg_mb']:.0f}" if pd.notna(row['rollout_phase_gpu01_mem_avg_mb']) else 'N/A'
        rollout_gpu23 = f"{row['rollout_phase_gpu23_mem_avg_mb']:.0f}" if pd.notna(row['rollout_phase_gpu23_mem_avg_mb']) else 'N/A'
        training_dur = f"{row['training_phase_duration_s']:.0f}" if pd.notna(row['training_phase_duration_s']) else 'N/A'
        training_gpu01 = f"{row['training_phase_gpu01_mem_avg_mb']:.0f}" if pd.notna(row['training_phase_gpu01_mem_avg_mb']) else 'N/A'
        training_gpu23 = f"{row['training_phase_gpu23_mem_avg_mb']:.0f}" if pd.notna(row['training_phase_gpu23_mem_avg_mb']) else 'N/A'

        html_content += f"""                    <tr>
                        <td>{row['run_id']}</td>
                        <td>{row['placement_mode']}</td>
                        <td class="numeric">{row['total_num_envs']}</td>
                        <td class="numeric">{row['micro_batch_size']}</td>
                        <td class="numeric">{rollout_dur}</td>
                        <td class="numeric">{rollout_gpu01}</td>
                        <td class="numeric">{rollout_gpu23}</td>
                        <td class="numeric">{training_dur}</td>
                        <td class="numeric">{training_gpu01}</td>
                        <td class="numeric">{training_gpu23}</td>
                    </tr>
"""

    html_content += """                </tbody>
            </table>
        </div>
    </div>

    <script>
        function sortTable(tableId, columnIndex) {
            const table = document.getElementById(tableId);
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Determine sort direction
            const currentSort = table.dataset.sortColumn;
            const currentDir = table.dataset.sortDir || 'asc';
            const newDir = (currentSort == columnIndex && currentDir === 'asc') ? 'desc' : 'asc';

            rows.sort((a, b) => {
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();

                // Handle N/A values
                if (aVal === 'N/A') return 1;
                if (bVal === 'N/A') return -1;

                // Try numeric comparison
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);

                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return newDir === 'asc' ? aNum - bNum : bNum - aNum;
                }

                // String comparison
                return newDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            });

            // Clear and re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));

            // Store sort state
            table.dataset.sortColumn = columnIndex;
            table.dataset.sortDir = newDir;
        }
    </script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    print(f"HTML report generated: {output_path}")


def generate_csv_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Generate a CSV report for further analysis."""

    df = pd.DataFrame(results)
    df = df.sort_values(["placement_mode", "total_num_envs", "micro_batch_size"])
    df.to_csv(output_path, index=False)
    print(f"CSV report generated: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze training experiments and generate reports"
    )
    parser.add_argument(
        "--trainlog-dir",
        type=Path,
        default=Path("yunzhe/trainlog"),
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--monitor-dir",
        type=Path,
        default=Path("yunzhe/monitor/runs"),
        help="Directory containing monitor data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_analysis.html"),
        help="Output HTML report path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("experiment_analysis.csv"),
        help="Output CSV report path",
    )

    args = parser.parse_args()

    print(f"Analyzing experiments from {args.trainlog_dir}")
    print(f"Using monitor data from {args.monitor_dir}")

    results = analyze_all_experiments(args.trainlog_dir, args.monitor_dir)

    if not results:
        print("No experiments found!")
        return 1

    print(f"\nAnalyzed {len(results)} experiments")

    generate_html_report(results, args.output)
    generate_csv_report(results, args.csv)

    print(f"\nReports generated:")
    print(f"  HTML: {args.output}")
    print(f"  CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
