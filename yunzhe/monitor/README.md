# System Monitor

一个轻量的本地监控脚本，用于按固定间隔采样：

- 所有 GPU 的利用率和显存占用
- 所有 CPU 核心的利用率
- 系统内存使用情况

采样结果会保存为精简日志，并在结束后生成一份静态 HTML 可视化报告。

## 功能

- 默认每秒采样一次
- GPU 数据直接来自 `nvidia-smi`
- CPU 和内存数据来自 Linux 的 `/proc`
- 输出 `CSV + runtime.log + report.html`
- 标准输出尽量精简，只显示开始和结束信息

## 运行方式

先在 monitor 目录创建本地虚拟环境（只做一次）：

```bash
cd /home/yunzhe/RLinf/RLinf/yunzhe/monitor
python3 -m venv .venv
source .venv/bin/activate
```

然后在仓库根目录执行：

```bash
/home/yunzhe/RLinf/RLinf/yunzhe/monitor/.venv/bin/python -m yunzhe.monitor --duration 60
```

注意：上面这条 `-m yunzhe.monitor` 必须在仓库根目录运行。

如果你当前就在 `yunzhe/monitor` 目录，可以这样执行：

```bash
cd /home/yunzhe/RLinf/RLinf/yunzhe/monitor
PYTHONPATH=/home/yunzhe/RLinf/RLinf /home/yunzhe/RLinf/RLinf/yunzhe/monitor/.venv/bin/python -m yunzhe.monitor --duration 60
```

也可以指定采样次数：

```bash
/home/yunzhe/RLinf/RLinf/yunzhe/monitor/.venv/bin/python -m yunzhe.monitor --max-samples 30
```

指定输出目录和运行名：

```bash
/home/yunzhe/RLinf/RLinf/yunzhe/monitor/.venv/bin/python -m yunzhe.monitor \
  --interval 1 \
  --max-samples 120 \
  --output-dir /home/yunzhe/RLinf/RLinf/yunzhe/monitor/runs \
  --name train-monitor
```

## 参数

- `--interval`: 采样间隔，单位秒，默认 `1.0`
- `--duration`: 总运行时长，单位秒
- `--max-samples`: 最大采样次数
- `--output-dir`: 输出目录，默认在当前目录下的 `runs/`
- `--name`: 本次运行的目录名

说明：

- 如果 `--duration` 和 `--max-samples` 都不传，脚本默认采样 `60` 次
- 按 `Ctrl+C` 中断时，已采集的数据仍会落盘，并生成报告

## 输出文件

每次运行会生成一个独立目录，通常包含：

- `gpu_metrics.csv`: 每张 GPU 每次采样一行
- `host_metrics.csv`: 每次采样一行，包含系统内存、CPU 平均利用率和每个 CPU 核心利用率
- `runtime.log`: 运行日志
- `report.html`: 静态可视化报告

## CSV 字段

`gpu_metrics.csv` 主要字段：

- `timestamp`
- `elapsed_seconds`
- `gpu_index`
- `gpu_name`
- `gpu_util_percent`
- `memory_used_mb`
- `memory_total_mb`
- `memory_util_percent`

`host_metrics.csv` 主要字段：

- `timestamp`
- `elapsed_seconds`
- `mem_used_mb`
- `mem_total_mb`
- `mem_available_mb`
- `mem_util_percent`
- `cpu_avg_util_percent`
- `cpu0_util_percent`, `cpu1_util_percent`, ...

## 环境要求

- Linux
- `nvidia-smi` 可执行
- NVIDIA 驱动正常工作
- Python 可运行当前仓库中的 `yunzhe.monitor`

如果机器上没有 GPU 或者 `nvidia-smi` 不可用，脚本会报错退出。

## 报告内容

`report.html` 默认包含：

- GPU 利用率折线图
- GPU 显存占用比例折线图
- CPU 平均利用率折线图
- 系统内存占用比例折线图
- CPU 热力图

直接用浏览器打开即可查看。

## 示例

快速检查 10 秒：

```bash
/home/yunzhe/RLinf/RLinf/yunzhe/monitor/.venv/bin/python -m yunzhe.monitor --duration 10 --name quick-check
```

高频短采样：

```bash
/home/yunzhe/RLinf/RLinf/yunzhe/monitor/.venv/bin/python -m yunzhe.monitor --interval 0.2 --max-samples 50 --name burst-check
```

## 代码位置

- 入口: [yunzhe/monitor/__main__.py](/home/yunzhe/RLinf/RLinf/yunzhe/monitor/__main__.py)
- 主逻辑: [yunzhe/monitor/monitor.py](/home/yunzhe/RLinf/RLinf/yunzhe/monitor/monitor.py)
- 单测: [tests/unit_tests/test_system_monitor.py](/home/yunzhe/RLinf/RLinf/tests/unit_tests/test_system_monitor.py)