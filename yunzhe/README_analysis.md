# 实验分析工具

## 功能概述

`analyze_experiments.py` 脚本用于整理和分析不同参数配置下的训练性能和资源占用情况。

## 使用方法

### 基本用法

```bash
python yunzhe/analyze_experiments.py
```

### 自定义路径

```bash
python yunzhe/analyze_experiments.py \
    --trainlog-dir yunzhe/trainlog \
    --monitor-dir yunzhe/monitor/runs \
    --output analysis_report.html \
    --csv analysis_data.csv
```

## 输出文件

1. **HTML 报告** (`experiment_analysis.html`)
   - 交互式可视化报告
   - 可点击表头进行排序
   - 包含多个视图：性能指标、GPU 内存、GPU 利用率、阶段分析

2. **CSV 数据** (`experiment_analysis.csv`)
   - 所有指标的原始数据
   - 可用于进一步分析（Excel、Python、R 等）

## 提取的指标

### 性能指标（从训练日志）

- **rollout_avg_time_s**: Rollout 过程的平均时间（秒/epoch）
- **step_time_s**: 每个训练 step 的总时间
- **generate_rollouts_time_s**: 生成 rollout 的时间
- **actor_run_training_time_s**: Actor 训练的时间
- **env_interact_time_s**: 环境交互的时间

### 资源占用指标（从监控数据）

#### GPU 内存（按 GPU 组）

- **gpu01_mem_avg_mb**: GPU 0-1 平均显存占用（rollout/env 在 split 模式）
- **gpu01_mem_max_mb**: GPU 0-1 峰值显存占用
- **gpu01_mem_stable_mb**: GPU 0-1 稳定显存占用（后 50% 样本平均）
- **gpu23_mem_avg_mb**: GPU 2-3 平均显存占用（actor 在 split 模式）
- **gpu23_mem_max_mb**: GPU 2-3 峰值显存占用
- **gpu23_mem_stable_mb**: GPU 2-3 稳定显存占用

#### GPU 利用率

- **gpu01_util_avg_pct**: GPU 0-1 平均利用率（%）
- **gpu01_util_max_pct**: GPU 0-1 峰值利用率（%）
- **gpu23_util_avg_pct**: GPU 2-3 平均利用率（%）
- **gpu23_util_max_pct**: GPU 2-3 峰值利用率（%）

#### 阶段分析（Rollout vs Training）

- **rollout_phase_duration_s**: Rollout 阶段持续时间
- **rollout_phase_gpu01_mem_avg_mb**: Rollout 阶段 GPU 0-1 平均显存
- **rollout_phase_gpu23_mem_avg_mb**: Rollout 阶段 GPU 2-3 平均显存
- **training_phase_duration_s**: Training 阶段持续时间
- **training_phase_gpu01_mem_avg_mb**: Training 阶段 GPU 0-1 平均显存
- **training_phase_gpu23_mem_avg_mb**: Training 阶段 GPU 2-3 平均显存

#### 主机资源

- **cpu_avg_util_pct**: CPU 平均利用率（%）
- **cpu_max_util_pct**: CPU 峰值利用率（%）
- **mem_avg_util_pct**: 系统内存平均利用率（%）
- **mem_max_util_pct**: 系统内存峰值利用率（%）
- **monitor_duration_s**: 监控总时长（环境启动时间）

## HTML 报告功能

### 可交互表格

- 点击表头可对该列进行升序/降序排序
- 数值列右对齐，便于比较
- 悬停行高亮显示

### 多视图分析

1. **性能指标表**：横向比较训练速度
2. **GPU 内存表**：比较显存占用情况
3. **GPU 利用率 & 主机指标表**：评估资源利用效率
4. **阶段分析表**：区分 Rollout 和 Training 阶段的资源使用

## 数据匹配逻辑

脚本通过以下规则匹配训练日志和监控数据：

- 日志文件名格式：`env{N}_mbs{M}-{timestamp}.log`
- 监控目录格式：`monitor-{mode}-*-env{N}_mbs{M}-{timestamp}`
- 自动识别 placement_mode（split 或 all）

## 典型分析场景

### 1. 找出最优配置

在 HTML 报告中：
- 按 `Step Time` 升序排序，找出最快的配置
- 查看对应的 `total_num_envs` 和 `micro_batch_size`

### 2. 分析 OOM 边界

- 按 `total_num_envs` 和 `micro_batch_size` 升序排序
- 查看 `gpu23_mem_max_mb`（actor GPU）
- L40 GPU 显存上限：46068 MB

### 3. 对比 Split 和 All 模式

- 按 `placement_mode` 分组查看
- 比较相同参数下的性能和资源占用

### 4. 发现瓶颈

- 比较 `generate_rollouts_time_s` 和 `actor_run_training_time_s`
- 看哪个占比更大，针对性优化

## 示例：在 CSV 中进行分析

```python
import pandas as pd

# 加载数据
df = pd.read_csv('yunzhe/experiment_analysis.csv')

# 找出 step time 最小的配置
best_config = df.loc[df['step_time_s'].idxmin()]
print(f"最优配置: env={best_config['total_num_envs']}, mbs={best_config['micro_batch_size']}")
print(f"Step Time: {best_config['step_time_s']:.2f}s")

# 分析显存随环境数量的增长
import matplotlib.pyplot as plt

split_data = df[df['placement_mode'] == 'split']
plt.plot(split_data['total_num_envs'], split_data['gpu01_mem_max_mb'], label='GPU 0-1 (rollout)')
plt.plot(split_data['total_num_envs'], split_data['gpu23_mem_max_mb'], label='GPU 2-3 (actor)')
plt.xlabel('Total Num Envs')
plt.ylabel('Peak Memory (MB)')
plt.legend()
plt.savefig('memory_scaling.png')
```

## 故障排查

### 缺少监控数据警告

```
Warning: No monitor data found for env64_mbs32
```

**原因**：训练日志存在，但对应的监控目录未找到  
**影响**：该实验的资源指标将显示为 N/A  
**解决**：检查 monitor_runs_dir 路径是否正确

### 日志解析错误

```
Error processing env64_mbs32-20260401-182308.log: Cannot parse filename
```

**原因**：日志文件命名不符合预期格式  
**影响**：该日志将被跳过  
**解决**：确保日志文件名格式为 `env{N}_mbs{M}-{timestamp}.log`

## 扩展脚本

如果需要添加新的指标，修改以下函数：

1. `parse_train_log()` - 添加日志解析规则
2. `parse_monitor_data()` - 添加监控数据处理
3. `PerformanceMetrics` / `ResourceMetrics` - 添加字段定义
4. `generate_html_report()` - 更新 HTML 表格

## 依赖

- Python 3.7+
- pandas

安装依赖：
```bash
pip install pandas
```
