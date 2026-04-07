# 断点续传功能说明

## 功能概述

脚本会自动读取 `yunzhe/monitor/runs/completed_experiments.csv`，跳过已完成的实验。

## 已完成实验记录

- **配置文件**: `yunzhe/monitor/runs/completed_experiments.csv`
- **格式**: CSV，包含 `env` 和 `mbs` 两列
- **当前状态**: 30 个已完成实验

```csv
env,mbs
2,1
2,2
...
```

## 使用方法

### 1. 查看当前状态

```bash
cat yunzhe/monitor/runs/completed_experiments.csv
cat yunzhe/monitor/runs/experiments_status.md
```

### 2. 运行实验（自动跳过已完成）

```bash
cd yunzhe/multirun

# 运行所有实验（会自动跳过已完成的30个）
bash run_sweep_split.sh

# 或指定特定配置
ENV_LIST="16,96" MBS_LIST="1,2,4,8,16,32,64,128,256" bash run_sweep_split.sh
```

### 3. 查看运行日志

脚本启动时会显示：
```
[matrix] loading completed experiments from: /path/to/completed_experiments.csv
[matrix] loaded 30 completed experiments
```

每个已完成的实验会输出：
```
[matrix] skip env2_mbs1 (already completed)
```

## 当前实验状态

| 配置 | 状态 | 数量 |
|------|------|------|
| 已完成 | ✅ | 30/63 (47.6%) |
| 待运行 | ⭕ | 33/63 (52.4%) |

### 待运行的实验

**env=2,4,8,32,64** (各需要 3 个):
- mbs=64, 128, 256

**env=16** (需要全部 9 个):
- mbs=1,2,4,8,16,32,64,128,256

**env=96** (需要全部 9 个):
- mbs=1,2,4,8,16,32,64,128,256

## 手动更新已完成列表

如果需要手动标记某些实验为已完成：

```bash
# 编辑配置文件
vim yunzhe/monitor/runs/completed_experiments.csv

# 添加新的配置，例如：
# 64,64
# 96,1
```

## 注意事项

1. **不要删除** `completed_experiments.csv` 文件，否则会重新运行所有实验
2. 文件格式必须严格遵循 CSV 格式：`env,mbs`（无空格）
3. 已完成的实验会被跳过，不会重新运行
4. 跳过的实验不会出现在新的 `manifest.csv` 中
