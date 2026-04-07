# 批量实验脚本使用说明

## 功能特性

✅ **断点续传**：自动跳过已完成的实验，支持中断后继续  
✅ **自动监控**：每个实验自动启动 GPU/CPU/内存监控  
✅ **日志管理**：训练日志和监控日志自动关联  
✅ **状态追踪**：manifest.csv 记录所有实验的状态  

---

## 快速开始

### 基本用法（使用默认配置）

```bash
cd /home/yunzhe/RLinf/RLinf/yunzhe/multirun
bash run_sweep_split.sh
```

**当前配置：**
- 环境数量：2, 4, 8, 16, 32, 64, 96
- MBS：1, 2, 4, 8, 16, 32, 64, 128, 256
- 总实验数：**7 × 9 = 63 个**

---

## 断点续传验证

### 验证断点续传功能

**测试步骤：**

1. **启动测试（小规模）**
```bash
ENV_LIST="2,4" MBS_LIST="1,2" MAX_STEPS=10 bash run_sweep_split.sh
```

2. **等待第1个实验完成后按 Ctrl+C 中断**

3. **检查 manifest.csv**
```bash
# 查看最新的 multirun 目录
ls -lt logs/ | head -5

# 查看 manifest
cat logs/multirun-split-libero_spatial_ppo_openpi_pi05-*/results/manifest.csv
```

应该看到类似：
```csv
run_id,placement_mode,total_num_envs,micro_batch_size,...,status
env2_mbs1,split,2,1,256,...,ok
env2_mbs2,split,2,2,256,...,interrupted
```

4. **重新运行（断点续传）**
```bash
ENV_LIST="2,4" MBS_LIST="1,2" MAX_STEPS=10 bash run_sweep_split.sh
```

**预期行为：**
```
[matrix] skip env2_mbs1 (already completed)
[matrix] start env2_mbs2
```

✅ **断点续传工作正常！**

---

## 配置参数

### 通过环境变量自定义

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ENV_LIST` | `auto` (2,4,8,16,32,64,96) | 环境数量 |
| `MBS_LIST` | `1,2,4,8,16,32,64,128,256` | MBS 值 |
| `MAX_STEPS` | `50` | 训练步数 |
| `RESUME_ENABLED` | `1` | 断点续传（1=启用） |

### 使用示例

```bash
# 只测试大 MBS
MBS_LIST="64,128,256" bash run_sweep_split.sh

# 只测试 env=96
ENV_LIST="96" bash run_sweep_split.sh

# 快速测试
ENV_LIST="2,4" MBS_LIST="1,2" MAX_STEPS=10 bash run_sweep_split.sh

# 禁用断点续传（强制重跑）
RESUME_ENABLED=0 bash run_sweep_split.sh
```

---

## 输出文件

```
logs/multirun-split-libero_spatial_ppo_openpi_pi05-TIMESTAMP/
├── results/
│   └── manifest.csv              # 实验状态清单 ⭐
├── train/
│   ├── env2_mbs1-TIMESTAMP.log          # 训练日志
│   ├── env2_mbs1-TIMESTAMP.monitor.log  # 监控日志
│   └── ...
└── jobs/
    └── env2_mbs1-TIMESTAMP/tensorboard/
```

### manifest.csv 状态

| 状态 | 含义 | 下次运行 |
|------|------|---------|
| `ok` | 成功完成 | ✅ 跳过 |
| `failed` | 失败 | 🔄 重跑 |
| `interrupted` | 中断 | 🔄 重跑 |

---

## 监控数据位置

```
yunzhe/monitor/runs/
└── monitor-split-CONFIG-envN_mbsM-TIMESTAMP/
    ├── gpu_metrics.csv    # GPU 时序数据
    ├── host_metrics.csv   # CPU/内存时序数据
    ├── runtime.log
    └── report.html        # 可视化报告
```

---

## 使用场景

### 场景 1：完整测试（默认）

```bash
bash run_sweep_split.sh
```
- 63 个实验
- 预计 5-10 小时

### 场景 2：测试大 MBS

```bash
MBS_LIST="64,128,256" bash run_sweep_split.sh
```
- 7 × 3 = 21 个实验
- 预计 2-3 小时

### 场景 3：测试 env=96（继续）

```bash
ENV_LIST="96" bash run_sweep_split.sh
```
- 1 × 9 = 9 个实验
- 预计 1-2 小时
- ⚠️ 某些 MBS 可能 OOM

---

## 中断与恢复

### 中断

```bash
# 按 Ctrl+C
^C
[matrix] interrupted by user. partial results kept at: logs/...
```

### 恢复

```bash
# 再次运行相同命令
bash run_sweep_split.sh
```

✅ 自动跳过已完成的实验

---

## 分析结果

```bash
python yunzhe/analyze_experiments.py \
    --trainlog-dir logs/multirun-*/train \
    --monitor-dir yunzhe/monitor/runs \
    --output yunzhe/experiment_analysis.html \
    --csv yunzhe/experiment_analysis.csv
```

---

## 推荐工作流

1. **小规模验证**
```bash
ENV_LIST="2,4" MBS_LIST="1,2" MAX_STEPS=10 bash run_sweep_split.sh
```

2. **完整运行（后台）**
```bash
# 使用 tmux
tmux new -s exp
bash run_sweep_split.sh
# Ctrl+B D 分离

# 或使用 nohup
nohup bash run_sweep_split.sh > sweep.log 2>&1 &
```

3. **查看进度**
```bash
# 查看 manifest
tail -f logs/multirun-*/results/manifest.csv

# 统计完成数
grep ",ok$" logs/multirun-*/results/manifest.csv | wc -l
```

4. **断点续传**
```bash
# 如果中断，重新运行即可
bash run_sweep_split.sh
```

5. **分析结果**
```bash
python yunzhe/analyze_experiments.py
```

---

## 故障排查

### 修改某个实验状态

```bash
# 编辑 manifest.csv
vim logs/multirun-*/results/manifest.csv

# 将状态从 ok 改为 failed，下次会重跑
# 或删除该行
```

### 强制重跑所有实验

```bash
RESUME_ENABLED=0 bash run_sweep_split.sh
# 或删除 logs 目录
```

---

## 配置已修改项

✅ MBS 列表：添加了 64, 128, 256  
✅ ENV 列表：包含 96（测试 OOM 边界）  
✅ Checkpoint：已关闭保存（save_interval=-1）  
✅ Train Video：已关闭（env/libero_spatial.yaml）  
✅ Eval Video：保持开启  

