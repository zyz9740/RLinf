# Stable 显存计算方法说明

## 一、计算公式

```python
stable_start = len(gpu_df) // 2
gpu_mem_stable_mb = gpu_df.iloc[stable_start:]["memory_used_mb"].mean()
```

**Stable 显存 = 监控数据后 50% 样本的平均值**

---

## 二、为什么要取后 50%？

### 实际数据示例（env=2, mbs=1）

| 统计范围 | 平均值 | 最小值 | 最大值 | 标准差 |
|---------|--------|--------|--------|--------|
| **全程** | 5863.6 MB | 19 MB | 16528 MB | 4219.6 MB |
| **前半段** | 3126.8 MB | 19 MB | 16528 MB | **4545.4 MB** ⚠️ |
| **后半段** | **8600.5 MB** ✓ | 8097 MB | 8639 MB | **140.1 MB** ✓ |

### 关键发现

1. **前半段波动巨大**（std = 4545 MB）
   - 最小值 19 MB → 峰值 16528 MB
   - 训练启动期：模型加载、环境初始化、缓冲区分配

2. **后半段极其稳定**（std = 140 MB）
   - 波动范围：8097 - 8639 MB（仅 6% 变化）
   - 训练稳定期：显存分配已固定

3. **前后差异达 93%**
   - 如果用全程平均（5863 MB），会严重低估稳定期占用
   - 如果用峰值（16528 MB），包含了启动期的异常尖峰

---

## 三、时间线演示

### 显存占用随时间变化（env=2, mbs=1）

```
时刻          显存占用        阶段
---------------------------------------------
0s            19 MB          监控启动
10s           20 MB          系统初始化
...
180s          8600 MB        ← Rollout 开始
240s          8639 MB        
300s          8639 MB        稳定期
310s          8639 MB        ← Training 开始
360s          8639 MB        
...
530s          8639 MB        
550s          8639 MB        训练结束
---------------------------------------------
              ↑
              后 50% 从这里开始计算
              （第 56/112 个样本）
```

### 阶段划分

| 阶段 | 时间范围 | 显存特征 | 是否纳入 stable |
|------|---------|---------|---------------|
| 🔵 启动期 | 0-180s | 19-8600 MB（剧烈波动） | ❌ 排除 |
| 🟢 稳定期 | 180-550s | 8600-8639 MB（稳定） | ✓ 计入 |

---

## 四、与其他指标的对比

### 以 env=2, mbs=1 为例

| 指标 | 值 | 含义 | 适用场景 |
|------|-----|------|---------|
| **Min** | 19 MB | 监控启动时的最小值 | 无实际意义 |
| **Avg** | 5863 MB | 全程平均（含启动期） | ❌ 低估真实占用 |
| **Stable** | **8600 MB** ✓ | 稳定期平均 | ✓ **推荐用于容量规划** |
| **Max** | 16528 MB | 启动期的异常峰值 | ⚠️ 可能高估 |

### 为什么不用 Avg？

**全程平均值被启动期拉低：**
```
Avg = (前半段 3127 MB + 后半段 8600 MB) / 2 = 5863 MB
      ↑ 包含大量启动期低值
```

### 为什么不用 Max？

**峰值可能包含启动期的短暂尖峰：**
```
Max = 16528 MB  （仅在 t=180s 附近出现一次）

但稳定期实际只需要 8600 MB
```

---

## 五、适用性分析

### ✅ Stable 值适用场景

1. **容量规划**：估算需要多少显存
   - 例：env=64 时 stable=32GB → 需要 L40 GPU（46GB）

2. **配置对比**：横向比较不同参数的显存需求
   - 例：env=2 (8.6GB) vs env=64 (32GB)

3. **OOM 预测**：判断是否会超出硬件限制
   - 例：stable=43GB → 接近 L40 上限 46GB

### ⚠️ Stable 值的局限性

1. **无法预测极端峰值**
   - 如果代码有 bug 导致显存泄漏，stable 无法捕捉

2. **依赖监控时长**
   - 如果训练时间 <10 分钟，后 50% 可能仍在启动期

3. **假设稳定性**
   - 如果训练过程显存持续增长（内存泄漏），stable 会低估

---

## 六、代码实现

### 在 analyze_experiments.py 中的实现

```python
# GPU 0-1 metrics
stable_start = len(gpu01_df) // 2
metrics.gpu01_mem_stable_mb = gpu01_df.iloc[stable_start:]["memory_used_mb"].mean()

# GPU 2-3 metrics
stable_start = len(gpu23_df) // 2
metrics.gpu23_mem_stable_mb = gpu23_df.iloc[stable_start:]["memory_used_mb"].mean()
```

### 逐步分解

```python
# 1. 读取监控数据
gpu_df = pd.read_csv("gpu_metrics.csv")

# 2. 过滤特定 GPU
gpu01_df = gpu_df[gpu_df["gpu_index"].isin([0, 1])]

# 3. 计算切分点（后 50% 的起始位置）
total_samples = len(gpu01_df)  # 例：112 个样本
stable_start = total_samples // 2  # 例：56

# 4. 提取后半段数据
second_half = gpu01_df.iloc[stable_start:]  # 第 56-112 个样本

# 5. 计算平均值
stable_mem = second_half["memory_used_mb"].mean()  # 例：8600.5 MB
```

---

## 七、验证方法

### 如何验证某个实验的 stable 值？

```bash
python3 << 'EOF'
import pandas as pd

# 读取监控数据
df = pd.read_csv("yunzhe/monitor/runs/monitor-split-.../gpu_metrics.csv")
gpu01 = df[df['gpu_index'].isin([0, 1])]

# 计算 stable
stable_start = len(gpu01) // 2
stable_mem = gpu01.iloc[stable_start:]['memory_used_mb'].mean()

print(f"Stable 显存: {stable_mem:.1f} MB")
EOF
```

### 与 CSV 报告对比

```bash
# 在 experiment_analysis.csv 中找到对应的 gpu01_mem_stable_mb
grep "env2_mbs1" yunzhe/experiment_analysis.csv
```

应该一致！

---

## 八、改进建议

### 当前方法的优点

- ✓ 简单直观
- ✓ 排除启动期干扰
- ✓ 对大多数场景有效

### 可能的改进方向

1. **自适应窗口**：根据显存标准差自动检测稳定期
   ```python
   # 找到 std < 阈值 的连续区间
   stable_window = find_stable_window(gpu_df, std_threshold=500)
   ```

2. **分位数法**：取 75% 分位数而非平均值
   ```python
   stable_mem = gpu_df.iloc[stable_start:]['memory_used_mb'].quantile(0.75)
   ```

3. **移动平均**：平滑短期波动
   ```python
   stable_mem = gpu_df.iloc[stable_start:]['memory_used_mb'].rolling(10).mean().mean()
   ```

---

**总结：**

Stable 显存 = **后 50% 样本的平均值**，用于反映训练稳定期的真实显存占用，是容量规划和配置对比的推荐指标。
