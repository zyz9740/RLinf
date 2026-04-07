# 为什么 Micro Batch Size 对 RL Training 显存影响很小？

## 一、实验观察

### 数据总结

| GPU | MBS 范围 | 显存变化 | 边际成本 |
|-----|---------|---------|---------|
| **GPU01 (Rollout/Env)** | 1 → 32 | ~0 MB | **0 MB/MBS** |
| **GPU23 (Actor/Training)** | 1 → 32 | +1514 MB | **18-36 MB/MBS** |

### 具体案例（env=16）

| MBS | GPU01 稳定显存 | GPU23 稳定显存 | GPU23 增长 |
|-----|---------------|---------------|-----------|
| 1   | 13933 MB      | 18810 MB      | 基准 |
| 2   | 13899 MB      | 18571 MB      | -239 MB |
| 4   | 13893 MB      | 18664 MB      | -146 MB |
| 8   | 13679 MB      | 18823 MB      | +13 MB |
| 16  | 13853 MB      | 18469 MB      | -341 MB |
| 32  | 13849 MB      | 19891 MB      | **+1081 MB** |

**结论：MBS 从 1 增加到 32（32倍），GPU23 显存仅增长 1.1GB（5.8%）**

---

## 二、理论原因

### 1. RL Training 显存构成

在 GPU23（Actor/Training）上，显存主要被以下部分占据：

| 组成部分 | 典型大小 | 是否随 MBS 变化 |
|---------|---------|----------------|
| **模型权重** | 10-12 GB | ❌ 固定 |
| **优化器状态** (Adam) | 10-12 GB | ❌ 固定（与模型权重同大小） |
| **梯度缓冲** | 2-3 GB | ❌ 固定（与模型权重同大小） |
| **前向激活值** | 0.5-2 GB | ✓ 随 MBS 增长 |
| **Rollout Buffer** | 0-1 GB | ❌ 在 GPU01，不在 GPU23 |
| **框架开销** | 1-2 GB | ❌ 固定 |

**总计：约 24-30 GB 是固定的，只有 0.5-2 GB 随 MBS 变化**

---

### 2. 为什么前向激活值占用小？

#### A. 梯度累积（Gradient Accumulation）

RL 训练通常使用梯度累积策略：

```python
# 伪代码
for micro_batch in split_batch(rollout_buffer, micro_batch_size):
    logits = model(micro_batch.obs)        # 前向传播
    loss = compute_loss(logits, micro_batch)
    loss.backward()                        # 反向传播
    # ⬆️ 前向激活值在 backward() 后立即释放
    # ⬇️ 只保留累积的梯度（固定大小）

optimizer.step()  # 一次性更新权重
```

**关键点：**
- 每个 micro batch **顺序处理**，不同时存在于 GPU 上
- 前向激活值在反向传播后立即释放
- 累积的梯度大小 = 模型参数大小（与 MBS 无关）

#### B. 激活值大小估算

假设模型参数量 = 1B（10亿参数）：

| MBS | Batch 数据量 | 激活值估算 | 实际占用 |
|-----|-------------|-----------|---------|
| 1   | 1 × Seq × Hidden | ~200 MB | 单个 micro batch |
| 32  | 32 × Seq × Hidden | ~200 MB | 仍是单个 micro batch（顺序处理）|

**由于梯度累积，显存中只存一个 micro batch 的激活值！**

---

### 3. 为什么 GPU01 完全不受 MBS 影响？

GPU01 主要存储：
- 环境状态（observations）
- 环境模型（如果有）
- Rollout Buffer（采样的经验）

**MBS 只影响 training 阶段如何消费 buffer，不影响 buffer 本身的存储！**

```
GPU01: 存储 N 个环境的完整 Rollout Buffer
       ↓
       数据传输到 GPU23
       ↓
GPU23: 按 MBS 分批处理，但同时只处理一个 micro batch
```

---

## 三、与监督学习的对比

### 监督学习（Batch Size 影响大）

```python
# 所有数据同时在 GPU 上
images = load_batch(batch_size=32)    # 32 × 3 × 224 × 224
logits = model(images)                # 所有样本的激活值同时存在
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

**显存占用 ∝ batch_size**（所有样本同时前向）

### RL Training（Micro Batch Size 影响小）

```python
# 数据在 GPU01 的 Rollout Buffer 中
for micro_batch in rollout_buffer.iter(mbs=32):
    obs = micro_batch.obs             # 只取一个 micro batch
    logits = model(obs)               # 只有这 32 个样本的激活值
    loss = compute_ppo_loss(...)
    loss.backward()                   # 激活值释放
    # 梯度累积，不清零
    
optimizer.step()  # 最后一次性更新
```

**显存占用 ≈ 固定（单个 micro batch 的激活值）**

---

## 四、实验数据验证

### GPU23 显存与 MBS 的线性关系

| 环境数 | 拟合公式 | 边际成本 |
|-------|---------|---------|
| env=2  | GPU23 = 1.4 × MBS + 19353 | 1.4 MB/MBS |
| env=4  | GPU23 = 18.6 × MBS + 19071 | 18.6 MB/MBS |
| env=8  | GPU23 = 36.2 × MBS + 18475 | 36.2 MB/MBS |
| env=16 | GPU23 = 34.6 × MBS + 18508 | 34.6 MB/MBS |

**平均边际成本：约 23 MB/MBS = 0.022 GB/MBS**

### 显存增长率

| MBS 变化 | GPU23 显存增长 | 增长率 |
|---------|---------------|--------|
| 1 → 2   | ~0 MB         | 0% |
| 1 → 4   | ~-200 MB      | -1% |
| 1 → 8   | ~50 MB        | +0.3% |
| 1 → 16  | ~200 MB       | +1% |
| 1 → 32  | ~1500 MB      | +8% |

**MBS 增加 32 倍，显存仅增长 8%**

---

## 五、为什么 MBS=32 时显存会突增？

### 观察数据

| 环境数 | MBS 16→32 增长 |
|-------|---------------|
| env=16 | +1422 MB     |
| env=32 | +2291 MB     |
| env=64 | +2908 MB     |

### 可能的原因

#### 1. 超过梯度累积阈值

当 MBS 较大时，框架可能切换策略：
- MBS ≤ 16：严格梯度累积（顺序处理）
- MBS = 32：允许部分并行（牺牲显存换速度）

#### 2. 缓冲区扩容

PyTorch 可能预分配更大的临时缓冲区：
```python
if micro_batch_size >= 32:
    allocate_larger_workspace()  # 优化大 batch 性能
```

#### 3. 优化器状态膨胀

Adam 优化器的动量缓冲可能随 batch size 调整：
```python
# 一阶动量
m_t = beta1 * m_{t-1} + (1-beta1) * grad

# 当 batch 很大时，可能需要更多中间缓冲
```

---

## 六、核心结论

### ✓ 理论解释

1. **固定占用主导**：模型权重 + 优化器状态 ≈ 24GB（占总显存 80%）
2. **梯度累积机制**：多个 micro batch 顺序处理，激活值不累积
3. **数据在 GPU01**：Rollout Buffer 存储在 GPU01，GPU23 只拉取当前 micro batch

### ✓ 量化结论

```
GPU23 显存 ≈ 18.5 GB (固定) + 23 MB × MBS (可变)
                ↑                    ↑
            占 98.5%             仅占 1.5%
```

### ✓ 实践建议

1. **调整 MBS 对显存影响很小**（<2GB）
2. **可以安全地增大 MBS** 来提升训练效率
3. **如果 OOM，应该减少环境数**，而不是减少 MBS
4. **MBS 主要影响训练速度**，而非显存

---

## 七、与环境数的对比

| 参数 | 对 GPU01 影响 | 对 GPU23 影响 | 总结 |
|------|-------------|-------------|------|
| **环境数** | **378 MB/env** | ~0 | 环境数是 GPU01 显存的决定因素 |
| **MBS** | 0 | **23 MB/MBS** | MBS 对显存影响极小 |

**环境数对显存的影响是 MBS 的 16 倍！**

---

## 八、附录：监督学习 vs RL Training

| 特性 | 监督学习 | RL Training (本实验) |
|------|---------|---------------------|
| Batch 来源 | 磁盘/CPU 加载 | GPU01 的 Rollout Buffer |
| Batch 处理方式 | 所有样本并行 | 梯度累积（顺序处理） |
| 显存主要占用 | 激活值 | 模型权重 + 优化器 |
| Batch Size 影响 | **显著**（~1GB/32samples） | **很小**（~23MB/32MBS） |
| 建议调参优先级 | 先调 Batch Size | 先调环境数 |

---

**数据来源：** yunzhe/experiment_analysis.csv  
**实验范围：** env=2~64, MBS=1~32, 48+ 组配置
