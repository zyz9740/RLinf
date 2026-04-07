#!/usr/bin/env python3
"""
分析 Micro Batch Size (MBS) 与 GPU 显存的关系
"""

import numpy as np

def linear_regression(x, y):
    """简单线性回归: y = slope * x + intercept"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return slope, intercept, r2

# 从 experiment_analysis.html 中提取的 Training Phase GPU 2-3 显存数据 (MB)
# 格式: (env, mbs, training_gpu23_mem_mb)
data = [
    # env=2
    (2, 1, 19788), (2, 2, 19615), (2, 4, 18718), (2, 8, 19256), (2, 16, 19154), (2, 32, 19728),
    # env=4
    (4, 1, 19016), (4, 2, 19267), (4, 4, 18632), (4, 8, 19094), (4, 16, 19431), (4, 32, 19942),
    # env=8
    (8, 1, 18363), (8, 2, 18337), (8, 4, 18499), (8, 8, 18242), (8, 16, 18905), (8, 32, 19832),
    # env=16
    (16, 1, 18539), (16, 2, 18391), (16, 4, 18717), (16, 8, 18872), (16, 16, 18059), (16, 32, 19663),
    # env=32
    (32, 1, 18391), (32, 2, 18508), (32, 4, 18296), (32, 8, 18574), (32, 16, 18868), (32, 32, 21270),
    # env=64
    (64, 1, 18319), (64, 2, 18333), (64, 4, 18370), (64, 8, 18364), (64, 16, 18689), (64, 32, 21577),
]

# 转换为字典格式便于按env分组
data_by_env = {}
for env, mbs, mem in data:
    if env not in data_by_env:
        data_by_env[env] = {'mbs': [], 'mem': []}
    data_by_env[env]['mbs'].append(mbs)
    data_by_env[env]['mem'].append(mem)

print("=" * 80)
print("MBS 与训练显存关系分析 (GPU 2-3, Training Phase)")
print("=" * 80)
print()

# 1. 按 env 分组，分析 MBS 的影响
print("📊 各环境数下 MBS 对显存的影响:")
print("-" * 80)

env_results = []
for env in sorted(data_by_env.keys()):
    mbs_values = np.array(data_by_env[env]['mbs'])
    mem_values = np.array(data_by_env[env]['mem'])

    # 线性回归
    slope, intercept, r2 = linear_regression(mbs_values, mem_values)

    print(f"\nenv={env:2d}:")
    print(f"  公式: Memory(MB) = {intercept:.1f} + {slope:.3f} × MBS")
    print(f"  R² = {r2:.6f}")
    print(f"  MBS=1: {intercept + slope*1:.0f} MB")
    print(f"  MBS=32: {intercept + slope*32:.0f} MB")
    print(f"  增量: {slope*31:.0f} MB (+{slope*31/intercept*100:.1f}%)")

    env_results.append({
        'env': env,
        'intercept': intercept,
        'slope': slope,
        'r2': r2
    })

# 2. 整体分析
print("\n" + "=" * 80)
print("🔍 整体线性模型分析")
print("=" * 80)

mbs_all = np.array([mbs for env_data in data_by_env.values() for mbs in env_data['mbs']])
mem_all = np.array([mem for env_data in data_by_env.values() for mem in env_data['mem']])

slope_all, intercept_all, r2_all = linear_regression(mbs_all, mem_all)

print(f"\n整体公式: Memory(MB) = {intercept_all:.1f} + {slope_all:.3f} × MBS")
print(f"R² = {r2_all:.6f}")
print()

# 3. 分段分析: MBS 小范围 vs 大范围
print("=" * 80)
print("📈 分段分析")
print("=" * 80)

# 小 MBS (1-16)
mbs_small = []
mem_small = []
for env_data in data_by_env.values():
    for i, mbs in enumerate(env_data['mbs']):
        if mbs <= 16:
            mbs_small.append(mbs)
            mem_small.append(env_data['mem'][i])

slope_s, intercept_s, r2_s = linear_regression(np.array(mbs_small), np.array(mem_small))
print(f"\nMBS ∈ [1, 16]:")
print(f"  Memory(MB) = {intercept_s:.1f} + {slope_s:.3f} × MBS")
print(f"  R² = {r2_s:.6f}")

# 大 MBS (16-32)
mbs_large = []
mem_large = []
for env_data in data_by_env.values():
    for i, mbs in enumerate(env_data['mbs']):
        if mbs >= 16:
            mbs_large.append(mbs)
            mem_large.append(env_data['mem'][i])

slope_l, intercept_l, r2_l = linear_regression(np.array(mbs_large), np.array(mem_large))
print(f"\nMBS ∈ [16, 32]:")
print(f"  Memory(MB) = {intercept_l:.1f} + {slope_l:.3f} × MBS")
print(f"  R² = {r2_l:.6f}")

# 4. 按 env 分组的平均斜率
print("\n" + "=" * 80)
print("📊 MBS 影响系数汇总")
print("=" * 80)
print()

print(f"{'env':>5} {'intercept':>12} {'slope':>10} {'R²':>10}")
print("-" * 40)
for result in env_results:
    print(f"{result['env']:5d} {result['intercept']:12.1f} {result['slope']:10.3f} {result['r2']:10.6f}")

slopes = np.array([r['slope'] for r in env_results])
print(f"\n平均斜率: {slopes.mean():.3f} MB/MBS")
print(f"标准差: {slopes.std():.3f} MB/MBS")

# 5. 给出实用公式
print("\n" + "=" * 80)
print("🎯 实用显存预估公式")
print("=" * 80)
print()

# 基础显存（MBS=1时的平均值）
mbs1_values = [env_data['mem'][env_data['mbs'].index(1)] for env_data in data_by_env.values() if 1 in env_data['mbs']]
base_mem = np.mean(mbs1_values)

# MBS增长系数（按平均斜率）
avg_slope = slopes.mean()

print("**训练阶段 GPU 2-3 显存公式**:")
print()
print(f"    Memory(GB) = {base_mem/1024:.2f} + {avg_slope/1024:.4f} × MBS")
print()
print("或")
print()
print(f"    Memory(MB) = {base_mem:.0f} + {avg_slope:.2f} × MBS")
print()

print("实际应用:")
print("-" * 40)
for mbs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    mem_mb = base_mem + avg_slope * mbs
    print(f"  MBS = {mbs:3d}  →  {mem_mb:5.0f} MB ({mem_mb/1024:5.2f} GB)")

print("\n" + "=" * 80)
print("⚠️  注意事项")
print("=" * 80)
print("""
1. 此公式仅适用于训练阶段的 GPU 2-3 (Actor/Training GPU)
2. Rollout GPU (GPU 0-1) 主要受 env 影响，与 MBS 无关
3. MBS > 32 的数据为推测值，实际可能有非线性效应
4. 建议预留 20% buffer 处理峰值显存
""")
