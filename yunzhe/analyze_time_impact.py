#!/usr/bin/env python3
"""
分析 env 和 MBS 对 Generate Rollout 和 Actor Training 时间的影响
"""

import numpy as np

def linear_regression(x, y):
    """简单线性回归"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return slope, intercept, r2

# 从 experiment_analysis.html 提取的数据
# 格式: (env, mbs, generate_rollouts_s, actor_training_s)
data = [
    # env=2
    (2, 1, 183.90, 76.97), (2, 2, 186.20, 50.61), (2, 4, 185.10, 37.94),
    (2, 8, 184.20, 32.17), (2, 16, 194.70, 30.30), (2, 32, 190.40, 29.51),
    # env=4
    (4, 1, 248.90, 150.30), (4, 2, 249.40, 98.17), (4, 4, 253.90, 73.68),
    (4, 8, 249.70, 62.13), (4, 16, 257.30, 57.23), (4, 32, 242.20, 55.71),
    # env=8
    (8, 1, 343.30, 299.70), (8, 2, 335.30, 192.90), (8, 4, 335.50, 141.80),
    (8, 8, 350.50, 122.10), (8, 16, 354.50, 117.20), (8, 32, 320.00, 114.20),
    # env=16
    (16, 1, 547.30, 589.40), (16, 2, 557.20, 384.90), (16, 4, 554.50, 291.10),
    (16, 8, 551.20, 273.90), (16, 16, 562.10, 260.30), (16, 32, 541.20, 259.70),
    # env=32
    (32, 1, 1005.60, 1187.40), (32, 2, 986.90, 835.90), (32, 4, 1021.20, 709.20),
    (32, 8, 958.10, 680.50), (32, 16, 983.50, 700.10), (32, 32, 975.80, 686.80),
    # env=64
    (64, 1, 1924.40, 2379.80), (64, 2, 1815.70, 1568.40), (64, 4, 1776.80, 1450.60),
    (64, 8, 1889.00, 1389.00), (64, 16, 1878.80, 1458.80), (64, 32, 1839.60, 1518.70),
]

print("=" * 80)
print("ENV 和 MBS 对执行时间的影响分析")
print("=" * 80)
print()

# 按 env 和 mbs 分组
data_by_env = {}
data_by_mbs = {}

for env, mbs, rollout_time, training_time in data:
    if env not in data_by_env:
        data_by_env[env] = {'mbs': [], 'rollout': [], 'training': []}
    data_by_env[env]['mbs'].append(mbs)
    data_by_env[env]['rollout'].append(rollout_time)
    data_by_env[env]['training'].append(training_time)

    if mbs not in data_by_mbs:
        data_by_mbs[mbs] = {'env': [], 'rollout': [], 'training': []}
    data_by_mbs[mbs]['env'].append(env)
    data_by_mbs[mbs]['rollout'].append(rollout_time)
    data_by_mbs[mbs]['training'].append(training_time)

# ============================================================================
# 1. ENV 对时间的影响
# ============================================================================
print("📊 1. ENV 数量对执行时间的影响")
print("=" * 80)

# 对每个 MBS，分析 env 的影响
print("\n【Rollout 时间 vs ENV】")
print("-" * 80)
print(f"{'MBS':>5} {'公式':>35} {'R²':>10}")
print("-" * 80)

for mbs in sorted(data_by_mbs.keys()):
    env_vals = np.array(data_by_mbs[mbs]['env'])
    rollout_vals = np.array(data_by_mbs[mbs]['rollout'])

    slope, intercept, r2 = linear_regression(env_vals, rollout_vals)
    print(f"{mbs:5d} {'Rollout(s) = ' + f'{intercept:.1f} + {slope:.2f} × env':>35} {r2:10.6f}")

# 整体分析
all_env = np.array([env for env, mbs, _, _ in data])
all_rollout = np.array([rollout for _, _, rollout, _ in data])
slope_r, intercept_r, r2_r = linear_regression(all_env, all_rollout)

print("-" * 80)
print(f"{'整体':>5} {'Rollout(s) = ' + f'{intercept_r:.1f} + {slope_r:.2f} × env':>35} {r2_r:10.6f}")
print()

print("\n【Training 时间 vs ENV】")
print("-" * 80)
print(f"{'MBS':>5} {'公式':>35} {'R²':>10}")
print("-" * 80)

for mbs in sorted(data_by_mbs.keys()):
    env_vals = np.array(data_by_mbs[mbs]['env'])
    training_vals = np.array(data_by_mbs[mbs]['training'])

    slope, intercept, r2 = linear_regression(env_vals, training_vals)
    print(f"{mbs:5d} {'Training(s) = ' + f'{intercept:.1f} + {slope:.2f} × env':>35} {r2:10.6f}")

# 整体分析
all_training = np.array([training for _, _, _, training in data])
slope_t, intercept_t, r2_t = linear_regression(all_env, all_training)

print("-" * 80)
print(f"{'整体':>5} {'Training(s) = ' + f'{intercept_t:.1f} + {slope_t:.2f} × env':>35} {r2_t:10.6f}")

# ============================================================================
# 2. MBS 对时间的影响
# ============================================================================
print("\n\n" + "=" * 80)
print("📊 2. MBS 对执行时间的影响")
print("=" * 80)

print("\n【Rollout 时间 vs MBS】")
print("-" * 80)
print(f"{'ENV':>5} {'公式':>35} {'R²':>10}")
print("-" * 80)

for env in sorted(data_by_env.keys()):
    mbs_vals = np.array(data_by_env[env]['mbs'])
    rollout_vals = np.array(data_by_env[env]['rollout'])

    slope, intercept, r2 = linear_regression(mbs_vals, rollout_vals)
    print(f"{env:5d} {'Rollout(s) = ' + f'{intercept:.1f} + {slope:.3f} × MBS':>35} {r2:10.6f}")

# 整体分析
all_mbs = np.array([mbs for _, mbs, _, _ in data])
slope_rm, intercept_rm, r2_rm = linear_regression(all_mbs, all_rollout)

print("-" * 80)
print(f"{'整体':>5} {'Rollout(s) = ' + f'{intercept_rm:.1f} + {slope_rm:.3f} × MBS':>35} {r2_rm:10.6f}")
print()

print("\n【Training 时间 vs MBS】")
print("-" * 80)
print(f"{'ENV':>5} {'公式':>35} {'R²':>10}")
print("-" * 80)

for env in sorted(data_by_env.keys()):
    mbs_vals = np.array(data_by_env[env]['mbs'])
    training_vals = np.array(data_by_env[env]['training'])

    slope, intercept, r2 = linear_regression(mbs_vals, training_vals)
    print(f"{env:5d} {'Training(s) = ' + f'{intercept:.1f} + {slope:.3f} × MBS':>35} {r2:10.6f}")

# 整体分析
slope_tm, intercept_tm, r2_tm = linear_regression(all_mbs, all_training)

print("-" * 80)
print(f"{'整体':>5} {'Training(s) = ' + f'{intercept_tm:.1f} + {slope_tm:.3f} × MBS':>35} {r2_tm:10.6f}")

# ============================================================================
# 3. 汇总统计
# ============================================================================
print("\n\n" + "=" * 80)
print("📈 3. 关键统计数据")
print("=" * 80)

print("\n【Rollout 时间统计】")
print("-" * 80)
for env in sorted(data_by_env.keys()):
    rollout_vals = np.array(data_by_env[env]['rollout'])
    print(f"env={env:2d}: 平均={rollout_vals.mean():6.1f}s, "
          f"最小={rollout_vals.min():6.1f}s, 最大={rollout_vals.max():6.1f}s, "
          f"变异={rollout_vals.std()/rollout_vals.mean()*100:4.1f}%")

print("\n【Training 时间统计】")
print("-" * 80)
for env in sorted(data_by_env.keys()):
    training_vals = np.array(data_by_env[env]['training'])
    print(f"env={env:2d}: 平均={training_vals.mean():6.1f}s, "
          f"最小={training_vals.min():6.1f}s, 最大={training_vals.max():6.1f}s, "
          f"变异={training_vals.std()/training_vals.mean()*100:4.1f}%")

# ============================================================================
# 4. 瓶颈分析
# ============================================================================
print("\n\n" + "=" * 80)
print("🎯 4. 瓶颈分析")
print("=" * 80)

print("\n【各配置的主要瓶颈】")
print("-" * 80)
print(f"{'env':>5} {'MBS':>5} {'Rollout(s)':>12} {'Training(s)':>12} {'瓶颈':>10} {'比例':>10}")
print("-" * 80)

for env, mbs, rollout_time, training_time in data:
    if rollout_time > training_time:
        bottleneck = "Rollout"
        ratio = rollout_time / training_time
    else:
        bottleneck = "Training"
        ratio = training_time / rollout_time

    print(f"{env:5d} {mbs:5d} {rollout_time:12.1f} {training_time:12.1f} {bottleneck:>10} {ratio:9.2f}x")

# ============================================================================
# 5. 实用结论
# ============================================================================
print("\n\n" + "=" * 80)
print("💡 5. 实用结论与建议")
print("=" * 80)

print("""
【ENV 的影响】
1. Rollout 时间:
   - 公式: Rollout(s) = {:.1f} + {:.2f} × env (R²={:.4f})
   - env 每增加 1，Rollout 增加 {:.1f} 秒
   - **强线性关系，env 是主导因素**

2. Training 时间:
   - 公式: Training(s) = {:.1f} + {:.2f} × env (R²={:.4f})
   - env 每增加 1，Training 增加 {:.1f} 秒
   - **超强线性关系，env 是决定性因素**

【MBS 的影响】
3. Rollout 时间:
   - 公式: Rollout(s) = {:.1f} + {:.3f} × MBS (R²={:.4f})
   - **MBS 对 Rollout 几乎无影响** (R² 接近 0)
   - Rollout 阶段不涉及梯度计算，MBS 不相关

4. Training 时间:
   - 公式: Training(s) = {:.1f} + {:.3f} × MBS (R²={:.4f})
   - MBS 每增加 1，Training **减少** {:.1f} 秒
   - **MBS 越大，Training 越快** (梯度累积次数减少)

【瓶颈转换点】
5. env ≤ 8:  Rollout 和 Training 相当
6. env = 16: 转折点
7. env ≥ 32: **Rollout 成为主要瓶颈**

【优化建议】
8. 小规模 (env≤8): 提高 MBS 可以加速 Training，整体提升有限
9. 大规模 (env≥32): Rollout 是瓶颈，优化重点在并行环境交互
10. 最佳 MBS: 建议 16-32，平衡性能和显存
""".format(
    intercept_r, slope_r, r2_r, slope_r,
    intercept_t, slope_t, r2_t, slope_t,
    intercept_rm, slope_rm, r2_rm,
    intercept_tm, slope_tm, r2_tm, -slope_tm
))
