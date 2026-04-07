#!/usr/bin/env bash
#
# 测试跳过已完成实验的功能
#

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================="
echo "测试跳过已完成实验功能"
echo "=========================================="
echo ""

# 测试读取逻辑
COMPLETED_EXPERIMENTS_FILE="${REPO_PATH}/yunzhe/monitor/runs/completed_experiments.csv"

if [[ ! -f "${COMPLETED_EXPERIMENTS_FILE}" ]]; then
  echo "❌ 未找到 completed_experiments.csv"
  exit 1
fi

echo "✓ 找到配置文件: ${COMPLETED_EXPERIMENTS_FILE}"
echo ""

# 读取已完成的实验
declare -A completed_experiments
while IFS=',' read -r env_val mbs_val; do
  if [[ "${env_val}" == "env" ]]; then
    continue  # skip header
  fi
  key="env${env_val}_mbs${mbs_val}"
  completed_experiments["${key}"]="1"
done < "${COMPLETED_EXPERIMENTS_FILE}"

echo "已加载 ${#completed_experiments[@]} 个已完成实验"
echo ""

# 测试一些样例
test_cases=("env2_mbs1" "env2_mbs64" "env16_mbs1" "env96_mbs256")

echo "测试样例检查:"
for run_id in "${test_cases[@]}"; do
  if [[ -n "${completed_experiments[${run_id}]+x}" ]]; then
    echo "  ✅ ${run_id} - 已完成 (将跳过)"
  else
    echo "  ⭕ ${run_id} - 未完成 (将运行)"
  fi
done
echo ""

# 统计应该运行的实验数量
ENV_VALUES=(2 4 8 16 32 64 96)
MBS_VALUES=(1 2 4 8 16 32 64 128 256)

total_count=0
skip_count=0
run_count=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实验运行预览"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for env_n in "${ENV_VALUES[@]}"; do
  echo "环境数 = ${env_n}:"
  for mbs in "${MBS_VALUES[@]}"; do
    run_id="env${env_n}_mbs${mbs}"
    ((total_count++))

    if [[ -n "${completed_experiments[${run_id}]+x}" ]]; then
      echo "  [跳过] ${run_id}"
      ((skip_count++))
    else
      echo "  [运行] ${run_id}"
      ((run_count++))
    fi
  done
  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "统计汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "总实验数:   ${total_count}"
echo "已完成:     ${skip_count} (将跳过)"
echo "需要运行:   ${run_count}"
echo ""

if [[ ${skip_count} -eq 30 ]] && [[ ${run_count} -eq 33 ]]; then
  echo "✅ 测试通过！"
  echo ""
  echo "脚本将正确跳过 30 个已完成的实验，运行 33 个新实验。"
  exit 0
else
  echo "❌ 测试失败"
  echo "  期望跳过: 30"
  echo "  实际跳过: ${skip_count}"
  echo "  期望运行: 33"
  echo "  实际运行: ${run_count}"
  exit 1
fi
