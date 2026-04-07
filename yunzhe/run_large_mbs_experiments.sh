#!/bin/bash
#
# 大 MBS 实验脚本（MBS=64, 128, 256）
# 用法: ./run_large_mbs_experiments.sh
#

set -e

REPO_PATH="/home/yunzhe/RLinf/RLinf"

# 定义实验组合：每个 env 测试 MBS=64, 128, 256
EXPERIMENTS=(
    # env=2
    "2,64"
    "2,128"
    "2,256"
    # env=4
    "4,64"
    "4,128"
    "4,256"
    # env=8
    "8,64"
    "8,128"
    "8,256"
    # env=16
    "16,64"
    "16,128"
    "16,256"
    # env=32
    "32,64"
    "32,128"
    "32,256"
    # env=64
    "64,64"
    "64,128"
    "64,256"
    # env=96 (继续测试)
    "96,1"
    "96,2"
    "96,4"
    "96,8"
    "96,16"
    "96,32"
    "96,64"
    "96,128"
    "96,256"
)

echo "=========================================="
echo "大 MBS 实验 + env=96 重测"
echo "=========================================="
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "配置: libero_spatial_ppo_openpi_pi05"
echo ""

# 调用通用批量脚本
cd "${REPO_PATH}/yunzhe"

# 临时修改批量脚本的实验列表
cat > /tmp/custom_experiments.sh <<'EOF'
#!/bin/bash
# 临时生成的实验列表

declare -a experiments=(
    "2,64" "2,128" "2,256"
    "4,64" "4,128" "4,256"
    "8,64" "8,128" "8,256"
    "16,64" "16,128" "16,256"
    "32,64" "32,128" "32,256"
    "64,64" "64,128" "64,256"
    "96,1" "96,2" "96,4" "96,8" "96,16" "96,32" "96,64" "96,128" "96,256"
)

total_exps=${#experiments[@]}
EOF

echo "开始运行实验..."
echo "可使用 Ctrl+C 中断，下次运行会自动跳过已完成的实验"
echo ""

# 使用通用批量脚本
bash "${REPO_PATH}/yunzhe/batch_experiments.sh"

echo ""
echo "全部完成！"
