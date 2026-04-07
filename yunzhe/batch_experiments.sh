#!/bin/bash
#
# 批量实验脚本 - 支持断点续传
# 用法: ./batch_experiments.sh [start_index]
#

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 实验参数组合
ENV_NUMS=(2 4 8 16 32 64 96)
MBS_VALUES=(1 2 4 8 16 32 64 128 256)

# 项目路径
REPO_PATH="/home/yunzhe/RLinf/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
MONITOR_SCRIPT="${REPO_PATH}/yunzhe/monitor/.venv/bin/python"
MONITOR_MODULE="${REPO_PATH}/yunzhe/monitor"

# 输出路径
TRAINLOG_DIR="${REPO_PATH}/yunzhe/trainlog"
MONITOR_RUNS_DIR="${REPO_PATH}/yunzhe/monitor/runs"
BATCH_LOG_DIR="${REPO_PATH}/yunzhe/batch_logs"

# 配置文件
BASE_CONFIG="libero_spatial_ppo_openpi_pi05"

# 监控参数
MONITOR_INTERVAL=1.0  # 采样间隔（秒）

# ==================== 函数定义 ====================

# 获取时间戳
get_timestamp() {
    date +'%Y%m%d-%H%M%S'
}

# 创建目录
mkdir -p "${TRAINLOG_DIR}"
mkdir -p "${MONITOR_RUNS_DIR}"
mkdir -p "${BATCH_LOG_DIR}"

# 检查实验是否已完成
check_experiment_done() {
    local env=$1
    local mbs=$2

    # 检查训练日志是否存在且包含完成标志
    local log_pattern="${TRAINLOG_DIR}/env${env}_mbs${mbs}-*.log"

    if ls ${log_pattern} 1> /dev/null 2>&1; then
        # 找到日志文件，检查是否包含完成标志
        local log_file=$(ls -t ${log_pattern} | head -1)

        # 检查是否包含训练完成的标志（例如："Training completed" 或 "max_epochs" 达到）
        if grep -q "Epoch 1/1" "${log_file}" 2>/dev/null; then
            # 检查是否有错误或异常退出
            if ! grep -q "Error\|Exception\|Traceback" "${log_file}" 2>/dev/null; then
                echo "1"  # 已完成
                return
            fi
        fi
    fi

    echo "0"  # 未完成
}

# 启动监控
start_monitoring() {
    local env=$1
    local mbs=$2
    local timestamp=$3
    local placement_mode="split"  # 固定为 split 模式

    local monitor_name="monitor-${placement_mode}-${BASE_CONFIG}-env${env}_mbs${mbs}-${timestamp}"
    local monitor_log="${MONITOR_RUNS_DIR}/${monitor_name}/runtime.log"

    echo "  [$(date +'%H:%M:%S')] 启动监控: ${monitor_name}"

    # 使用 nohup 在后台启动监控
    PYTHONPATH="${REPO_PATH}" nohup ${MONITOR_SCRIPT} -m yunzhe.monitor \
        --interval ${MONITOR_INTERVAL} \
        --output-dir "${MONITOR_RUNS_DIR}" \
        --name "${monitor_name}" \
        > "${MONITOR_RUNS_DIR}/${monitor_name}.monitor.log" 2>&1 &

    local monitor_pid=$!
    echo ${monitor_pid} > "${MONITOR_RUNS_DIR}/${monitor_name}.pid"
    echo "  [$(date +'%H:%M:%S')] 监控进程 PID: ${monitor_pid}"

    # 等待监控启动
    sleep 3

    echo ${monitor_pid}
}

# 停止监控
stop_monitoring() {
    local monitor_pid=$1

    if [ -n "${monitor_pid}" ] && kill -0 ${monitor_pid} 2>/dev/null; then
        echo "  [$(date +'%H:%M:%S')] 停止监控进程: ${monitor_pid}"
        kill -SIGINT ${monitor_pid}

        # 等待监控优雅退出（最多 10 秒）
        for i in {1..10}; do
            if ! kill -0 ${monitor_pid} 2>/dev/null; then
                echo "  [$(date +'%H:%M:%S')] 监控已停止"
                return
            fi
            sleep 1
        done

        # 强制杀死
        kill -9 ${monitor_pid} 2>/dev/null || true
        echo "  [$(date +'%H:%M:%S')] 监控已强制停止"
    fi
}

# 运行单个实验
run_experiment() {
    local env=$1
    local mbs=$2
    local exp_index=$3
    local total_exps=$4

    echo ""
    echo "=========================================="
    echo "实验 ${exp_index}/${total_exps}: env=${env}, mbs=${mbs}"
    echo "=========================================="

    # 检查是否已完成
    local is_done=$(check_experiment_done ${env} ${mbs})
    if [ "${is_done}" == "1" ]; then
        echo "  ✓ 实验已完成，跳过"
        return 0
    fi

    local timestamp=$(get_timestamp)
    local log_file="${TRAINLOG_DIR}/env${env}_mbs${mbs}-${timestamp}.log"
    local monitor_log="${TRAINLOG_DIR}/env${env}_mbs${mbs}-${timestamp}.monitor.log"

    echo "  [$(date +'%H:%M:%S')] 开始实验"
    echo "  训练日志: ${log_file}"

    # 启动监控
    local monitor_pid=$(start_monitoring ${env} ${mbs} ${timestamp})

    # 设置环境变量
    export EMBODIED_PATH="${EMBODIED_PATH}"
    export REPO_PATH="${REPO_PATH}"
    export MUJOCO_GL="egl"
    export PYOPENGL_PLATFORM="egl"
    export ROBOT_PLATFORM="LIBERO"
    export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"

    # 构建命令
    local cmd="python ${EMBODIED_PATH}/train_embodied_agent.py \
        --config-path ${EMBODIED_PATH}/config/ \
        --config-name ${BASE_CONFIG} \
        env.train.total_num_envs=${env} \
        actor.micro_batch_size=${mbs} \
        runner.logger.log_path=${TRAINLOG_DIR}"

    echo "  命令: ${cmd}"

    # 运行训练（记录开始时间）
    local start_time=$(date +%s)

    # 执行训练，捕获退出码
    set +e  # 临时关闭错误退出
    ${cmd} > "${log_file}" 2>&1
    local exit_code=$?
    set -e  # 恢复错误退出

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # 停止监控
    stop_monitoring ${monitor_pid}

    # 保存监控日志路径到训练日志旁边
    echo "Monitor log: ${MONITOR_RUNS_DIR}/monitor-split-${BASE_CONFIG}-env${env}_mbs${mbs}-${timestamp}" > "${monitor_log}"

    # 检查训练结果
    if [ ${exit_code} -eq 0 ]; then
        echo "  ✓ 实验完成 (用时: ${duration}s)"
        return 0
    else
        echo "  ✗ 实验失败 (退出码: ${exit_code}, 用时: ${duration}s)"
        echo "  日志: ${log_file}"

        # 如果是 OOM，记录并继续下一个实验
        if grep -q "OutOfMemoryError\|CUDA out of memory" "${log_file}" 2>/dev/null; then
            echo "  ⚠️  检测到 OOM，跳过此配置"
            return 1
        fi

        # 其他错误，询问是否继续
        echo ""
        echo "  是否继续下一个实验? (y/n)"
        read -t 30 -n 1 answer || answer="y"
        echo ""

        if [ "${answer}" != "y" ] && [ "${answer}" != "Y" ]; then
            echo "  用户中止批量实验"
            exit 1
        fi

        return 1
    fi
}

# ==================== 主程序 ====================

echo "=========================================="
echo "批量实验脚本"
echo "=========================================="
echo "配置: ${BASE_CONFIG}"
echo "环境数量: ${ENV_NUMS[@]}"
echo "MBS 值: ${MBS_VALUES[@]}"
echo ""

# 生成所有实验组合
declare -a experiments
for env in "${ENV_NUMS[@]}"; do
    for mbs in "${MBS_VALUES[@]}"; do
        experiments+=("${env},${mbs}")
    done
done

total_exps=${#experiments[@]}
echo "总实验数: ${total_exps}"
echo ""

# 支持断点续传：从指定索引开始
START_INDEX=${1:-0}
if [ ${START_INDEX} -gt 0 ]; then
    echo "从实验 ${START_INDEX} 开始（断点续传）"
    echo ""
fi

# 统计
completed_count=0
failed_count=0
skipped_count=0

# 保存批量日志
batch_log="${BATCH_LOG_DIR}/batch_$(get_timestamp).log"
echo "批量日志: ${batch_log}"
echo ""

# 运行实验
for i in $(seq ${START_INDEX} $((total_exps - 1))); do
    exp_params="${experiments[$i]}"
    IFS=',' read -r env mbs <<< "${exp_params}"

    # 先检查是否已完成
    is_done=$(check_experiment_done ${env} ${mbs})
    if [ "${is_done}" == "1" ]; then
        echo "实验 $((i+1))/${total_exps}: env=${env}, mbs=${mbs} - ✓ 已完成，跳过"
        ((skipped_count++))
        continue
    fi

    # 运行实验
    if run_experiment ${env} ${mbs} $((i+1)) ${total_exps}; then
        ((completed_count++))
    else
        ((failed_count++))
    fi

    # 记录到批量日志
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 实验 $((i+1))/${total_exps}: env=${env}, mbs=${mbs} - 完成" >> "${batch_log}"

    # 简短延迟，避免系统过载
    sleep 5
done

# ==================== 总结 ====================

echo ""
echo "=========================================="
echo "批量实验完成"
echo "=========================================="
echo "总实验数: ${total_exps}"
echo "完成: ${completed_count}"
echo "失败: ${failed_count}"
echo "跳过: ${skipped_count}"
echo ""
echo "训练日志目录: ${TRAINLOG_DIR}"
echo "监控数据目录: ${MONITOR_RUNS_DIR}"
echo "批量日志: ${batch_log}"
echo ""

# 运行分析脚本
if [ ${completed_count} -gt 0 ]; then
    echo "运行分析脚本..."
    python ${REPO_PATH}/yunzhe/analyze_experiments.py \
        --trainlog-dir "${TRAINLOG_DIR}" \
        --monitor-dir "${MONITOR_RUNS_DIR}" \
        --output "${REPO_PATH}/yunzhe/experiment_analysis.html" \
        --csv "${REPO_PATH}/yunzhe/experiment_analysis.csv"

    echo ""
    echo "分析完成！查看报告:"
    echo "  HTML: ${REPO_PATH}/yunzhe/experiment_analysis.html"
    echo "  CSV:  ${REPO_PATH}/yunzhe/experiment_analysis.csv"
fi

echo ""
echo "全部完成！"
