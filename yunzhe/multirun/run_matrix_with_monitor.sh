#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_PATH}"

: "${EMBODIED_PATH:=${REPO_PATH}/examples/embodiment}"
export EMBODIED_PATH

: "${ROBOT_PLATFORM:=LIBERO}"
: "${MUJOCO_GL:=egl}"
: "${PYOPENGL_PLATFORM:=egl}"
export ROBOT_PLATFORM MUJOCO_GL PYOPENGL_PLATFORM
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

CONFIG_PATH="${EMBODIED_PATH}/config"
CONFIG_NAME="${1:-libero_spatial_ppo_openpi_pi05}"
PLACEMENT_MODE="${PLACEMENT_MODE:-split}"
MAX_STEPS="${MAX_STEPS:-50}"
ENV_LIST="${ENV_LIST:-auto}"
MBS_LIST="${MBS_LIST:-1,2,4,8,16,32,64,128}"
GBS="${GBS:-256}"
RUN_TAG="${RUN_TAG:-${PLACEMENT_MODE}}"
ENABLE_MONITOR="${ENABLE_MONITOR:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-10.0}"
MONITOR_MAX_SAMPLES="${MONITOR_MAX_SAMPLES:-0}"
MONITOR_PYTHON="${MONITOR_PYTHON:-${REPO_PATH}/yunzhe/monitor/.venv/bin/python}"
MONITOR_OUTPUT_DIR="${MONITOR_OUTPUT_DIR:-${REPO_PATH}/yunzhe/monitor/runs}"
MONITOR_SHUTDOWN_GRACE_SECONDS="${MONITOR_SHUTDOWN_GRACE_SECONDS:-30}"

COMPLETED_EXPERIMENTS_FILE="${REPO_PATH}/yunzhe/monitor/runs/completed_experiments.csv"

STAMP="$(date +'%Y%m%d-%H%M%S')"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_PATH}/logs/multirun-${RUN_TAG}-${CONFIG_NAME}-${STAMP}}"

RESULTS_DIR="${SWEEP_ROOT}/results"
TRAIN_LOG_DIR="${SWEEP_ROOT}/train"
mkdir -p "${SWEEP_ROOT}" "${RESULTS_DIR}" "${TRAIN_LOG_DIR}" "${MONITOR_OUTPUT_DIR}"

MANIFEST_CSV="${RESULTS_DIR}/manifest.csv"
if [[ ! -f "${MANIFEST_CSV}" ]]; then
  echo "run_id,placement_mode,total_num_envs,micro_batch_size,global_batch_size,train_start,train_end,train_exit_code,train_log_file,monitor_run_name,status" > "${MANIFEST_CSV}"
fi

IFS=',' read -r -a MBS_VALUES <<< "${MBS_LIST}"

# Load completed experiments
declare -A completed_experiments
if [[ -f "${COMPLETED_EXPERIMENTS_FILE}" ]]; then
  echo "[matrix] loading completed experiments from: ${COMPLETED_EXPERIMENTS_FILE}"
  while IFS=',' read -r env_val mbs_val; do
    if [[ "${env_val}" == "env" ]]; then
      continue  # skip header
    fi
    key="env${env_val}_mbs${mbs_val}"
    completed_experiments["${key}"]="1"
  done < "${COMPLETED_EXPERIMENTS_FILE}"
  echo "[matrix] loaded ${#completed_experiments[@]} completed experiments"
else
  echo "[matrix] no completed experiments file found, will run all experiments"
fi

monitor_pid=""
stop_requested=0

cleanup_monitor() {
  local force="${1:-0}"
  local pid="${monitor_pid}"
  if [[ -z "${pid}" ]]; then
    return
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    # Ask monitor to flush CSV/report and exit cleanly.
    kill -INT "${pid}" 2>/dev/null || true

    loops=$((MONITOR_SHUTDOWN_GRACE_SECONDS * 5))
    if (( loops < 1 )); then
      loops=1
    fi
    for _ in $(seq 1 "${loops}"); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
  fi

  if [[ "${force}" == "1" ]] && kill -0 "${pid}" 2>/dev/null; then
    # Fast escalation path used on interrupts and abnormal exits.
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  fi

  if [[ "${force}" == "1" ]] && kill -0 "${pid}" 2>/dev/null; then
    # Last resort to avoid hanging script shutdown.
    kill -KILL "${pid}" 2>/dev/null || true
  fi

  wait "${pid}" 2>/dev/null || true
  monitor_pid=""
}

on_interrupt() {
  stop_requested=1
  cleanup_monitor 1
}

trap 'cleanup_monitor 1' EXIT
trap on_interrupt INT TERM

placement_override() {
  case "$1" in
    split)
      printf '%s\n' 'cluster.component_placement.env=0-1' 'cluster.component_placement.rollout=0-1' 'cluster.component_placement.actor=2-3'
      ;;
    all)
      printf '%s\n' 'cluster.component_placement.env=all' 'cluster.component_placement.rollout=all' 'cluster.component_placement.actor=all'
      ;;
    *)
      echo "Unsupported PLACEMENT_MODE: $1 (expected split or all)" >&2
      exit 2
      ;;
  esac
}

placement_gpu_multiple() {
  case "$1" in
    split)
      printf '%s' '2'
      ;;
    all)
      printf '%s' '4'
      ;;
    *)
      echo "Unsupported PLACEMENT_MODE: $1 (expected split or all)" >&2
      exit 2
      ;;
  esac
}

actor_world_size_for_mode() {
  case "$1" in
    split)
      # split placement uses 2 actor GPUs: 2-3
      printf '%s' '2'
      ;;
    all)
      # all placement uses all 4 GPUs for actor
      printf '%s' '4'
      ;;
    *)
      echo "Unsupported PLACEMENT_MODE: $1 (expected split or all)" >&2
      exit 2
      ;;
  esac
}

default_env_list_for_mode() {
  case "$1" in
    split)
      # split uses 2 GPUs for env+rollout
      printf '%s' '2,4,8,16,32,64,96,128'
      ;;
    all)
      # all uses 4 GPUs collocated
      printf '%s' '4,8,16,32,64,96,128'
      ;;
    *)
      echo "Unsupported PLACEMENT_MODE: $1 (expected split or all)" >&2
      exit 2
      ;;
  esac
}

if [[ "${ENABLE_MONITOR}" == "1" ]] && [[ ! -x "${MONITOR_PYTHON}" ]]; then
  echo "Monitor python not executable: ${MONITOR_PYTHON}" >&2
  exit 2
fi

echo "[matrix] repo: ${REPO_PATH}"
echo "[matrix] config: ${CONFIG_NAME}"
echo "[matrix] placement: ${PLACEMENT_MODE}"
echo "[matrix] sweep root: ${SWEEP_ROOT}"
echo "[matrix] micro_batch_size: ${MBS_LIST}"
echo "[matrix] monitor enabled: ${ENABLE_MONITOR}"

mapfile -t PLACEMENT_OVERRIDES < <(placement_override "${PLACEMENT_MODE}")
GPU_MULTIPLE="$(placement_gpu_multiple "${PLACEMENT_MODE}")"
ACTOR_WORLD_SIZE="$(actor_world_size_for_mode "${PLACEMENT_MODE}")"

if [[ "${ENV_LIST}" == "auto" ]]; then
  ENV_LIST="$(default_env_list_for_mode "${PLACEMENT_MODE}")"
fi

echo "[matrix] total_num_envs(resolved): ${ENV_LIST}"
echo "[matrix] actor_world_size: ${ACTOR_WORLD_SIZE}"

for mbs in "${MBS_VALUES[@]}"; do
  mbs="${mbs// /}"
  if [[ ! "${mbs}" =~ ^[0-9]+$ ]]; then
    echo "[matrix][error] invalid micro_batch_size value: ${mbs}" >&2
    exit 2
  fi
  if (( mbs <= 0 )); then
    echo "[matrix][error] micro_batch_size must be > 0, got ${mbs}" >&2
    exit 2
  fi
  if (( GBS % (mbs * ACTOR_WORLD_SIZE) != 0 )); then
    echo "[matrix][error] invalid micro_batch_size=${mbs} for global_batch_size=${GBS} and actor_world_size=${ACTOR_WORLD_SIZE}; require global_batch_size % (micro_batch_size * actor_world_size) == 0" >&2
    exit 2
  fi
done

IFS=',' read -r -a ENV_VALUES <<< "${ENV_LIST}"
for env_n in "${ENV_VALUES[@]}"; do
  env_n="${env_n// /}"
  if [[ ! "${env_n}" =~ ^[0-9]+$ ]]; then
    echo "[matrix][error] invalid env value: ${env_n}" >&2
    exit 2
  fi
  if (( env_n < GPU_MULTIPLE )); then
    echo "[matrix][error] env value ${env_n} is below minimum ${GPU_MULTIPLE} for placement=${PLACEMENT_MODE}" >&2
    exit 2
  fi
  if (( env_n % GPU_MULTIPLE != 0 )); then
    echo "[matrix][error] env value ${env_n} must be a multiple of ${GPU_MULTIPLE} for placement=${PLACEMENT_MODE}" >&2
    exit 2
  fi
done

for env_n in "${ENV_VALUES[@]}"; do
  for mbs in "${MBS_VALUES[@]}"; do
    if [[ ${stop_requested} -eq 1 ]]; then
      break 2
    fi
    env_n="${env_n// /}"
    mbs="${mbs// /}"
    run_id="env${env_n}_mbs${mbs}"
    run_stamp="$(date +'%Y%m%d-%H%M%S')"
    train_start="$(date -Iseconds)"
    train_log_file="${TRAIN_LOG_DIR}/${run_id}-${run_stamp}.log"
    monitor_run_name="monitor-${RUN_TAG}-${CONFIG_NAME}-${run_id}-${run_stamp}"
    train_exit_code=0
    status="ok"

    # Check if experiment is already completed
    if [[ -n "${completed_experiments[${run_id}]+x}" ]]; then
      echo "[matrix] skip ${run_id} (already completed)"
      continue
    fi

    echo "[matrix] start ${run_id}"

    if [[ "${ENABLE_MONITOR}" == "1" ]]; then
      monitor_log_file="${TRAIN_LOG_DIR}/${run_id}-${run_stamp}.monitor.log"
      monitor_cmd=(
        "${MONITOR_PYTHON}" -m yunzhe.monitor
        --interval "${MONITOR_INTERVAL}"
        --output-dir "${MONITOR_OUTPUT_DIR}"
        --name "${monitor_run_name}"
      )
      if [[ "${MONITOR_MAX_SAMPLES}" != "0" ]]; then
        monitor_cmd+=(--max-samples "${MONITOR_MAX_SAMPLES}")
      else
        monitor_cmd+=(--duration 86400)
      fi
      # Stream monitor logs to terminal and persist to file at the same time.
      "${monitor_cmd[@]}" > >(tee -a "${monitor_log_file}") 2>&1 &
      monitor_pid=$!
      sleep 1
      if ! kill -0 "${monitor_pid}" 2>/dev/null; then
        echo "[matrix][error] monitor exited unexpectedly for ${run_id}" >&2
        status="failed"
        train_end="$(date -Iseconds)"
        echo "${run_id},${PLACEMENT_MODE},${env_n},${mbs},${GBS},${train_start},${train_end},1,${train_log_file},${monitor_run_name},${status}" >> "${MANIFEST_CSV}"
        exit 1
      fi
    else
      monitor_run_name=""
    fi

    set +e
    train_cmd=(
      python examples/embodiment/train_embodied_agent.py
      --config-path "${CONFIG_PATH}"
      --config-name "${CONFIG_NAME}"
      "${PLACEMENT_OVERRIDES[@]}"
      "env.train.total_num_envs=${env_n}"
      "actor.micro_batch_size=${mbs}"
      "actor.global_batch_size=${GBS}"
      "runner.max_steps=${MAX_STEPS}"
      "runner.logger.log_path=${SWEEP_ROOT}/jobs/${run_id}-${run_stamp}"
    )

    printf '[matrix] train cmd:' | tee -a "${train_log_file}"
    printf ' %q' "${train_cmd[@]}" | tee -a "${train_log_file}"
    printf '\n' | tee -a "${train_log_file}"

    "${train_cmd[@]}" > >(tee -a "${train_log_file}") 2>&1
    train_exit_code=$?
    set -e

    cleanup_monitor

    train_end="$(date -Iseconds)"
    if [[ ${train_exit_code} -eq 130 ]] || [[ ${stop_requested} -eq 1 ]]; then
      status="interrupted"
      stop_requested=1
    elif [[ ${train_exit_code} -ne 0 ]]; then
      status="failed"
    fi

    echo "${run_id},${PLACEMENT_MODE},${env_n},${mbs},${GBS},${train_start},${train_end},${train_exit_code},${train_log_file},${monitor_run_name},${status}" >> "${MANIFEST_CSV}"
    echo "[matrix] done ${run_id} exit=${train_exit_code}"

    if [[ ${train_exit_code} -ne 0 ]] && [[ ${train_exit_code} -ne 130 ]]; then
      echo "[matrix][error] train command failed for ${run_id} (exit=${train_exit_code}), aborting sweep" >&2
      exit ${train_exit_code}
    fi

    if [[ ${stop_requested} -eq 1 ]]; then
      break 2
    fi
  done
done

if [[ ${stop_requested} -eq 1 ]]; then
  echo "[matrix] interrupted by user. partial results kept at: ${SWEEP_ROOT}"
  exit 130
fi

echo "[matrix] finished. manifest: ${MANIFEST_CSV}"
