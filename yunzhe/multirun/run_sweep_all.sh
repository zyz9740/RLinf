#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

: "${PLACEMENT_MODE:=all}"
: "${RUN_TAG:=all}"

export PLACEMENT_MODE RUN_TAG

bash "${SCRIPT_DIR}/run_matrix_with_monitor.sh" "$@"
