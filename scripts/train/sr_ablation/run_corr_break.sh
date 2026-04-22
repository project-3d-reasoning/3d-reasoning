#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_NAME="corr_break"
GEOMETRY_ABLATION_MODE="corr_break"
export EXPERIMENT_NAME GEOMETRY_ABLATION_MODE
exec "${SCRIPT_DIR}/train_single_ablation.sh"
