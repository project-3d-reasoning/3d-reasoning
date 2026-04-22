#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_NAME="2d_only"
GEOMETRY_ABLATION_MODE="2d_only"
export EXPERIMENT_NAME GEOMETRY_ABLATION_MODE
exec "${SCRIPT_DIR}/train_single_ablation.sh"
