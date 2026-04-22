#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/sr_ablation}
export DATASETS=${DATASETS:-spar_234k,llava_hound_64k}

exec "${SCRIPT_DIR}/../geometry_ablation/train_single_ablation.sh"
