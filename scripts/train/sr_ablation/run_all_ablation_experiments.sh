#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SCRIPTS=(
    run_2d_only.sh
    run_depth_noise.sh
    run_corr_break.sh
    run_pos_break.sh
    run_layout_break.sh
)

for script in "${SCRIPTS[@]}"; do
    echo "[RUN] ${script}"
    "${SCRIPT_DIR}/${script}"
    echo "[DONE] ${script}"
    echo
done

echo "All SR ablation experiments finished."
