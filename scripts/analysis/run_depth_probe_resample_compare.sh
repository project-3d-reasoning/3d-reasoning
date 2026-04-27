#!/usr/bin/env bash
set -euo pipefail

# One-command launcher:
# 1) reuse a baseline depth-probe config
# 2) rerun the probe with a new seed
# 3) compare depth-related channel overlap

BASELINE_LOG_PATH="${BASELINE_LOG_PATH:-logs/depth_probe/depth_probe_20260421_053625.json}"
RESAMPLE_SEED="${RESAMPLE_SEED:-43}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-logs/depth_probe_resample}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-logs/depth_probe_compare}"
SAMPLE_RATIO_OVERRIDE="${SAMPLE_RATIO_OVERRIDE:--1}"
MAX_COLLECT_STEPS_OVERRIDE="${MAX_COLLECT_STEPS_OVERRIDE:--1}"
PROBE_TRAIN_STEPS_OVERRIDE="${PROBE_TRAIN_STEPS_OVERRIDE:-1000}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-}"
VGGT_MODEL_PATH_OVERRIDE="${VGGT_MODEL_PATH_OVERRIDE:-}"

CMD=(
  python3 scripts/analysis/depth_probe_resample_compare.py
  --baseline_log_path "${BASELINE_LOG_PATH}"
  --seed "${RESAMPLE_SEED}"
  --probe_output_dir "${PROBE_OUTPUT_DIR}"
  --compare_output_dir "${COMPARE_OUTPUT_DIR}"
  --sample_ratio "${SAMPLE_RATIO_OVERRIDE}"
  --max_collect_steps "${MAX_COLLECT_STEPS_OVERRIDE}"
  --probe_train_steps "${PROBE_TRAIN_STEPS_OVERRIDE}"
)

if [[ -n "${DEVICE_OVERRIDE}" ]]; then
  CMD+=(--device "${DEVICE_OVERRIDE}")
fi

if [[ -n "${VGGT_MODEL_PATH_OVERRIDE}" ]]; then
  CMD+=(--vggt_model_path "${VGGT_MODEL_PATH_OVERRIDE}")
fi

CMD+=("$@")

echo "[INFO] Running depth probe resample+compare..."
echo "[INFO] Command: ${CMD[*]}"
"${CMD[@]}"
