#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for depth channel probe.
# You can override defaults by exporting env vars before running:
#   ANNOTATION_PATH, MEDIA_ROOT, SAMPLE_RATIO, VGGT_MODEL_PATH, DEVICE, OUTPUT_DIR
#   PROBE_TRAIN_STEPS, PROBE_LR, PROBE_BATCH_SIZE, MAX_PATCH_SAMPLES
# Example:
#   ANNOTATION_PATH=data/train/scan2cap_train_32frames.json \
#   SAMPLE_RATIO=0.1 \
#   bash scripts/analysis/run_depth_channel_probe.sh

ANNOTATION_PATH="${ANNOTATION_PATH:-data/train/scanrefer_train_32frames.json}"
MEDIA_ROOT="${MEDIA_ROOT:-data/media}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.1}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-inspire/hdd/project/qproject-fundationmodel/public/yxliu/test/Demongorgan/VG-LLM/models/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/depth_probe}"
USE_INTRINSICS="${USE_INTRINSICS:-1}"
PROBE_TRAIN_STEPS="${PROBE_TRAIN_STEPS:-1000}"
PROBE_LR="${PROBE_LR:-0.02}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-8192}"
MAX_PATCH_SAMPLES="${MAX_PATCH_SAMPLES:-0}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${OUTPUT_DIR}/depth_probe_${TIMESTAMP}.json"

CMD=(
  python3 scripts/analysis/depth_channel_probe.py
  --annotation_path "${ANNOTATION_PATH}"
  --media_root "${MEDIA_ROOT}"
  --sample_ratio "${SAMPLE_RATIO}"
  --vggt_model_path "${VGGT_MODEL_PATH}"
  --device "${DEVICE}"
  --log_path "${LOG_PATH}"
  --probe_train_steps "${PROBE_TRAIN_STEPS}"
  --probe_lr "${PROBE_LR}"
  --probe_batch_size "${PROBE_BATCH_SIZE}"
  --max_patch_samples "${MAX_PATCH_SAMPLES}"
)

if [[ "${USE_INTRINSICS}" == "1" ]]; then
  CMD+=(--use_intrinsics)
fi

# Forward any extra CLI args to the python script.
CMD+=("$@")

echo "[INFO] Running depth probe..."
echo "[INFO] Log path: ${LOG_PATH}"
echo "[INFO] Command: ${CMD[*]}"
"${CMD[@]}"

echo "[DONE] Depth probe finished."
echo "[DONE] Log saved to: ${LOG_PATH}"
