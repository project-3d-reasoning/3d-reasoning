#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the 2D feature probe overlap script.
# Defaults are set to run on the ScanNet detection training split.
#
# Example:
#   CUDA_VISIBLE_DEVICES=4 \
#   bash scripts/analysis/run_feature_2d_probe_resample_compare.sh
#
# Override defaults via env vars if needed:
#   ANNOTATION_PATH, MEDIA_ROOT, SAMPLE_RATIO, SEED, RESAMPLE_SEED,
#   MAX_COLLECT_STEPS, MAX_PATCH_SAMPLES, TARGET_FEATURE_LEVEL, DEVICE, OUTPUT_DIR, PYTHON_BIN
#   AUTO_SPLIT_DEVICES, QWEN_DEVICE, VGGT_DEVICE, PROBE_DEVICE

if [[ -x /data2/miniconda3/envs/vg/bin/python3.11 ]]; then
  DEFAULT_PYTHON_BIN="/data2/miniconda3/envs/vg/bin/python3.11"
else
  DEFAULT_PYTHON_BIN="python3"
fi

ANNOTATION_PATH="${ANNOTATION_PATH:-data/train/scannet_det_train_4frames.json}"
MEDIA_ROOT="${MEDIA_ROOT:-data/media}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.1}"
SEED="${SEED:-42}"
RESAMPLE_SEED="${RESAMPLE_SEED:-43}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data7t-root/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TARGET_FEATURE_LEVEL="${TARGET_FEATURE_LEVEL:-merged}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/feature_2d_probe_compare/scannet}"
PROBE_TRAIN_STEPS="${PROBE_TRAIN_STEPS:-1000}"
PROBE_LR="${PROBE_LR:-0.002}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4096}"
CHANNEL_RATIO="${CHANNEL_RATIO:-0.2}"
MAX_COLLECT_STEPS="${MAX_COLLECT_STEPS:-1000}"
MAX_FRAMES_PER_ENTRY="${MAX_FRAMES_PER_ENTRY:-4}"
FRAMES_PER_FORWARD="${FRAMES_PER_FORWARD:-4}"
PATCHES_PER_FRAME="${PATCHES_PER_FRAME:-256}"
MAX_PATCH_SAMPLES="${MAX_PATCH_SAMPLES:-120000}"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
AUTO_SPLIT_DEVICES="${AUTO_SPLIT_DEVICES:-auto}"
QWEN_DEVICE="${QWEN_DEVICE:-}"
VGGT_DEVICE="${VGGT_DEVICE:-}"
PROBE_DEVICE="${PROBE_DEVICE:-}"

VISIBLE_GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#_VISIBLE_GPU_LIST[@]}"
fi

if [[ "${AUTO_SPLIT_DEVICES}" == "auto" ]]; then
  if [[ "${DEVICE}" == cuda* && "${VISIBLE_GPU_COUNT}" -ge 2 ]]; then
    AUTO_SPLIT_DEVICES="1"
  else
    AUTO_SPLIT_DEVICES="0"
  fi
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${OUTPUT_DIR}/feature_2d_probe_compare_scannet_${TIMESTAMP}.json"

CMD=(
  "${PYTHON_BIN}" scripts/analysis/feature_2d_probe_resample_compare.py
  --annotation_path "${ANNOTATION_PATH}"
  --media_root "${MEDIA_ROOT}"
  --sample_ratio "${SAMPLE_RATIO}"
  --seed "${SEED}"
  --resample_seed "${RESAMPLE_SEED}"
  --qwen_model_path "${QWEN_MODEL_PATH}"
  --vggt_model_path "${VGGT_MODEL_PATH}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --target_feature_level "${TARGET_FEATURE_LEVEL}"
  --probe_train_steps "${PROBE_TRAIN_STEPS}"
  --probe_lr "${PROBE_LR}"
  --probe_batch_size "${PROBE_BATCH_SIZE}"
  --channel_ratio "${CHANNEL_RATIO}"
  --max_collect_steps "${MAX_COLLECT_STEPS}"
  --max_frames_per_entry "${MAX_FRAMES_PER_ENTRY}"
  --frames_per_forward "${FRAMES_PER_FORWARD}"
  --patches_per_frame "${PATCHES_PER_FRAME}"
  --max_patch_samples "${MAX_PATCH_SAMPLES}"
  --log_path "${LOG_PATH}"
)

if [[ "${AUTO_SPLIT_DEVICES}" == "1" ]]; then
  CMD+=(--auto_split_devices)
fi

if [[ -n "${QWEN_DEVICE}" ]]; then
  CMD+=(--qwen_device "${QWEN_DEVICE}")
fi

if [[ -n "${VGGT_DEVICE}" ]]; then
  CMD+=(--vggt_device "${VGGT_DEVICE}")
fi

if [[ -n "${PROBE_DEVICE}" ]]; then
  CMD+=(--probe_device "${PROBE_DEVICE}")
fi

# Forward any extra CLI args to the python script.
CMD+=("$@")

echo "[INFO] Running 2D feature probe overlap on ScanNet..."
echo "[INFO] Annotation path: ${ANNOTATION_PATH}"
echo "[INFO] Auto split devices: ${AUTO_SPLIT_DEVICES}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] Log path: ${LOG_PATH}"
echo "[INFO] Command: ${CMD[*]}"
"${CMD[@]}"

echo "[DONE] 2D feature probe overlap finished."
echo "[DONE] Log saved to: ${LOG_PATH}"
