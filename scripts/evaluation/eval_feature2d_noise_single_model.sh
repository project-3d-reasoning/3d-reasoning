#!/usr/bin/env bash
# Evaluate a single checkpoint with Gaussian noise injected into the
# feature-2d-sensitive VGGT channels on ScanRefer and ScanNet detection.
#
# Notes:
# - The ScanNet detection eval task name in lmms_eval is `scannet_4frames`.
# - The hard-coded channels come from
#   `logs/feature_2d_probe_compare/scannet/feature_2d_probe_compare_scannet_20260424_043440.json`
#   (`seed_42`, top-20% channels).
#
# Usage:
#   bash scripts/evaluation/eval_feature2d_noise_single_model.sh
#   bash scripts/evaluation/eval_feature2d_noise_single_model.sh /path/to/model
#
# Optional env vars:
#   NUM_GPUS=8
#   EVAL_OUTPUT_ROOT=logs/eval_feature2d_noise_single_model
#   GEOMETRY_ABLATION_NOISE_SIGMA=1.0
#   MAX_NUM_FRAMES=32
#   MAX_LENGTH=12800
#   BATCH_SIZE=1
#   TASK_SCANNET=scannet_4frames
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

DEFAULT_MODEL_PATH="/data7t-root/huggingface/hub/models--zd11024--vgllm-3d-vggt-8b/snapshots/82e48eee69a4e168b1a25e26ab752d965db3b408"
MODEL_PATH=${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}
if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: bash scripts/evaluation/eval_feature2d_noise_single_model.sh <model_path>" >&2
    exit 1
fi

export LMMS_EVAL_LAUNCHER=${LMMS_EVAL_LAUNCHER:-accelerate}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

EXP_NAME=${EXP_NAME:-feature2d_noise}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-logs/eval_feature2d_noise_single_model}
GEOMETRY_ABLATION_NOISE_SIGMA=${GEOMETRY_ABLATION_NOISE_SIGMA:-1.0}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-32}
MAX_LENGTH=${MAX_LENGTH:-12800}
BATCH_SIZE=${BATCH_SIZE:-1}
TASK_SCANNET=${TASK_SCANNET:-scannet_4frames}
DATE_STR=$(TZ="${EVAL_TZ:-Asia/Shanghai}" date "+%Y%m%d_%H%M%S")
OUTPUT_BASE="${EVAL_OUTPUT_ROOT}/${DATE_STR}"

if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
        NUM_GPUS=${#visible_gpu_array[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        NUM_GPUS=1
    fi
fi

run_eval() {
    local task_name="$1"
    local task_output_dir="$2"
    local log_file="$3"
    local model_args="pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=${MAX_NUM_FRAMES},max_length=${MAX_LENGTH},geometry_ablation_mode=${EXP_NAME},geometry_ablation_noise_sigma=${GEOMETRY_ABLATION_NOISE_SIGMA}"

    if [ "${task_name}" = "scanrefer" ]; then
        model_args="${model_args},add_frame_index=true"
    fi

    {
        echo "----------------------------------------"
        echo "Task: ${task_name}"
        echo "Ablation Mode: ${EXP_NAME}"
        echo "Model Path: ${MODEL_PATH}"
        echo "Noise Sigma: ${GEOMETRY_ABLATION_NOISE_SIGMA}"
        echo "Output Dir: ${task_output_dir}"
        echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "----------------------------------------"
    } | tee -a "${log_file}"

    accelerate launch \
        --num_processes="${NUM_GPUS}" \
        --main_process_port "$(shuf -i 29500-29999 -n 1)" \
        -m lmms_eval \
        --model vgllm \
        --model_args "${model_args}" \
        --tasks "${task_name}" \
        --batch_size "${BATCH_SIZE}" \
        --log_samples_suffix "${EXP_NAME}_${task_name}" \
        --log_samples \
        --output_path "${task_output_dir}" \
        2>&1 | tee -a "${log_file}"
}

mkdir -p "${OUTPUT_BASE}"
LOG_FILE="${OUTPUT_BASE}/eval.log"

{
    echo "========================================"
    echo "Experiment: ${EXP_NAME}"
    echo "Model Path: ${MODEL_PATH}"
    echo "Num GPUs: ${NUM_GPUS}"
    echo "Noise Sigma: ${GEOMETRY_ABLATION_NOISE_SIGMA}"
    echo "ScanNet Task Name: ${TASK_SCANNET}"
    echo "Output Base: ${OUTPUT_BASE}"
    echo "========================================"
} | tee -a "${LOG_FILE}"

run_eval "scanrefer" "${OUTPUT_BASE}/scanrefer" "${LOG_FILE}"
run_eval "${TASK_SCANNET}" "${OUTPUT_BASE}/${TASK_SCANNET}" "${LOG_FILE}"

{
    echo "Completed experiment: ${EXP_NAME}"
    echo "End Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${LOG_FILE}"

echo "feature2d_noise evaluation finished."
echo "Results saved to: ${OUTPUT_BASE}"
