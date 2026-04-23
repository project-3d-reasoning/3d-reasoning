#!/usr/bin/env bash
# Evaluate a single trained checkpoint under five geometry-ablation modes on
# ScanRefer and ScanNet 4-frames.
#
# Usage:
#   bash scripts/evaluation/eval_geometry_ablation_single_model.sh /path/to/model
#
# Optional env vars:
#   NUM_GPUS=8
#   EVAL_OUTPUT_ROOT=logs/eval_geometry_ablation_single_model
#   GEOMETRY_ABLATION_NOISE_SIGMA=1.0
#   GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE=4
#   MAX_NUM_FRAMES=32
#   MAX_LENGTH=12800
#   BATCH_SIZE=1
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

DEFAULT_MODEL_PATH="/data7t-root/huggingface/hub/models--zd11024--vgllm-3d-vggt-8b/snapshots/82e48eee69a4e168b1a25e26ab752d965db3b408"
MODEL_PATH=${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}
if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: bash scripts/evaluation/eval_geometry_ablation_single_model.sh <model_path>" >&2
    exit 1
fi

export LMMS_EVAL_LAUNCHER=${LMMS_EVAL_LAUNCHER:-accelerate}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-logs/eval_geometry_ablation_single_model}
GEOMETRY_ABLATION_NOISE_SIGMA=${GEOMETRY_ABLATION_NOISE_SIGMA:-1.0}
GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE=${GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE:-4}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-32}
MAX_LENGTH=${MAX_LENGTH:-12800}
BATCH_SIZE=${BATCH_SIZE:-1}
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
    local exp_name="$1"
    local task_name="$2"
    local task_output_dir="$3"
    local log_file="$4"
    local model_args="pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=${MAX_NUM_FRAMES},max_length=${MAX_LENGTH},geometry_ablation_mode=${exp_name},geometry_ablation_noise_sigma=${GEOMETRY_ABLATION_NOISE_SIGMA},geometry_ablation_layout_block_size=${GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE}"

    if [ "${task_name}" = "scanrefer" ]; then
        model_args="${model_args},add_frame_index=true"
    fi

    {
        echo "----------------------------------------"
        echo "Task: ${task_name}"
        echo "Ablation Mode: ${exp_name}"
        echo "Model Path: ${MODEL_PATH}"
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
        --log_samples_suffix "${exp_name}_${task_name}" \
        --log_samples \
        --output_path "${task_output_dir}" \
        2>&1 | tee -a "${log_file}"
}

declare -a EXPERIMENTS=(
    "2d_only"
    "depth_noise"
    "corr_break"
    "pos_break"
    "layout_break"
)

mkdir -p "${OUTPUT_BASE}"

for exp_name in "${EXPERIMENTS[@]}"; do
    exp_output_dir="${OUTPUT_BASE}/${exp_name}"
    mkdir -p "${exp_output_dir}"
    log_file="${exp_output_dir}/eval.log"

    {
        echo "========================================"
        echo "Experiment: ${exp_name}"
        echo "Model Path: ${MODEL_PATH}"
        echo "Num GPUs: ${NUM_GPUS}"
        echo "Noise Sigma: ${GEOMETRY_ABLATION_NOISE_SIGMA}"
        echo "Layout Block Size: ${GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE}"
        echo "Experiment Output: ${exp_output_dir}"
        echo "========================================"
    } | tee -a "${log_file}"

    run_eval "${exp_name}" "scanrefer" "${exp_output_dir}/scanrefer" "${log_file}"
    run_eval "${exp_name}" "scannet_4frames" "${exp_output_dir}/scannet_4frames" "${log_file}"

    {
        echo "Completed experiment: ${exp_name}"
        echo "End Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
    } | tee -a "${log_file}"
done

echo "All single-model geometry ablation evaluations finished."
echo "Results saved to: ${OUTPUT_BASE}"
