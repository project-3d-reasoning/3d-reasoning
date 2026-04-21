#!/usr/bin/env bash
# Evaluate all geometry ablation checkpoints on ScanRefer and ScanNet 4-frames.
# You can either:
# 1) export MODEL_DIR_* env vars to explicit run directories, or
# 2) rely on auto-discovery of the latest run under TRAIN_OUTPUT_ROOT/<exp_name>/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export LMMS_EVAL_LAUNCHER=${LMMS_EVAL_LAUNCHER:-accelerate}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-experiments/geometry_ablation}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-logs/eval_geometry_ablation}
DATE_STR=$(TZ="${EVAL_TZ:-Asia/Shanghai}" date "+%Y%m%d_%H%M%S")
OUTPUT_BASE="${EVAL_OUTPUT_ROOT}/${DATE_STR}"

MODEL_DIR_2D_ONLY=${MODEL_DIR_2D_ONLY:-}
MODEL_DIR_DEPTH_NOISE=${MODEL_DIR_DEPTH_NOISE:-}
MODEL_DIR_CORR_BREAK=${MODEL_DIR_CORR_BREAK:-}
MODEL_DIR_POS_BREAK=${MODEL_DIR_POS_BREAK:-}
MODEL_DIR_LAYOUT_BREAK=${MODEL_DIR_LAYOUT_BREAK:-}

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

find_latest_run() {
    local root="$1"
    if [ ! -d "${root}" ]; then
        return 1
    fi
    find "${root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

resolve_model_dir() {
    local explicit_dir="$1"
    local exp_name="$2"
    if [ -n "${explicit_dir}" ]; then
        printf "%s\n" "${explicit_dir}"
        return 0
    fi
    find_latest_run "${TRAIN_OUTPUT_ROOT}/${exp_name}"
}

run_eval() {
    local exp_name="$1"
    local model_dir="$2"
    local task_name="$3"
    local task_output_dir="$4"
    local log_file="$5"
    local model_args="pretrained=${model_dir},use_flash_attention_2=true,max_num_frames=32,max_length=12800"

    if [ "${task_name}" = "scanrefer" ]; then
        model_args="${model_args},add_frame_index=true"
    fi

    {
        echo "----------------------------------------"
        echo "Task: ${task_name}"
        echo "Model Dir: ${model_dir}"
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
        --batch_size 1 \
        --log_samples_suffix "${task_name}" \
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

declare -A MODEL_DIRS
MODEL_DIRS["2d_only"]="$(resolve_model_dir "${MODEL_DIR_2D_ONLY}" "2d_only" || true)"
MODEL_DIRS["depth_noise"]="$(resolve_model_dir "${MODEL_DIR_DEPTH_NOISE}" "depth_noise" || true)"
MODEL_DIRS["corr_break"]="$(resolve_model_dir "${MODEL_DIR_CORR_BREAK}" "corr_break" || true)"
MODEL_DIRS["pos_break"]="$(resolve_model_dir "${MODEL_DIR_POS_BREAK}" "pos_break" || true)"
MODEL_DIRS["layout_break"]="$(resolve_model_dir "${MODEL_DIR_LAYOUT_BREAK}" "layout_break" || true)"

mkdir -p "${OUTPUT_BASE}"

for exp_name in "${EXPERIMENTS[@]}"; do
    model_dir="${MODEL_DIRS[$exp_name]}"
    if [ -z "${model_dir}" ] || [ ! -d "${model_dir}" ]; then
        echo "Missing model directory for ${exp_name}." >&2
        echo "Set the matching MODEL_DIR_* env var or place runs under ${TRAIN_OUTPUT_ROOT}/${exp_name}/" >&2
        exit 1
    fi

    exp_output_dir="${OUTPUT_BASE}/${exp_name}"
    mkdir -p "${exp_output_dir}"
    log_file="${exp_output_dir}/eval.log"

    {
        echo "========================================"
        echo "Experiment: ${exp_name}"
        echo "Model Dir: ${model_dir}"
        echo "Num GPUs: ${NUM_GPUS}"
        echo "Experiment Output: ${exp_output_dir}"
        echo "========================================"
    } | tee -a "${log_file}"

    run_eval "${exp_name}" "${model_dir}" "scanrefer" "${exp_output_dir}/scanrefer" "${log_file}"
    run_eval "${exp_name}" "${model_dir}" "scannet_4frames" "${exp_output_dir}/scannet_4frames" "${log_file}"

    {
        echo "Completed experiment: ${exp_name}"
        echo "End Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
    } | tee -a "${log_file}"
done

echo "All geometry ablation evaluations finished."
echo "Results saved to: ${OUTPUT_BASE}"
