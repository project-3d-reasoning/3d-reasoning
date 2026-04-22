#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

EXPERIMENT_NAME=${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}
GEOMETRY_ABLATION_MODE=${GEOMETRY_ABLATION_MODE:?GEOMETRY_ABLATION_MODE is required}

OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/geometry_ablation}
DATASETS=${DATASETS:-scan2cap,scanrefer,scannet_det}
CACHE_DIR=${CACHE_DIR:-./cache}
SEED=${SEED:-42}
LR=${LR:-2e-5}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-64}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero2_opt.json}
FEATURE_FUSION_METHOD=${FEATURE_FUSION_METHOD:-add}
GEOMETRY_ABLATION_NOISE_SIGMA=${GEOMETRY_ABLATION_NOISE_SIGMA:-1.0}
GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE=${GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE:-4}
MODEL_PATH="/inspire/hdd/project/qproject-fundationmodel/public/yxliu/test/Demongorgan/VG-LLM/models/Qwen2.5-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_PATH="/inspire/hdd/project/qproject-fundationmodel/public/yxliu/test/Demongorgan/VG-LLM/models/VGGT-1B"
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
if [ -z "${NPROC_PER_NODE:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
        NPROC_PER_NODE=${#visible_gpu_array[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    else
        NPROC_PER_NODE=1
    fi
fi

GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / NPROC_PER_NODE))
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=1
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"

{
    echo "========================================"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Ablation Mode: ${GEOMETRY_ABLATION_MODE}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "Model Path: ${MODEL_PATH}"
    echo "Geometry Encoder Path: ${GEOMETRY_ENCODER_PATH}"
    echo "Datasets: ${DATASETS}"
    echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "========================================"
} | tee -a "${LOG_FILE}"

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

CMD=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    src/qwen_vl/train/train_qwen.py
    --model_name_or_path "${MODEL_PATH}"
    --tune_mm_llm True
    --tune_mm_vision False
    --tune_mm_mlp False
    --dataset_use "${DATASETS}"
    --output_dir "${OUTPUT_DIR}"
    --cache_dir "${CACHE_DIR}"
    --bf16
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LR}"
    --mm_projector_lr 1e-5
    --vision_tower_lr 1e-6
    --optim adamw_torch
    --model_max_length 12800
    --data_flatten False
    --max_pixels "$((576 * 28 * 28))"
    --min_pixels "$((16 * 28 * 28))"
    --base_interval 2
    --video_max_frames 8
    --video_min_frames 4
    --video_max_frame_pixels "$((1664 * 28 * 28))"
    --video_min_frame_pixels "$((256 * 28 * 28))"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --deepspeed "${DEEPSPEED_CONFIG}"
    --gradient_checkpointing
    --dataloader_num_workers 4
    --group_by_modality_length true
    --seed "${SEED}"
    --report_to none
    --use_geometry_encoder True
    --geometry_encoder_type vggt
    --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}"
    --feature_fusion_method "${FEATURE_FUSION_METHOD}"
    --geometry_ablation_mode "${GEOMETRY_ABLATION_MODE}"
    --geometry_ablation_noise_sigma "${GEOMETRY_ABLATION_NOISE_SIGMA}"
    --geometry_ablation_layout_block_size "${GEOMETRY_ABLATION_LAYOUT_BLOCK_SIZE}"
)

if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi

if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${TRAIN_EXTRA_ARGS})
    CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

{
    echo "========================================"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "End Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Saved to: ${OUTPUT_DIR}"
    echo "========================================"
} | tee -a "${LOG_FILE}"
