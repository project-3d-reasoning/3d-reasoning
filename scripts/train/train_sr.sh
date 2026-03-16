#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NUM_GPUS="${NUM_GPUS:-}"                    # Set e.g. "4" to use 4 GPUs; leave empty to use all local GPUs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils/select_available_gpus.sh"

# Set NUM_GPUS to request N GPUs. If unset, request all local GPUs.
if [ -n "${NUM_GPUS:-}" ]; then
    REQUESTED_GPUS="$NUM_GPUS"
else
    REQUESTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi
select_available_gpus "$REQUESTED_GPUS" || exit 1
NPROC_PER_NODE="$SELECTED_GPU_COUNT"

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"   # [ModelArguments] Pretrained model path
MODEL_PATH="${MODEL_PATH%/}"               # Safety: strip trailing slash for HF repo id validation


GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
TUNE_MM_VISION=False
TUNE_MM_VISION_LORA=False
TUNE_GEOMETRY_ENCODER=False
TUNE_GEOMETRY_ENCODER_LORA=False
FEATURE_FUSION_METHOD="decompose_add"      # choices: add/concat/cross_attention/gated/weighted/decompose_add/decompose_concat
DECOMPOSE_HIDDEN_SIZE=2048                 # Set empty to follow hidden_size (e.g. DECOMPOSE_HIDDEN_SIZE="")
FUSION_ALIGN_MODE="cosine"                 # choices: cosine/infonce
FUSION_ORTHO_MODE="cosine"                 # choices: cosine/mine
FUSION_LAMBDA_ALIGN=1.0
FUSION_LAMBDA_ORTHO=1.0
FUSION_LAMBDA_RECON=1.0
FUSION_LAMBDA_WARMUP=True
FUSION_LAMBDA_WARMUP_STEPS=100
OUTPUT_DIR="PATH_TO_OUTPUT_DIR"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="spar_234k,llava_hound_64k"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
LR=1e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))
if [ "$GRADIENT_ACCUMULATION_STEPS" -lt 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=1
fi

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE"

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision $TUNE_MM_VISION \
            --tune_mm_vision_lora $TUNE_MM_VISION_LORA \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 12800 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 50 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_geometry_encoder true \
            --tune_geometry_encoder $TUNE_GEOMETRY_ENCODER \
            --tune_geometry_encoder_lora $TUNE_GEOMETRY_ENCODER_LORA \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --feature_fusion_method $FEATURE_FUSION_METHOD \
            --decompose_hidden_size $DECOMPOSE_HIDDEN_SIZE \
            --fusion_align_mode $FUSION_ALIGN_MODE \
            --fusion_ortho_mode $FUSION_ORTHO_MODE \
            --fusion_lambda_align $FUSION_LAMBDA_ALIGN \
            --fusion_lambda_ortho $FUSION_LAMBDA_ORTHO \
            --fusion_lambda_recon $FUSION_LAMBDA_RECON \
            --fusion_lambda_warmup $FUSION_LAMBDA_WARMUP \
            --fusion_lambda_warmup_steps $FUSION_LAMBDA_WARMUP_STEPS \
            > ${OUTPUT_DIR}/train.log 2>&1
