#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=4                            # Set this manually to match the number of GPUs you want to use

if [ "${NPROC_PER_NODE}" -le 0 ]; then
    echo "Failed to determine a valid NPROC_PER_NODE. Set it to a positive integer."
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
    VISIBLE_GPU_COUNT=${#CUDA_DEVICE_LIST[@]}
    if [ "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]; then
        echo "NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPUs (${VISIBLE_GPU_COUNT}) from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        exit 1
    fi
fi

# ======================
# Path Configuration
# ======================
MODEL_PATH="/data7t-root/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
OUTPUT_DIR="7b-pe-16patch"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="scan2cap,scanrefer,scannet_det"                  # [DataArguments] Dataset with sampling rate
USE_COORD_PE=${USE_COORD_PE:-True}                        # [ModelArguments] Defaults to True; set to False to disable coordinate positional encoding

# ======================
# Training Hyperparameters
# ======================
export NCCL_NVLS_ENABLE=0
LR=2e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
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
            --logging_steps 10 \
            --save_steps 500 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_geometry_encoder True \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --feature_fusion_method "add" \
            --use_coord_pe $USE_COORD_PE \
            > ${OUTPUT_DIR}/train.log 2>&1
