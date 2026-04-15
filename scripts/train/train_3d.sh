#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=1  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="/data7t-root/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
OUTPUT_DIR="PATH_TO_OUTPUT_DIR"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="scan2cap,scanrefer,scannet_det"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
export NCCL_NVLS_ENABLE=0
LR=2e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

# ======================
# unique_3d Prefix Branch
# ======================
USE_UNIQUE_3D_PREFIX=True
UNIQUE_3D_NUM_QUERIES=20
UNIQUE_3D_PREFIX_NUM_HEADS=8
UNIQUE_3D_PREFIX_DROPOUT=0
UNIQUE_3D_HSIC_WEIGHT=10
UNIQUE_3D_HSIC_SIGMA_2D=-1
UNIQUE_3D_HSIC_SIGMA_3D=-1
UNIQUE_3D_HSIC_MAX_SAMPLES=-1

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
            --use_unique_3d_prefix $USE_UNIQUE_3D_PREFIX \
            --unique_3d_num_queries $UNIQUE_3D_NUM_QUERIES \
            --unique_3d_prefix_num_heads $UNIQUE_3D_PREFIX_NUM_HEADS \
            --unique_3d_prefix_dropout $UNIQUE_3D_PREFIX_DROPOUT \
            --unique_3d_hsic_weight $UNIQUE_3D_HSIC_WEIGHT \
            --unique_3d_hsic_sigma_2d $UNIQUE_3D_HSIC_SIGMA_2D \
            --unique_3d_hsic_sigma_3d $UNIQUE_3D_HSIC_SIGMA_3D \
            --unique_3d_hsic_max_samples $UNIQUE_3D_HSIC_MAX_SAMPLES \
            > ${OUTPUT_DIR}/train.log 2>&1
