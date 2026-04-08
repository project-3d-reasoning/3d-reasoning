#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="/inspire/hdd/project/qproject-fundationmodel/public/yxliu/test/Demongorgan/VG-LLM/models/Qwen2.5-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="/inspire/hdd/project/qproject-fundationmodel/public/yxliu/test/Demongorgan/VG-LLM/models/VGGT-1B"
OUTPUT_DIR="output/7b-add-newtokens-residual"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="scan2cap,scanrefer,scannet_det"                  # [DataArguments] Dataset with sampling rate
USE_BBOX_SPECIAL_TOKENS=True
TOKENIZER_USE_FAST=True
MODEL_MAX_LENGTH=12800
NUM_TRAIN_EPOCHS=1

# BBox token / residual experiment knobs
BBOX_COORDINATE_LABEL_SMOOTHING=0.06
BBOX_COORDINATE_SMOOTHING_NEIGHBOR_RADIUS=2
USE_BBOX_RESIDUAL_HEAD=True
BBOX_RESIDUAL_LOSS_WEIGHT=10
BBOX_RESIDUAL_LOSS_WEIGHT_WARMUP_RATIO=0.3
BBOX_RESIDUAL_LOSS_RATIO_TARGET=0.1
BBOX_RESIDUAL_LOSS_RATIO_START=0.5
BBOX_RESIDUAL_LOSS_WEIGHT_MAX=50.0

# Probe is diagnostic only; keep this high enough for scannet_det long outputs
BBOX_PROBE_INTERVAL=500
BBOX_PROBE_NUM_SAMPLES=2
BBOX_PROBE_MAX_NEW_TOKENS=12800

# ======================
# Training Hyperparameters
# ======================
export NCCL_NVLS_ENABLE=0
LR=3e-5
BBOX_RESIDUAL_HEAD_LR=3e-5
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
            --bbox_residual_head_lr $BBOX_RESIDUAL_HEAD_LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length $MODEL_MAX_LENGTH \
            --tokenizer_use_fast $TOKENIZER_USE_FAST \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs $NUM_TRAIN_EPOCHS \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 500 \
            --save_total_limit 3 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_bbox_special_tokens $USE_BBOX_SPECIAL_TOKENS \
            --bbox_coordinate_label_smoothing $BBOX_COORDINATE_LABEL_SMOOTHING \
            --bbox_coordinate_smoothing_neighbor_radius $BBOX_COORDINATE_SMOOTHING_NEIGHBOR_RADIUS \
            --use_bbox_residual_head $USE_BBOX_RESIDUAL_HEAD \
            --bbox_residual_loss_weight $BBOX_RESIDUAL_LOSS_WEIGHT \
            --bbox_residual_loss_weight_warmup_ratio $BBOX_RESIDUAL_LOSS_WEIGHT_WARMUP_RATIO \
            --bbox_residual_loss_ratio_target $BBOX_RESIDUAL_LOSS_RATIO_TARGET \
            --bbox_residual_loss_ratio_start $BBOX_RESIDUAL_LOSS_RATIO_START \
            --bbox_residual_loss_weight_max $BBOX_RESIDUAL_LOSS_WEIGHT_MAX \
            --bbox_probe_interval $BBOX_PROBE_INTERVAL \
            --bbox_probe_num_samples $BBOX_PROBE_NUM_SAMPLES \
            --bbox_probe_max_new_tokens $BBOX_PROBE_MAX_NEW_TOKENS \
            --use_geometry_encoder True \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --feature_fusion_method "add" \
            > ${OUTPUT_DIR}/train.log 2>&1
