set -e
export LMMS_EVAL_LAUNCHER="accelerate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils/select_available_gpus.sh"

export NCCL_NVLS_ENABLE=0
benchmark=scan2cap # choices: [scan2cap, scanrefer, scannet_4frames, scannet_6frames]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=/data7t-root/r1/dmgg/VG-LLM/3b-decompose-add-infonce-cosine-ffff
FEATURE_FUSION_METHOD="decompose_add" # choices: add/concat/cross_attention/gated/weighted/decompose_add/decompose_concat
FUSION_ORTHO_MODE="mine"
FUSION_LAMBDA_ORTHO=0.002
USE_LEARNABLE_PREFIX=false
LEARNABLE_PREFIX_LEN=0
TUNE_MM_VISION=false
TUNE_MM_VISION_LORA=false
TUNE_GEOMETRY_ENCODER=false
TUNE_GEOMETRY_ENCODER_LORA=false
NUM_GPUS="${NUM_GPUS:-4}" # Set evaluation GPU count directly here

# Set NUM_GPUS to request N GPUs.
REQUESTED_GPUS="$NUM_GPUS"
select_available_gpus "$REQUESTED_GPUS"
NUM_PROCESSES="$SELECTED_GPU_COUNT"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using num_processes=$NUM_PROCESSES"

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,feature_fusion_method=$FEATURE_FUSION_METHOD,fusion_ortho_mode=$FUSION_ORTHO_MODE,fusion_lambda_ortho=$FUSION_LAMBDA_ORTHO,tune_mm_vision=$TUNE_MM_VISION,tune_mm_vision_lora=$TUNE_MM_VISION_LORA,tune_geometry_encoder=$TUNE_GEOMETRY_ENCODER,tune_geometry_encoder_lora=$TUNE_GEOMETRY_ENCODER_LORA,use_learnable_prefix=$USE_LEARNABLE_PREFIX,learnable_prefix_len=$LEARNABLE_PREFIX_LEN"
if [ "$benchmark" = "scanrefer" ]; then
    model_args_str="${model_args_str},add_frame_index=true"
fi

accelerate launch --num_processes="$NUM_PROCESSES" --main_process_port 29501 -m lmms_eval \
    --model vgllm \
    --model_args "$model_args_str" \
    --tasks ${benchmark} \
    --batch_size 1 \
    --log_samples_suffix original \
    --log_samples \
    --output_path $output_path
