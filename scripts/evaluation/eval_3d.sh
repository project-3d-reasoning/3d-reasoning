set -e
export LMMS_EVAL_LAUNCHER="accelerate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils/select_available_gpus.sh"

export NCCL_NVLS_ENABLE=0
benchmark=scan2cap # choices: [scan2cap, scanrefer, scannet_4frames, scannet_6frames]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=/data7t-root/r1/dmgg/VG-LLM/3b-hsic
FEATURE_FUSION_METHOD="decompose_add" # choices: add/concat/cross_attention/gated/weighted/decompose_add/decompose_concat/nrsr_add/nrsr_concat/knn_concat
FUSION_NUM_LAYERS=2
FUSION_KNN_K=9
FUSION_KNN_MIN_VALID_RATIO=0.5
FUSION_KNN_POS_MLP_HIDDEN_SIZE=1024
FUSION_ORTHO_MODE="hsic"
FUSION_LAMBDA_ORTHO=0.02 # target loss_ortho_weighted / loss_ce ratio; reused as eval-time lambda for config compatibility
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

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,feature_fusion_method=$FEATURE_FUSION_METHOD,fusion_num_layers=$FUSION_NUM_LAYERS,fusion_knn_k=$FUSION_KNN_K,fusion_knn_min_valid_ratio=$FUSION_KNN_MIN_VALID_RATIO,fusion_knn_pos_mlp_hidden_size=$FUSION_KNN_POS_MLP_HIDDEN_SIZE,fusion_ortho_mode=$FUSION_ORTHO_MODE,fusion_lambda_ortho=$FUSION_LAMBDA_ORTHO,fusion_ortho_target_ratio=$FUSION_LAMBDA_ORTHO,tune_mm_vision=$TUNE_MM_VISION,tune_mm_vision_lora=$TUNE_MM_VISION_LORA,tune_geometry_encoder=$TUNE_GEOMETRY_ENCODER,tune_geometry_encoder_lora=$TUNE_GEOMETRY_ENCODER_LORA,use_learnable_prefix=$USE_LEARNABLE_PREFIX,learnable_prefix_len=$LEARNABLE_PREFIX_LEN"
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
