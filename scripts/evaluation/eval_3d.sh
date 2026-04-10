set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=scannet_4frames # choices: [scan2cap, scanrefer, scannet_4frames, scannet_6frames]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=zd11024/vgllm-3d-vggt-4b
USE_HSIC_FUSION=true
HSIC_LOSS_WEIGHT=10
HSIC_RBF_SIGMA_2D=-1  # Set to -1 to auto-estimate sigma with the median heuristic
HSIC_RBF_SIGMA_3D=-1  # Set to -1 to auto-estimate sigma with the median heuristic
UNIQUE_3D_HSIC_MAX_SAMPLES=-1  # Max random token/feature points per sample for HSIC; <= 0 uses all points

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,use_hsic_fusion=$USE_HSIC_FUSION,hsic_loss_weight=$HSIC_LOSS_WEIGHT,hsic_rbf_sigma_2d=$HSIC_RBF_SIGMA_2D,hsic_rbf_sigma_3d=$HSIC_RBF_SIGMA_3D,unique_3d_hsic_max_samples=$UNIQUE_3D_HSIC_MAX_SAMPLES"
if [ "$benchmark" = "scanrefer" ]; then
    model_args_str="${model_args_str},add_frame_index=true"
fi

accelerate launch --num_processes=8  --main_process_port 29501 -m lmms_eval \
    --model vgllm \
    --model_args "$model_args_str" \
    --tasks ${benchmark} \
    --batch_size 1 \
    --log_samples_suffix original \
    --log_samples \
    --output_path $output_path
