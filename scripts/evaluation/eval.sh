set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=zd11024/vgllm-qa-vggt-4b
USE_HSIC_FUSION=true
HSIC_LOSS_WEIGHT=1e-3
HSIC_RBF_SIGMA_2D=1.0  # Set to -1 to auto-estimate sigma with the median heuristic
HSIC_RBF_SIGMA_3D=1.0  # Set to -1 to auto-estimate sigma with the median heuristic
UNIQUE_3D_HSIC_MAX_SAMPLES=-1  # Max random token/feature points per sample for HSIC; <= 0 uses all points

accelerate launch --num_processes=8 -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,use_hsic_fusion=$USE_HSIC_FUSION,hsic_loss_weight=$HSIC_LOSS_WEIGHT,hsic_rbf_sigma_2d=$HSIC_RBF_SIGMA_2D,hsic_rbf_sigma_3d=$HSIC_RBF_SIGMA_3D,unique_3d_hsic_max_samples=$UNIQUE_3D_HSIC_MAX_SAMPLES \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path
