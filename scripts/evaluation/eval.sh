set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=zd11024/vgllm-qa-vggt-4b
USE_GEOMETRY_LASTVIT_SELECTOR=false
GEOMETRY_LASTVIT_TOP_K=1
USE_UNIQUE_3D_PREFIX=false
UNIQUE_3D_NUM_QUERIES=8
UNIQUE_3D_PREFIX_NUM_HEADS=8
UNIQUE_3D_PREFIX_DROPOUT=0.1
UNIQUE_3D_HSIC_WEIGHT=0.01
UNIQUE_3D_HSIC_SIGMA_2D=-1
UNIQUE_3D_HSIC_SIGMA_3D=-1
UNIQUE_3D_HSIC_MAX_SAMPLES=256

accelerate launch --num_processes=8 -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,use_geometry_lastvit_selector=$USE_GEOMETRY_LASTVIT_SELECTOR,geometry_lastvit_top_k=$GEOMETRY_LASTVIT_TOP_K,use_unique_3d_prefix=$USE_UNIQUE_3D_PREFIX,unique_3d_num_queries=$UNIQUE_3D_NUM_QUERIES,unique_3d_prefix_num_heads=$UNIQUE_3D_PREFIX_NUM_HEADS,unique_3d_prefix_dropout=$UNIQUE_3D_PREFIX_DROPOUT,unique_3d_hsic_weight=$UNIQUE_3D_HSIC_WEIGHT,unique_3d_hsic_sigma_2d=$UNIQUE_3D_HSIC_SIGMA_2D,unique_3d_hsic_sigma_3d=$UNIQUE_3D_HSIC_SIGMA_3D,unique_3d_hsic_max_samples=$UNIQUE_3D_HSIC_MAX_SAMPLES \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path
