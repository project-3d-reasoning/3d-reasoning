set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=scannet_4frames # choices: [scan2cap, scanrefer, scannet_4frames, scannet_6frames]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=output/7b-add-newtokens
USE_BBOX_SPECIAL_TOKENS=True

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,use_bbox_special_tokens=$USE_BBOX_SPECIAL_TOKENS"
if [ "$benchmark" = "scanrefer" ]; then
    model_args_str="${model_args_str},add_frame_index=true"
fi

accelerate launch --num_processes=8  --main_process_port 29501 -m lmms_eval \
    --model vgllm \
    --model_args "$model_args_str" \
    --tasks ${benchmark} \
    --batch_size 8 \
    --log_samples_suffix original \
    --log_samples \
    --output_path $output_path
