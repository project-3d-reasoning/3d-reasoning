set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=${BENCHMARK:-scannet_4frames} # choices: [scan2cap, scanrefer, scanrefer_first_frame, scannet_4frames, scannet_6frames]
output_path=${OUTPUT_PATH:-logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")}
model_path=${MODEL_PATH:-/data7t-root/r1/dmgg/VG-LLM/7b-pe}
batch_size=${BATCH_SIZE:-8}
num_processes=${NUM_PROCESSES:-4}
main_process_port=${MAIN_PROCESS_PORT:-29501}

model_args_str="pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800"
if [ "$benchmark" = "scanrefer" ] || [ "$benchmark" = "scanrefer_first_frame" ]; then
    model_args_str="${model_args_str},add_frame_index=true"
fi

cmd=(
    accelerate launch --num_processes "$num_processes" --main_process_port "$main_process_port" -m lmms_eval
    --model vgllm
    --model_args "$model_args_str"
    --tasks "$benchmark"
    --batch_size "$batch_size"
    --output_path "$output_path"
)

if [ "${LOG_SAMPLES:-0}" = "1" ]; then
    cmd+=(--log_samples_suffix original --log_samples)
fi

"${cmd[@]}"
