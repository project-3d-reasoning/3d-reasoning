#!/bin/bash

# Select available GPUs for launch scripts.
#
# Priority:
# 1) GPUs that satisfy idle thresholds:
#    - utilization.gpu <= GPU_MAX_UTIL (default 20%)
# 2) If idle GPUs are insufficient:
#    - default: fail fast
#    - if GPU_ALLOW_BACKFILL=1: fill remaining slots with GPUs that have the
#      highest free memory.
#
# Inputs:
#   $1: requested number of GPUs (integer > 0)
#
# Outputs:
#   export CUDA_VISIBLE_DEVICES
#   export SELECTED_GPU_COUNT
select_available_gpus() {
    local requested="$1"
    local max_util="${GPU_MAX_UTIL:-20}"
    local allow_backfill="${GPU_ALLOW_BACKFILL:-0}"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[GPU-SELECT] nvidia-smi not found." >&2
        return 1
    fi

    if ! [[ "$requested" =~ ^[0-9]+$ ]] || [ "$requested" -le 0 ]; then
        echo "[GPU-SELECT] Invalid requested GPU count: '$requested'." >&2
        return 1
    fi

    # Respect manually specified CUDA_VISIBLE_DEVICES.
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        SELECTED_GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
        echo "[GPU-SELECT] Using preset CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        export SELECTED_GPU_COUNT
        return 0
    fi

    local gpu_stats
    gpu_stats=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
    if [ -z "$gpu_stats" ]; then
        echo "[GPU-SELECT] Failed to query GPU stats." >&2
        return 1
    fi

    local idle_candidates
    idle_candidates=$(echo "$gpu_stats" | awk -F',' -v max_util="$max_util" '
        {
            gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
            # Force numeric comparison so values are never compared lexically.
            idx=$1 + 0; used=$2 + 0; free=$3 + 0; util=$4 + 0;
            # Force numeric comparison. Some drivers may report N/A; in awk, (value + 0)
            # safely coerces non-numeric strings to 0 instead of doing lexicographic compare.
            used_num=(used + 0); free_num=(free + 0); util_num=(util + 0);
            if (util_num <= max_util) {
                print idx "," used_num "," free_num "," util_num;
            }
        }
    ' | sort -t',' -k2,2n -k4,4n -k3,3nr)

    local selected=()
    local selected_set=","
    local idx used free util

    while IFS=',' read -r idx used free util; do
        [ -z "$idx" ] && continue
        selected+=("$idx")
        selected_set="${selected_set}${idx},"
        [ "${#selected[@]}" -ge "$requested" ] && break
    done <<< "$idle_candidates"

    # If idle GPUs are not enough, optionally backfill from all GPUs by free memory.
    if [ "${#selected[@]}" -lt "$requested" ] && [ "$allow_backfill" = "1" ]; then
        local all_candidates
        all_candidates=$(echo "$gpu_stats" | awk -F',' '
            {
                gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
                print ($1 + 0) "," ($2 + 0) "," ($3 + 0) "," ($4 + 0);
            }
        ' | sort -t',' -k3,3nr -k4,4n -k2,2n)

        while IFS=',' read -r idx used free util; do
            [ -z "$idx" ] && continue
            if [[ "$selected_set" == *",$idx,"* ]]; then
                continue
            fi
            selected+=("$idx")
            selected_set="${selected_set}${idx},"
            [ "${#selected[@]}" -ge "$requested" ] && break
        done <<< "$all_candidates"
    fi

    if [ "${#selected[@]}" -lt "$requested" ]; then
        echo "[GPU-SELECT] Requested $requested idle GPUs, but only ${#selected[@]} matched thresholds (util<=${max_util}%)." >&2
        echo "[GPU-SELECT] Raw stats: $gpu_stats" >&2
        if [ "$allow_backfill" != "1" ]; then
            echo "[GPU-SELECT] Set GPU_ALLOW_BACKFILL=1 to allow non-idle fallback by free memory." >&2
        fi
        return 1
    fi

    CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${selected[*]}")
    SELECTED_GPU_COUNT="${#selected[@]}"

    export CUDA_VISIBLE_DEVICES
    export SELECTED_GPU_COUNT

    echo "[GPU-SELECT] Selected GPUs: $CUDA_VISIBLE_DEVICES" >&2
}
