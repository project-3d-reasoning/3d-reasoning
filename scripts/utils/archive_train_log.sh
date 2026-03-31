#!/bin/bash

setup_train_log_archive() {
    local repo_root="$1"
    local output_dir="$2"

    TRAIN_LOG_ARCHIVE_DIR="${repo_root%/}/train_log"
    TRAIN_LOG_FILE="${output_dir%/}/train.log"
    TRAIN_LOG_OUTPUT_NAME="$(basename "${output_dir%/}")"
    TRAIN_LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
    TRAIN_LOG_ARCHIVED=0

    trap archive_train_log EXIT
}

archive_train_log() {
    if [ "${TRAIN_LOG_ARCHIVED:-0}" -eq 1 ]; then
        return 0
    fi
    TRAIN_LOG_ARCHIVED=1

    if [ -z "${TRAIN_LOG_FILE:-}" ] || [ ! -f "$TRAIN_LOG_FILE" ]; then
        return 0
    fi

    mkdir -p "$TRAIN_LOG_ARCHIVE_DIR"

    local archive_path="${TRAIN_LOG_ARCHIVE_DIR}/${TRAIN_LOG_OUTPUT_NAME}_${TRAIN_LOG_TIMESTAMP}.log"
    local suffix=1
    while [ -e "$archive_path" ]; do
        archive_path="${TRAIN_LOG_ARCHIVE_DIR}/${TRAIN_LOG_OUTPUT_NAME}_${TRAIN_LOG_TIMESTAMP}_${suffix}.log"
        suffix=$((suffix + 1))
    done

    cp "$TRAIN_LOG_FILE" "$archive_path"
    echo "Archived training log to $archive_path"
}
