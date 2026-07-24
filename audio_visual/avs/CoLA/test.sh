#!/bin/bash
set -e

# ============================================================
# CONFIG — edit these for your setup
# ============================================================
MAIN_DIR="./"

VISUAL_BACKBONE="pvt"        # "resnet" or "pvt"
LR="0.0005"

LORA_R=16
LORA_ALPHA=8

# absolute path to the trained model checkpoint
WEIGHT="/path/to/model_checkpoint.pth"
# ============================================================

LAST_DIR_NAME="$(basename "$(pwd)")"
OUTPUT_DIR="${MAIN_DIR}checkpoints/${LAST_DIR_NAME}/"
SESSION_NAME="${LAST_DIR_NAME}"

mkdir -p "${OUTPUT_DIR}"

python test.py \
    --visual_backbone "${VISUAL_BACKBONE}" \
    --lr ${LR} \
    --tpavi_stages 0 1 2 3 \
    --session_name "${SESSION_NAME}" \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --output_dir "${OUTPUT_DIR}" \
    --weight "${WEIGHT}" \
    --save_pred_mask \
    --tpavi_va_flag