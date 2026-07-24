#!/bin/bash
set -e

# ============================================================
# CONFIG — edit these for your setup
# ============================================================
MAIN_DIR="./"

VISUAL_BACKBONE="pvt"        # "resnet" or "pvt"
SWIN_SIZE="swin_l"
LR="0.0002"
TRAIN_BATCH_SIZE=2
GRAD_ACCUM=4

LORA_R=16
LORA_ALPHA=84
## Paramters match settting for CoLA and LoRA
#LORA_R=48
#LORA_ALPHA=24
# ============================================================

LAST_DIR_NAME="$(basename "$(pwd)")"
OUTPUT_DIR="${MAIN_DIR}checkpoints/${LAST_DIR_NAME}/"
SESSION_NAME="${LAST_DIR_NAME}"

mkdir -p "${OUTPUT_DIR}"

python train.py \
    --visual_backbone "${VISUAL_BACKBONE}" \
    --lr ${LR} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --tpavi_stages 0 1 2 3 \
    --session_name "${SESSION_NAME}" \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --grad_accum ${GRAD_ACCUM} \
    --output_dir "${OUTPUT_DIR}" \
    --tpavi_va_flag 1 \
    --swin_size "${SWIN_SIZE}"