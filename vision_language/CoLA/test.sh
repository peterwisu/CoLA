#!/bin/bash
set -e

# ============================================================
# CONFIG — edit these paths for your setup
# ============================================================
MAIN_DIR="./"
DATASET="unc+"
EVAL_SET="testB"

DATA_ROOT="./ln_data/"
SPLIT_ROOT="./mask_data/"
BACKBONE_CKPT_DIR="./bb_ckpt/"

NUM_GPUS=1
NUM_WORKERS=6
BATCH_SIZE=20
# The rank and alpha of CoLA and LoRA that match the training setup.
LORA_R=16
LORA_ALPHA=8
# ============================================================

LAST_DIR_NAME="$(basename "$(pwd)")"
OUTPUT_DIR="${MAIN_DIR}checkpoints/${LAST_DIR_NAME}/${DATASET}/"
EVAL_MODEL="${OUTPUT_DIR}best_mask_checkpoint.pth"

mkdir -p "${OUTPUT_DIR}"

python -m torch.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --use_env \
    --master_port 12345 \
    eval.py \
    --bert_enc_num 12 \
    --imsize 448 \
    --data_root "${DATA_ROOT}" \
    --split_root "${SPLIT_ROOT}" \
    --batch_size ${BATCH_SIZE} \
    --dataset "${DATASET}" \
    --eval_set "${EVAL_SET}" \
    --vl_enc_layers 3 \
    --max_query_len 40 \
    --dim_feedforward 1024 \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers ${NUM_WORKERS} \
    --eval_model "${EVAL_MODEL}" \
    --backbone ViTDet \
    --is_segment \
    --bb_ckpt_dir "${BACKBONE_CKPT_DIR}" \
    --is_eliminate \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA}