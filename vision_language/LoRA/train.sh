#!/bin/bash
set -e

# ============================================================
# CONFIG — edit these paths for your setup
# ============================================================
MAIN_DIR="./"
DATASET="unc+"

DATA_ROOT="./ln_data/"
SPLIT_ROOT="./mask_data/"
BACKBONE_CKPT_DIR="./bb_ckpt/"

NUM_GPUS=4
NUM_WORKERS=6
BATCH_SIZE=20
GRAD_ACCUM_STEPS=1
EPOCHS=150


LORA_R=16
LORA_ALPHA=8
## Paramters match settting for CoLA and LoRA
# LORA_R=54
# LORA_ALPHA=27
# ============================================================

LAST_DIR_NAME="$(basename "$(pwd)")exp_rank_27"
OUTPUT_DIR="${MAIN_DIR}checkpoints/${LAST_DIR_NAME}/${DATASET}/"

mkdir -p "${OUTPUT_DIR}"

python -m torch.distributed.launch \
    --nproc_per_node ${NUM_GPUS} \
    --master_port 12345 \
    --use_env \
    train.py \
    --bb_ckpt_dir "${BACKBONE_CKPT_DIR}" \
    --data_root "${DATA_ROOT}" \
    --split_root "${SPLIT_ROOT}" \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lr_adapter 0.0001 \
    --batch_size ${BATCH_SIZE} \
    --lr 0.000025 \
    --lr_bert 0.000005 \
    --lr_visual 0.00001 \
    --aug_scale \
    --aug_translate \
    --aug_crop \
    --backbone ViTDet \
    --is_eliminate \
    --imsize 448 \
    --bert_enc_num 12 \
    --dataset "${DATASET}" \
    --max_query_len 40 \
    --lr_scheduler poly \
    --is_segment \
    --vl_enc_layers 3 \
    --dim_feedforward 1024 \
    --loss_alpha 0.1 \
    --epochs ${EPOCHS} \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers ${NUM_WORKERS}