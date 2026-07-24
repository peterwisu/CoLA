#!/bin/bash
set -e

# ============================================================
# CONFIG — edit these for your setup
# ============================================================
MAIN_DIR="./"

MODEL_TYPE="vit_b"           # ["vit_b", "dino_b", "dino_l", "eva_l"]
SSLAM_NORM=0                

VIDEO_FOLDER="./AVE_Dataset/video_frames"
AUDIO_FOLDER="./AVE_Dataset/raw_audio"

BATCH_SIZE=2
ACCUM_ITR=1
LR="5e-6"
LR_MLP="4e-6"
EARLY_STOP=10
EPOCHS=50
NUM_WORKERS=4

LORA_R=16
LORA_ALPHA=8
LORA_C_SCALING="0.1"
LORA_REDUCTION=16
# ============================================================

LAST_DIR_NAME="$(basename "$(pwd)")"
OUTPUT_DIR="${MAIN_DIR}checkpoints/${LAST_DIR_NAME}/"

mkdir -p "${OUTPUT_DIR}"

python main_trans.py \
    --model_save_dir "${OUTPUT_DIR}" \
    --working_dir "${MAIN_DIR}" \
    --audio_folder "${AUDIO_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --vis_encoder_type "${MODEL_TYPE}" \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lr ${LR} \
    --lr_mlp ${LR_MLP} \
    --batch_size ${BATCH_SIZE} \
    --early_stop ${EARLY_STOP} \
    --epochs ${EPOCHS} \
    --accum_itr ${ACCUM_ITR} \
    --num_workers ${NUM_WORKERS} \
    --sslam_norm ${SSLAM_NORM} \
    --lora_c_scaling ${LORA_C_SCALING} \
    --lora_reduction ${LORA_REDUCTION}