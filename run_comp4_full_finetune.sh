#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/vePFS-0x0d/home/cx/cx/hw/project/project/赛题四数据集及说明文档/训练集}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vePFS-0x0d/home/cx/cx/hw/project/project/赛题四数据集及说明文档/mdjpt_comp4_125hz}"
RUN_NAME="${RUN_NAME:-pretrain}"
CKPT_EPOCH="${CKPT_EPOCH:-20}"
FT_GPU="${FT_GPU:-1}"
FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"
FT_TIME_LEN="${FT_TIME_LEN:-5}"
FT_TIME_STEP="${FT_TIME_STEP:-5}"
FT_HEAD_WARMUP_EPOCHS="${FT_HEAD_WARMUP_EPOCHS:-2}"
FT_UNFREEZE_MLLA_EPOCH="${FT_UNFREEZE_MLLA_EPOCH:--1}"
FT_ADAPTER_LR="${FT_ADAPTER_LR:-0.00005}"
FT_MLLA_LR="${FT_MLLA_LR:-0.00001}"
FT_HEAD_LR="${FT_HEAD_LR:-0.0005}"
FT_DROPOUT="${FT_DROPOUT:-0.4}"
FT_WD="${FT_WD:-0.005}"

python preprocess/prepare_comp4_mdjpt.py \
  --input-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1

CUDA_VISIBLE_DEVICES="$FT_GPU" python train_finetune_full.py \
  data@data_val=COMP4 \
  data_val.timeLen2="$FT_TIME_LEN" \
  data_val.timeStep2="$FT_TIME_STEP" \
  log.run_name="$RUN_NAME" \
  val.extractor.ckpt_epoch="$CKPT_EPOCH" \
  train.num_workers="$FT_NUM_WORKERS" \
  full_ft.num_workers="$FT_NUM_WORKERS" \
  full_ft.head_warmup_epochs="$FT_HEAD_WARMUP_EPOCHS" \
  full_ft.unfreeze_mlla_epoch="$FT_UNFREEZE_MLLA_EPOCH" \
  full_ft.adapter_lr="$FT_ADAPTER_LR" \
  full_ft.mlla_lr="$FT_MLLA_LR" \
  full_ft.head_lr="$FT_HEAD_LR" \
  full_ft.dropout="$FT_DROPOUT" \
  full_ft.wd="$FT_WD"
