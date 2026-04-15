#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/vePFS-0x0d/home/cx/cx/hw/project/project/赛题四数据集及说明文档/训练集}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/vePFS-0x0d/home/cx/cx/hw/project/project/赛题四数据集及说明文档/mdjpt_comp4_125hz}"
RUN_NAME="${RUN_NAME:-pretrain}"
CKPT_EPOCH="${CKPT_EPOCH:-20}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
EXT_GPU="${EXT_GPU:-0}"
MLP_GPU="${MLP_GPU:-1}"
LP_TIME_LEN="${LP_TIME_LEN:-5}"
LP_TIME_STEP="${LP_TIME_STEP:-5}"
LP_ARTIFACT_TAG="${LP_ARTIFACT_TAG:-len5_step5_lp}"

python preprocess/prepare_comp4_mdjpt.py \
  --input-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1

CUDA_VISIBLE_DEVICES="$EXT_GPU" python ext_fea.py \
  data@data_val=COMP4 \
  data_val.timeLen2="$LP_TIME_LEN" \
  data_val.timeStep2="$LP_TIME_STEP" \
  log.run_name="$RUN_NAME" \
  val.extractor.ckpt_epoch="$CKPT_EPOCH" \
  val.artifact_tag="$LP_ARTIFACT_TAG" \
  train.num_workers="$TRAIN_NUM_WORKERS"

CUDA_VISIBLE_DEVICES="$MLP_GPU" python train_mlp_full.py \
  data@data_val=COMP4 \
  data_val.timeLen2="$LP_TIME_LEN" \
  data_val.timeStep2="$LP_TIME_STEP" \
  log.run_name="$RUN_NAME" \
  val.extractor.ckpt_epoch="$CKPT_EPOCH" \
  val.artifact_tag="$LP_ARTIFACT_TAG" \
  train.num_workers="$TRAIN_NUM_WORKERS"
