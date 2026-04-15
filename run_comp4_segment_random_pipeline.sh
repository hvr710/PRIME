#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/comp4_segment_random_pipeline}"
H5_PATH="${H5_PATH:-$ROOT_DIR/data/comp4_len5_step5_mapped60.h5}"
SEGMENT_SPLIT_JSON="${SEGMENT_SPLIT_JSON:-$ROOT_DIR/data/comp4_len5_step5_segment_random_split_seed42.json}"
RUN_NAME="${RUN_NAME:-pretrain}"
CKPT_EPOCH="${CKPT_EPOCH:-20}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
EXT_GPU="${EXT_GPU:-0}"
MLP_GPU="${MLP_GPU:-1}"
FT_GPU="${FT_GPU:-1}"
ARTIFACT_TAG="${ARTIFACT_TAG:-len5_step5_seg_random_seed42}"

python preprocess/create_comp4_segment_random_split.py \
  --h5-path "$H5_PATH" \
  --output-json "$SEGMENT_SPLIT_JSON" \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1

CUDA_VISIBLE_DEVICES="$EXT_GPU" python ext_fea.py \
  data@data_val=COMP4 \
  data_val.data_dir="$OUTPUT_ROOT" \
  data_val.split_json='' \
  data_val.timeLen2=5 \
  data_val.timeStep2=5 \
  log.run_name="$RUN_NAME" \
  val.extractor.ckpt_epoch="$CKPT_EPOCH" \
  val.artifact_tag="$ARTIFACT_TAG" \
  train.num_workers="$TRAIN_NUM_WORKERS"

CUDA_VISIBLE_DEVICES="$MLP_GPU" python train_mlp_full.py \
  data@data_val=COMP4 \
  data_val.data_dir="$OUTPUT_ROOT" \
  data_val.split_json='' \
  data_val.timeLen2=5 \
  data_val.timeStep2=5 \
  log.run_name="$RUN_NAME" \
  val.extractor.ckpt_epoch="$CKPT_EPOCH" \
  val.artifact_tag="$ARTIFACT_TAG" \
  val.segment_split_json="$SEGMENT_SPLIT_JSON" \
  train.num_workers="$TRAIN_NUM_WORKERS"

CUDA_VISIBLE_DEVICES="$FT_GPU" python train_finetune_full.py \
  data@data_val=COMP4 \
  data_val.data_dir="$OUTPUT_ROOT" \
  data_val.split_json='' \
  data_val.timeLen2=5 \
  data_val.timeStep2=5 \
  log.run_name="$RUN_NAME" \
  val.extractor.ckpt_epoch="$CKPT_EPOCH" \
  val.segment_split_json="$SEGMENT_SPLIT_JSON" \
  train.num_workers="$TRAIN_NUM_WORKERS" \
  full_ft.num_workers="$TRAIN_NUM_WORKERS"
