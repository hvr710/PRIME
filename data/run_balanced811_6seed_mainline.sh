#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="$REPO_ROOT/data"
OUTPUT_ROOT="$REPO_ROOT/outputs/balanced811_6seed_adaptsmoothall_mainline"
SPLIT_DIR="$OUTPUT_ROOT/splits"
CACHE_DIR="$OUTPUT_ROOT/adapt_cache"
BASELINE_DIR="$OUTPUT_ROOT/baseline_results"
FUSION_DIR="$OUTPUT_ROOT/fusion_results"
MOE_DIR="$OUTPUT_ROOT/moe_results"
COMPARISON_DIR="$OUTPUT_ROOT/comparison"
FULLFT_DIR="$BASELINE_DIR/full_finetune_results"

SEEDS=(42 3407 2025 666 777 888)

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p "$OUTPUT_ROOT" "$BASELINE_DIR" "$FUSION_DIR" "$MOE_DIR" "$COMPARISON_DIR"

echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] balanced811 mainline start"

python "$DATA_DIR/run_balanced811_6seed_mainline.py" make_splits \
  --output-root "$OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --baseline-dir "$BASELINE_DIR" \
  --fusion-dir "$FUSION_DIR" \
  --moe-dir "$MOE_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --fullft-dir "$FULLFT_DIR" \
  --seeds "${SEEDS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/make_splits.log"

python "$DATA_DIR/run_balanced811_6seed_mainline.py" build_cache \
  --output-root "$OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --baseline-dir "$BASELINE_DIR" \
  --fusion-dir "$FUSION_DIR" \
  --moe-dir "$MOE_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --fullft-dir "$FULLFT_DIR" \
  --seeds "${SEEDS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/build_cache.log"

(
  export CUDA_VISIBLE_DEVICES=0
  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu0: full_finetune start"
  PYTHONUNBUFFERED=1 python "$DATA_DIR/train_mdjpt_only_step5.py" \
    --eeg-h5 "$DATA_DIR/comp4_len5_step5_mapped60.h5" \
    --embed-h5 "$DATA_DIR/mdjpt_step5_embeddings.h5" \
    --split-dir "$SPLIT_DIR" \
    --result-dir "$FULLFT_DIR" \
    --seeds "${SEEDS[@]}" \
    --split-kinds subject segment \
    --modes full_finetune \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/gpu0_full_finetune.log"

  while [[ ! -f "$MOE_DIR/selected_lr.json" ]]; do
    echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu0: waiting for selected_lr.json"
    sleep 60
  done

  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu0: token-MoE start"
  PYTHONUNBUFFERED=1 python "$DATA_DIR/run_balanced811_6seed_mainline.py" train_moe \
    --output-root "$OUTPUT_ROOT" \
    --split-dir "$SPLIT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --fusion-dir "$FUSION_DIR" \
    --moe-dir "$MOE_DIR" \
    --comparison-dir "$COMPARISON_DIR" \
    --fullft-dir "$FULLFT_DIR" \
    --seeds "${SEEDS[@]}" \
    --split-kinds subject segment \
    --variants cross_attn_moe_token_sup_l03 cross_attn_moe_token_sup_l05 \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/gpu0_token_moe.log"
) &
GPU0_PID=$!

(
  export CUDA_VISIBLE_DEVICES=1
  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu1: baselines start"
  PYTHONUNBUFFERED=1 python "$DATA_DIR/run_balanced811_6seed_mainline.py" train_baselines \
    --output-root "$OUTPUT_ROOT" \
    --split-dir "$SPLIT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --fusion-dir "$FUSION_DIR" \
    --moe-dir "$MOE_DIR" \
    --comparison-dir "$COMPARISON_DIR" \
    --fullft-dir "$FULLFT_DIR" \
    --seeds "${SEEDS[@]}" \
    --split-kinds subject segment \
    --models logistic_regression svm xgboost frozen_mlp \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/gpu1_baselines.log"

  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu1: cross-attn start"
  PYTHONUNBUFFERED=1 python "$DATA_DIR/run_balanced811_6seed_mainline.py" train_fusion \
    --output-root "$OUTPUT_ROOT" \
    --split-dir "$SPLIT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --fusion-dir "$FUSION_DIR" \
    --moe-dir "$MOE_DIR" \
    --comparison-dir "$COMPARISON_DIR" \
    --fullft-dir "$FULLFT_DIR" \
    --seeds "${SEEDS[@]}" \
    --split-kinds subject segment \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/gpu1_fusion.log"

  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu1: moe lr screening start"
  PYTHONUNBUFFERED=1 python "$DATA_DIR/run_balanced811_6seed_mainline.py" screen_moe_lr \
    --output-root "$OUTPUT_ROOT" \
    --split-dir "$SPLIT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --fusion-dir "$FUSION_DIR" \
    --moe-dir "$MOE_DIR" \
    --comparison-dir "$COMPARISON_DIR" \
    --fullft-dir "$FULLFT_DIR" \
    --seeds "${SEEDS[@]}" \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/gpu1_moe_screen.log"

  echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] gpu1: moe D/E start"
  PYTHONUNBUFFERED=1 python "$DATA_DIR/run_balanced811_6seed_mainline.py" train_moe \
    --output-root "$OUTPUT_ROOT" \
    --split-dir "$SPLIT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --fusion-dir "$FUSION_DIR" \
    --moe-dir "$MOE_DIR" \
    --comparison-dir "$COMPARISON_DIR" \
    --fullft-dir "$FULLFT_DIR" \
    --seeds "${SEEDS[@]}" \
    --split-kinds subject segment \
    --variants cross_attn_moe_unsup cross_attn_moe_sup_l03 cross_attn_moe_sup_l05 \
    --device cuda \
    2>&1 | tee "$OUTPUT_ROOT/gpu1_moe_de.log"
) &
GPU1_PID=$!

wait "$GPU0_PID"
wait "$GPU1_PID"

python "$DATA_DIR/run_balanced811_6seed_mainline.py" summarize \
  --output-root "$OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --baseline-dir "$BASELINE_DIR" \
  --fusion-dir "$FUSION_DIR" \
  --moe-dir "$MOE_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --fullft-dir "$FULLFT_DIR" \
  --seeds "${SEEDS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/summarize.log"

echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] balanced811 mainline finished"
