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
PID_FILE="$OUTPUT_ROOT/launched_pids.txt"

SEEDS=(42 3407 2025 666 777 888)

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p "$OUTPUT_ROOT" "$BASELINE_DIR" "$FUSION_DIR" "$MOE_DIR" "$COMPARISON_DIR"
: > "$PID_FILE"

echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] preparing splits and caches"

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
  --split-kinds subject segment \
  2>&1 | tee "$OUTPUT_ROOT/build_cache.log"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_baselines \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --split-kinds subject segment \
    --models logistic_regression svm xgboost frozen_mlp \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_baselines.log" 2>&1 &
echo "gpu1_baselines $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_fusion \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --split-kinds subject segment \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_fusion.log" 2>&1 &
echo "gpu1_fusion $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" screen_moe_lr \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_moe_screen.log" 2>&1 &
echo "gpu1_moe_screen $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  while [[ ! -f \"$MOE_DIR/selected_lr.json\" ]]; do sleep 30; done
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --split-kinds subject segment \
    --variants cross_attn_moe_unsup cross_attn_moe_sup_l03 cross_attn_moe_sup_l05 \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_moe_de.log" 2>&1 &
echo "gpu1_moe_de $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=0
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/train_mdjpt_only_step5.py\" \
    --eeg-h5 \"$DATA_DIR/comp4_len5_step5_mapped60.h5\" \
    --embed-h5 \"$DATA_DIR/mdjpt_step5_embeddings.h5\" \
    --split-dir \"$SPLIT_DIR\" \
    --result-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --split-kinds subject segment \
    --modes full_finetune \
    --device cuda
" > "$OUTPUT_ROOT/gpu0_full_finetune.log" 2>&1 &
echo "gpu0_full_finetune $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=0
  while [[ ! -f \"$MOE_DIR/selected_lr.json\" ]]; do sleep 30; done
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --split-kinds subject segment \
    --variants cross_attn_moe_token_sup_l03 cross_attn_moe_token_sup_l05 \
    --device cuda
" > "$OUTPUT_ROOT/gpu0_moe_token.log" 2>&1 &
echo "gpu0_moe_token $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  while pgrep -f 'run_balanced811_6seed_mainline.py train_baselines|run_balanced811_6seed_mainline.py train_fusion|run_balanced811_6seed_mainline.py train_moe|train_mdjpt_only_step5.py' >/dev/null; do
    sleep 120
  done
  python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" summarize \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds ${SEEDS[*]} \
    --split-kinds subject segment
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] launched parallel jobs"
cat "$PID_FILE"
