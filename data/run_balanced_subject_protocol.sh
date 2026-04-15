#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <protocol_name> <gpu_id>" >&2
  exit 1
fi

PROTOCOL="$1"
GPU_ID="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="$REPO_ROOT/data"
OUTPUT_ROOT="$REPO_ROOT/outputs/$PROTOCOL"
SPLIT_DIR="$OUTPUT_ROOT/splits"

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p "$OUTPUT_ROOT"

echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] protocol=$PROTOCOL gpu=$GPU_ID output_root=$OUTPUT_ROOT"

python "$DATA_DIR/create_balanced_subject_splits.py" \
  --output-root "$REPO_ROOT/outputs" \
  --protocols "$PROTOCOL"

python "$DATA_DIR/run_feature_baselines.py" \
  --feature-h5 "$DATA_DIR/features_comp4_len5_step5_mapped60.h5" \
  --result-dir "$OUTPUT_ROOT/feature_baseline_results" \
  --split-dir "$SPLIT_DIR" \
  --split-kinds subject \
  --seeds 42 3407 2025 \
  2>&1 | tee "$OUTPUT_ROOT/feature_baseline_results.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 python "$DATA_DIR/train_mdjpt_only_step5.py" \
  --eeg-h5 "$DATA_DIR/comp4_len5_step5_mapped60.h5" \
  --embed-h5 "$DATA_DIR/mdjpt_step5_embeddings.h5" \
  --split-dir "$SPLIT_DIR" \
  --result-dir "$OUTPUT_ROOT/mdjpt_only_results" \
  --split-kinds subject \
  --seeds 42 3407 2025 \
  --device cuda \
  2>&1 | tee "$OUTPUT_ROOT/mdjpt_only_results.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 python "$DATA_DIR/train_fusion_step5.py" \
  --embed-h5 "$DATA_DIR/mdjpt_step5_embeddings.h5" \
  --feature-h5 "$DATA_DIR/features_comp4_len5_step5_mapped60.h5" \
  --split-dir "$SPLIT_DIR" \
  --result-dir "$OUTPUT_ROOT/step5_fusion_results" \
  --split-kinds subject \
  --seeds 42 3407 2025 \
  --device cuda \
  2>&1 | tee "$OUTPUT_ROOT/step5_fusion_results.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 python "$DATA_DIR/train_fusion_step6_lora.py" \
  --workflow all \
  --feature-h5 "$DATA_DIR/features_comp4_len5_step5_mapped60.h5" \
  --eeg-h5 "$DATA_DIR/comp4_len5_step5_mapped60.h5" \
  --split-dir "$SPLIT_DIR" \
  --result-dir "$OUTPUT_ROOT/step6_lora_results" \
  --split-kinds subject \
  --seeds 42 3407 2025 \
  --device cuda \
  2>&1 | tee "$OUTPUT_ROOT/step6_lora_results.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 python "$DATA_DIR/train_step7_moe.py" \
  --workflow all \
  --embed-h5 "$DATA_DIR/mdjpt_step5_embeddings.h5" \
  --feature-h5 "$DATA_DIR/features_comp4_len5_step5_mapped60.h5" \
  --split-dir "$SPLIT_DIR" \
  --result-dir "$OUTPUT_ROOT/step7_moe_results" \
  --split-kinds subject \
  --seeds 42 3407 2025 \
  --device cuda \
  2>&1 | tee "$OUTPUT_ROOT/step7_moe_results.log"

echo "[`date -u '+%Y-%m-%d %H:%M:%S UTC'`] finished protocol=$PROTOCOL"
