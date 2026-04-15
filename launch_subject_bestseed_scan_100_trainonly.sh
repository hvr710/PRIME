#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
DATA_DIR="$REPO_ROOT/data"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/subject_bestseed_100scan_crossattn_tokenmoe_20260415}"

SPLIT_DIR="$OUTPUT_ROOT/splits"
CACHE_DIR="$OUTPUT_ROOT/adapt_cache"
BASELINE_DIR="$OUTPUT_ROOT/baseline_results_unused"
FUSION_SHARD_ROOT="$OUTPUT_ROOT/fusion_shards"
MOE_L03_SHARD_ROOT="$OUTPUT_ROOT/moe_token_l03_shards"
MOE_L05_SHARD_ROOT="$OUTPUT_ROOT/moe_token_l05_shards"
FUSION_DIR="$OUTPUT_ROOT/fusion_results"
MOE_DIR="$OUTPUT_ROOT/moe_results"
COMPARISON_DIR="$OUTPUT_ROOT/comparison"
FULLFT_DIR="$OUTPUT_ROOT/fullft_unused"
META_DIR="$OUTPUT_ROOT/meta"
PID_FILE="$OUTPUT_ROOT/launched_pids.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"
MOE_BASE_LR="${MOE_BASE_LR:-5e-4}"

if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "Missing output root: $OUTPUT_ROOT" >&2
  exit 1
fi

if [[ ! -d "$CACHE_DIR/subject" ]]; then
  echo "Missing subject cache directory: $CACHE_DIR/subject" >&2
  exit 1
fi

mkdir -p \
  "$FUSION_SHARD_ROOT" "$MOE_L03_SHARD_ROOT" "$MOE_L05_SHARD_ROOT" \
  "$FUSION_DIR" "$MOE_DIR" "$COMPARISON_DIR"

find "$OUTPUT_ROOT" -maxdepth 1 -type f \( -name 'gpu*_*.log' -o -name 'summarize*.log' \) -delete
: > "$PID_FILE"

for shard_id in 0 1 2 3; do
  gpu_id="$shard_id"
  shard_seeds="$(cat "$META_DIR/shard${shard_id}.txt")"

  mkdir -p \
    "$FUSION_SHARD_ROOT/shard${shard_id}" \
    "$MOE_L03_SHARD_ROOT/shard${shard_id}" \
    "$MOE_L05_SHARD_ROOT/shard${shard_id}"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$DATA_DIR/run_balanced811_6seed_mainline.py" train_fusion \
      --output-root "$OUTPUT_ROOT" \
      --split-dir "$SPLIT_DIR" \
      --cache-dir "$CACHE_DIR" \
      --baseline-dir "$BASELINE_DIR" \
      --fusion-dir "$FUSION_SHARD_ROOT/shard${shard_id}" \
      --moe-dir "$MOE_DIR" \
      --comparison-dir "$COMPARISON_DIR" \
      --fullft-dir "$FULLFT_DIR" \
      --seeds $shard_seeds \
      --split-kinds subject \
      --device cuda \
      > "$OUTPUT_ROOT/gpu${gpu_id}_fusion_shard${shard_id}.log" 2>&1 &
  echo "gpu${gpu_id}_fusion_shard${shard_id} $!" | tee -a "$PID_FILE"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$DATA_DIR/run_balanced811_6seed_mainline.py" train_moe \
      --output-root "$OUTPUT_ROOT" \
      --split-dir "$SPLIT_DIR" \
      --cache-dir "$CACHE_DIR" \
      --baseline-dir "$BASELINE_DIR" \
      --fusion-dir "$FUSION_DIR" \
      --moe-dir "$MOE_L03_SHARD_ROOT/shard${shard_id}" \
      --comparison-dir "$COMPARISON_DIR" \
      --fullft-dir "$FULLFT_DIR" \
      --seeds $shard_seeds \
      --split-kinds subject \
      --variants cross_attn_moe_token_sup_l03 \
      --base-lr "$MOE_BASE_LR" \
      --device cuda \
      > "$OUTPUT_ROOT/gpu${gpu_id}_tokenl03_shard${shard_id}.log" 2>&1 &
  echo "gpu${gpu_id}_tokenl03_shard${shard_id} $!" | tee -a "$PID_FILE"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$DATA_DIR/run_balanced811_6seed_mainline.py" train_moe \
      --output-root "$OUTPUT_ROOT" \
      --split-dir "$SPLIT_DIR" \
      --cache-dir "$CACHE_DIR" \
      --baseline-dir "$BASELINE_DIR" \
      --fusion-dir "$FUSION_DIR" \
      --moe-dir "$MOE_L05_SHARD_ROOT/shard${shard_id}" \
      --comparison-dir "$COMPARISON_DIR" \
      --fullft-dir "$FULLFT_DIR" \
      --seeds $shard_seeds \
      --split-kinds subject \
      --variants cross_attn_moe_token_sup_l05 \
      --base-lr "$MOE_BASE_LR" \
      --device cuda \
      > "$OUTPUT_ROOT/gpu${gpu_id}_tokenl05_shard${shard_id}.log" 2>&1 &
  echo "gpu${gpu_id}_tokenl05_shard${shard_id} $!" | tee -a "$PID_FILE"
done

nohup bash -lc "
  while read -r _ pid; do
    while kill -0 \"\$pid\" 2>/dev/null; do sleep 60; done
  done < \"$PID_FILE\"

  \"$PYTHON_BIN\" - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('$OUTPUT_ROOT')
fusion_shards = sorted((root / 'fusion_shards').glob('shard*'))
moe_l03_shards = sorted((root / 'moe_token_l03_shards').glob('shard*'))
moe_l05_shards = sorted((root / 'moe_token_l05_shards').glob('shard*'))
fusion_dir = root / 'fusion_results'
moe_dir = root / 'moe_results'
comparison_dir = root / 'comparison'
fusion_dir.mkdir(parents=True, exist_ok=True)
moe_dir.mkdir(parents=True, exist_ok=True)
comparison_dir.mkdir(parents=True, exist_ok=True)

fusion_frames = [pd.read_csv(p / 'all_results.csv') for p in fusion_shards if (p / 'all_results.csv').exists()]
if fusion_frames:
    pd.concat(fusion_frames, ignore_index=True).to_csv(fusion_dir / 'all_results.csv', index=False)

moe_frames = []
router_frames = []
for shard_list in [moe_l03_shards, moe_l05_shards]:
    for p in shard_list:
        if (p / 'all_results.csv').exists():
            moe_frames.append(pd.read_csv(p / 'all_results.csv'))
        if (p / 'router_metrics.csv').exists():
            router_frames.append(pd.read_csv(p / 'router_metrics.csv'))
if moe_frames:
    pd.concat(moe_frames, ignore_index=True).to_csv(moe_dir / 'all_results.csv', index=False)
if router_frames:
    pd.concat(router_frames, ignore_index=True).to_csv(moe_dir / 'router_metrics.csv', index=False)

(moe_dir / 'selected_lr.json').write_text(json.dumps({
    'selected_base_lr': 5e-4,
    'selection_rule': 'fixed_from_current_mainline'
}, indent=2), encoding='utf-8')
(moe_dir / 'merge_manifest.json').write_text(json.dumps({
    'fusion_shards': [p.name for p in fusion_shards],
    'moe_l03_shards': [p.name for p in moe_l03_shards],
    'moe_l05_shards': [p.name for p in moe_l05_shards],
    'n_fusion_rows': int(sum(len(f) for f in fusion_frames)) if fusion_frames else 0,
    'n_moe_rows': int(sum(len(f) for f in moe_frames)) if moe_frames else 0,
    'n_router_rows': int(sum(len(f) for f in router_frames)) if router_frames else 0,
    'models': [
        'cross_attn_single_head',
        'cross_attn_moe_token_sup_l03',
        'cross_attn_moe_token_sup_l05'
    ],
    'split_kind': 'subject',
    'seed_count': 100,
}, indent=2), encoding='utf-8')
PY

  \"$PYTHON_BIN\" \"$DATA_DIR/run_balanced811_6seed_mainline.py\" summarize \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds \$(cat \"$META_DIR/seeds_100.txt\" | tr '\n' ' ') \
    --split-kinds subject \
    > \"$OUTPUT_ROOT/summarize.log\" 2>&1
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] relaunched train-only best-seed scan"
cat "$PID_FILE"
