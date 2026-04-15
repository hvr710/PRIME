#!/usr/bin/env bash
set -euo pipefail

# Scan 100 seeds for best single-seed subject-split result.
#
# Models:
#   1. cross_attn_single_head
#   2. cross_attn_moe_token_sup_l03
#   3. cross_attn_moe_token_sup_l05
#
# Protocol:
#   - subject split only
#   - balanced 8:1:1 subject split
#   - pre-split per-subject z-score + complete-trial causal EWMA smoothing
#   - frozen mdJPT features only
#   - fixed MoE base_lr = 5e-4 (taken from the current mainline)
#
# GPU layout:
#   - 4 shards
#   - each GPU runs 3 processes concurrently: CrossAttn + TokenMoE l03 + TokenMoE l05
#   - intended memory target is roughly 30 GiB per process

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

SEEDS=(
  42 666 777 888 2025 3407 8279 9331 19986 43719 46005 47344 88814 94099 100121
  105739 106082 106802 118601 125583 139759 142157 151005 153945 180835 210040
  214408 238145 245001 247102 264476 274405 281188 288509 302677 322084 326477
  347091 368641 405133 405977 413721 428290 429289 434767 453848 454429 485255
  496606 500792 506658 516141 517530 553298 566843 570793 593173 600620 615804
  626115 642130 646388 646440 650489 661045 677654 697734 711037 714782 717700
  718005 744380 767492 769388 772793 773747 775903 786041 789779 813041 829732
  830655 835148 843618 866429 891469 903961 918746 921973 936024 939179 945029
  956563 961007 965813 966505 972357 981999 993707 996237
)

SPLIT_KINDS=(subject)
MOE_BASE_LR="5e-4"

if [[ -e "$OUTPUT_ROOT" && "${OVERWRITE:-0}" != "1" ]]; then
  cat <<EOF
Output directory already exists:
  $OUTPUT_ROOT

Use a new directory:
  OUTPUT_ROOT=$REPO_ROOT/outputs/subject_bestseed_100scan_\$(date +%Y%m%d_%H%M%S) bash run_subject_bestseed_scan_100.sh

or force overwrite:
  OVERWRITE=1 bash run_subject_bestseed_scan_100.sh
EOF
  exit 1
fi

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p \
  "$OUTPUT_ROOT" "$SPLIT_DIR" "$CACHE_DIR" "$BASELINE_DIR" "$FUSION_SHARD_ROOT" \
  "$MOE_L03_SHARD_ROOT" "$MOE_L05_SHARD_ROOT" "$FUSION_DIR" "$MOE_DIR" \
  "$COMPARISON_DIR" "$FULLFT_DIR" "$META_DIR"
: > "$PID_FILE"

SEED_COUNT="${#SEEDS[@]}"
printf '%s\n' "${SEEDS[@]}" > "$META_DIR/seeds_100.txt"

python - <<'PY' "$META_DIR" "${SEEDS[@]}"
from pathlib import Path
import json
import sys

meta = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2:]]
n_shards = 4
shards = [seeds[i::n_shards] for i in range(n_shards)]
payload = {
    "n_seeds": len(seeds),
    "seeds": seeds,
    "n_shards": n_shards,
    "shards": {f"shard{i}": shard for i, shard in enumerate(shards)},
    "models": [
        "cross_attn_single_head",
        "cross_attn_moe_token_sup_l03",
        "cross_attn_moe_token_sup_l05",
    ],
    "split_kinds": ["subject"],
    "transform_mode": "presplit_subject_zscore_smooth_all",
    "moe_base_lr": 5e-4,
}
(meta / "seed_scan_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
for i, shard in enumerate(shards):
    (meta / f"shard{i}.txt").write_text(" ".join(str(x) for x in shard) + "\n", encoding="utf-8")
PY

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] subject best-seed scan start"
echo "Output: $OUTPUT_ROOT"
echo "Seeds: $SEED_COUNT"

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
  --split-kinds subject \
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
  --split-kinds subject \
  --transform-mode presplit_subject_zscore_smooth_all \
  --smooth-alpha 0.65 \
  2>&1 | tee "$OUTPUT_ROOT/build_cache.log"

for shard_id in 0 1 2 3; do
  gpu_id="$shard_id"
  shard_seeds="$(cat "$META_DIR/shard${shard_id}.txt")"

  mkdir -p "$FUSION_SHARD_ROOT/shard${shard_id}" "$MOE_L03_SHARD_ROOT/shard${shard_id}" "$MOE_L05_SHARD_ROOT/shard${shard_id}"

  nohup bash -lc "
    source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export OMP_NUM_THREADS=4
    PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_fusion \
      --output-root \"$OUTPUT_ROOT\" \
      --split-dir \"$SPLIT_DIR\" \
      --cache-dir \"$CACHE_DIR\" \
      --baseline-dir \"$BASELINE_DIR\" \
      --fusion-dir \"$FUSION_SHARD_ROOT/shard${shard_id}\" \
      --moe-dir \"$MOE_DIR\" \
      --comparison-dir \"$COMPARISON_DIR\" \
      --fullft-dir \"$FULLFT_DIR\" \
      --seeds $shard_seeds \
      --split-kinds subject \
      --device cuda
  " > "$OUTPUT_ROOT/gpu${gpu_id}_fusion_shard${shard_id}.log" 2>&1 &
  echo "gpu${gpu_id}_fusion_shard${shard_id} $!" | tee -a "$PID_FILE"

  nohup bash -lc "
    source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export OMP_NUM_THREADS=4
    PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
      --output-root \"$OUTPUT_ROOT\" \
      --split-dir \"$SPLIT_DIR\" \
      --cache-dir \"$CACHE_DIR\" \
      --baseline-dir \"$BASELINE_DIR\" \
      --fusion-dir \"$FUSION_DIR\" \
      --moe-dir \"$MOE_L03_SHARD_ROOT/shard${shard_id}\" \
      --comparison-dir \"$COMPARISON_DIR\" \
      --fullft-dir \"$FULLFT_DIR\" \
      --seeds $shard_seeds \
      --split-kinds subject \
      --variants cross_attn_moe_token_sup_l03 \
      --base-lr \"$MOE_BASE_LR\" \
      --device cuda
  " > "$OUTPUT_ROOT/gpu${gpu_id}_tokenl03_shard${shard_id}.log" 2>&1 &
  echo "gpu${gpu_id}_tokenl03_shard${shard_id} $!" | tee -a "$PID_FILE"

  nohup bash -lc "
    source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export OMP_NUM_THREADS=4
    PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
      --output-root \"$OUTPUT_ROOT\" \
      --split-dir \"$SPLIT_DIR\" \
      --cache-dir \"$CACHE_DIR\" \
      --baseline-dir \"$BASELINE_DIR\" \
      --fusion-dir \"$FUSION_DIR\" \
      --moe-dir \"$MOE_L05_SHARD_ROOT/shard${shard_id}\" \
      --comparison-dir \"$COMPARISON_DIR\" \
      --fullft-dir \"$FULLFT_DIR\" \
      --seeds $shard_seeds \
      --split-kinds subject \
      --variants cross_attn_moe_token_sup_l05 \
      --base-lr \"$MOE_BASE_LR\" \
      --device cuda
  " > "$OUTPUT_ROOT/gpu${gpu_id}_tokenl05_shard${shard_id}.log" 2>&1 &
  echo "gpu${gpu_id}_tokenl05_shard${shard_id} $!" | tee -a "$PID_FILE"
done

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  while read -r _ pid; do
    while kill -0 \"\$pid\" 2>/dev/null; do sleep 60; done
  done < \"$PID_FILE\"

  python - <<'PY'
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
    --split-kinds subject \
    2>&1 | tee \"$OUTPUT_ROOT/summarize.log\"
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launched subject best-seed scan"
cat "$PID_FILE"
