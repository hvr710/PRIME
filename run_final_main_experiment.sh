#!/usr/bin/env bash
set -euo pipefail

# Final COMP4 mainline runner.
#
# Protocol:
#   - 6 seeds: 42, 3407, 2025, 666, 777, 888
#   - balanced 8:1:1 subject split and segment split
#   - frozen mdJPT features only, no full fine-tuning
#   - pre-split per-subject z-score + complete-trial causal EWMA smoothing
#   - baselines + CrossAttn Single Head + MoE ablations
#
# Usage:
#   bash run_final_main_experiment.sh
#
# Optional:
#   OUTPUT_ROOT=/path/to/new/output bash run_final_main_experiment.sh
#   OVERWRITE=1 bash run_final_main_experiment.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
DATA_DIR="$REPO_ROOT/data"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/balanced811_6seed_presplit_subjectzscore_smoothall_frozen_moe_mainline}"

SPLIT_DIR="$OUTPUT_ROOT/splits"
CACHE_DIR="$OUTPUT_ROOT/adapt_cache"
BASELINE_DIR="$OUTPUT_ROOT/baseline_results"
FUSION_DIR="$OUTPUT_ROOT/fusion_results"
MOE_SCREEN_DIR="$OUTPUT_ROOT/moe_screen"
MOE_DE_DIR="$OUTPUT_ROOT/moe_results_de"
MOE_TOKEN_DIR="$OUTPUT_ROOT/moe_results_token"
MOE_MERGED_DIR="$OUTPUT_ROOT/moe_results"
COMPARISON_DIR="$OUTPUT_ROOT/comparison"
FULLFT_DIR="$BASELINE_DIR/full_finetune_results"
PID_FILE="$OUTPUT_ROOT/final_mainline_pids.txt"

SEEDS=(42 3407 2025 666 777 888)
SPLIT_KINDS=(subject segment)

if [[ -e "$OUTPUT_ROOT" && "${OVERWRITE:-0}" != "1" ]]; then
  cat <<EOF
Output directory already exists:
  $OUTPUT_ROOT

To avoid accidentally overwriting existing results, either choose a new directory:
  OUTPUT_ROOT=$REPO_ROOT/outputs/final_rerun_\$(date +%Y%m%d_%H%M%S) bash run_final_main_experiment.sh

or explicitly allow overwrite:
  OVERWRITE=1 bash run_final_main_experiment.sh
EOF
  exit 1
fi

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p "$OUTPUT_ROOT" "$BASELINE_DIR" "$FUSION_DIR" "$MOE_SCREEN_DIR" "$MOE_DE_DIR" "$MOE_TOKEN_DIR" "$MOE_MERGED_DIR" "$COMPARISON_DIR"
: > "$PID_FILE"

common_args=(
  --output-root "$OUTPUT_ROOT"
  --split-dir "$SPLIT_DIR"
  --cache-dir "$CACHE_DIR"
  --baseline-dir "$BASELINE_DIR"
  --fusion-dir "$FUSION_DIR"
  --comparison-dir "$COMPARISON_DIR"
  --fullft-dir "$FULLFT_DIR"
  --seeds "${SEEDS[@]}"
  --split-kinds "${SPLIT_KINDS[@]}"
)

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] final mainline start"
echo "Output: $OUTPUT_ROOT"

python "$DATA_DIR/run_balanced811_6seed_mainline.py" make_splits \
  "${common_args[@]}" \
  --moe-dir "$MOE_MERGED_DIR" \
  2>&1 | tee "$OUTPUT_ROOT/make_splits.log"

python "$DATA_DIR/run_balanced811_6seed_mainline.py" build_cache \
  "${common_args[@]}" \
  --moe-dir "$MOE_MERGED_DIR" \
  --transform-mode presplit_subject_zscore_smooth_all \
  --smooth-alpha 0.65 \
  2>&1 | tee "$OUTPUT_ROOT/build_cache.log"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_baselines \
    ${common_args[*]} \
    --moe-dir \"$MOE_MERGED_DIR\" \
    --models logistic_regression svm xgboost frozen_mlp \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_baselines.log" 2>&1 &
echo "gpu1_baselines $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_fusion \
    ${common_args[*]} \
    --moe-dir \"$MOE_MERGED_DIR\" \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_fusion.log" 2>&1 &
echo "gpu1_fusion $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" screen_moe_lr \
    ${common_args[*]} \
    --moe-dir \"$MOE_SCREEN_DIR\" \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_moe_screen.log" 2>&1 &
echo "gpu1_moe_screen $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=1
  while [[ ! -f \"$MOE_SCREEN_DIR/selected_lr.json\" ]]; do sleep 30; done
  SELECTED_LR=\$(python - <<'PY'
import json
from pathlib import Path
p = Path('$MOE_SCREEN_DIR') / 'selected_lr.json'
print(json.loads(p.read_text())['selected_base_lr'])
PY
)
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
    ${common_args[*]} \
    --moe-dir \"$MOE_DE_DIR\" \
    --variants cross_attn_moe_unsup cross_attn_moe_sup_l03 cross_attn_moe_sup_l05 \
    --base-lr \"\$SELECTED_LR\" \
    --device cuda
" > "$OUTPUT_ROOT/gpu1_moe_de.log" 2>&1 &
echo "gpu1_moe_de $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=0
  while [[ ! -f \"$MOE_SCREEN_DIR/selected_lr.json\" ]]; do sleep 30; done
  SELECTED_LR=\$(python - <<'PY'
import json
from pathlib import Path
p = Path('$MOE_SCREEN_DIR') / 'selected_lr.json'
print(json.loads(p.read_text())['selected_base_lr'])
PY
)
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
    ${common_args[*]} \
    --moe-dir \"$MOE_TOKEN_DIR\" \
    --variants cross_attn_moe_token_sup_l03 cross_attn_moe_token_sup_l05 \
    --base-lr \"\$SELECTED_LR\" \
    --device cuda
" > "$OUTPUT_ROOT/gpu0_moe_token.log" 2>&1 &
echo "gpu0_moe_token $!" | tee -a "$PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  while read -r _ pid; do
    while kill -0 \"\$pid\" 2>/dev/null; do sleep 60; done
  done < \"$PID_FILE\"

  mkdir -p \"$MOE_MERGED_DIR\"
  python - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path('$OUTPUT_ROOT')
screen = Path('$MOE_SCREEN_DIR')
out = Path('$MOE_MERGED_DIR')
sources = [Path('$MOE_DE_DIR'), Path('$MOE_TOKEN_DIR')]
out.mkdir(parents=True, exist_ok=True)

all_frames = [pd.read_csv(src / 'all_results.csv') for src in sources if (src / 'all_results.csv').exists()]
router_frames = [pd.read_csv(src / 'router_metrics.csv') for src in sources if (src / 'router_metrics.csv').exists()]
if all_frames:
    pd.concat(all_frames, ignore_index=True).to_csv(out / 'all_results.csv', index=False)
if router_frames:
    pd.concat(router_frames, ignore_index=True).to_csv(out / 'router_metrics.csv', index=False)
selected = json.loads((screen / 'selected_lr.json').read_text())
(out / 'selected_lr.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
(out / 'merge_manifest.json').write_text(json.dumps({
    'sources': [src.name for src in sources],
    'n_all_rows': int(sum(len(frame) for frame in all_frames)),
    'n_router_rows': int(sum(len(frame) for frame in router_frames)),
    'selected_lr': selected,
}, indent=2), encoding='utf-8')
PY

  python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" summarize \
    ${common_args[*]} \
    --moe-dir \"$MOE_MERGED_DIR\" \
    2>&1 | tee \"$OUTPUT_ROOT/summarize.log\"
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launched final mainline jobs"
cat "$PID_FILE"
