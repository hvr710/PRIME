#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
DATA_DIR="$REPO_ROOT/data"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/top3seed_l05_fullrerun_with_preproc_ablation}"

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
PREPROC_ROOT="$OUTPUT_ROOT/preproc_ablation"
PID_FILE="$OUTPUT_ROOT/top3_l05_pids.txt"
WAIT_PID_FILE="$OUTPUT_ROOT/top3_l05_wait_pids.txt"

ALL_SEEDS=(453402 1344584 1202623)
SPLIT_KINDS=(subject segment)

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  SEEDS=(453402)
  MAIN_SPLITS=(subject)
  BASELINE_MODELS=(frozen_mlp)
  MAIN_MOE_DE_VARIANTS=()
  MAIN_MOE_TOKEN_VARIANTS=(cross_attn_moe_token_sup_l05)
  PREPROC_MODES=(raw_no_preprocess presplit_subject_zscore_smooth_all)
  PREPROC_SPLITS=(subject)
  SCREEN_SEED=453402
else
  SEEDS=("${ALL_SEEDS[@]}")
  MAIN_SPLITS=("${SPLIT_KINDS[@]}")
  BASELINE_MODELS=(logistic_regression svm xgboost frozen_mlp)
  MAIN_MOE_DE_VARIANTS=(
    cross_attn_moe_unsup
    cross_attn_moe_sup_l03
    cross_attn_moe_sup_l05
  )
  MAIN_MOE_TOKEN_VARIANTS=(
    cross_attn_moe_token_sup_l03
    cross_attn_moe_token_sup_l05
  )
  PREPROC_MODES=(
    presplit_subject_zscore_smooth_all
    presplit_subject_zscore_all
    presplit_subject_smooth_all
    raw_no_preprocess
  )
  PREPROC_SPLITS=("${SPLIT_KINDS[@]}")
  SCREEN_SEED=453402
fi

if [[ -e "$OUTPUT_ROOT" && "${OVERWRITE:-0}" != "1" ]]; then
  cat <<EOF
Output directory already exists:
  $OUTPUT_ROOT

Use a new directory:
  OUTPUT_ROOT=$REPO_ROOT/outputs/top3_l05_rerun_\$(date +%Y%m%d_%H%M%S) bash run_top3_l05_fullrerun_with_preproc_ablation.sh

or explicitly allow overwrite:
  OVERWRITE=1 bash run_top3_l05_fullrerun_with_preproc_ablation.sh
EOF
  exit 1
fi

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p \
  "$OUTPUT_ROOT" "$BASELINE_DIR" "$FUSION_DIR" "$MOE_SCREEN_DIR" "$MOE_DE_DIR" \
  "$MOE_TOKEN_DIR" "$MOE_MERGED_DIR" "$COMPARISON_DIR" "$PREPROC_ROOT"
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
  --split-kinds "${MAIN_SPLITS[@]}"
)

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] top3 l05 rerun start"
echo "Output: $OUTPUT_ROOT"
echo "Seeds: ${SEEDS[*]}"
echo "Main splits: ${MAIN_SPLITS[*]}"
echo "Preproc modes: ${PREPROC_MODES[*]}"

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
    --models ${BASELINE_MODELS[*]} \
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
  export CUDA_VISIBLE_DEVICES=2
  PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" screen_moe_lr \
    ${common_args[*]} \
    --moe-dir \"$MOE_SCREEN_DIR\" \
    --screen-seed $SCREEN_SEED \
    --device cuda
" > "$OUTPUT_ROOT/gpu2_moe_screen.log" 2>&1 &
echo "gpu2_moe_screen $!" | tee -a "$PID_FILE"

if [[ ${#MAIN_MOE_DE_VARIANTS[@]} -gt 0 ]]; then
nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=2
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
    --variants ${MAIN_MOE_DE_VARIANTS[*]} \
    --base-lr \"\$SELECTED_LR\" \
    --device cuda
" > "$OUTPUT_ROOT/gpu2_moe_de.log" 2>&1 &
echo "gpu2_moe_de $!" | tee -a "$PID_FILE"
fi

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  export CUDA_VISIBLE_DEVICES=3
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
    --variants ${MAIN_MOE_TOKEN_VARIANTS[*]} \
    --base-lr \"\$SELECTED_LR\" \
    --device cuda
" > "$OUTPUT_ROOT/gpu3_moe_token.log" 2>&1 &
echo "gpu3_moe_token $!" | tee -a "$PID_FILE"

for mode in "${PREPROC_MODES[@]}"; do
  MODE_ROOT="$PREPROC_ROOT/$mode"
  MODE_CACHE_DIR="$MODE_ROOT/adapt_cache"
  MODE_BASELINE_DIR="$MODE_ROOT/baseline_results_unused"
  MODE_FUSION_DIR="$MODE_ROOT/fusion_results_unused"
  MODE_MOE_DIR="$MODE_ROOT/moe_results"
  MODE_COMPARISON_DIR="$MODE_ROOT/comparison"
  MODE_FULLFT_DIR="$MODE_ROOT/fullft_unused"
  MODE_COMMON_ARGS=(
    --output-root "$MODE_ROOT"
    --split-dir "$SPLIT_DIR"
    --cache-dir "$MODE_CACHE_DIR"
    --baseline-dir "$MODE_BASELINE_DIR"
    --fusion-dir "$MODE_FUSION_DIR"
    --comparison-dir "$MODE_COMPARISON_DIR"
    --fullft-dir "$MODE_FULLFT_DIR"
    --seeds "${SEEDS[@]}"
    --split-kinds "${PREPROC_SPLITS[@]}"
  )

  mkdir -p "$MODE_ROOT" "$MODE_MOE_DIR" "$MODE_COMPARISON_DIR"
  python "$DATA_DIR/run_balanced811_6seed_mainline.py" build_cache \
    "${MODE_COMMON_ARGS[@]}" \
    --moe-dir "$MODE_MOE_DIR" \
    --transform-mode "$mode" \
    --smooth-alpha 0.65 \
    > "$MODE_ROOT/build_cache.log" 2>&1

  case "$mode" in
    presplit_subject_zscore_smooth_all) ABLATION_GPU=0 ;;
    presplit_subject_zscore_all) ABLATION_GPU=1 ;;
    presplit_subject_smooth_all) ABLATION_GPU=2 ;;
    raw_no_preprocess) ABLATION_GPU=3 ;;
    *) ABLATION_GPU=0 ;;
  esac

  nohup bash -lc "
    source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
    while [[ ! -f \"$MOE_SCREEN_DIR/selected_lr.json\" ]]; do sleep 30; done
    SELECTED_LR=\$(python - <<'PY'
import json
from pathlib import Path
p = Path('$MOE_SCREEN_DIR') / 'selected_lr.json'
print(json.loads(p.read_text())['selected_base_lr'])
PY
)
    export CUDA_VISIBLE_DEVICES=$ABLATION_GPU
    PYTHONUNBUFFERED=1 python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" train_moe \
      ${MODE_COMMON_ARGS[*]} \
      --moe-dir \"$MODE_MOE_DIR\" \
      --variants cross_attn_moe_token_sup_l05 \
      --base-lr \"\$SELECTED_LR\" \
      --device cuda
  " > "$MODE_ROOT/gpu${ABLATION_GPU}_train_moe.log" 2>&1 &
  echo "gpu${ABLATION_GPU}_ablation_${mode} $!" | tee -a "$PID_FILE"
done

cp "$PID_FILE" "$WAIT_PID_FILE"

nohup bash -lc "
  source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba
  while read -r _ pid; do
    while kill -0 \"\$pid\" 2>/dev/null; do sleep 60; done
  done < \"$WAIT_PID_FILE\"

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
    > \"$OUTPUT_ROOT/summarize.log\" 2>&1
  if [[ -f \"$OUTPUT_ROOT/REPORT.md\" ]]; then
    cp \"$OUTPUT_ROOT/REPORT.md\" \"$OUTPUT_ROOT/REPORT_mainline.md\"
  fi

  for mode in ${PREPROC_MODES[*]}; do
    MODE_ROOT=\"$PREPROC_ROOT/\$mode\"
    MODE_CACHE_DIR=\"\$MODE_ROOT/adapt_cache\"
    MODE_BASELINE_DIR=\"\$MODE_ROOT/baseline_results_unused\"
    MODE_FUSION_DIR=\"\$MODE_ROOT/fusion_results_unused\"
    MODE_MOE_DIR=\"\$MODE_ROOT/moe_results\"
    MODE_COMPARISON_DIR=\"\$MODE_ROOT/comparison\"
    MODE_FULLFT_DIR=\"\$MODE_ROOT/fullft_unused\"
    python \"$DATA_DIR/run_balanced811_6seed_mainline.py\" summarize \
      --output-root \"\$MODE_ROOT\" \
      --split-dir \"$SPLIT_DIR\" \
      --cache-dir \"\$MODE_CACHE_DIR\" \
      --baseline-dir \"\$MODE_BASELINE_DIR\" \
      --fusion-dir \"\$MODE_FUSION_DIR\" \
      --moe-dir \"\$MODE_MOE_DIR\" \
      --comparison-dir \"\$MODE_COMPARISON_DIR\" \
      --fullft-dir \"\$MODE_FULLFT_DIR\" \
      --seeds ${SEEDS[*]} \
      --split-kinds ${PREPROC_SPLITS[*]} \
      > \"\$MODE_ROOT/summarize.log\" 2>&1
  done

  python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('$OUTPUT_ROOT')
preproc_root = root / 'preproc_ablation'
comparison_dir = root / 'comparison'
comparison_dir.mkdir(parents=True, exist_ok=True)
main_summary_path = comparison_dir / 'mean_std_summary.csv'
best_seed_path = comparison_dir / 'best_single_seed.csv'

frames = []
summary_rows = []
for mode_root in sorted(preproc_root.iterdir()):
    if not mode_root.is_dir():
        continue
    mode = mode_root.name
    result_path = mode_root / 'moe_results' / 'all_results.csv'
    if not result_path.exists():
        continue
    df = pd.read_csv(result_path)
    if len(df) == 0:
        continue
    df['transform_mode'] = mode
    frames.append(df)
    for split_kind, sub in df.groupby('split_kind'):
        row = {
            'transform_mode': mode,
            'split_kind': split_kind,
            'n_rows': int(len(sub)),
            'test_window_acc': f\"{sub['test_window_acc'].mean()*100:.2f} ± {sub['test_window_acc'].std(ddof=0)*100:.2f}\",
            'test_window_f1': f\"{sub['test_window_f1'].mean()*100:.2f} ± {sub['test_window_f1'].std(ddof=0)*100:.2f}\",
            'test_window_auroc': f\"{sub['test_window_auroc'].mean()*100:.2f} ± {sub['test_window_auroc'].std(ddof=0)*100:.2f}\",
            'test_trial_auroc': '-',
        }
        if split_kind == 'subject':
            vals = sub['test_trial_auroc'].dropna().astype(float)
            row['test_trial_auroc'] = f\"{vals.mean()*100:.2f} ± {vals.std(ddof=0)*100:.2f}\" if len(vals) else '-'
        summary_rows.append(row)

if frames:
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(comparison_dir / 'preproc_ablation_all_results.csv', index=False)
pd.DataFrame(summary_rows).to_csv(comparison_dir / 'preproc_ablation_summary.csv', index=False)

mode_labels = {
    'presplit_subject_zscore_smooth_all': 'subject z-score + smoothing',
    'presplit_subject_zscore_all': 'subject z-score only',
    'presplit_subject_smooth_all': 'smoothing only',
    'raw_no_preprocess': 'raw frozen features',
}

report_lines = [
    '# Top-3 L05 Seeds Mainline Re-run + Preprocessing Ablation',
    '',
    '- Seeds: `453402, 1344584, 1202623`',
    '- Main rerun: baselines + CrossAttn + formal MoE + LR screening',
    '- Extra ablation: `cross_attn_moe_token_sup_l05` under 4 preprocessing modes',
    '',
    '## Main Files',
    '',
    f'- Main summary: `{comparison_dir / \"mean_std_summary.csv\"}`',
    f'- Main best seeds: `{comparison_dir / \"best_single_seed.csv\"}`',
    f'- Preproc ablation summary: `{comparison_dir / \"preproc_ablation_summary.csv\"}`',
    '',
    '## Mainline Output',
    '',
    '- `REPORT_mainline.md` contains the main experiment matrix summary generated by `summarize`.',
    '- `comparison/per_seed_results.csv` stores all per-seed rows for the top-3 rerun.',
    '- `comparison/mean_std_summary.csv` stores the 3-seed mean ± std summary.',
    '',
    '## Preprocessing Modes',
    '',
    '- `presplit_subject_zscore_smooth_all`: z-score + smoothing',
    '- `presplit_subject_zscore_all`: z-score only',
    '- `presplit_subject_smooth_all`: smoothing only',
    '- `raw_no_preprocess`: no z-score, no smoothing',
]
(root / 'REPORT.md').write_text('\\n'.join(report_lines) + '\\n', encoding='utf-8')

if summary_rows:
    report_lines.extend([
        '',
        '## Preprocessing Ablation Summary',
        '',
        '| Transform | Split | Test Acc | Test F1 | Test AUROC | Test Trial AUROC |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for row in summary_rows:
        report_lines.append(
            f\"| {mode_labels.get(row['transform_mode'], row['transform_mode'])} | {row['split_kind']} | {row['test_window_acc']} | {row['test_window_f1']} | {row['test_window_auroc']} | {row['test_trial_auroc']} |\"
        )

(root / 'REPORT.md').write_text('\\n'.join(report_lines) + '\\n', encoding='utf-8')
(comparison_dir / 'run_manifest.json').write_text(json.dumps({
    'seeds': [453402, 1344584, 1202623],
    'main_splits': ${#MAIN_SPLITS[@]},
    'preproc_modes': ${#PREPROC_MODES[@]},
    'screen_seed': $SCREEN_SEED,
    'smoke_only': ${SMOKE_ONLY:-0},
}, indent=2), encoding='utf-8')
PY
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launched top3 l05 rerun jobs"
cat "$PID_FILE"
