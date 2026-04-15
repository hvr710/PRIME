#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
DATA_DIR="$REPO_ROOT/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

PREV_ROOT="${PREV_ROOT:-$REPO_ROOT/outputs/subject_bestseed_100scan_crossattn_tokenmoe_20260415}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/subject_bestseed_extra1000_crossattn_tokenmoe_20260415}"

SPLIT_DIR="$OUTPUT_ROOT/splits"
CACHE_DIR="$OUTPUT_ROOT/adapt_cache"
BASELINE_DIR="$OUTPUT_ROOT/baseline_results_unused"
FUSION_SHARD_ROOT="$OUTPUT_ROOT/fusion_shards"
MOE_L03_SHARD_ROOT="$OUTPUT_ROOT/moe_token_l03_shards"
MOE_L05_SHARD_ROOT="$OUTPUT_ROOT/moe_token_l05_shards"
FUSION_DIR="$OUTPUT_ROOT/fusion_results"
MOE_DIR="$OUTPUT_ROOT/moe_results"
COMPARISON_DIR="$OUTPUT_ROOT/comparison"
COMBINED_DIR="$OUTPUT_ROOT/combined_with_prev100"
FULLFT_DIR="$OUTPUT_ROOT/fullft_unused"
META_DIR="$OUTPUT_ROOT/meta"
PID_FILE="$OUTPUT_ROOT/launched_pids.txt"

N_EXTRA_SEEDS="${N_EXTRA_SEEDS:-1000}"
N_SHARDS="${N_SHARDS:-8}"
MOE_BASE_LR="${MOE_BASE_LR:-5e-4}"
TRANSFORM_MODE="${TRANSFORM_MODE:-presplit_subject_zscore_smooth_all}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.65}"

if [[ -e "$OUTPUT_ROOT" && "${OVERWRITE:-0}" != "1" ]]; then
  cat <<EOF
Output directory already exists:
  $OUTPUT_ROOT

Use a new directory:
  OUTPUT_ROOT=$REPO_ROOT/outputs/subject_bestseed_extra1000_\$(date +%Y%m%d_%H%M%S) bash run_subject_bestseed_scan_extra1000.sh

or force overwrite:
  OVERWRITE=1 bash run_subject_bestseed_scan_extra1000.sh
EOF
  exit 1
fi

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p \
  "$OUTPUT_ROOT" "$SPLIT_DIR" "$CACHE_DIR" "$BASELINE_DIR" "$FUSION_SHARD_ROOT" \
  "$MOE_L03_SHARD_ROOT" "$MOE_L05_SHARD_ROOT" "$FUSION_DIR" "$MOE_DIR" \
  "$COMPARISON_DIR" "$COMBINED_DIR" "$FULLFT_DIR" "$META_DIR"
: > "$PID_FILE"

python - <<'PY' "$META_DIR" "$PREV_ROOT" "$N_EXTRA_SEEDS" "$N_SHARDS"
from pathlib import Path
import json
import numpy as np
import sys

meta = Path(sys.argv[1])
prev_root = Path(sys.argv[2])
n_extra = int(sys.argv[3])
n_shards = int(sys.argv[4])

prev_seed_file = prev_root / "meta" / "seeds_100.txt"
prev_seeds = []
if prev_seed_file.exists():
    prev_seeds = [int(x) for x in prev_seed_file.read_text(encoding="utf-8").split()]

rng = np.random.default_rng(20260416)
seen = set(prev_seeds)
extra = []
while len(extra) < n_extra:
    cand = int(rng.integers(1, 2_000_000))
    if cand in seen:
        continue
    seen.add(cand)
    extra.append(cand)

shards = [extra[i::n_shards] for i in range(n_shards)]
(meta / "seeds_1000.txt").write_text("\n".join(str(x) for x in extra) + "\n", encoding="utf-8")
(meta / "prev_seeds_100.txt").write_text("\n".join(str(x) for x in prev_seeds) + ("\n" if prev_seeds else ""), encoding="utf-8")
payload = {
    "seed_generation_rng": 20260416,
    "n_extra_seeds": n_extra,
    "extra_seeds": extra,
    "previous_seed_count": len(prev_seeds),
    "previous_seeds": prev_seeds,
    "combined_seed_count": len(prev_seeds) + len(extra),
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

SEED_COUNT="$(wc -l < "$META_DIR/seeds_1000.txt")"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] extra-1000 seed scan start"
echo "Output: $OUTPUT_ROOT"
echo "Prev root: $PREV_ROOT"
echo "Extra seeds: $SEED_COUNT"
echo "Shards: $N_SHARDS"

"$PYTHON_BIN" "$DATA_DIR/run_balanced811_6seed_mainline.py" make_splits \
  --output-root "$OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --baseline-dir "$BASELINE_DIR" \
  --fusion-dir "$FUSION_DIR" \
  --moe-dir "$MOE_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --fullft-dir "$FULLFT_DIR" \
  --seeds $(tr '\n' ' ' < "$META_DIR/seeds_1000.txt") \
  --split-kinds subject \
  > "$OUTPUT_ROOT/make_splits.log" 2>&1

"$PYTHON_BIN" "$DATA_DIR/run_balanced811_6seed_mainline.py" build_cache \
  --output-root "$OUTPUT_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --baseline-dir "$BASELINE_DIR" \
  --fusion-dir "$FUSION_DIR" \
  --moe-dir "$MOE_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --fullft-dir "$FULLFT_DIR" \
  --seeds $(tr '\n' ' ' < "$META_DIR/seeds_1000.txt") \
  --split-kinds subject \
  --transform-mode "$TRANSFORM_MODE" \
  --smooth-alpha "$SMOOTH_ALPHA" \
  > "$OUTPUT_ROOT/build_cache.log" 2>&1

for shard_id in $(seq 0 $((N_SHARDS - 1))); do
  gpu_id=$(( shard_id % 4 ))
  shard_seeds="$(cat "$META_DIR/shard${shard_id}.txt")"

  mkdir -p \
    "$FUSION_SHARD_ROOT/shard${shard_id}" \
    "$MOE_L03_SHARD_ROOT/shard${shard_id}" \
    "$MOE_L05_SHARD_ROOT/shard${shard_id}"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
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
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
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
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
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
fusion_dir.mkdir(parents=True, exist_ok=True)
moe_dir.mkdir(parents=True, exist_ok=True)

fusion_frames = [pd.read_csv(p / 'all_results.csv') for p in fusion_shards if (p / 'all_results.csv').exists()]
if fusion_frames:
    fusion_df = pd.concat(fusion_frames, ignore_index=True)
    fusion_df.to_csv(fusion_dir / 'all_results.csv', index=False)
else:
    fusion_df = pd.DataFrame()

moe_frames = []
router_frames = []
for shard_list in [moe_l03_shards, moe_l05_shards]:
    for p in shard_list:
        if (p / 'all_results.csv').exists():
            moe_frames.append(pd.read_csv(p / 'all_results.csv'))
        if (p / 'router_metrics.csv').exists():
            router_frames.append(pd.read_csv(p / 'router_metrics.csv'))
if moe_frames:
    moe_df = pd.concat(moe_frames, ignore_index=True)
    moe_df.to_csv(moe_dir / 'all_results.csv', index=False)
else:
    moe_df = pd.DataFrame()
if router_frames:
    pd.concat(router_frames, ignore_index=True).to_csv(moe_dir / 'router_metrics.csv', index=False)

(moe_dir / 'selected_lr.json').write_text(json.dumps({
    'selected_base_lr': 5e-4,
    'selection_rule': 'fixed_from_current_mainline'
}, indent=2), encoding='utf-8')
(moe_dir / 'merge_manifest.json').write_text(json.dumps({
    'fusion_rows': int(len(fusion_df)),
    'moe_rows': int(len(moe_df)),
    'seed_count': 1000,
    'n_shards': $N_SHARDS,
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
    --seeds \$(tr '\n' ' ' < \"$META_DIR/seeds_1000.txt\") \
    --split-kinds subject \
    > \"$OUTPUT_ROOT/summarize.log\" 2>&1

  \"$PYTHON_BIN\" - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('$OUTPUT_ROOT')
prev_root = Path('$PREV_ROOT')
combined_dir = root / 'combined_with_prev100'
combined_dir.mkdir(parents=True, exist_ok=True)

prev_path = prev_root / 'comparison' / 'per_seed_results.csv'
curr_path = root / 'comparison' / 'per_seed_results.csv'
if not prev_path.exists() or not curr_path.exists():
    raise SystemExit(0)

prev_df = pd.read_csv(prev_path)
curr_df = pd.read_csv(curr_path)
df = pd.concat([prev_df, curr_df], ignore_index=True)
df.to_csv(combined_dir / 'per_seed_results.csv', index=False)

metrics = [
    'test_window_acc',
    'test_window_f1',
    'test_window_auroc',
    'test_trial_acc',
    'test_trial_f1',
    'test_trial_auroc',
]
summary_rows = []
for (split_kind, method), sub in sorted(df.groupby(['split_kind', 'method'])):
    row = {'split_kind': split_kind, 'method': method}
    for metric in metrics:
        vals = sub[metric].dropna().astype(float)
        if len(vals) == 0:
            row[metric] = ''
        else:
            row[metric] = f'{vals.mean() * 100:.2f} ± {vals.std(ddof=0) * 100:.2f}'
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(combined_dir / 'mean_std_summary.csv', index=False)

best_rows = []
for (split_kind, method), sub in sorted(df.groupby(['split_kind', 'method'])):
    best = sub.sort_values(['test_window_acc', 'test_window_auroc', 'seed'], ascending=[False, False, True]).iloc[0]
    best_rows.append({
        'split_kind': split_kind,
        'method': method,
        'best_seed': int(best['seed']),
        'test_window_acc': best['test_window_acc'],
        'test_window_f1': best['test_window_f1'],
        'test_window_auroc': best['test_window_auroc'],
        'test_trial_acc': best.get('test_trial_acc', ''),
        'test_trial_f1': best.get('test_trial_f1', ''),
        'test_trial_auroc': best.get('test_trial_auroc', ''),
    })
pd.DataFrame(best_rows).to_csv(combined_dir / 'best_single_seed.csv', index=False)

(combined_dir / 'run_manifest.json').write_text(json.dumps({
    'previous_root': str(prev_root),
    'current_root': str(root),
    'combined_rows': int(len(df)),
    'previous_rows': int(len(prev_df)),
    'current_rows': int(len(curr_df)),
}, indent=2), encoding='utf-8')

report = [
    '# Combined 1100-Seed Scan',
    '',
    '- Previous scan: 100 seeds',
    '- Extra scan: 1000 seeds',
    '- Combined total: 1100 seeds',
    '',
    '## Best Single Seed',
    '',
]
best_df = pd.DataFrame(best_rows)
for _, row in best_df.sort_values(['split_kind', 'method']).iterrows():
    report.append(
        '- {method}: seed={seed}, test_window_acc={acc:.2f}, test_window_auroc={auc:.2f}'.format(
            method=row['method'],
            seed=int(row['best_seed']),
            acc=float(row['test_window_acc']) * 100,
            auc=float(row['test_window_auroc']) * 100,
        )
    )
report.append('')
report.append('## Summary Files')
report.append('')
report.append('- `best_single_seed.csv`')
report.append('- `mean_std_summary.csv`')
report.append('- `per_seed_results.csv`')
(combined_dir / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
PY
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launched extra-1000 seed scan"
cat "$PID_FILE"
