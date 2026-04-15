#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/outputs/subject_bestseed_extra1000_crossattn_tokenmoe_20260415}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/subject_bestseed_extra1000_preview_cached120_20260415}"
N_PREVIEW_SEEDS="${N_PREVIEW_SEEDS:-120}"
N_SHARDS="${N_SHARDS:-4}"
MOE_BASE_LR="${MOE_BASE_LR:-5e-4}"
EXCLUDE_SEEDS_FILE="${EXCLUDE_SEEDS_FILE:-}"

SPLIT_DIR="$SOURCE_ROOT/splits"
CACHE_DIR="$SOURCE_ROOT/adapt_cache"
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

if [[ -e "$OUTPUT_ROOT" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Output directory already exists: $OUTPUT_ROOT" >&2
  echo "Use OVERWRITE=1 to replace it." >&2
  exit 1
fi

source /vePFS-0x0d/home/cx/cx/miniconda3/bin/activate labram_mamba

mkdir -p \
  "$OUTPUT_ROOT" "$BASELINE_DIR" "$FUSION_SHARD_ROOT" "$MOE_L03_SHARD_ROOT" \
  "$MOE_L05_SHARD_ROOT" "$FUSION_DIR" "$MOE_DIR" "$COMPARISON_DIR" \
  "$FULLFT_DIR" "$META_DIR"
: > "$PID_FILE"

"$PYTHON_BIN" - <<'PY' "$SOURCE_ROOT" "$META_DIR" "$N_PREVIEW_SEEDS" "$N_SHARDS" "$EXCLUDE_SEEDS_FILE"
from pathlib import Path
import json
import re
import sys

source_root = Path(sys.argv[1])
meta_dir = Path(sys.argv[2])
n_preview = int(sys.argv[3])
n_shards = int(sys.argv[4])
exclude_file_arg = sys.argv[5].strip()

seed_file = source_root / "meta" / "seeds_1000.txt"
if not seed_file.exists():
    raise FileNotFoundError(f"Missing seed file: {seed_file}")

cache_dir = source_root / "adapt_cache" / "subject"
pattern = re.compile(r"comp4_subject_seed(\d+)\.h5$")
cached = set()
for path in cache_dir.glob("comp4_subject_seed*.h5"):
    match = pattern.match(path.name)
    if match:
        cached.add(int(match.group(1)))

ordered = [int(x.strip()) for x in seed_file.read_text(encoding="utf-8").splitlines() if x.strip()]
excluded = set()
if exclude_file_arg:
    exclude_path = Path(exclude_file_arg)
    if not exclude_path.exists():
        raise FileNotFoundError(f"Missing exclude seed file: {exclude_path}")
    excluded = {int(x.strip()) for x in exclude_path.read_text(encoding="utf-8").splitlines() if x.strip()}

available = [seed for seed in ordered if seed in cached and seed not in excluded]
selected = available[:n_preview] if n_preview > 0 else available
if not selected:
    raise RuntimeError("No cached seeds available for preview run.")

shards = [selected[i::n_shards] for i in range(n_shards)]
(meta_dir / "preview_seeds.txt").write_text("\n".join(str(x) for x in selected) + "\n", encoding="utf-8")
payload = {
    "source_root": str(source_root),
    "n_preview_seeds_requested": n_preview,
    "n_preview_seeds_selected": len(selected),
    "n_excluded_seeds": len(excluded),
    "exclude_seeds_file": exclude_file_arg or None,
    "selected_seeds": selected,
    "n_shards": n_shards,
    "shards": {f"shard{i}": shard for i, shard in enumerate(shards)},
    "models": [
        "cross_attn_single_head",
        "cross_attn_moe_token_sup_l03",
        "cross_attn_moe_token_sup_l05",
    ],
}
(meta_dir / "preview_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
for i, shard in enumerate(shards):
    (meta_dir / f"shard{i}.txt").write_text(" ".join(str(x) for x in shard) + "\n", encoding="utf-8")
print(len(selected))
PY

SELECTED_COUNT="$(tail -n 1 "$META_DIR/preview_manifest.json" >/dev/null 2>&1; wc -l < "$META_DIR/preview_seeds.txt")"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] preview scan start"
echo "Source root: $SOURCE_ROOT"
echo "Preview output: $OUTPUT_ROOT"
echo "Selected cached seeds: $SELECTED_COUNT"

for shard_id in $(seq 0 $((N_SHARDS - 1))); do
  gpu_id=$(( shard_id % 4 ))
  shard_seeds="$(cat "$META_DIR/shard${shard_id}.txt")"
  if [[ -z "${shard_seeds// }" ]]; then
    continue
  fi

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
    "$PYTHON_BIN" "$REPO_ROOT/data/run_balanced811_6seed_mainline.py" train_fusion \
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
    "$PYTHON_BIN" "$REPO_ROOT/data/run_balanced811_6seed_mainline.py" train_moe \
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
    "$PYTHON_BIN" "$REPO_ROOT/data/run_balanced811_6seed_mainline.py" train_moe \
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
    while kill -0 \"\$pid\" 2>/dev/null; do sleep 20; done
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
    'selection_rule': 'fixed_for_preview'
}, indent=2), encoding='utf-8')
PY

  \"$PYTHON_BIN\" \"$REPO_ROOT/data/run_balanced811_6seed_mainline.py\" summarize \
    --output-root \"$OUTPUT_ROOT\" \
    --split-dir \"$SPLIT_DIR\" \
    --cache-dir \"$CACHE_DIR\" \
    --baseline-dir \"$BASELINE_DIR\" \
    --fusion-dir \"$FUSION_DIR\" \
    --moe-dir \"$MOE_DIR\" \
    --comparison-dir \"$COMPARISON_DIR\" \
    --fullft-dir \"$FULLFT_DIR\" \
    --seeds \$(tr '\n' ' ' < \"$META_DIR/preview_seeds.txt\") \
    --split-kinds subject \
    > \"$OUTPUT_ROOT/summarize.log\" 2>&1
" > "$OUTPUT_ROOT/summarize_async.log" 2>&1 &
echo "summarize_async $!" | tee -a "$PID_FILE"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launched preview scan"
cat "$PID_FILE"
