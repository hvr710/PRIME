#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_dir> [interval_seconds]" >&2
  exit 1
fi

OUT="$1"
INTERVAL="${2:-300}"
LOG="$OUT/monitor_progress.log"

mkdir -p "$OUT"
touch "$LOG"

while true; do
  ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  total="$(wc -l < "$OUT/meta/seeds_1000.txt" 2>/dev/null || echo 0)"
  done_cache="$(grep -c '^cache subject seed=' "$OUT/build_cache.log" 2>/dev/null || echo 0)"
  fusion_rows="$(find "$OUT/fusion_shards" -type f -name 'all_results.csv' -exec wc -l {} + 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  moe_rows="$(find "$OUT/moe_token_l03_shards" "$OUT/moe_token_l05_shards" -type f -name 'all_results.csv' -exec wc -l {} + 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  train_ps="$(ps -eo cmd | awk '/train_fusion|train_moe|run_balanced811_6seed_mainline.py train/ && !/awk/ {c++} END{print c+0}')"
  build_ps="$(ps -eo cmd | awk -v out="$OUT" '/run_balanced811_6seed_mainline.py build_cache/ && index($0, out) > 0 && !/awk/ {c++} END{print c+0}')"
  gpu="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '; ')"
  printf '[%s] cache=%s/%s train_ps=%s build_ps=%s fusion_lines=%s moe_lines=%s gpu=%s\n' \
    "$ts" "$done_cache" "$total" "$train_ps" "$build_ps" "$fusion_rows" "$moe_rows" "$gpu" >> "$LOG"
  sleep "$INTERVAL"
done
