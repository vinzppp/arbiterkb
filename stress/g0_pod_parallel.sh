#!/usr/bin/env bash
# G0 parallel-width measurement on the RunPod A40 (model-free; no key in the environment).
# Usage on the pod, from the repo root, with MEMI at $MEMI (pinned bb6c37c):
#   MEMI=/workspace/memi bash stress/g0_pod_parallel.sh "1 2 4 6 8" results/memory_or_policy/g0/pod
# For each width k it starts k copies of each of the 5 tasks' val preflight (5k episodes,
# k at a time per task wave), records wall time per episode and the machine's load, and
# writes one JSON line per width.  Episodes are the ones G0 already used (no new val read).
set -euo pipefail
WIDTHS=${1:-"1 2 4 6 8"}
OUT=${2:-results/memory_or_policy/g0/pod}
MEMI=${MEMI:?set MEMI to the pinned MEMI checkout}
if [ -n "${OPENAI_API_KEY:-}${OPENROUTER_API_KEY:-}" ]; then echo "unset API keys first" >&2; exit 1; fi
mkdir -p "$OUT"
{ nproc; free -g | sed -n 2p; df -h /workspace | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } \
  > "$OUT/machine.txt" 2>&1 || true
TASKS="VideoUnmaskSwap:0 VideoRepick:0 RouteStick:49 SwingXtimes:49 StopCube:49"
for k in $WIDTHS; do
  start=$(date +%s)
  ( nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 5 > "$OUT/gpu_k$k.csv" 2>/dev/null & echo $! > "$OUT/.smi" )
  ( while sleep 5; do cat /proc/loadavg; done > "$OUT/load_k$k.txt" & echo $! > "$OUT/.load" )
  for te in $TASKS; do
    for i in $(seq 1 "$k"); do
      echo "${te%%:*} ${te##*:} $i"
    done
  done | xargs -P "$k" -L 1 sh -c \
    'python stress/g0_preflight.py --task $0 --episode $1 --memi-root "'"$MEMI"'" --out "'"$OUT"'/k'"$k"'_copy$2" | tail -1' \
    > "$OUT/episodes_k$k.jsonl"
  kill "$(cat "$OUT/.smi")" "$(cat "$OUT/.load")" 2>/dev/null || true
  end=$(date +%s)
  echo "{\"width\": $k, \"episodes\": $((5 * k)), \"wall_s\": $((end - start)), \"episodes_per_hour\": $(( 5 * k * 3600 / (end - start + 1) ))}" \
    | tee -a "$OUT/widths.jsonl"
done
du -sh "$OUT" | tee "$OUT/disk.txt"
