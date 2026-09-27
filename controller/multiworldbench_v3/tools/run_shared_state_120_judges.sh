#!/usr/bin/env bash
set -Eeuo pipefail

RUN=${MWB3_SHARED_STATE_RUN:?set MWB3_SHARED_STATE_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
PY=${QWEN_JUDGE_PY:-/data2/xuzhangbo/.venvs/qwen-judge/bin/python}
CASES=${MWB3_SHARED_STATE_CASES:-$REPO/MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json}
ASSETS=${MWB3_SHARED_STATE_ASSETS:-$RUN/assets/interaction}
SHARDS=8
EXPECTED=1080

mkdir -p "$RUN/logs/vlm"
exec >>"$RUN/logs/vlm_manager.log" 2>&1
echo "[$(date -Is)] vlm_manager_start"
cd "$REPO"

export MWB3_SHARED_STATE_RUN="$RUN"
export MWB3_SHARED_STATE_CASES="$CASES"
export MWB3_SHARED_STATE_ASSETS="$ASSETS"

"$PY" controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py prepare
actual=$("$PY" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['MWB3_SHARED_STATE_RUN'])/'evaluation/requests.json'
print(len(json.loads(p.read_text())))
PY
)
[[ "$actual" == "$EXPECTED" ]] || { echo "expected $EXPECTED requests, got $actual"; exit 30; }

declare -a pids
for i in $(seq 0 $((SHARDS-1))); do
  CUDA_VISIBLE_DEVICES="$i" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$PY" controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py \
    judge --shard "$i" --num-shards "$SHARDS" --max-memory-gib 43 \
    >"$RUN/logs/vlm/shard_$(printf '%02d' "$i").log" 2>&1 &
  pids[$i]=$!
  echo "[$(date -Is)] judge_start shard=$i gpu=$i pid=${pids[$i]}"
done

failed=0
for i in $(seq 0 $((SHARDS-1))); do
  wait "${pids[$i]}" || failed=1
done
(( failed == 0 )) || exit 31

outputs=$(find "$RUN/evaluation/vlm" -type f -name '*.json' | wc -l)
[[ "$outputs" == "$EXPECTED" ]] || { echo "expected $EXPECTED outputs, got $outputs"; exit 32; }
"$PY" controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py aggregate
"$PY" controller/multiworldbench_v3/tools/report_shared_state_comparison.py
touch "$RUN/.evaluation_complete"
echo "[$(date -Is)] vlm_manager_complete outputs=$outputs"
