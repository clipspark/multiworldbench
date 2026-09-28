#!/usr/bin/env bash
set -Eeuo pipefail
RUN=${MWB3_PERSISTENCE_RUN:?set MWB3_PERSISTENCE_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
PY=${QWEN_JUDGE_PY:-/data2/xuzhangbo/.venvs/qwen-judge/bin/python}
CASES=${MWB3_PERSISTENCE_CASES:-$REPO/MultiWorldBench/v3/cases/shared_world/shared_state_persistence.json}
ASSETS=${MWB3_PERSISTENCE_ASSETS:-$RUN/assets/interaction}
EXPECTED=150
IFS=',' read -r -a GPUS <<<"${MWB3_VLM_GPUS:-0,1,2,3,4,5,6,7}"
SHARDS=${#GPUS[@]}
(( SHARDS > 0 )) || { echo "MWB3_VLM_GPUS contains no GPUs"; exit 29; }
mkdir -p "$RUN/logs/vlm"
exec >>"$RUN/logs/vlm_manager.log" 2>&1
echo "[$(date -Is)] vlm_manager_start"
cd "$REPO"
export MWB3_PERSISTENCE_RUN="$RUN" MWB3_PERSISTENCE_CASES="$CASES" MWB3_PERSISTENCE_ASSETS="$ASSETS"
"$PY" controller/multiworldbench_v3/tools/evaluate_shared_state_persistence.py prepare
actual=$("$PY" - <<'PY'
import json,os
from pathlib import Path
print(len(json.loads((Path(os.environ['MWB3_PERSISTENCE_RUN'])/'evaluation_change_detection/requests.json').read_text())))
PY
)
[[ "$actual" == "$EXPECTED" ]] || { echo "expected $EXPECTED requests, got $actual"; exit 30; }
declare -a pids
for i in $(seq 0 $((SHARDS-1))); do
  gpu=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$PY" controller/multiworldbench_v3/tools/evaluate_shared_state_persistence.py judge \
    --shard "$i" --num-shards "$SHARDS" --max-memory-gib 43 \
    >"$RUN/logs/vlm/shard_$(printf '%02d' "$i").log" 2>&1 &
  pids[$i]=$!;echo "[$(date -Is)] judge_start shard=$i gpu=$gpu pid=${pids[$i]}"
done
failed=0
for i in $(seq 0 $((SHARDS-1))); do wait "${pids[$i]}" || failed=1; done
(( failed==0 )) || exit 31
outputs=$(find "$RUN/evaluation_change_detection/vlm" -type f -name '*.json' | wc -l)
[[ "$outputs" == "$EXPECTED" ]] || { echo "expected $EXPECTED outputs, got $outputs"; exit 32; }
"$PY" controller/multiworldbench_v3/tools/evaluate_shared_state_persistence.py aggregate
"$PY" controller/multiworldbench_v3/tools/report_shared_state_persistence.py
touch "$RUN/.evaluation_change_detection_complete"
echo "[$(date -Is)] vlm_manager_complete outputs=$outputs"
