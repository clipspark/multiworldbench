#!/usr/bin/env bash
set -Eeuo pipefail

RUN=${MWB3_BUILDING_REVEAL_RUN:?set MWB3_BUILDING_REVEAL_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
JUDGE_PY=${MWB3_JUDGE_PY:-/data2/xuzhangbo/.venvs/qwen-judge/bin/python}
EVALUATOR=controller/multiworldbench_v3/tools/evaluate_building_reveal_75.py
GEOMETRY=controller/multiworldbench_v3/tools/measure_building_reveal_75_geometry.py
REPORT=controller/multiworldbench_v3/tools/report_building_reveal_75.py
PRIMARY=/data2/xuzhangbo/models/Qwen2.5-VL-72B-Instruct-AWQ
FALLBACK=/data2/xuzhangbo/models/Qwen2.5-VL-3B-Instruct
IFS=',' read -r -a GPU_IDS <<< "${MWB3_EVALUATION_GPUS:-0,1,2,3,4,5,6,7}"
SHARDS=${#GPU_IDS[@]}
(( SHARDS > 0 )) || { echo "at least one evaluation GPU is required" >&2; exit 2; }

test -f "$RUN/.inference_complete"
mkdir -p "$RUN/logs/evaluation"
exec >>"$RUN/logs/evaluation_manager.log" 2>&1
echo "[$(date -Is)] evaluation_start"
cd "$REPO"
export MWB3_BUILDING_REVEAL_RUN="$RUN"

"$JUDGE_PY" "$EVALUATOR" prepare

run_judges() {
  local checkpoint=$1 label=$2 memory=$3
  local failed=0
  declare -a pids=()
  for i in $(seq 0 $((SHARDS-1))); do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" "$JUDGE_PY" "$EVALUATOR" judge \
      --shard-index "$i" --num-shards "$SHARDS" --max-memory-gib "$memory" \
      --checkpoint "$checkpoint" \
      >"$RUN/logs/evaluation/${label}_shard_$(printf '%02d' "$i").log" 2>&1 &
    pids[$i]=$!
    echo "[$(date -Is)] ${label}_start shard=$i gpu=${GPU_IDS[$i]} pid=${pids[$i]}"
  done
  for i in $(seq 0 $((SHARDS-1))); do
    wait "${pids[$i]}" || failed=1
  done
  (( failed == 0 ))
}

run_judges "$PRIMARY" qwen72b 43
missing=$("$JUDGE_PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); req=json.loads((root/'evaluation/requests.json').read_text())
print(sum(not (p:=root/'evaluation/vlm'/f"{r['request_id']}.json").is_file() or json.loads(p.read_text()).get('measurement') is None for r in req))
PY
)
echo "[$(date -Is)] primary_complete missing=$missing"
if (( missing > 0 )); then
  run_judges "$FALLBACK" qwen3b 16
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$JUDGE_PY" "$GEOMETRY" >"$RUN/logs/evaluation/geometry.log" 2>&1
"$JUDGE_PY" "$EVALUATOR" aggregate
python3 "$REPORT"
touch "$RUN/.evaluation_complete"
echo "[$(date -Is)] evaluation_complete"
