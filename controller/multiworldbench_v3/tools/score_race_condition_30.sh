#!/usr/bin/env bash
set -Eeuo pipefail
RUN=${MWB3_RACE_RUN:?set MWB3_RACE_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
JUDGE_PY=${MWB3_JUDGE_PY:-/data2/xuzhangbo/.venvs/qwen-judge/bin/python}
SHARDS=4
cd "$REPO"
test -f "$RUN/.inference_complete"
mkdir -p "$RUN/logs/evaluation"
exec >>"$RUN/logs/evaluation_manager.log" 2>&1
echo "[$(date -Is)] evaluation_start"
for model in solaris gamma; do
  "$JUDGE_PY" controller/multiworldbench_v3/tools/judge_race_condition.py "$RUN" "$model" prepare
 done
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  CUDA_VISIBLE_DEVICES="$i" "$JUDGE_PY" controller/multiworldbench_v3/tools/judge_race_condition.py "$RUN" solaris judge "$i" "$SHARDS" >"$RUN/logs/evaluation/solaris_$i.log" 2>&1 & pids+=("$!")
done
for i in $(seq 0 $((SHARDS-1))); do
  gpu=$((i+SHARDS))
  CUDA_VISIBLE_DEVICES="$gpu" "$JUDGE_PY" controller/multiworldbench_v3/tools/judge_race_condition.py "$RUN" gamma judge "$i" "$SHARDS" >"$RUN/logs/evaluation/gamma_$i.log" 2>&1 & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1;done
(( failed==0 )) || exit 31
for model in solaris gamma; do "$JUDGE_PY" controller/multiworldbench_v3/tools/judge_race_condition.py "$RUN" "$model" aggregate;done
python3 controller/multiworldbench_v3/tools/report_race_condition.py "$RUN"
touch "$RUN/.evaluation_complete"
echo "[$(date -Is)] evaluation_complete"
