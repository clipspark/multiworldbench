#!/usr/bin/env bash
set -Eeuo pipefail

RUN=${MWB3_BUILDING_REVEAL_RUN:?set MWB3_BUILDING_REVEAL_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
GAMMA=${GAMMA_ROOT:-/data2/xuzhangbo/multihuman-animate/Gamma-World}
GAMMA_BIN=${GAMMA_BIN:-/data2/xuzhangbo/.conda/envs/gamma-world/bin}
SOLARIS=${SOLARIS_ROOT:-/data2/xuzhangbo/multihuman-animate/solaris}
SOLARIS_PY=${SOLARIS_PY:-/data2/xuzhangbo/.conda/envs/solaris/bin/python}
SOURCE="$RUN/assets/reveal"
N_CASES=75
FRAMES=129
IFS=',' read -r -a GPU_IDS <<< "${MWB3_INFERENCE_GPUS:-0,1,2,3,4,5,6,7}"
GPU_COUNT=${#GPU_IDS[@]}
(( GPU_COUNT >= 3 )) || { echo "at least three GPUs are required" >&2; exit 2; }
N_SHARDS=$GPU_COUNT
GAMMA_WORKERS=$((GPU_COUNT-2))
GAMMA_GPU_CSV=$(IFS=,; echo "${GPU_IDS[*]:0:GAMMA_WORKERS}")
MAX_SHARD=$(((N_CASES+N_SHARDS-1)/N_SHARDS))
EVAL_NAME=eval_mwb3_building_reveal_129

test -f "$RUN/.gt_capture_complete"
mkdir -p "$RUN/logs/gamma" "$RUN/logs/solaris" "$RUN/models/gamma/reveal" \
  "$RUN/models/solaris/raw" "$RUN/models/solaris/processed_shards" "$RUN/hydra/solaris"
exec >>"$RUN/logs/inference_manager.log" 2>&1
echo "[$(date -Is)] manager_start"

cd "$REPO"
python3 - "$SOURCE" "$RUN" "$N_SHARDS" "$N_CASES" <<'PY'
from pathlib import Path
import json, sys
source, run, shards, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ids=sorted(p.name for p in source.glob('MWB3-S-RV-S*-B*-A*') if p.is_dir())
if len(ids)!=expected: raise RuntimeError(f'expected {expected} cases, found {len(ids)}')
lists=run/'prepared/solaris_lists';lists.mkdir(parents=True,exist_ok=True)
for i in range(shards): (lists/f'shard_{i:02d}.txt').write_text('\n'.join(ids[i::shards])+'\n')
(run/'prepared/gamma_manifest.json').write_text(json.dumps({'schema_version':'mwb.v3.gamma_input.building_reveal.v1','self_contained':True,'frames':129,'cases':ids},indent=2)+'\n')
print({'cases':len(ids),'shards':[len(ids[i::shards]) for i in range(shards)]})
PY

rm -rf "$RUN/prepared/solaris"
for i in $(seq 0 $((N_SHARDS-1))); do
  shard=$(printf '%02d' "$i")
  python3 controller/multiworldbench_v3/tools/prepare_solaris_inputs.py \
    --source-root "$SOURCE" --case-list "$RUN/prepared/solaris_lists/shard_$shard.txt" \
    --output "$RUN/prepared/solaris/shard_$shard" --frames "$FRAMES" --video-prefix-frames 1 \
    >"$RUN/logs/solaris/prepare_$shard.log" 2>&1
done

# Runtime evaluation indices are V3-specific data, not model code.
python3 - "$SOLARIS/src/data/eval_ids/eval_ids_${EVAL_NAME}.json" "$MAX_SHARD" "$FRAMES" <<'PY'
from pathlib import Path
import json,sys
p,n,frames=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
p.write_text(json.dumps([[i,0,frames,0,frames] for i in range(n)],indent=2)+'\n')
print(p)
PY
echo "[$(date -Is)] inputs_ready"

start_solaris(){
  local i=$1 gpu=$2 shard raw samples
  shard=$(printf '%02d' "$i"); raw="$RUN/models/solaris/raw/shard_$shard"; mkdir -p "$raw"
  samples=$(grep -cve '^[[:space:]]*$' "$RUN/prepared/solaris_lists/shard_$shard.txt")
  test "$samples" -gt 0
  (
    cd "$SOLARIS"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      PYTHONPATH="$SOLARIS" "$SOLARIS_PY" src/inference.py \
      --config-dir "$RUN/prepared/solaris/shard_$shard" --config-name inference_mwb3 \
      experiment_name="mwb3_building_reveal_75_s$shard" \
      device.eval_data_dir="$RUN/prepared/solaris/shard_$shard/dataset" \
      device.output_dir="$raw" device.pretrained_model_dir="$SOLARIS/pretrained" \
      device.eval_num_samples="$samples" runner.num_frames_eval="$FRAMES" \
      runner.params.eval_dataloader_config.params.batch_size=1 \
      runner.params.eval_dataloader_config.params.num_workers=0 \
      enable_jax_cache=true device.jax_cache_dir=/data2/xuzhangbo/.cache/solaris-jax \
      shuffle_bots=false eval_datasets.eval_mwb3.name="$EVAL_NAME" \
      hydra.run.dir="$RUN/hydra/solaris/shard_$shard" \
      >"$RUN/logs/solaris/shard_$shard.log" 2>&1
  ) & LAST_PID=$!
  echo "[$(date -Is)] solaris_start shard=$shard gpu=$gpu pid=$LAST_PID"
}

(
  cd "$GAMMA"
  CUDA_VISIBLE_DEVICES="$GAMMA_GPU_CSV" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$GAMMA_BIN/torchrun" --standalone --nproc-per-node="$GAMMA_WORKERS" --master-port=29765 scripts/inference.py \
    --mode causal_few_step \
    --checkpoint /data2/xuzhangbo/models/Gamma-World/causal-few-step/model.safetensors \
    --vae /data2/xuzhangbo/models/Gamma-World/tokenizer.pth \
    --text-encoder /data2/xuzhangbo/multihuman_motion/Gamma-World/checkpoints/Cosmos-Reason1-7B \
    --eval-dir "$SOURCE" --output "$RUN/models/gamma/reveal" \
    --num-frames "$FRAMES" --n-players 2 --width 480 --height 320 \
    --guidance 5 --shift 5 --seed 1 --fps 20 \
    --cache-text-embeddings --text-encoder-offload release \
    >"$RUN/logs/gamma/inference.log" 2>&1
) & gamma_pid=$!
echo "[$(date -Is)] gamma_start pid=$gamma_pid gpus=$GAMMA_GPU_CSV"

declare -a solaris_pids
for i in $(seq "$GAMMA_WORKERS" $((N_SHARDS-1))); do
  start_solaris "$i" "${GPU_IDS[$i]}"; solaris_pids[$i]=$LAST_PID
 done

gamma_status=0; wait "$gamma_pid" || gamma_status=$?
gamma_count=$(find "$RUN/models/gamma/reveal" -type f -name generated.mp4 | wc -l)
echo "[$(date -Is)] gamma_end status=$gamma_status outputs=$gamma_count"
(( gamma_status == 0 && gamma_count == N_CASES )) || exit 20

for i in $(seq 0 $((GAMMA_WORKERS-1))); do start_solaris "$i" "${GPU_IDS[$i]}"; solaris_pids[$i]=$LAST_PID; done
failed=0
for i in $(seq 0 $((N_SHARDS-1))); do wait "${solaris_pids[$i]}" || failed=1; done
(( failed == 0 )) || exit 21

cd "$REPO"
rm -rf "$RUN/models/solaris/processed_shards" "$RUN/models/solaris/reveal"
mkdir -p "$RUN/models/solaris/processed_shards" "$RUN/models/solaris/reveal"
for i in $(seq 0 $((N_SHARDS-1))); do
  shard=$(printf '%02d' "$i")
  leaf=$(dirname "$(find "$RUN/models/solaris/raw/shard_$shard" -type f -name video_0_side_by_side.mp4 -print -quit)")
  test -n "$leaf"
  python3 controller/multiworldbench_v3/tools/postprocess_solaris.py \
    --input "$RUN/prepared/solaris/shard_$shard" --raw "$leaf" \
    --output "$RUN/models/solaris/processed_shards/shard_$shard" \
    >"$RUN/logs/solaris/postprocess_$shard.log" 2>&1
done
python3 - "$RUN" "$N_SHARDS" "$N_CASES" <<'PY'
from pathlib import Path
import json,os,shutil,sys
run,shards,expected=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]);out=run/'models/solaris/reveal';ids=[]
for i in range(shards):
  for src in sorted((run/f'models/solaris/processed_shards/shard_{i:02d}').glob('MWB3-S-RV-S*-B*-A*')):
    shutil.copytree(src,out/src.name,copy_function=os.link);ids.append(src.name)
if len(ids)!=expected or len(set(ids))!=expected: raise RuntimeError(f'invalid Solaris outputs: {len(ids)}/{len(set(ids))}')
(out/'manifest.json').write_text(json.dumps({'schema_version':'mwb.v3.solaris.building_reveal.v1','self_contained':True,'frames':129,'cases':sorted(ids)},indent=2)+'\n')
PY
touch "$RUN/.inference_complete"
echo "[$(date -Is)] manager_complete gamma=$gamma_count solaris=$N_CASES"
