#!/usr/bin/env bash
set -Eeuo pipefail
RUN=${MWB3_RACE_RUN:?set MWB3_RACE_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
GAMMA=${GAMMA_ROOT:-/data2/xuzhangbo/multihuman-animate/Gamma-World}
GAMMA_BIN=${GAMMA_BIN:-/data2/xuzhangbo/.conda/envs/gamma-world/bin}
SOLARIS=${SOLARIS_ROOT:-/data2/xuzhangbo/multihuman-animate/solaris}
SOLARIS_PY=${SOLARIS_PY:-/data2/xuzhangbo/.conda/envs/solaris/bin/python}
SOURCE="$RUN/assets/race"
N_CASES=30
N_SHARDS=8
SOLARIS_FRAMES=101
GAMMA_FRAMES=129
EVAL_NAME=eval_mwb3_race_condition_101

mkdir -p "$RUN/logs/gamma" "$RUN/logs/solaris" "$RUN/models/gamma/race" "$RUN/models/solaris/raw" "$RUN/hydra/solaris"
exec >>"$RUN/logs/inference_manager.log" 2>&1
echo "[$(date -Is)] manager_start gt_collected=false"
cd "$REPO"
python3 - "$SOURCE" "$RUN" "$N_SHARDS" "$N_CASES" <<'PY'
from pathlib import Path
import json,sys
source,run,n_shards,expected=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
ids=sorted(p.name for p in source.glob('MWB3-SW-RC*') if p.is_dir())
if len(ids)!=expected: raise RuntimeError(f'expected {expected} cases, found {len(ids)}')
lists=run/'prepared/solaris_lists';lists.mkdir(parents=True,exist_ok=True)
for i in range(n_shards): (lists/f'shard_{i:02d}.txt').write_text('\n'.join(ids[i::n_shards])+'\n')
(run/'prepared/gamma_manifest.json').write_text(json.dumps({'schema_version':'mwb.v3.gamma_input.race.v1','gt_collected':False,'frames':129,'cases':ids},indent=2)+'\n')
print({'cases':len(ids),'shards':[len(ids[i::n_shards]) for i in range(n_shards)]})
PY

rm -rf "$RUN/prepared/solaris"
for i in $(seq 0 $((N_SHARDS-1))); do
  shard=$(printf '%02d' "$i")
  python3 controller/multiworldbench_v3/tools/prepare_solaris_inputs.py \
    --source-root "$SOURCE" --case-list "$RUN/prepared/solaris_lists/shard_$shard.txt" \
    --output "$RUN/prepared/solaris/shard_$shard" --frames "$SOLARIS_FRAMES" --video-prefix-frames 1 \
    >"$RUN/logs/solaris/prepare_$shard.log" 2>&1
done
python3 - "$SOLARIS/src/data/eval_ids/eval_ids_${EVAL_NAME}.json" <<'PY'
from pathlib import Path
import json,sys
Path(sys.argv[1]).write_text(json.dumps([[i,0,101,0,101] for i in range(4)],indent=2)+'\n')
PY
echo "[$(date -Is)] inputs_ready"

start_solaris(){
  local i=$1 gpu=$2 shard raw samples
  shard=$(printf '%02d' "$i");raw="$RUN/models/solaris/raw/shard_$shard";rm -rf "$raw" "$RUN/hydra/solaris/shard_$shard";mkdir -p "$raw"
  samples=$(grep -cve '^[[:space:]]*$' "$RUN/prepared/solaris_lists/shard_$shard.txt")
  (
    cd "$SOLARIS"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="$SOLARIS" \
      "$SOLARIS_PY" src/inference.py --config-dir "$RUN/prepared/solaris/shard_$shard" --config-name inference_mwb3 \
      experiment_name="mwb3_race_condition_30_s$shard" device.eval_data_dir="$RUN/prepared/solaris/shard_$shard/dataset" \
      device.output_dir="$raw" device.pretrained_model_dir="$SOLARIS/pretrained" device.eval_num_samples="$samples" \
      runner.num_frames_eval="$SOLARIS_FRAMES" runner.params.eval_dataloader_config.params.batch_size=1 \
      runner.params.eval_dataloader_config.params.num_workers=0 enable_jax_cache=true \
      device.jax_cache_dir=/data2/xuzhangbo/.cache/solaris-jax shuffle_bots=false \
      eval_datasets.eval_mwb3.name="$EVAL_NAME" hydra.run.dir="$RUN/hydra/solaris/shard_$shard" \
      >"$RUN/logs/solaris/shard_$shard.log" 2>&1
  ) & LAST_PID=$!
  echo "[$(date -Is)] solaris_start shard=$shard gpu=$gpu pid=$LAST_PID cases=$samples"
}

rm -rf "$RUN/models/gamma/race";mkdir -p "$RUN/models/gamma/race"
(
  cd "$GAMMA"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$GAMMA_BIN/torchrun" --standalone --nproc-per-node=6 --master-port=29769 scripts/inference.py \
    --mode causal_few_step --checkpoint /data2/xuzhangbo/models/Gamma-World/causal-few-step/model.safetensors \
    --vae /data2/xuzhangbo/models/Gamma-World/tokenizer.pth \
    --text-encoder /data2/xuzhangbo/multihuman_motion/Gamma-World/checkpoints/Cosmos-Reason1-7B \
    --eval-dir "$SOURCE" --output "$RUN/models/gamma/race" --num-frames "$GAMMA_FRAMES" \
    --n-players 2 --width 480 --height 320 --guidance 5 --shift 5 --seed 1 --fps 20 \
    --cache-text-embeddings --text-encoder-offload release >"$RUN/logs/gamma/inference.log" 2>&1
) & gamma_pid=$!
echo "[$(date -Is)] gamma_start pid=$gamma_pid gpus=0-5"

declare -a solaris_pids
start_solaris 6 6;solaris_pids[6]=$LAST_PID
start_solaris 7 7;solaris_pids[7]=$LAST_PID

gamma_status=0;wait "$gamma_pid" || gamma_status=$?
gamma_count=$(find "$RUN/models/gamma/race" -type f -name generated.mp4 | wc -l)
echo "[$(date -Is)] gamma_end status=$gamma_status outputs=$gamma_count/$N_CASES"
(( gamma_status==0 && gamma_count==N_CASES )) || exit 20
for i in $(seq 0 5); do start_solaris "$i" "$i";solaris_pids[$i]=$LAST_PID;done
failed=0
for i in $(seq 0 $((N_SHARDS-1))); do wait "${solaris_pids[$i]}" || failed=1;done
(( failed==0 )) || exit 21

cd "$REPO"
rm -rf "$RUN/models/solaris/processed_shards" "$RUN/models/solaris/race"
mkdir -p "$RUN/models/solaris/processed_shards" "$RUN/models/solaris/race"
for i in $(seq 0 $((N_SHARDS-1))); do
  shard=$(printf '%02d' "$i")
  leaf=$(dirname "$(find "$RUN/models/solaris/raw/shard_$shard" -type f -name video_0_side_by_side.mp4 -print -quit)")
  test -n "$leaf"
  python3 controller/multiworldbench_v3/tools/postprocess_solaris.py --input "$RUN/prepared/solaris/shard_$shard" \
    --raw "$leaf" --output "$RUN/models/solaris/processed_shards/shard_$shard" >"$RUN/logs/solaris/postprocess_$shard.log" 2>&1
done
python3 - "$RUN" "$N_SHARDS" "$N_CASES" <<'PY'
from pathlib import Path
import json,os,shutil,sys
run,n_shards,expected=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]);out=run/'models/solaris/race';ids=[]
for i in range(n_shards):
  for src in sorted((run/f'models/solaris/processed_shards/shard_{i:02d}').glob('MWB3-SW-RC*')):
    dst=out/src.name
    if dst.exists(): raise RuntimeError(f'duplicate {src.name}')
    shutil.copytree(src,dst,copy_function=os.link);ids.append(src.name)
if len(ids)!=expected or len(set(ids))!=expected: raise RuntimeError(f'invalid Solaris outputs: {len(ids)}/{len(set(ids))}')
(out/'manifest.json').write_text(json.dumps({'schema_version':'mwb.v3.solaris.race.v1','gt_collected':False,'frames':101,'cases':sorted(ids)},indent=2)+'\n')
PY
touch "$RUN/.inference_complete"
echo "[$(date -Is)] manager_complete gamma=$gamma_count solaris=$N_CASES gt_collected=false"
