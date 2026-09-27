#!/usr/bin/env bash
set -Eeuo pipefail

RUN=${MWB3_SHARED_STATE_RUN:?set MWB3_SHARED_STATE_RUN}
REPO=${MWB3_REPO:-/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine}
GAMMA=${GAMMA_ROOT:-/data2/xuzhangbo/multihuman-animate/Gamma-World}
GAMMA_BIN=${GAMMA_BIN:-/data2/xuzhangbo/.conda/envs/gamma-world/bin}
SOLARIS=${SOLARIS_ROOT:-/data2/xuzhangbo/multihuman-animate/solaris}
SOLARIS_PY=${SOLARIS_PY:-/data2/xuzhangbo/.conda/envs/solaris/bin/python}
SOURCE="$RUN/assets/interaction"
N_SHARDS=8
N_CASES=120

mkdir -p "$RUN/logs/gamma" "$RUN/logs/solaris" \
  "$RUN/models/gamma/interaction" "$RUN/models/solaris/raw" \
  "$RUN/models/solaris/processed_shards" "$RUN/hydra/solaris"
exec >>"$RUN/logs/inference_manager.log" 2>&1
echo "[$(date -Is)] manager_start"

cd "$REPO"
python3 - "$SOURCE" "$RUN" "$N_SHARDS" "$N_CASES" <<'PY'
from pathlib import Path
import json, sys
source, run, n_shards, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ids = sorted(p.name for p in source.glob("MWB3-S-SS*") if p.is_dir())
if len(ids) != expected:
    raise RuntimeError(f"expected {expected} self-contained cases, found {len(ids)}")
lists = run / "prepared/solaris_lists"
lists.mkdir(parents=True, exist_ok=True)
sizes = []
for i in range(n_shards):
    shard = ids[i::n_shards]
    sizes.append(len(shard))
    (lists / f"shard_{i:02d}.txt").write_text("\n".join(shard) + "\n", encoding="utf-8")
(run / "prepared/gamma_manifest.json").write_text(json.dumps({
    "schema_version": "mwb.v3.gamma_input.shared_state.v2",
    "self_contained": True,
    "source": str(source),
    "cases": ids,
}, indent=2) + "\n", encoding="utf-8")
print({"cases": len(ids), "solaris_shards": sizes})
PY

rm -rf "$RUN/prepared/solaris"
for i in $(seq 0 $((N_SHARDS-1))); do
  shard=$(printf '%02d' "$i")
  python3 controller/multiworldbench_v3/tools/prepare_solaris_inputs.py \
    --source-root "$SOURCE" \
    --case-list "$RUN/prepared/solaris_lists/shard_$shard.txt" \
    --output "$RUN/prepared/solaris/shard_$shard" --frames 81 --video-prefix-frames 1 \
    >"$RUN/logs/solaris/prepare_$shard.log" 2>&1
done

python3 - "$N_CASES" "$N_SHARDS" <<'PY'
from pathlib import Path
import json, sys
n, shards = int(sys.argv[1]), int(sys.argv[2])
size = (n + shards - 1) // shards
p = Path('/data2/xuzhangbo/multihuman-animate/solaris/src/data/eval_ids/eval_ids_eval_mwb3_shared_state_81.json')
p.write_text(json.dumps([[i, 0, 81, 0, 81] for i in range(size)], indent=2) + '\n')
PY
echo "[$(date -Is)] inputs_ready"

start_solaris() {
  local i=$1 gpu=$2 shard raw
  shard=$(printf '%02d' "$i")
  raw="$RUN/models/solaris/raw/shard_$shard"
  mkdir -p "$raw"
  (
    cd "$SOLARIS"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      PYTHONPATH="$SOLARIS" "$SOLARIS_PY" src/inference.py \
      --config-dir "$RUN/prepared/solaris/shard_$shard" --config-name inference_mwb3 \
      experiment_name="mwb3_shared_state_120_s$shard" \
      device.eval_data_dir="$RUN/prepared/solaris/shard_$shard/dataset" \
      device.output_dir="$raw" device.pretrained_model_dir="$SOLARIS/pretrained" \
      device.eval_num_samples=15 runner.num_frames_eval=81 \
      runner.params.eval_dataloader_config.params.batch_size=1 \
      runner.params.eval_dataloader_config.params.num_workers=0 \
      enable_jax_cache=true device.jax_cache_dir=/data2/xuzhangbo/.cache/solaris-jax \
      shuffle_bots=false eval_datasets.eval_mwb3.name=eval_mwb3_shared_state_81 \
      hydra.run.dir="$RUN/hydra/solaris/shard_$shard" \
      >"$RUN/logs/solaris/shard_$shard.log" 2>&1
  ) &
  LAST_PID=$!
  echo "[$(date -Is)] solaris_start shard=$shard gpu=$gpu pid=$LAST_PID"
}

# Gamma-World needs six ranks. Solaris begins on the remaining two GPUs.
(
  cd "$GAMMA"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "$GAMMA_BIN/torchrun" --standalone --nproc-per-node=6 --master-port=29764 scripts/inference.py \
    --mode causal_few_step \
    --checkpoint /data2/xuzhangbo/models/Gamma-World/causal-few-step/model.safetensors \
    --vae /data2/xuzhangbo/models/Gamma-World/tokenizer.pth \
    --text-encoder /data2/xuzhangbo/multihuman_motion/Gamma-World/checkpoints/Cosmos-Reason1-7B \
    --eval-dir "$SOURCE" --output "$RUN/models/gamma/interaction" \
    --num-frames 81 --n-players 2 --width 480 --height 320 \
    --guidance 5 --shift 5 --seed 1 --fps 20 \
    --cache-text-embeddings --text-encoder-offload release \
    >"$RUN/logs/gamma/inference.log" 2>&1
) & gamma_pid=$!
echo "[$(date -Is)] gamma_start pid=$gamma_pid gpus=0-5"

declare -a solaris_pids
start_solaris 6 6; solaris_pids[6]=$LAST_PID
start_solaris 7 7; solaris_pids[7]=$LAST_PID

gamma_status=0
wait "$gamma_pid" || gamma_status=$?
gamma_count=$(find "$RUN/models/gamma/interaction" -type f -name generated.mp4 | wc -l)
echo "[$(date -Is)] gamma_end status=$gamma_status outputs=$gamma_count"
(( gamma_status == 0 && gamma_count == N_CASES )) || exit 20

# Reuse the six released GPUs for the remaining Solaris shards.
for i in $(seq 0 5); do
  start_solaris "$i" "$i"; solaris_pids[$i]=$LAST_PID
done

failed=0
for i in $(seq 0 $((N_SHARDS-1))); do
  wait "${solaris_pids[$i]}" || failed=1
done
(( failed == 0 )) || exit 21

cd "$REPO"
rm -rf "$RUN/models/solaris/processed_shards" "$RUN/models/solaris/interaction"
mkdir -p "$RUN/models/solaris/processed_shards" "$RUN/models/solaris/interaction"
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
import json, os, shutil, sys
run, n_shards, expected = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out = run / 'models/solaris/interaction'
ids = []
for i in range(n_shards):
    for src in sorted((run / f'models/solaris/processed_shards/shard_{i:02d}').glob('MWB3-S-SS*')):
        shutil.copytree(src, out / src.name, copy_function=os.link)
        ids.append(src.name)
if len(ids) != expected or len(set(ids)) != expected:
    raise RuntimeError(f'invalid Solaris outputs: {len(ids)} / {len(set(ids))}')
(out / 'manifest.json').write_text(json.dumps({
    'schema_version': 'mwb.v3.solaris.shared_state.v2',
    'self_contained': True,
    'cases': sorted(ids),
}, indent=2) + '\n')
PY
touch "$RUN/.inference_complete"
echo "[$(date -Is)] manager_complete gamma=$gamma_count solaris=$N_CASES"
