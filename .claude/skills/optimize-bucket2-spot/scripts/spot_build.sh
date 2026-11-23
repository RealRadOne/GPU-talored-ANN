#!/bin/bash
# One out-of-core build (bucket2 -> prune_windowed -> optimize_chunked), no quality eval, optionally under the spot
# simulator.
#   spot_build.sh DATASET G OUTDIR [TRACE]
#     TRACE unset / ""  : spot OFF (plain multi-GPU build, = scale skill's mgpu_run.sh)
#     TRACE = none      : spot ON, no events (scheduler path, overhead check)
#     TRACE = <csv>     : spot ON, replay the trace with spot_sim.py (SPOT_SCALE, default 60; SPOT_NOTICE_REAL_S optional)
# Env: BIN / B2BIN (binary dirs, default this worktree's build/), GPU_LIST=a,b (force devices), BUCKET2_SPOT_MODE,
#      BUCKET2_ASSIGN_DETERMINISTIC (det check). Takes the shared bench lock, waits for G free GPUs (GPU_WAIT lines).
#      optimize_chunked always runs with --num-gpus 1 when spot is ON (user 2026-11-20: optimize has no spot / multi-GPU).
# Leaves OUTDIR/work in place (callers compare / delete it). Last line: "DONE ..." or "<stage> FAILED".
set -uo pipefail
DS=$1; G=$2; OUT=$3; TRACE=${4:-}
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$T/../../../.." && pwd)
S=$ROOT/.claude/skills/optimize-bucket2/scripts
BIN=${BIN:-$ROOT/bucketDemo/buildBucket/build}
B2BIN=${B2BIN:-$BIN}
PY=/home/lanlu/miniconda3/envs/cuvs-bench/bin/python
source $S/params.sh
eval "$($PY -I $S/dataset_env.py $DS)"
source $S/gpu_select.sh
BUCKET2_EXTRA="$BUCKET2_EXTRA --kmeans-train-fraction $KMEANS_TRAIN_FRACTION --cache-mb $CACHE_MB --bucket-mem-mb $BUCKET_MEM_MB --gpu-bucket-cache-mb $GPU_BUCKET_CACHE_MB"
(( ${ARRIVAL_MEM_MB:-0} > 0 )) && BUCKET2_EXTRA="$BUCKET2_EXTRA --arrival-mem-mb $ARRIVAL_MEM_MB"
(( ${BUCKET2_HOST_BUDGET_MB:-0} > 0 )) && BUCKET2_EXTRA="$BUCKET2_EXTRA --host-budget-mb $BUCKET2_HOST_BUDGET_MB"
CHUNKED_EXTRA="$CHUNKED_EXTRA --host-budget-mb $CHUNKED_HOST_BUDGET_MB"
export BUCKET2_A1_PASSES=$A1_PASSES BUCKET2_A7_UNWIN=$A7_UNWIN
TAG="k${KNN_K}p${NPROBE}m${NEIGHBORS_M}t${ITERATIONS}"

exec 9>>/home/lanlu/.bench_locks/bucket2_bench.lock
flock 9
echo "$$ $(date +%F_%T) spot_build $DS G=$G trace=${TRACE:-off} -> $OUT" > /home/lanlu/.bench_locks/bucket2_bench.lock.owner

if [[ -z ${GPU_LIST:-} ]]; then   # G free GPUs in $GPU_ORDER; for G=2 a same-NUMA pair first (2,3 then 0,1)
  W=0
  while :; do
    free=(); for g in $GPU_ORDER; do gpu_is_free "$g" && free+=("$g"); done
    fs=" ${free[*]} "
    if (( G == 2 )); then
      for pr in "2 3" "0 1"; do read -r ga gb <<< "$pr"; [[ $fs == *" $ga "* && $fs == *" $gb "* ]] && { GPU_LIST="$ga,$gb"; break; }; done
    fi
    [[ -z ${GPU_LIST:-} ]] && (( ${#free[@]} >= G )) && GPU_LIST=$(IFS=,; echo "${free[*]:0:G}")
    [[ -n ${GPU_LIST:-} ]] && break
    (( W % 600 == 0 )) && echo "GPU_WAIT $(date +%F_%T): need $G free GPU(s), free: ${free[*]:-none}"
    sleep 60; W=$((W + 60))
  done
fi
export CUDA_VISIBLE_DEVICES=$GPU_LIST
mkdir -p $OUT/work; echo "$GPU_LIST" > $OUT/gpus.txt
timed() { local n=$1; shift; /usr/bin/time -f "%e %M" -o $OUT/$n.time stdbuf -oL -eL "$@" > $OUT/$n.log 2>&1; }

SIM_PID=""; SPOT_DIR=""
stop_sim() {
  [[ -n $SIM_PID ]] && { kill $SIM_PID 2>/dev/null; wait $SIM_PID 2>/dev/null; }
  [[ -n $SPOT_DIR ]] && rm -rf $SPOT_DIR
  SIM_PID=""; SPOT_DIR=""; unset BUCKET2_SPOT_DIR
}
trap stop_sim EXIT
OC_G=$G
if [[ -n $TRACE ]]; then
  SPOT_DIR=/dev/shm/spot_sim/build_$$; rm -rf $SPOT_DIR; mkdir -p $SPOT_DIR
  tr=""; [[ $TRACE != none ]] && tr=$TRACE
  na=(); [[ -n ${SPOT_NOTICE_REAL_S:-} ]] && na=(--notice-real-s $SPOT_NOTICE_REAL_S)
  $PY -I $T/spot_sim.py run --trace "$tr" --dir $SPOT_DIR --n-inst $G --scale ${SPOT_SCALE:-60} "${na[@]}" \
    --log $OUT/spot_events.log > /dev/null 2>&1 9>&- &
  SIM_PID=$!
  export BUCKET2_SPOT_DIR=$SPOT_DIR
  OC_G=1
  echo "trace=$TRACE scale=${SPOT_SCALE:-60} notice_real_s=${SPOT_NOTICE_REAL_S:-scaled} mode=${BUCKET2_SPOT_MODE:-default}" > $OUT/spot.txt
fi
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

timed bucket2 $B2BIN/bucket2 -i $BASE -o $OUT/work --centroid-ratio $CENTROID_RATIO --seed $SEED \
  --knn-k $KNN_K --nprobe $NPROBE --neighbors-m $NEIGHBORS_M -t $ITERATIONS \
  --fused-optimize-degree $OUTPUT_DEGREE --num-gpus $G $BUCKET2_EXTRA || { echo "bucket2 FAILED"; exit 3; }
O=$OUT/work/$TAG
if [[ -f $O/final_graph.bin ]]; then
  echo "DONE $DS G=$G gpus=$GPU_LIST bucket2=$(cut -d' ' -f1 $OUT/bucket2.time) (resident: no spot stages)"; exit 0
fi
timed prune $BIN/prune_windowed -k $O/vector_knn.bin -i $O/data_reordered$EXT \
  --bucket-offsets $O/bucket_offsets.bin --bucket-process-order $O/bucket_process_order.bin \
  -c $O/centroid_knn.bin -o $O/forward_graph_reordered.bin \
  --output-degree $OUTPUT_DEGREE --cache-mb $PRUNE_CACHE_MB --num-gpus $G $PRUNE_EXTRA || { echo "prune FAILED"; exit 3; }
stop_sim
timed optimize_chunked $BIN/optimize_chunked -g $O/forward_graph_reordered.bin \
  -o $O/graph_new_ids.bin -m E --bucket-offsets $O/bucket_offsets.bin \
  --gpu-budget-mb $CHUNKED_GPU_BUDGET_MB --edge-buffer-mb $CHUNKED_EDGE_BUFFER_MB \
  --tmp-dir $OUT/work/tmp --inverse-perm $O/inverse_perm.bin --final-output $O/final_graph.bin \
  --num-gpus $OC_G $CHUNKED_EXTRA || { echo "optimize_chunked FAILED"; exit 3; }
ev=""; [[ -f $OUT/spot_events.log ]] && ev="events: $(grep -c ' preempt ' $OUT/spot_events.log) preempt / $(grep -c ' notice ' $OUT/spot_events.log) notice / $(grep -c ' join ' $OUT/spot_events.log) join"
echo "DONE $DS G=$G gpus=$GPU_LIST trace=${TRACE:-off} $(for n in bucket2 prune optimize_chunked; do echo -n "$n=$(cut -d' ' -f1 $OUT/$n.time) "; done)$ev"
