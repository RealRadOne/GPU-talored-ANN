#!/bin/bash
# One out-of-core build (bucket2 -> prune_windowed -> optimize_chunked) with --num-gpus G, no quality eval.
#   mgpu_run.sh DATASET G OUTDIR
# Same flags as run_pipeline.sh (auto mode). Uses G free GPUs (GPU_LIST=a,b,... to force), waits until they
# are free, takes the bench lock. B2BIN=<dir> overrides the bucket2 binary dir (A/B against HEAD in build_wip/).
# Leaves OUTDIR/work in place (callers compare / delete it).
set -uo pipefail
DS=$1; G=$2; OUT=$3
S=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2/scripts
BIN=/home/lanlu/microBenchmark/bucketDemo/buildBucket/build
B2BIN=${B2BIN:-$BIN}
PY=/home/lanlu/miniconda3/envs/cuvs-bench/bin/python
source $S/params.sh
eval "$($PY -I $S/dataset_env.py $DS)"
source $S/gpu_select.sh
BUCKET2_EXTRA="$BUCKET2_EXTRA --kmeans-train-fraction $KMEANS_TRAIN_FRACTION --cache-mb $CACHE_MB --bucket-mem-mb $BUCKET_MEM_MB --gpu-bucket-cache-mb $GPU_BUCKET_CACHE_MB"
CHUNKED_EXTRA="$CHUNKED_EXTRA --host-budget-mb $CHUNKED_HOST_BUDGET_MB"
export BUCKET2_A1_PASSES=$A1_PASSES BUCKET2_A7_UNWIN=$A7_UNWIN
TAG="k${KNN_K}p${NPROBE}m${NEIGHBORS_M}t${ITERATIONS}"

exec 9>>/home/lanlu/.bench_locks/bucket2_bench.lock
flock 9
echo "$$ $(date +%F_%T) mgpu_run $DS G=$G -> $OUT" > /home/lanlu/.bench_locks/bucket2_bench.lock.owner

if [[ -z ${GPU_LIST:-} ]]; then   # G free GPUs; for G=2 a same-NUMA pair first (0,1 / 2,3)
  W=0
  while :; do
    free=(); for g in 0 1 2 3; do gpu_is_free "$g" && free+=("$g"); done
    fs=" ${free[*]} "
    if (( G == 2 )); then
      for pr in "0 1" "2 3"; do read -r ga gb <<< "$pr"; [[ $fs == *" $ga "* && $fs == *" $gb "* ]] && { GPU_LIST="$ga,$gb"; break; }; done
    fi
    [[ -z ${GPU_LIST:-} ]] && (( ${#free[@]} >= G )) && GPU_LIST=$(IFS=,; echo "${free[*]:0:G}")
    [[ -n ${GPU_LIST:-} ]] && break
    (( W % 600 == 0 )) && echo "GPU_WAIT $(date +%F_%T): need $G free GPU(s), free: ${free[*]:-none}"
    sleep 60; W=$((W + 60))
  done
fi
export CUDA_VISIBLE_DEVICES=$GPU_LIST
mkdir -p $OUT/work; echo "$GPU_LIST" > $OUT/gpus.txt
timed() { local n=$1; shift; /usr/bin/time -f "%e %M" -o $OUT/$n.time "$@" > $OUT/$n.log 2>&1; }
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

timed bucket2 $B2BIN/bucket2 -i $BASE -o $OUT/work --centroid-ratio $CENTROID_RATIO --seed $SEED \
  --knn-k $KNN_K --nprobe $NPROBE --neighbors-m $NEIGHBORS_M -t $ITERATIONS \
  --fused-optimize-degree $OUTPUT_DEGREE --num-gpus $G $BUCKET2_EXTRA || { echo "bucket2 FAILED"; exit 3; }
O=$OUT/work/$TAG
if [[ -f $O/final_graph.bin ]]; then   # resident path finished inside bucket2 (small datasets)
  echo "DONE $DS G=$G gpus=$GPU_LIST bucket2=$(cut -d' ' -f1 $OUT/bucket2.time) (resident)"; exit 0
fi
timed prune $BIN/prune_windowed -k $O/vector_knn.bin -i $O/data_reordered$EXT \
  --bucket-offsets $O/bucket_offsets.bin --bucket-process-order $O/bucket_process_order.bin \
  -c $O/centroid_knn.bin -o $O/forward_graph_reordered.bin \
  --output-degree $OUTPUT_DEGREE --cache-mb $PRUNE_CACHE_MB --num-gpus $G $PRUNE_EXTRA || { echo "prune FAILED"; exit 3; }
timed optimize_chunked $BIN/optimize_chunked -g $O/forward_graph_reordered.bin \
  -o $O/graph_new_ids.bin -m E --bucket-offsets $O/bucket_offsets.bin \
  --gpu-budget-mb $CHUNKED_GPU_BUDGET_MB --edge-buffer-mb $CHUNKED_EDGE_BUFFER_MB \
  --tmp-dir $OUT/work/tmp --inverse-perm $O/inverse_perm.bin --final-output $O/final_graph.bin \
  --num-gpus $G $CHUNKED_EXTRA || { echo "optimize_chunked FAILED"; exit 3; }
echo "DONE $DS G=$G gpus=$GPU_LIST $(for n in bucket2 prune optimize_chunked; do echo -n "$n=$(cut -d' ' -f1 $OUT/$n.time) "; done)"
