# GPU selection for timed runs (user decision 2026-11-01), sourced by run_pipeline.sh / resume_large_tail.sh.
# select_gpu keeps $GPU if it is free, else takes the first free GPU in $GPU_ORDER; if none is free it
# waits, printing GPU_WAIT lines (the loop's Monitor turns them into a notification to the user).
# Free = no compute process on it and < GPU_FREE_MIB used. Order: same NUMA node as GPU 0 first
# (GPU 0, 1 = node 0; GPU 2, 3 = node 1), timings across nodes are not comparable (judge.py).
: "${GPU_ORDER:=0 1 2 3}"
: "${GPU_FREE_MIB:=1000}"
GPU_WAIT_MARK=/home/lanlu/.bench_locks/GPU_WAIT

gpu_is_free() {
  local used procs
  used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -n $used && $used -lt $GPU_FREE_MIB ]] || return 1
  procs=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  [[ -z $procs ]]
}

select_gpu() {
  local g waited=0 first=1
  while :; do
    if [[ -n ${GPU:-} ]] && gpu_is_free "$GPU"; then break; fi
    for g in $GPU_ORDER; do
      if gpu_is_free "$g"; then
        [[ -n ${GPU:-} ]] && echo "[gpu] GPU $GPU is taken, moving to GPU $g"
        GPU=$g; break 2
      fi
    done
    if (( first || waited >= 600 )); then
      echo "GPU_WAIT $(date +%F_%T): no free GPU (order: $GPU_ORDER), waiting. $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
      date +%F_%T > "$GPU_WAIT_MARK" 2>/dev/null
      first=0; waited=0
    fi
    sleep 60; waited=$((waited + 60))
  done
  if [[ -e $GPU_WAIT_MARK ]]; then rm -f "$GPU_WAIT_MARK"; echo "GPU_WAIT over $(date +%F_%T): using GPU $GPU"; fi
  export CUDA_VISIBLE_DEVICES=$GPU
}
