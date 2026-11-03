#!/usr/bin/env bash
# Build Lan's bucket2 (with the task-trace patch) under your own account and record Step 6 task traces.
# No access to /home/lanlu needed. Steps skip themselves when already done, so re-running is safe.
#
# Usage (run from this buildBucket folder, SOCKS tunnel up for steps 1-3):
#   ./setup_bucket2.sh sift10M          env + build + data + both trace runs
#   ./setup_bucket2.sh sift100M
#   ./setup_bucket2.sh msTuring100M
#   GPU=1 ./setup_bucket2.sh sift10M    (default GPU 0)
#   KEEP=1 ...                          keep bucket2's intermediate files (default: delete after each run)
#   RUNS="normal" ...                   only the normal run (default: "normal iso")
#
# Outputs, per dataset:  ~/traces/<ds>_normal.csv + .log   (all streams, as Lan runs it)
#                        ~/traces/<ds>_iso.csv + .log      (BUCKET2_SLOTS=1: one bucket at a time on the GPU)

DS=$1
GPU=${GPU:-0}
KEEP=${KEEP:-0}
RUNS=${RUNS:-"normal iso"}
CONDA_DIR=$HOME/miniforge3
ENV_NAME=bucket2-cuvs
DATA_DIR=$HOME/data
WORK_DIR=$HOME/bucket2_work
TRACE_DIR=$HOME/traces
BB=$(cd "$(dirname "$0")" && pwd)
JOBS=${JOBS:-16}

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() {
  log "FAILED: $*"
  exit 1
}
run() {
  log "RUN: $*"
  "$@" || fail "exit $? from: $*"
}

# ---- dataset table: source, dims, and bucket2 parameters (same rules as Lan's dataset_env.py) ----
SIFT_URL=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
TURING_URL=https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin
case $DS in
  sift10M)      URL=$SIFT_URL   N=10000000  DIM=128 ITEM=1 EXT=u8bin RATIO=0.00195312 NPROBE=40 ITERS=2 CACHE=15000 BMEM=16000 ;;
  sift100M)     URL=$SIFT_URL   N=100000000 DIM=128 ITEM=1 EXT=u8bin RATIO=0.00195312 NPROBE=40 ITERS=2 CACHE=15000 BMEM=16000 ;;
  msTuring10M)  URL=$TURING_URL N=10000000  DIM=100 ITEM=4 EXT=fbin  RATIO=0.001953125 NPROBE=64 ITERS=4 CACHE=15000 BMEM=16000 ;;
  msTuring100M) URL=$TURING_URL N=100000000 DIM=100 ITEM=4 EXT=fbin  RATIO=0.00195312 NPROBE=48 ITERS=2 CACHE=45000 BMEM=46000 ;;
  *) fail "usage: ./setup_bucket2.sh sift10M|sift100M|msTuring10M|msTuring100M" ;;
esac
BASE=$DATA_DIR/$DS/base.$EXT
DATA_GB=$(( N * DIM * ITEM / 1000000000 ))

log "==== $DS: $N x $DIM, ${DATA_GB} GB of vectors, GPU $GPU, runs: $RUNS"
log "Machine: $(hostname -s), user $USER, proxy: ${https_proxy:-none}"
log "Free disk in home: $(df -h --output=avail "$HOME" | tail -1)"

# ---- Step 1/4: conda env with cuVS 26.10, nvcc, gcc 14, Boost (what Lan builds with) ----
log "==== Step 1/4: conda env $ENV_NAME"
[ -x "$CONDA_DIR/bin/conda" ] || fail "no conda at $CONDA_DIR (setup_ann.sh installs it)"
source "$CONDA_DIR/etc/profile.d/conda.sh" || fail "could not load conda"
if conda env list | grep -q "^$ENV_NAME "; then
  log "Env exists, skipping"
else
  run conda create -y -n "$ENV_NAME" -c rapidsai -c conda-forge \
    python=3.11 "libcuvs=26.10" "cuda-version=12.9" cuda-nvcc cuda-cudart-dev \
    libcublas-dev libcurand-dev libcusolver-dev libcusparse-dev \
    "cmake>=4" make "gxx_linux-64=14" libboost-devel
fi
run conda activate "$ENV_NAME"
log "nvcc: $(nvcc --version | tail -1)"

# ---- Step 2/4: build bucket2 against this env ----
log "==== Step 2/4: build bucket2"
if [ -x "$BB/build/bucket2" ] && [ "$BB/build/bucket2" -nt "$BB/src/bucket.cu" ]; then
  log "bucket2 is up to date, skipping"
else
  run cmake -S "$BB" -B "$BB/build" -DCUVS_ENV="$CONDA_PREFIX" -DTOOL_ENV="$CONDA_PREFIX" \
    -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  run make -C "$BB/build" -j"$JOBS" bucket2
fi
grep -q BUCKET2_TASK_TRACE "$BB/src/bucket.cu" || fail "src/bucket.cu has no task-trace patch"

# ---- Step 3/4: first N vectors of the 1B file, downloaded in 1 GB pieces (resumable) ----
log "==== Step 3/4: dataset $BASE"
WANT=$(( 8 + N * DIM * ITEM ))
if [ -f "$BASE" ] && [ "$(stat -c %s "$BASE")" = "$WANT" ]; then
  log "Already downloaded, skipping"
else
  run mkdir -p "$DATA_DIR/$DS/parts"
  run curl -fsSI --max-time 30 -o /dev/null "$URL"
  BYTES=$(( N * DIM * ITEM )) PIECE=1000000000 i=0
  for (( off = 0; off < BYTES; off += PIECE, i++ )); do
    end=$(( off + PIECE < BYTES ? off + PIECE : BYTES ))
    part=$DATA_DIR/$DS/parts/$(printf %04d $i)
    [ -f "$part" ] && [ "$(stat -c %s "$part")" = $(( end - off )) ] && continue
    log "piece $i: bytes $off-$(( end - 1 )) of $BYTES"
    run curl -fL --retry 5 --retry-delay 10 -r $(( 8 + off ))-$(( 8 + end - 1 )) -o "$part" "$URL"
  done
  log "Assembling $BASE"
  python -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<ii', $N, $DIM))" > "$BASE.tmp" || fail "header"
  for part in "$DATA_DIR/$DS"/parts/*; do
    cat "$part" >> "$BASE.tmp" || fail "append $part"
    rm "$part"
  done
  [ "$(stat -c %s "$BASE.tmp")" = "$WANT" ] || fail "size mismatch in $BASE.tmp"
  run mv "$BASE.tmp" "$BASE"
  rmdir "$DATA_DIR/$DS/parts"
fi

# ---- Step 4/4: trace runs ----
log "==== Step 4/4: bucket2 trace runs"
run mkdir -p "$TRACE_DIR" "$WORK_DIR"
log "Free disk for bucket2 temp files: $(df -h --output=avail "$WORK_DIR" | tail -1) (100M runs need ~2-3x the vector size)"
export CUDA_VISIBLE_DEVICES=$GPU
for mode in $RUNS; do
  W=$WORK_DIR/${DS}_$mode
  CSV=$TRACE_DIR/${DS}_$mode.csv
  LOG=$TRACE_DIR/${DS}_$mode.log
  rm -rf "$W" "$CSV"
  mkdir -p "$W"
  SLOTS=""
  [ "$mode" = iso ] && SLOTS=1
  log "RUN ($mode): bucket2 on $DS -> $CSV"
  t0=$(date +%s)
  env BUCKET2_TASK_TRACE="$CSV" ${SLOTS:+BUCKET2_SLOTS=$SLOTS} \
    "$BB/build/bucket2" -i "$BASE" -o "$W" --centroid-ratio "$RATIO" --seed 42 \
    --knn-k 32 --nprobe "$NPROBE" --neighbors-m 64 -t "$ITERS" --out-of-core \
    --skip-npy --skip-dists --kmeans-train-fraction 0.1 \
    --cache-mb "$CACHE" --bucket-mem-mb "$BMEM" --gpu-bucket-cache-mb 8192 2>&1 | tee "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || fail "bucket2 ($mode) failed, see $LOG"
  log "($mode) done in $(( $(date +%s) - t0 )) s; $(( $(wc -l < "$CSV") - 1 )) task rows"
  grep -E "\[TaskTrace\]|bucket loop:" "$LOG"
  [ "$KEEP" = 1 ] || rm -rf "$W"
done
log "==== Done. Send back: $TRACE_DIR/${DS}_*.csv and .log (gzip the 100M ones)"
