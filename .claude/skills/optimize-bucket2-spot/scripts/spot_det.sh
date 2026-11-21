#!/bin/bash
# Spot correctness check (hard gate): deterministic build (BUCKET2_ASSIGN_DETERMINISTIC=1) with G GPUs, once without
# spot and once under the spot simulator replaying TRACE; every artifact must be byte-identical (Step 4 top-2, Step 6
# buckets and prune windows are deterministic per unit, so preemption / resume / stealing must not change any byte).
#   spot_det.sh DATASET G TRACE [WORKROOT]     (DATASET sift10M recommended; TRACE e.g. traces/d1_g4.csv)
# Env: SPOT_SCALE (default 60), SPOT_NOTICE_REAL_S, BUCKET2_SPOT_MODE pass through.
# Prints one line per artifact, the spot event / scheduler summary, and a final "EQUAL" or "MISMATCH" line
# (exit 0 = EQUAL). Also fails if the trace caused no preemption during the spot stages (the check would be vacuous).
# Deletes its work dirs.
set -uo pipefail
DS=$1; G=$2; TRACE=$3
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$T/../../../.." && pwd)
W=${4:-$ROOT/bucketDemo/buildBucket/runs/spot_det_$$}
export BUCKET2_ASSIGN_DETERMINISTIC=1 BUCKET2_STEP4_CHECKSUM=1
bash $T/spot_build.sh $DS $G $W/plain | tee >(grep --line-buffered GPU_WAIT >&2) | tail -1
[[ -f $W/plain/optimize_chunked.time ]] || { echo "MISMATCH (plain build failed)"; rm -rf $W; exit 1; }
# same devices for the spot run (Step 4 batch sizes follow free GPU memory)
GPU_LIST=$(cat $W/plain/gpus.txt) bash $T/spot_build.sh $DS $G $W/spot $TRACE | tee >(grep --line-buffered GPU_WAIT >&2) | tail -1
[[ -f $W/spot/optimize_chunked.time ]] || { echo "MISMATCH (spot build failed)"; tail -20 $W/spot/*.log; rm -rf $W; exit 1; }
a=$(ls -d $W/plain/work/k*/ | head -1); b=$(ls -d $W/spot/work/k*/ | head -1)
ok=1
for f in inverse_perm.bin bucket_offsets.bin vector_knn.bin forward_graph_reordered.bin final_graph.bin; do
  if [[ ! -f $a/$f && ! -f $b/$f ]]; then continue; fi
  if cmp -s $a/$f $b/$f; then echo "$f same"; else echo "$f DIFF"; ok=0; fi
done
echo "step4 checksums plain: $(grep -o 'checksum [0-9a-f]*' $W/plain/bucket2.log | awk '{print $2}' | tr '\n' ' ') spot: $(grep -o 'checksum [0-9a-f]*' $W/spot/bucket2.log | awk '{print $2}' | tr '\n' ' ')"
grep -h '^\[spot\] stage=' $W/spot/bucket2.log $W/spot/prune.log 2>/dev/null
pre=$(grep -h '^\[spot\] stage=' $W/spot/bucket2.log $W/spot/prune.log 2>/dev/null | grep -o 'preempts=[0-9]*' | cut -d= -f2 | awk '{s+=$1} END {print s+0}')
echo "events: $(grep -c ' preempt ' $W/spot/spot_events.log 2>/dev/null) preempt / $(grep -c ' join ' $W/spot/spot_events.log 2>/dev/null) join; scheduler preempts in spot stages: ${pre:-0}"
for x in plain spot; do echo "$x times: $(for n in bucket2 prune optimize_chunked; do [[ -f $W/$x/$n.time ]] && echo -n "$n=$(head -1 $W/$x/$n.time | cut -d' ' -f1) "; done) gpus=$(cat $W/$x/gpus.txt)"; done
rm -rf $W
if (( ok )) && (( ${pre:-0} == 0 )); then echo "MISMATCH-VACUOUS ($DS, G=$G: no preemption hit a spot stage; use a denser trace / larger SPOT_SCALE)"; exit 2; fi
if (( ok )); then echo "EQUAL ($DS, G=$G, spot $TRACE vs plain)"; exit 0; else echo "MISMATCH ($DS, G=$G, spot $TRACE vs plain)"; exit 1; fi
