#!/bin/bash
# Multi-GPU equivalence check: deterministic build (BUCKET2_ASSIGN_DETERMINISTIC=1: sequential Step 4d) with
# 1 GPU and with G GPUs on the same dataset; every artifact must be byte-identical.
#   det_check.sh DATASET G [WORKROOT]      (DATASET: sift10M recommended, ~3 min for both builds)
# Prints one line per artifact and a final "EQUAL" or "MISMATCH" line; exit 0 = EQUAL. Deletes its work dirs.
# Only meaningful on free GPUs (Step 4 batch size follows free GPU memory); mgpu_run.sh waits for them.
set -uo pipefail
DS=$1; G=$2
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=${3:-/home/lanlu/microBenchmark/bucketDemo/buildBucket/runs/det_check_$$}
export BUCKET2_ASSIGN_DETERMINISTIC=1 BUCKET2_STEP4_CHECKSUM=1
bash $T/mgpu_run.sh $DS 1 $W/g1 | tail -1 || { echo "MISMATCH (G=1 build failed)"; exit 1; }
bash $T/mgpu_run.sh $DS $G $W/g$G | tail -1 || { echo "MISMATCH (G=$G build failed)"; rm -rf $W; exit 1; }
a=$(ls -d $W/g1/work/k*/ | head -1); b=$(ls -d $W/g$G/work/k*/ | head -1)
ok=1
for f in inverse_perm.bin bucket_offsets.bin vector_knn.bin forward_graph_reordered.bin final_graph.bin; do
  if [[ ! -f $a/$f && ! -f $b/$f ]]; then continue; fi
  if cmp -s $a/$f $b/$f; then echo "$f same"; else echo "$f DIFF"; ok=0; fi
done
echo "step4 checksums G=1: $(grep -o 'checksum [0-9a-f]*' $W/g1/bucket2.log | awk '{print $2}' | tr '\n' ' ') G=$G: $(grep -o 'checksum [0-9a-f]*' $W/g$G/bucket2.log | awk '{print $2}' | tr '\n' ' ')"
for g in 1 $G; do echo "G=$g times: $(for n in bucket2 prune optimize_chunked; do [[ -f $W/g$g/$n.time ]] && echo -n "$n=$(head -1 $W/g$g/$n.time | cut -d' ' -f1) "; done) gpus=$(cat $W/g$g/gpus.txt)"; done
rm -rf $W
if (( ok )); then echo "EQUAL ($DS, G=$G vs 1)"; exit 0; else echo "MISMATCH ($DS, G=$G vs 1)"; exit 1; fi
