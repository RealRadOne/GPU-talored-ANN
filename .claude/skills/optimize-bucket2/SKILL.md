---
name: optimize-bucket2
description: One iteration of the bucket2 build-pipeline optimization loop (bucket2 -> optimize | prune_windowed+optimize_chunked -> translate_ids) in bucketDemo/buildBucket, learning from cuVS CAGRA and CAGRA ACE (partitioned out-of-core CAGRA build) in the cuVS 26.12 source. Dataset-generic (datasets.json: uint8 / float32, size-adaptive params via dataset_env.py). Pick one idea, implement, build against cuVS 26.10, benchmark the bench set (SIFT1M auto/large, SIFT10M, SIFT100M, msTuring1M, msTuring10M; msTuring100M at milestones; median of 3 each) under the shared bench lock (GPU 0 preferred, other free GPUs when taken), accept/reject with judge.py (same standards as optimize-bucket2-tagore: ±3% per config; search gated against both CAGRA and TAGORE), log to bucketDemo/OPTIMIZATION_LOG.md. Run repeatedly with `/loop /optimize-bucket2`.
---

# bucket2 optimization loop (CAGRA + CAGRA ACE) — one iteration per invocation

Runs **fully automatically** (no confirmation per round). Each invocation does exactly
one idea end-to-end, then returns; `/loop /optimize-bucket2` keeps it going until the
user stops it. Never stop the loop on your own.

## Fixed context

| what | where |
|---|---|
| code under optimization | `bucketDemo/buildBucket/src/` — `bucket.cu` (bucket2), `optimize.cu` (resident prune+reverse), `prune_windowed.cpp` + `prune_windowed_core.hpp` + `prune_windowed_gpu.cu` (large-path prune, chunked GPU), `optimize_chunked.cu` (method E), `translate_ids.cpp` + `translate_graph_ids.hpp`, shared `*.hpp` |
| pipeline parameters | `scripts/params.sh` (fixed: degrees 64/32, seed, stage flags) + `scripts/dataset_env.py` (size-adaptive: CENTROID_RATIO = 1/B with avg bucket B = 512, NPROBE = ceil(20000/B), KMeans train fraction, host/GPU cache sizes, A1/A7 switches) + `scripts/datasets.json` (registry: dir, base file, reps, iterations, search tool, quality ref, bench modes, per-dataset `params`). Precedence: exported env var > datasets.json `params` > dataset_env rules. Tune per dataset in datasets.json, not in params.sh |
| reference source (read-only) | `/home/lanlu/cuvs` = cuVS **26.12**, `cpp/src/neighbors/`. See **What to learn from** below: CAGRA (NN-Descent / IVF-PQ build, `graph_core.cuh` optimize) and **CAGRA ACE** (`detail/cagra/cagra_ace.cuh`). raft headers: `/home/lanlu/miniconda3/envs/cuvs-bench/include/raft` |
| link-time dependency | conda env `cuvs-bench` = libcuvs / libraft / librmm **26.10**, nvcc 12.9. Only the **public** cuVS/raft API is available (no `detail::` from 26.10 headers). Porting a 26.12 kernel or routine = copy it into our source with a credit comment, don't include `/home/lanlu/cuvs/cpp/src` |
| toolchain | conda env `cuvs-build` (gcc 14, cmake 4, ninja, boost 1.87). Build: `cd bucketDemo/buildBucket && /home/lanlu/miniconda3/envs/cuvs-build/bin/ninja -C build` (reconfigure: `CONDA_PREFIX=/home/lanlu/miniconda3/envs/cuvs-bench /home/lanlu/miniconda3/envs/cuvs-build/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=/home/lanlu/miniconda3/envs/cuvs-build/bin/ninja -DCMAKE_BUILD_TYPE=Release`) |
| eval tools | `bucketDemo/evalCuvs/build/cagra_eval` (cuVS 26.10 CAGRA build / search sweep, uint8 / int8 / float32), `scripts/diskann_search.py` (DiskANN CPU in-memory search, for the 100M sets that do not fit the GPU) |
| datasets | `scripts/datasets.json`: `sift1M`, `sift10M`, `sift100M` (uint8, D=128), `msTuring1M`, `msTuring10M`, `msTuring100M` (float32, D=100); each dir holds `base.*`, `query.public.10K.*`, `groundtruth.ibin` (k=100), `base_knn_sample10K_k64.ibin` (exact KNN of 10K sampled base points; `scripts/prep_dataset.py` + `gen_gt.py` make them) |
| references | `state/cagra_baseline.json` — CAGRA NN-Descent 64→32 for 1M / 10M sets, **CAGRA ACE** 64→32 for SIFT100M (738 s) and msTuring100M (1205 s); `state/tagore_baseline.json` — TAGORE (GNN-Descent for 1M, largeIndex Vamana for 10M / 100M). Both with recall / QPS tables (from the TAGORE loop's step B / P1); judge.py gates search against **both** |
| current best | `state/best.json` (written only by `judge.py --update`). Pre-sync history: `state/best_pre_main_sync_2026-10-29.json` (not comparable) |
| log | `bucketDemo/OPTIMIZATION_LOG.md`; also read `bucketDemo/OPTIMIZATION_LOG_TAGORE.md` (the TAGORE loop's accepted + rejected ideas) so this loop does not repeat them |
| branch | `opt/bucket2-loop` (commit every accepted iteration there; never push) |
| GPU | timed runs prefer **GPU 0** under the bench lock; `scripts/gpu_select.sh` moves to a free GPU when it is taken (see **GPU availability**). SMOKE runs: `SMOKE_GPU` (default 2), falling back to 3, then 1 |

## What to learn from (cuVS 26.12)

**1. CAGRA build** — kernels and per-stage techniques for the in-memory work:
- NN-Descent: `detail/nn_descent.cuh`, `detail/nn_descent_gnnd.hpp` (fp16 WMMA distances for dim > 16, local join).
- IVF-PQ: `detail/cagra/cagra_build.cuh::build_knn_graph` (batched queries over a host dataset, pinned double buffers, overlap of search + refine, `refine_rate`), `ivf_pq/ivf_pq_build.cuh` (sampled balanced k-means, `process_and_fill_codes`), `ivf_pq/ivf_pq_search.cuh`, `detail/refine*`.
- `detail/cagra/graph_core.cuh` — `sort_knn_graph`, `optimize` (detour-count prune, reverse graph, merge), `kern_fused_prune`.
- Building blocks: `detail/knn_brute_force.cuh`, `detail/fused_l2_knn.cuh`, `detail/faiss_select/`, `matrix/select_k`, `../cluster/detail/kmeans*`, `all_neighbors/` (clustered / batched kNN).

**2. CAGRA ACE** (`detail/cagra/cagra_ace.cuh`, "augmented core extraction") — cuVS's own partitioned
out-of-core CAGRA build; the closest cuVS analogue of our large path. Map of its steps onto ours:

| ACE function | what it does | our stage to compare with |
|---|---|---|
| `ace_get_partition_labels` | sampled k-means centroids (host-side float conversion, 32K-row chunks to bound GPU memory); every vector gets its **2 closest partitions** (core + augmented) | Step 2 centroid selection, Step 4 assignment (our neighbor buckets / two-choice rule play the role of the augmented copy) |
| `ace_check_partition_sizes` | partition histogram, warns at > 5× the expected size | bucket-size outliers, Step 4 balance / cap |
| `ace_create_forward_and_backward_lists` | reordered ↔ original id maps (parallel in memory mode, ordered in disk mode) | inline reorder (`inverse_perm.bin`), `translate_ids` |
| `ace_reorder_and_store_dataset` | writes core + augmented partitions to disk: pre-allocated files, per-partition buffers sized from free memory (`0.5 × free / (2 × n_partitions)`), ≤ 64 MB writes "to saturate NVMe" | Step 4.5 reorder write, bucket-vector write, `data_reordered` |
| `ace_load_partition_dataset_from_disk` | per-partition load for its sub-build | Step 6 bucket loading / host + GPU bucket cache, prune chunk loading |
| per-partition `cagra::build` + optimize | in-memory CAGRA (NN-Descent or IVF-PQ) on core ∪ augmented, then optimize | Step 6 bucket-pair KNN + prune (we keep bucket-pair computation — see invariants) |
| `ace_adjust_sub_graph_ids(_disk)` | local → reordered → original ids on the device | `optimize_chunked` / `translate_ids` id handling |
| `ace_check_use_disk_mode`, `ace_suggest_partition_count`, `ace_validate_disk_mode_partitions` | explicit host / GPU memory model decides memory vs disk mode and the partition count | bucket2's fit check, `dataset_env.py` budgets (`--cache-mb`, `--bucket-mem-mb`, `--gpu-bucket-cache-mb`, `--host-budget-mb`) |

Reference numbers to beat (cuVS 26.10, V100, from `state/cagra_baseline.json`): ACE SIFT100M 738 s,
DiskANN-L recall@10 0.767 / 0.870 / 0.937 at L 32 / 64 / 128; ACE msTuring100M 1205 s, 0.630 / 0.751 / 0.846.
For every idea, name the CAGRA or ACE function it comes from and say in one line how our code does that step today.

## Synced with main (2026-10-29)

`opt/bucket2-loop` was merged with `main` (TAGORE loop #53–#89). The shared scripts come from
`optimize-bucket2-tagore` (`run_pipeline.sh`, `dataset_env.py`, `datasets.json`, `summarize.py`,
`knn_similarity.py`, `diskann_search.py`, `gen_gt.py`, `resume_large_tail.sh`, `prep_dataset.py`,
`params.sh`, `judge.py`). The code and defaults include the user-approved algorithm changes from that
loop (A2: avg bucket 512 + NPROBE rule; A7: unwindowed batched GPU prune on SIFT1M/SIFT10M/msTuring10M;
2 rounds on SIFT10M / the 100M sets; A1 local join on msTuring1M) — treat them as the design baseline.
The old best is archived; **the next invocation is the re-baseline** (step 0).

## GPU availability (user decision 2026-11-01)

Other users (e.g. vLLM) may take any GPU. **If the GPU in use is taken, rerun on another free GPU; if no GPU is free, wait and tell the user.**
- `run_pipeline.sh` does this itself before every rep (`select_gpu`): keeps the current GPU if free, else the first free one in order 0, 1, 2, 3 (free = no compute process and < 1000 MiB used). A rep during which another process is on its GPU at the end, or that failed while one is there, is discarded and rerun. The GPU of each rep is in `rep<i>/gpu.txt` and `metrics.json` `gpus`.
- **Do not add a "wait for GPU 0" guard to chain.sh** (the old `gpu_wait` / `GPU0_BUSY`): it would block while other GPUs are free. Do not treat a GPU that is taken as a reject.
- **No free GPU:** the run holds the lock and prints `GPU_WAIT ...` (again every 10 min), marker `/home/lanlu/.bench_locks/GPU_WAIT`. Every Monitor on a chain must match `GPU_WAIT` as well as `CHAIN_DONE|FAILED`. On the first `GPU_WAIT`, notify the user right away with PushNotification (load it with ToolSearch `select:PushNotification`) and say it in your text: which GPUs are taken, by whom (`nvidia-smi`, `ps`), what is waiting. Keep waiting; don't stop the loop.
- **Timings on any GPU count** (user decision 2026-11-01): judge.py compares them with the best like any other run and only reports the GPU. Say in the log entry which GPU was used when it was not GPU 0.

## Current priority (user decision 2026-11-01) — read first

**1. One-time re-record (do this first, then commit).** sift1M_large and msTuring10M are the noisy configs and
now use the median of 5 (±3% unchanged). Their recorded bests are medians of 3 (sift1M_large 7.31 s was a fast
draw that rejected #82, a change that does not even run on SIFT1M), so re-record only these two on the current
code, as a measurement-rule change:
`run_pipeline.sh -d sift1M -m large -o $R/sift1M_large`, `run_pipeline.sh -d msTuring10M -m auto -o $R/msTuring10M_auto`
(reps come from datasets.json), then `judge.py $R/sift1M_large $R/msTuring10M_auto --update --rebaseline --md`.
Log it under the iteration as "measurement-rule change", with the old and new values. Do not re-record other configs.

**2. Top optimization target: msTuring1M is slower than both references.** `msTuring1M_auto` 16.00 s vs CAGRA
11.48 s and TAGORE 9.35 s. Pick ideas for it first, until it is faster than **both** (≤ 9.35 s), search gates
unchanged. Where the time goes (`runs/iter80`): **Step 6 11.4 s of 16.0 s** (6 rounds × ~1.9 s, `gemm=FP32`, pool
~100K rows; SIFT1M's whole Step 6 is 1.9 s), Write 1.0, Step 1 0.9, A1 local join 0.68, FusedOptimize 0.69. Order:
- **Implementation first: port the float32 fp16 Tensor Core scan + exact fp32 re-rank to the resident Step 6.** The
  large path has it (TAGORE-loop #72, #84, #85: msTuring10M Step 6 63.8 → 45.6 s); the resident path still runs FP32
  GEMM. Output must stay identical to the fp32 path, as on the large path.
- Then other implementation work on the resident float path: GEMM + top-M fused kernel, row tile sizing, Step 1 / Write.
- **Parameters** (`datasets.json` msTuring1M: 6 rounds, NPROBE 96, A1 1 pass, B 512) only with the quality-margin rule.
  The margin is small: recall@10 at itopk 32 is 0.659 vs CAGRA floor 0.6528; TAGORE's graph is much weaker there (0.605),
  so the CAGRA gate binds and we may not trade down to TAGORE's quality. Read `optimize-bucket2-tagore/docs/dataset_tuning_params.md`
  (the earlier msTuring1M calibration) so already-failed points are not rerun.
- Anything that changes what is computed (e.g. skipping pairs seen in earlier rounds) is an algorithm change: propose under **Awaiting approval**.
Normal acceptance still applies (gain ≥ 3% on msTuring1M is a gain; no config > 3% slower). The other backlog items, including the #82 retry, come after this target.

## Goal and bench set (user decision 2026-10-29: same tests and standards as the TAGORE loop)

**Make the build faster on every dataset, with the 100M sets as the main target**, at unchanged search quality.

| config | when | reps | search tool | quality gates (judge.py; **all** must pass) |
|---|---|---|---|---|
| `sift1M_auto` (resident) | every iteration | 3 (median) | cuVS CAGRA (`cagra_eval`), itopk 32…512 | CAGRA − 0.01 **and** TAGORE − 0.01 |
| `sift1M_large` (large path forced) | every iteration | **5** (median) | cuVS CAGRA | CAGRA − 0.05 and TAGORE − 0.05 |
| `sift10M_auto` (→ large path, 2 rounds) | every iteration | 3 | cuVS CAGRA | CAGRA − 0.05 and TAGORE − 0.05 |
| `sift100M_auto` (2 rounds) | every iteration | 3 | DiskANN CPU, L 32…512 | own best − 0.05 and TAGORE − 0.05 (CAGRA ACE reported) |
| `msTuring1M_auto` (float32, resident) | every iteration | 3 | cuVS CAGRA | CAGRA − 0.01 and TAGORE − 0.01 |
| `msTuring10M_auto` (float32) | every iteration | **5** (median) | cuVS CAGRA | CAGRA − 0.05 and TAGORE − 0.05 |
| `msTuring100M_auto` (float32, 2 rounds) | **milestones only** | 3 | DiskANN CPU | CAGRA ACE − 0.05 and TAGORE − 0.05 |

Tolerance per path: resident − 0.01, large − 0.05 (same as the TAGORE loop). These live in `datasets.json`
(`bench`, `reps`, `search`, `ref` = list of references, `own_tol`). Same tests and standards as the
TAGORE loop; the only difference is that its registry gates each dataset against one reference, this
loop against CAGRA *and* TAGORE (user 2026-10-29).

**Milestones** (msTuring100M ×3, like the TAGORE loop): the re-baseline (step 0); before writing a summary
for the user; when the user asks; and after an accepted change to the float32 large path (Step 4 / Step 6
float kernels, chunked prune, `optimize_chunked`) — run it then as a check, judge it with the bench set
plus `$R/msTuring100M_auto`; a failed check there reverts the change like any reject.

**Rules (judge.py, identical to the TAGORE loop):**
- **Every config: build_total (sum of all stages, median of 3) ≥ 3% faster than best = gain, > 3% slower = regression.**
- **ACCEPT** iff the full bench set was run, at least one config gains, none regresses, search quality passes
  everywhere (recall@10 at every itopk / L vs the reference above), 0 graph defects, path selection unchanged,
  final degree 32 / pre-prune degree 64, large-path memory (host peak RSS per stage, GPU peak) ≤ 1.2× best.
- **Quality-margin rule:** a parameter choice must clear the quality gate reliably (≥ 4 timed runs, all
  above the floor; Step 4 is nondeterministic). A change that buys that robustness is accepted even if slower
  ("quality-margin, accepted by rule"; re-record with `--rebaseline`, put the multi-run spread table in the log).
- `--quick` pre-checks a subset (correctness, regressions, quality, memory; no "needs a gain").
- Pre-prune KNN similarity (degree-64 index vs exact KNN, recall@10 / @64) is recorded, not gated.
- After **Current priority**, pick ideas by the 100M stage times first (`state/best.json` → `sift100M_auto` / `msTuring100M_auto`
  `stages_s`, `bucket2_steps_s`), then the 10M sets. Ideas must scale with N.

## Design invariants — never violate

The pipeline is designed for datasets far larger than GPU memory. An optimization may
change *how* a stage is done, never remove or bypass a stage:

1. **Bucketing** — data is partitioned into buckets (centroid selection + assignment).
2. **Bucket order** — buckets are processed in the DiskJoin-style order derived from the centroid KNN graph.
3. **Bucket-pair computation** — per-vector KNN comes from bucket × neighbor-bucket computation, not from a global index.
4. **Index prune** (CAGRA-style rank-based pruning).
5. **Reverse-edge adding** (and merge).
6. **ID translation** back to the original order (`translate_ids`).

Also:
- Build index degree **64** (`NEIGHBORS_M`), final degree after prune + reverse edges **32** (`OUTPUT_DEGREE`). Fixed.
- **Two paths must both keep working:** data fits on the GPU → fully GPU-resident path; otherwise bucket2 out-of-core + `prune_windowed` + `optimize_chunked -m E`. The auto-selection must stay correct. Never make the large path depend on holding the full dataset or full graph in GPU memory; host memory stays bounded by the configured budgets.
- Never replace the pipeline with `cagra::build`, CAGRA ACE, nn_descent, or any other library build on the whole dataset — learn techniques from them, port them into our stages.
- **Change policy:** implementation optimizations (same results, or same semantics with exact re-ranking) may be tried directly. Algorithm / semantic changes (what is computed, approximations, new heuristics, parameter rules that trade quality) need a logged proposal (`A<k>` under **Awaiting approval** in the log) and the user's approval first. Parameter tuning within `datasets.json` is allowed when the quality gates pass with margin.
- Keep boundary checks: final graph must have no out-of-range ids, self loops, or duplicate edges. Add new bounds checks when you add new indexing code.

## One iteration

### 0. Orient (cheap)
- If `state/best.json` does not exist, this invocation is the **re-baseline**: skip steps 1–2, run the full bench set of step 3 into `runs/iter<N>/` (N = next free number, label "baseline after main sync"), then `judge.py <all 7 dirs (msTuring100M included: the re-baseline is a milestone)> --update --add-new --md`, fill the log's **Current best** block, commit, and return.
- `git -C /home/lanlu/microBenchmark status --short` on branch `opt/bucket2-loop`; the source tree must be clean (if a previous iteration was interrupted, `git checkout -- bucketDemo/buildBucket/src .claude/skills/optimize-bucket2/scripts/params.sh .claude/skills/optimize-bucket2/scripts/datasets.json` first and log it as "interrupted").
- Read `OPTIMIZATION_LOG.md` (**Current best**, **Ideas backlog**, **Awaiting approval**, **Tried & rejected**) and the Tried & rejected list of `OPTIMIZATION_LOG_TAGORE.md`. Never retry a rejected idea unless you write down what is different this time.
- Bench lock: `cat /home/lanlu/.bench_locks/bucket2_bench.lock.owner`. If the TAGORE loop is timing, edit / build / smoke-test gently and let the timed runs queue (run_pipeline blocks on the lock by itself).

### 1. Pick ONE idea
- Start from where time goes on the 100M sets (largest stage first), then the 10M sets.
- Ground it in CAGRA or CAGRA ACE (see **What to learn from**): name the file + function, and how our code does that step today.
- If it is an algorithm / semantic change and not approved: add it under **Awaiting approval** and pick the next implementation idea.
- One coherent change so the result is attributable. Write the plan in one or two sentences before editing (it becomes the log entry's "idea").
- Profiling: `nsys profile` on SIFT1M / msTuring1M is fine (output under the scratchpad, not the repo).

### 2. Implement
- Edit sources and/or `datasets.json` / `params.sh`. Match the surrounding code style (Chinese/English comment mix, naming). Credit the source, e.g. `// 参考 cuVS cagra_ace.cuh::ace_reorder_and_store_dataset: 预分配文件 + 64 MB 分块写`.
- Build with ninja (gently while the other loop is timing: `nice -n 19 taskset -c 1-79:2 ninja -j 8 -C build`). Compile error → fix; infeasible → revert and log it as rejected ("infeasible: ...").
- Smoke-test new code first: `SMOKE=1 [SMOKE_GPU=1] bash $S/run_pipeline.sh -d sift1M -m large -r 1 -o <dir>` (and `-d msTuring1M` for float code); SMOKE runs are unlocked, niced, never judged. Delete smoke dirs afterwards.

### 3. Test (full bench set, under the bench lock; GPU per **GPU availability**)
```bash
S=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2/scripts
R=/home/lanlu/microBenchmark/bucketDemo/buildBucket/runs/iter<N>
# quick sets first (reps from datasets.json = 3)
bash $S/run_pipeline.sh -d sift1M      -m auto  -o $R/sift1M_auto
bash $S/run_pipeline.sh -d sift1M      -m large -o $R/sift1M_large
bash $S/run_pipeline.sh -d sift10M     -m auto  -o $R/sift10M_auto
bash $S/run_pipeline.sh -d msTuring1M  -m auto  -o $R/msTuring1M_auto
bash $S/run_pipeline.sh -d msTuring10M -m auto  -o $R/msTuring10M_auto
python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/msTuring1M_auto $R/msTuring10M_auto --quick
# only if the quick check passes:
bash $S/run_pipeline.sh -d sift100M     -m auto -o $R/sift100M_auto
# milestones only (see Goal):
bash $S/run_pipeline.sh -d msTuring100M -m auto -o $R/msTuring100M_auto
```
- Launch the chain detached (`setsid nohup bash $R/chain.sh > $R/chain.out 2>&1 &`) so a harness restart cannot kill it; wait for completion with a background until-loop / Monitor, never sleep-poll. A normal iteration takes roughly 1 h (SIFT100M × 3 dominates); a milestone adds msTuring100M × 3 (~45 min).
- Any non-zero exit / "FAILED" = reject (after checking it isn't an environment issue: disk full — `df -h /home/lanlu` must keep > 50 GB free).
- Quality per run: (a) `knn_similarity` (pre-prune index vs exact KNN, recall@1/10/32/64); (b) `search` — recall@10 **and** QPS at every itopk / L, graph defect counts. Keep both recall and QPS in the log.

### 4. Judge
```bash
python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/sift100M_auto $R/msTuring1M_auto $R/msTuring10M_auto --md
# at a milestone add $R/msTuring100M_auto
```
Rules: see **Goal and bench set** and judge.py's docstring (config keys come from each run's metrics.json; order does not matter).

### 5a. ACCEPT
1. Same command with `--update --md` (writes `state/best.json`).
2. Append `### #<N> — <idea>` to `OPTIMIZATION_LOG.md`: idea, **CAGRA / ACE source** (file + function), class (implementation / approved `A<k>`), the `--md` tables (build, bucket2 steps, (a) KNN similarity, (b) search incl. the CAGRA / ACE rows); update **Current best**.
3. Commit on `opt/bucket2-loop`: sources, `params.sh` / `datasets.json`, `state/best.json`, the log. Message `bucket2 opt #<N>: <idea>` + the attribution line from the system prompt. Don't commit `runs/`.

### 5b. REJECT
Every failure in an optimization iteration (judge reject, build error, crash, timeout, wrong path, graph defects, quick check failed). Never stop the loop for it.
1. Keep the patch: `git diff > bucketDemo/patches/iter<N>_<short-name>.patch`.
2. `git checkout -- bucketDemo/buildBucket/src .claude/skills/optimize-bucket2/scripts/params.sh .claude/skills/optimize-bucket2/scripts/datasets.json` and rebuild.
3. One line under **Tried & rejected**: idea (CAGRA / ACE function), which config / stage failed, measured numbers vs best (time, recall@10, QPS, memory) or the error, the judge reason, the patch path. Commit the log and the patch.

### 6. Housekeeping & return
- Delete `runs/iter<N>` dirs older than the last ~5 iterations except those referenced by `state/best.json`; never delete the TAGORE loop's runs. Side runs (A/B, equivalence checks) must delete their own intermediates. Check `df -h /home/lanlu` (> 50 GB free).
- Refill the **Ideas backlog** (CAGRA / ACE map rows not yet tried first).
- End with a 2–3 line status: iteration N, idea, verdict, new best totals per dataset.
