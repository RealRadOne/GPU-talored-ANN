---
name: optimize-bucket2-scale
description: One iteration of the bucket2 multi-GPU scalability optimization loop (bucket2 Step 4 / Step 6, prune_windowed, optimize_chunked with --num-gpus G) in bucketDemo/buildBucket. Goal - make the SIFT100M build faster with 2 GPUs (always measured) and 4 GPUs (measured when all four are free, skipped otherwise), at unchanged search quality and without slowing the single-GPU path. Pick one scalability idea, implement, build against cuVS 26.10, check multi-GPU correctness (deterministic SIFT10M byte-identical to 1 GPU), benchmark SIFT100M with MGPU=2 (and MGPU=4) median of 3 plus a single-GPU guard, accept/reject with judge.py against state/best_scale.json, log to bucketDemo/OPTIMIZATION_LOG_SCALE.md. Learns from cuVS 26.12 multi-GPU (neighbors/mg/snmg.cuh) and TAGORE's GPU-CPU-disk pipeline. Run repeatedly with `/loop /optimize-bucket2-scale`.
---

# bucket2 multi-GPU scalability loop — one iteration per invocation

Runs **fully automatically** (no confirmation per round). Each invocation does exactly one idea
end-to-end, then returns; `/loop /optimize-bucket2-scale` keeps it going until the user stops it.
Never stop the loop on your own. Sister loop of `optimize-bucket2` (same code, scripts, invariants,
change policy); this one only optimizes **how the build scales over GPUs**.

## Fixed context

| what | where |
|---|---|
| code under optimization | `bucketDemo/buildBucket/src/` — multi-GPU parts: `bucket.cu` (`g_num_gpus`; Step 4 `batch_assign_with_cagra_anns` per-GPU `top2_worker`; Step 6 `run_step6` lambda → `build_vector_knn_with_tensorcore(..., pos_begin, pos_end, acc_lane)`; `ChunkedKnnAccumulator` arrival writes + `merge_iteration`), `prune_windowed.cpp` (`--num-gpus`: row ranges, one `run_chunked` per GPU), `optimize_chunked.cu` (`GpuReverseMerge`, `run_on_gpus`), `bucket_order.hpp` (order / future-access tables) |
| shared scripts (do not copy) | `S=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2/scripts` — `run_pipeline.sh` (`MGPU=G` → G free GPUs, `--num-gpus G` for all three binaries, waits until G GPUs are free), `judge.py` (`JUDGE_BEST`, `JUDGE_BENCH`, `--no-mem`; multi-GPU config key `<dataset>_<mode>_g<G>`), `summarize.py`, `params.sh`, `dataset_env.py`, `datasets.json` |
| this skill's scripts | `T=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2-scale/scripts` — `mgpu_run.sh` (one build without quality eval, any binary dir via `B2BIN`), `det_check.sh` (deterministic multi-GPU equivalence check) |
| current best | `state/best_scale.json` of this skill (written only by `judge.py --update` with `JUDGE_BEST` set). Single-GPU reference (read-only here): `optimize-bucket2/state/best.json` → `sift100M_auto` |
| log | `bucketDemo/OPTIMIZATION_LOG_SCALE.md`; also read `bucketDemo/OPTIMIZATION_LOG.md` (#96, #97 and the Ideas backlog) so this loop does not repeat work |
| branch | `opt/bucket2-loop` (commit every accepted iteration there; never push). Do not run this loop and `/optimize-bucket2` at the same time (same branch, same bench lock) |
| toolchain / build | same as `optimize-bucket2` (conda `cuvs-build`, `ninja -C bucketDemo/buildBucket/build`; cuVS / raft 26.10 public API only) |
| machine | 4 × V100-SXM2 16 GB, all pairs NVLink (NV2); GPU 0,1 on NUMA node 0, GPU 2,3 on node 1; 80 CPU threads; 251 GB RAM; `/` one LV, **~0.29 GB/s sustained write**, `vm.dirty_ratio` 20 / background 10 (≈ 39 / 19 GB dirty thresholds — check `/proc/vmstat nr_dirty_threshold`) |

## What to learn from

- **cuVS 26.12 multi-GPU** (`/home/lanlu/cuvs`, read-only): `cpp/src/neighbors/mg/snmg.cuh` — sharded build (`n_rows_per_shard`, one shard per rank), replicated vs sharded indexes, `sharded_search_with_direct_merge` (per-rank top-k then a merge over NCCL send / recv). Also `detail/cagra/cagra_build.cuh::build_knn_graph` (pinned double buffers between host and device), `detail/cagra/graph_core.cuh::optimize` (reverse graph).
- **TAGORE** (`/home/lanlu/Tagore/src/large_index.cu`, `devicelist` / per-device threads): async GPU–CPU–disk pipeline for large builds, overlapping disk I/O with GPU work.
- Name the function an idea comes from and say in one line how our code does that step today.

## Measured starting point (SIFT100M, 2026-11-16 / 19, see OPTIMIZATION_LOG.md #97)

| | 1 GPU (best #96) | 2 GPUs (#97) | 4 GPUs (#97) |
|---|---|---|---|
| total (s) | 429.6 | 317.7 | 347.8 |
| Step 4 (2 rounds) | 101.6 | 62.5 | 42.3 |
| Step 6 (2 rounds) | 159.8 | 109.9 | 121.8 |
| Wait | 33.8 | 25.2 | 50.7 |
| prune / optimize | 55.8 / 39.5 | 44.7 / 35.8 | 45.8 / 46.5 |

Known limits: Step 6 arrival records (~35 GB / round) all go through one appender; with per-GPU lanes the
writes become parallel but hit the kernel dirty-page throttle and turn Step 6 disk-bound (A/B in the log).
optimize G > 1 re-uploads the whole forward graph per GPU per segment. prune barely scales (load + sink
dominated). Step 4 scales (GPU search). Per-GPU setup (`gpu_buffer_alloc`, `future_access`) is replicated.

## GPU availability (user decisions 2026-11-01 / 2026-11-19)

- **GPU agreement (2026-11-19):** wangcy07 uses GPUs 0,1; ours are 2,3, but 0,1 may be used while they are free.
  `gpu_select.sh` `GPU_ORDER="2 3 0 1"`; `run_pipeline.sh` / `mgpu_run.sh` scan `$GPU_ORDER`, so single-GPU runs (guard, det
  G = 1) take GPU 2 or 3 first and 2-GPU runs take the pair **2,3** first, then 0,1. G = 4 only when all four happen to be
  free at the test phase; skipping it never blocks a verdict. The G = 4 best in `best_scale.json` stays as reference.
- **2 GPUs are always measured. 4 GPUs are measured only if all four GPUs are free when the test phase
  starts; otherwise skip G = 4 for this iteration** (log "G=4 skipped: GPU x,y used by <user>") and judge on
  G = 2 alone. Never wait for the 4th GPU.
- `MGPU=G` runs wait inside `run_pipeline.sh` until G GPUs are free (no compute process, < 1000 MiB). They
  never share a GPU with another user: shared GPUs change Step 4 batching (batch size follows free GPU
  memory) and invalidate both timing and the determinism check.
- If not even 2 GPUs are free, the run prints `GPU_WAIT ...` every 10 min. Every Monitor must match
  `GPU_WAIT` as well as `DONE|FAILED`. On the first `GPU_WAIT`, notify the user (PushNotification, via
  ToolSearch `select:PushNotification`) with who holds the GPUs (`nvidia-smi --query-compute-apps=pid,used_memory`, `ps -o user= -p <pid>`). Keep waiting.
- **Disk:** a SIFT100M run needs ~150 GB free at peak. Check `df -h /` before the test phase; below 200 GB,
  do not start — tell the user who is using the space if you can tell (`find / -xdev -size +2G -mtime -2`),
  and wait. Every Monitor also emits `DISK_LOW` when `/` drops below 40 GB; then stop the chain by PID (never
  `pkill -f` a pattern that also matches your own shell) and delete that run's `work/`.

## Goal and bench set

**Make SIFT100M build faster with 2 and 4 GPUs, at unchanged search quality, without slowing 1 GPU.**

| config (judge key) | when | reps | gate |
|---|---|---|---|
| `sift100M_auto_g2` = `MGPU=2 run_pipeline.sh -d sift100M -m auto` | every iteration | 3 (median) | speed vs `best_scale.json`; quality: own best − 0.05 and TAGORE − 0.05 (datasets.json `ref`), 0 graph defects |
| `sift100M_auto_g4` = `MGPU=4 ...` | every iteration **if all 4 GPUs are free** | 3 | same |
| multi-GPU equivalence (optional diagnostic, user 2026-11-20 — not a gate) | when useful | 1 | `det_check.sh sift10M G` (deterministic mode); differences are fine as long as search quality holds |
| single-GPU guard: `run_pipeline.sh -d sift10M -m auto` + `-d sift1M -m large` | every iteration | 3 / 5 | `judge.py --quick` against `optimize-bucket2/state/best.json` (default `JUDGE_BEST`): no config > 3% slower, quality + defects as there; **memory guard against `state/guard_mem_ref.json`** (`JUDGE_MEM_REF`, user 2026-11-20: the scale loop's current HEAD level, ≤ 1.2× — the sister loop's best.json is not changed) |
| `msTuring100M_auto_g2` | milestones (user asks, before a summary, after a change to the float32 path) | 1 | float32 large path still correct + quality (own best − 0.05, CAGRA ACE − 0.05) |

**User standard (2026-11-19):** single GPU unaffected or faster, multi-GPU unaffected or faster; **multi-GPU
optimization has priority**; never change the algorithm's semantics — log such ideas as `S<k>` under
**Awaiting approval** and move on to the next implementation idea.

**Rules:**
- **ACCEPT** iff the equivalence check passes, the single-GPU guard passes, every measured multi-GPU config
  passes quality / defects, at least one measured multi-GPU config is **≥ 3% faster** than its best, and none is
  **> 3% slower**. Memory guard ON (user 2026-11-20: judge default, peak RSS ≤ 1.2 × best + 0.2 GB; no `--no-mem`).
- **Memory fast paths (user 2026-11-20):** never gate an in-memory fast path on "the data fits" / "the machine has free RAM".
  All fast paths count against one user-given host memory budget (reuse / unify the existing host-budget parameters:
  `HOST_BUDGET_MB` in dataset_env.py → `--host-budget-mb` / `--bucket-mem-mb` / `--arrival-mem-mb` / `--cache-mb`); only within
  that budget use the fast path, otherwise the existing disk / out-of-core fallback. Default peak memory must not rise noticeably.
  Everything must scale to 1B / degree 512 (no design that needs the whole data in RAM).
- **Correctness (user 2026-11-20):** byte identity between paths or between G GPUs and 1 GPU is no longer required; the gate is
  search performance — recall@10 at every L within judge.py's quality gates and QPS not lower. Every iteration that adds a fast
  path runs at least one build *with search eval* on the fast path and one on the fallback (force it with a small budget, on
  sift10M or SIFT100M once). `det_check.sh` is an optional diagnostic.
- **Compute gain, disk-bound (user 2026-11-19, in force from S7 on):** the user will re-test on a high-throughput disk, so a
  change that really saves GPU / compute time but does not show in the total because `/` is slow (~0.29 GB/s) is also
  ACCEPTED when all of:
  (a) the saving is attributable and measured: the changed stage's compute metric (Step 6 per-GPU `bucket_loop` compute,
      Step 4 GPU time, prune `compute(parallel)`, optimize GPU reverse+merge count / fill / select / h2d, ...) drops by
      **≥ 5 s or ≥ 10 %** (median of 3), and the log names the disk wait that absorbed it (Wait / flush / sink growth);
  (b) the total is **not > 3 % slower** than best (noise gate), preferably with a same-period HEAD reference;
  (c) search quality (recall / QPS gates) and the single-GPU guard pass as usual (det_check optional).
  Judge with `--update --compute-gain` (writes only configs whose total beat their best; a slower total never replaces a
  best, so +3 % cannot accumulate). Log it as **"accepted (compute gain, disk-bound)"** with a before / after table of the
  compute metrics. Pure I/O moves / overlaps (e.g. S6 background write) are not compute savings.
- A config with no best yet (e.g. G = 4 never measured, or the first iteration) is recorded with `--add-new`
  and does not block the verdict.
- Report every iteration's **speedup vs 1 GPU** (`optimize-bucket2/state/best.json` `sift100M_auto`) per
  stage: Step 4, Step 6, Wait, prune, optimize, total. The scalability target is the 1-GPU / G-GPU ratio.
- Byte identity to G = 1 is required only in deterministic mode; normal runs differ through Step 4 randomness
  (user 2026-11-16: "multi-GPU search quality must not be affected" is the real requirement).

## Design invariants — never violate

Everything in `optimize-bucket2` **Design invariants** (bucketing, bucket order, bucket-pair computation,
prune, reverse edges, id translation; degrees 64 / 32; both paths; no library build on the whole dataset;
bounds checks; change policy) applies. In addition (user decisions 2026-11-15):
1. **One host CPU manages 1…N GPUs**; `--num-gpus G` stays generic (no hard-coded 2 / 4), one thread per GPU
   with its own streams / resources. `cudaSetDevice` is per thread — every new thread that touches CUDA sets
   its device first (#97 bug).
2. **Only Step 4, Step 6, prune and optimize go multi-GPU.** Steps 1, 2, 3, 4.5, 5 stay single-GPU (never split across GPUs),
   but (user 2026-11-20) they may be parallelised on the CPU (multi-threaded) or moved to background threads that overlap with
   GPU stages. Search quality unchanged (judge gates, QPS not lower; user 2026-11-20), single-GPU path not slower. Lesson S6: a
   background write of data_reordered competed with Step 6 for memory bandwidth / page cache — prefer overlapping with Step 4,
   multi-threading, or throttled writes.
3. **1B / degree-512 target:** no design that needs the whole graph or all arrival records in host RAM
   (size-gated fast paths are fine with an out-of-core fallback that keeps working); GPU memory per device
   bounded; device count and sizes from runtime queries.
4. Sliced inputs (a GPU's part of the order / rows) must not be used where code expects the full range
   (#97: future-access table sized by the slice → out-of-bounds writes). Add bounds checks for new indexing.
5. **Change policy:** implementation changes (same results; different record order / work split / overlap) may
   be tried directly. Changes to what is computed (approximations, different bucket order, different
   assignment) need an `S<k>` proposal under **Awaiting approval** in the log and the user's approval first.

## One iteration

### 0. Orient (cheap)
- If `state/best_scale.json` does not exist, this invocation is the **baseline**: skip steps 1–2; run step 3's
  multi-GPU configs (G = 2, and G = 4 if free) on the committed HEAD, judge with `--update --add-new`, fill
  **Current best** in the log, commit, return.
- `git -C /home/lanlu/microBenchmark status --short` on `opt/bucket2-loop`: the source tree must be clean (if an
  iteration was interrupted: save `git diff > bucketDemo/patches/scale<N>_interrupted.patch`, then
  `git checkout -- bucketDemo/buildBucket/src`, and log it).
- Read `OPTIMIZATION_LOG_SCALE.md` (**Current best**, **Ideas backlog**, **Awaiting approval**, **Tried &
  rejected**). Never retry a rejected idea without writing down what is different.
- `nvidia-smi`, `df -h /`, `cat /home/lanlu/.bench_locks/bucket2_bench.lock.owner`.

### 1. Pick ONE idea
- Largest multi-GPU gap first: the stage with the worst 1-GPU / G-GPU ratio weighted by its seconds (from the
  last accepted run's `bucket2_steps_s` / `stages_s` and the per-GPU lines in `bucket2.log`:
  `[Step4 gpuN]`, `[VectorKNN] ... lane N`, `function total`, `setup before bucket loop`).
- Ground it (cuVS snmg / CAGRA build / TAGORE pipeline), one coherent change, plan in 1–2 sentences.

### 2. Implement
- Edit sources; match style (Chinese/English comments); credit sources (`// 参考 cuVS mg/snmg.cuh::...`).
- Build `build/` with ninja (nice). Compile error → fix; infeasible → revert, log rejected.
- Smoke first: a sift10M build (and optionally `bash $T/det_check.sh sift10M 2` as a diagnostic, ≈ 3 min). Deterministic
  mismatch = bug → fix before timing. (Mismatch while another user is on the GPUs is not a verdict — rerun when free.)

### 3. Test (detached chain; GPUs per **GPU availability**; disk check first)
```bash
S=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2/scripts
T=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2-scale/scripts
R=/home/lanlu/microBenchmark/bucketDemo/buildBucket/runs/scale<N>
# optional: bash $T/det_check.sh sift10M 2 > $R/det2.txt   (diagnostic only)
# fast path added? one build with search eval on the fallback too (small budget), e.g. sift10M
bash $S/run_pipeline.sh -d sift10M -m auto  -o $R/sift10M_auto     # single-GPU guard
bash $S/run_pipeline.sh -d sift1M  -m large -o $R/sift1M_large
JUDGE_MEM_REF=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2-scale/state/guard_mem_ref.json \
python3 $S/judge.py $R/sift10M_auto $R/sift1M_large --quick   # speed / quality vs optimize-bucket2 best.json; memory vs the scale loop's own reference
MGPU=2 bash $S/run_pipeline.sh -d sift100M -m auto -o $R/sift100M_auto_g2
MGPU=4 bash $S/run_pipeline.sh -d sift100M -m auto -o $R/sift100M_auto_g4   # only if all 4 free at this point
```
- Write it as `$R/chain.sh`, launch with `setsid nohup bash $R/chain.sh > $R/chain.out 2>&1 < /dev/null &`, record
  the **script's own PID** (`ps -o pid,lstart,cmd -p $(pgrep -d, -f chain.sh)`; the setsid wrapper exits at once)
  and wait with a Monitor on that PID that matches `DONE|FAILED|GPU_WAIT|DISK_LOW|G4_SKIPPED`. Never sleep-poll.
  Stop a chain by PID — never `pkill -f` a pattern that also matches your own shell.
- Decide G = 4 in the chain itself, right before that step: `for g in 0 1 2 3; do gpu_is_free $g; done`
  (source `$S/gpu_select.sh`); skip with an `echo "G4_SKIPPED ..."` line when not all free.
- A/B against HEAD (same period, alternating) when a result is surprising or the machine is noisy: build HEAD
  into `build_wip/` and use `B2BIN=... $T/mgpu_run.sh` (timing only); delete each run's `work/` right away.

### 4. Judge
```bash
JUDGE_BEST=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2-scale/state/best_scale.json \
JUDGE_BENCH=sift100M_auto_g2[,sift100M_auto_g4] \
python3 $S/judge.py $R/sift100M_auto_g2 [$R/sift100M_auto_g4] --md
```
plus the single-GPU `--quick` verdict (det*.txt optional). Rules in **Goal and bench set**.

### 5a. ACCEPT
1. Same judge command with `--update` (add `--add-new` if G = 4 has no best yet). `--update` writes every judged config: if one config was slower than its best (accepted on the other), restore that entry from HEAD — never ratchet a best upwards (S26).
2. Append `### S<N> — <idea>` to `OPTIMIZATION_LOG_SCALE.md`: idea, source (cuVS / TAGORE function), class,
   GPUs used, judge tables, **per-stage speedup vs 1 GPU** table (G = 2, G = 4), det check, single-GPU guard.
   Update **Current best**.
3. Commit on `opt/bucket2-loop`: sources, `state/best_scale.json`, the log (and shared scripts if changed).
   Message `bucket2 scale S<N>: <idea>` + the attribution line from the system prompt. Don't commit `runs/`.

### 5b. REJECT
Any failure (judge reject, det mismatch on free GPUs, single-GPU regression, crash, build error). Never stop
the loop for it.
1. `git diff > bucketDemo/patches/scale<N>_<short-name>.patch`; `git checkout -- bucketDemo/buildBucket/src`; rebuild.
2. One line under **Tried & rejected** (idea, which config / stage, numbers vs best, reason, patch path). Commit log + patch.

### 6. Housekeeping & return
- Delete `runs/scale<N>/*/rep*/work` leftovers; keep `metrics.json` / logs. Keep the last ~5 `runs/scale*` dirs
  plus those referenced by `best_scale.json`. Never touch the other loops' runs. `df -h /` > 200 GB before the
  next iteration's test phase.
- Refill **Ideas backlog**.
- End with 2–3 lines: iteration, idea, verdict, SIFT100M G = 2 / G = 4 totals and speedup vs 1 GPU.
