---
name: optimize-bucket2-spot
description: One iteration of the bucket2 spot-instance loop (branch spot/bucket2, worktree /home/lanlu/microBenchmark_spot) - implement, test and optimize interruption + resume for the GPU-parallel stages (Step 4, Step 6, prune) under a local spot simulator (spot_sim.py replays a trace, sends SIGUSR1, workers stop / resume from per-segment cursors). First iterations integrate the stages (P1-P3, deterministic byte-identity gate), then a baseline (P4), then optimization ideas measured on SIFT100M with 1, 2 and 4 GPUs under a fixed trace. Accept/reject with judge.py against state/best_spot.json, log to bucketDemo/OPTIMIZATION_LOG_SPOT.md. Run repeatedly with `/loop /optimize-bucket2-spot`.
---

# bucket2 spot-instance loop — one iteration per invocation

Runs **fully automatically** (no confirmation per round). Each invocation does exactly one phase or idea end-to-end,
then returns; `/loop /optimize-bucket2-spot` keeps it going until the user stops it. Never stop the loop on your own.

**Read first, every iteration:** `bucketDemo/SPOT_DESIGN.md` (the design + user decisions — binding) and
`bucketDemo/OPTIMIZATION_LOG_SPOT.md` (plan / phase status, Current best, backlog, rejected).

## Fixed context

| what | where |
|---|---|
| worktree / branch | **`/home/lanlu/microBenchmark_spot`**, branch **`spot/bucket2`** (from main d0c2c0b). Work only here. Commit every accepted iteration on this branch; never push; never touch `/home/lanlu/microBenchmark` (the scale loop's worktree, branch opt/bucket2-loop) or `_tagore`. |
| code | `bucketDemo/buildBucket/src/`: `spot_sched.hpp` (Client + SegmentScheduler, done), `bucket.cu` (Step 4 `batch_assign_with_cagra_anns` / `top2_worker` ~L1700–2520; Step 6 `run_step6` ~L10140 → `build_vector_knn_with_tensorcore(..., pos_begin, pos_end, acc_lane)` ~L7090, scatter thread ~L7640, main pos loop ~L8206), `prune_windowed.cpp` (~L573) + `prune_windowed_gpu.cu::run_chunked` (~L565) |
| tests | `src/tests/test_spot_sched.cpp` (unit; also build with `-fsanitize=thread` after scheduler changes), `src/tests/test_spot_e2e.cpp` (with a running `spot_sim.py run`). Compiler: `/home/lanlu/miniconda3/envs/cuvs-build/bin/x86_64-conda-linux-gnu-g++ -std=c++17 -O2 -pthread -I..` |
| build | `cd bucketDemo/buildBucket && nice -n 19 taskset -c 1-79:2 /home/lanlu/miniconda3/envs/cuvs-build/bin/ninja -j 8 -C build` (cuVS / raft 26.10 public API only; never rebuild `build/` while one of *your* timed runs uses it) |
| this skill's scripts | `T=/home/lanlu/microBenchmark_spot/.claude/skills/optimize-bucket2-spot/scripts`: `spot_sim.py` (gen / run / show), `spot_build.sh DATASET G OUTDIR [TRACE|none]` (one build, no eval, bench lock), `spot_det.sh DATASET G TRACE` (deterministic plain vs spot byte check, the correctness gate) |
| traces | `$T/../traces/`: `t1_g1.csv` (spot instance 0, shared mode), `t1_g2.csv` (instance 1), `t1_g4.csv` (instances 1,2,3) — the **reference trace** for timing; `d1_g*.csv` dense stress traces for correctness. New traces: `spot_sim.py gen` (never overwrite t1 / d1). The user's real trace source is **Bamboo** (later: a converter to `time_s,instance,event`). |
| shared scripts (this worktree's copies) | `S=/home/lanlu/microBenchmark_spot/.claude/skills/optimize-bucket2/scripts`: `run_pipeline.sh` (`SPOT=1 SPOT_TRACE=<csv> SPOT_TAG=<tag> SPOT_SCALE=60 [SPOT_NOTICE_REAL_S] MGPU=G`; uses this worktree's `build/`), `judge.py` (key `<ds>_<mode>_g<G>_spot<TAG>`), `summarize.py` (`spot_reps` in metrics.json) |
| best / log | `state/best_spot.json` (only `judge.py --update` with `JUDGE_BEST` set writes it); `bucketDemo/OPTIMIZATION_LOG_SPOT.md` |
| machine | 4 × V100-SXM2 16 GB (NVLink), 80 threads, 251 GB RAM, `/` ~0.29 GB/s writes. **Shared with the scale loop** (`/optimize-bucket2-scale` in tmux `scaleloop`): every GPU run of this loop goes through the shared bench lock (`run_pipeline.sh` / `spot_build.sh` take it; never use `SMOKE=1` here), so runs of the two loops never overlap. |

## User decisions (binding — details in SPOT_DESIGN.md)

- Target 1B (design-level now); tests: SIFT10M correctness, SIFT100M timing.
- Spot only for **Step 4, Step 6, prune**. optimize_chunked: no spot, no multi-GPU (`--num-gpus 1` in spot mode — run_pipeline / spot_build do it).
- Local simulation: trace replay → SIGUSR1 + mmap state → GPU worker threads stop / resume. Time compressed (`SPOT_SCALE=60`); original intervals later on 1B.
- Serial steps on the on-demand device (CUDA device 0 of the run). **dedicated** mode (G ≥ 2): device 0 = on-demand worker that always participates but picks last and takes small segments; devices 1..G-1 spot. **shared** (G = 1): the one device is spot instance 0, the on-demand role only rescues.
- Resume = from per-segment cursors (bucket order is sequential → cursor = "everything before is done"; several GPUs → several cursors). GPU type may change on resume.
- Byte identity to an uninterrupted run is not required in normal mode, but **algorithm correctness must not change**: in deterministic mode spot + trace must equal plain byte-for-byte (`spot_det.sh`), and in normal mode recall / QPS pass judge's quality gates.
- **Change policy** (as in the other loops): implementation changes (scheduling, granularity, overlap, caching) may be done directly; anything that changes *what is computed* (assignment, bucket order, pruning, approximations) needs an `SP<k>` entry under **Awaiting approval** and the user's OK first.
- Spot OFF (no `BUCKET2_SPOT_DIR`) must keep the existing code path / behaviour and speed (the scale loop optimizes that path on main).

## One iteration

### 0. Orient
- `git -C /home/lanlu/microBenchmark_spot status --short` must be clean on `spot/bucket2` (interrupted iteration: save
  `git diff > bucketDemo/patches/spot<N>_interrupted.patch`, `git checkout -- bucketDemo/buildBucket/src`, log it).
- Read the log's phase table. The first phase not `done` is this iteration's task; after P4 pick one backlog idea.
- `nvidia-smi`, `df -h /` (≥ 200 GB free before a SIFT100M chain; the scale loop also needs space), `cat /home/lanlu/.bench_locks/bucket2_bench.lock.owner`.

### 1. Implement (P1–P3: the next stage; O*: one idea)
- Follow SPOT_DESIGN §3 protocol (`run_worker` / `may_start` / `commit` / `release`) and §4 for the stage. Keep spot OFF
  behaviour identical. `cudaSetDevice(worker.device)` in every worker thread (and every helper thread that touches CUDA).
  Setup / teardown per instance epoch (lazy setup is in `run_worker`); free everything a preempted worker allocated.
- Bounds checks for every new index (sliced ranges, stolen segments — scale invariant 4).
- Build. Run the unit test (and TSan if `spot_sched.hpp` changed).

### 2. Correctness (every iteration, hard gate)
Write a chain `R=/home/lanlu/microBenchmark_spot/bucketDemo/buildBucket/runs/spot<N>`, launch detached
(`setsid nohup bash $R/chain.sh > $R/chain.out 2>&1 < /dev/null &`, record the script's PID, Monitor on that PID matching
`DONE|FAILED|EQUAL|MISMATCH|GPU_WAIT|DISK_LOW|G4_SKIPPED`; never sleep-poll; stop chains by PID, never `pkill -f`):
```bash
bash $T/spot_det.sh sift10M 2 $T/../traces/d1_g2.csv > $R/det_g2.txt
bash $T/spot_det.sh sift10M 1 $T/../traces/d1_g1.csv > $R/det_g1.txt
# G=4 only if all four GPUs are free right now (source $S/gpu_select.sh; gpu_is_free 0..3), else echo G4_SKIPPED
bash $T/spot_det.sh sift10M 4 $T/../traces/d1_g4.csv > $R/det_g4.txt
```
- Every measured G must print `EQUAL`. `MISMATCH` = bug → fix and rerun. `MISMATCH-VACUOUS` (no preemption hit a spot
  stage) = make the check bite (denser trace via `spot_sim.py gen`, `SPOT_SCALE`), it does not count as a pass.
- Read the `[spot]` lines: preemptions really happened inside the stage you changed, rejected units > 0 somewhere, resumes
  continued from the cursor (log lines `reap` / `preempted ... committed [a,b)` / `steals`).
- Also one normal-mode build with search eval under the trace (quality), e.g.
  `SPOT=1 SPOT_TRACE=$T/../traces/d1_g2.csv SPOT_TAG=d1 MGPU=2 bash $S/run_pipeline.sh -d sift10M -m auto -r 1 -o $R/sift10M_auto_g2_spotd1`
  and `python3 $S/judge.py $R/sift10M_auto_g2_spotd1 --quick` style quality check (recall within gates of
  `optimize-bucket2/state/best.json` `sift10M_auto`; speed is not judged for spot runs at sift10M).

P1–P3 are **accepted** when the correctness gate passes (no speed gate yet) → mark the phase `done`, log, commit.

### 3. Timing (P4 and every O iteration)
```bash
for G in 2 4 1; do   # G=4 only if all 4 GPUs are free at that point (else echo G4_SKIPPED); G=1 = shared mode
  tr=$T/../traces/t1_g$G.csv; mg=$([[ $G -gt 1 ]] && echo "MGPU=$G")
  env SPOT=1 SPOT_TRACE=$tr SPOT_TAG=t1 $mg bash $S/run_pipeline.sh -d sift100M -m auto -o $R/sift100M_auto_g${G}_spott1
done
```
P4 additionally measures, per G: plain (spot off: `MGPU=G run_pipeline` without SPOT) and `SPOT=1 SPOT_TAG=none`
(no trace: overhead). Record all with `JUDGE_BEST=$T/../state/best_spot.json python3 $S/judge.py <dirs> --update --add-new`.
Overhead gate (P4 and whenever a hot path changes): `_spotnone` ≤ plain + 3% for every G (same period).

### 4. Judge (O iterations)
```bash
JUDGE_BEST=$T/../state/best_spot.json JUDGE_BENCH=sift100M_auto_g2_spott1,sift100M_auto_g1_spott1[,sift100M_auto_g4_spott1] \
python3 $S/judge.py $R/sift100M_auto_g2_spott1 $R/sift100M_auto_g1_spott1 [$R/sift100M_auto_g4_spott1] --md
```
- **ACCEPT** iff the correctness gate passed, every measured config passes quality / defects, at least one measured
  config is ≥ 3% faster than its best and none is > 3% slower (judge rules; memory guard on). A change that cuts the
  attributable lost work (rejected units × unit time, cold-start setup seconds, idle seconds) by ≥ 10% while the totals
  stay within ±3% may be accepted as **"accepted (loss cut)"** with a before / after table — spot timings are noisier
  than plain ones because preemptions land at different points; rerun once before rejecting on noise.
- Report per G: total, Step 4 / Step 6 / prune times, preempts, rejected units, setups, steals, drains, and the slowdown
  vs the same-period `_spotnone` and plain runs.

### 5a. ACCEPT
1. Judge with `--update` (`--add-new` for new keys); a config that got slower but was accepted on the others: restore
   its entry from HEAD (never ratchet a best upward).
2. Append `### SP<N> — <title>` to the log (what, why, results tables, det results, `[spot]` stats); update the phase
   table / Current best.
3. Commit on `spot/bucket2`: sources, scripts, `state/best_spot.json`, the log. Message `bucket2 spot SP<N>: <title>`
   + the attribution line from the system prompt. Don't commit `runs/`.

### 5b. REJECT
`git diff > bucketDemo/patches/spot<N>_<name>.patch`; `git checkout -- bucketDemo/buildBucket/src`; rebuild; one line
under **Tried & rejected**; commit log + patch. Never stop the loop for a rejection.

### 6. Housekeeping & return
- Delete `runs/spot<N>/*/rep*/work` and `spot_det` leftovers; keep logs / metrics. Keep the last ~5 `runs/spot*` dirs.
  `/dev/shm/spot_sim/*` of finished runs must be gone (spot_build / run_pipeline remove them; clean up after crashes).
- Refill the Ideas backlog.
- End with 2–3 lines: iteration, phase / idea, verdict, det results, SIFT100M G = 1 / 2 / 4 totals under t1.
