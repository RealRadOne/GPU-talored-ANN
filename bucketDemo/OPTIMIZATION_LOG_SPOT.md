# bucket2 spot-instance loop — log (`/optimize-bucket2-spot`, branch `spot/bucket2`, worktree `/home/lanlu/microBenchmark_spot`)

Design: `bucketDemo/SPOT_DESIGN.md` (user decisions 2026-11-20/21). Skill: `.claude/skills/optimize-bucket2-spot/SKILL.md`.

## Implementation plan (phases — do them in order, one per iteration)

| phase | what | status |
|---|---|---|
| P0 | infrastructure: `src/spot_sched.hpp` (Client + SegmentScheduler), unit test (`src/tests/test_spot_sched.cpp`, TSan clean), cross-process e2e (`src/tests/test_spot_e2e.cpp`), `spot_sim.py` (gen / run / show), run_pipeline `SPOT=1` hooks, summarize / judge `_spot<TAG>` keys, `spot_build.sh`, `spot_det.sh`, traces `t1_g{1,2,4}`, `d1_g{1,2,4}`; `Client::init()` at the top of bucket2 / prune_windowed `main()` | **done 2026-11-21** (set up interactively, see commit) |
| P1 | Step 4 on the scheduler (SPOT_DESIGN §4.1) + `spot_det.sh sift10M {1,2,4} d1_g*` EQUAL | **done 2026-11-21** (SP1; det with denser `d2_g*`) |
| P2 | Step 6 on the scheduler (§4.2) + det EQUAL | **done 2026-11-21** (SP2) |
| P3 | prune on the scheduler (§4.3) + det EQUAL | **done 2026-11-21** (SP3) |
| P4 | baseline: SIFT100M G=1/2/4 × {plain (spot off), SPOT none, SPOT t1} → `state/best_spot.json`, **Current best** | **done 2026-11-21** (SP4; no-event overhead gate fails at G=4 → first O iteration) |
| O* | optimization iterations (Ideas backlog) | — |

## Current best

SP4 baseline (2026-11-21, `runs/spot4`, SIFT100M auto, median of 3; preempts = sum over the 3 stages and both rounds, median rep):

| config | total (s) | bucket2 | Step 4 | Step 6 | prune | optimize | preempts S4/S6/prune | notes |
|---|---|---|---|---|---|---|---|---|
| sift100M_auto_g2_spott1 | 280.84 | 209.3 | 53.8 | 102.1 | 48.6 | 22.9 | 0 / 2 / 1 | |
| sift100M_auto_g4_spott1 | 274.00 | 175.2 | 63.9 | 58.8 | 75.4 | 23.2 | 6 / 4 / 5 | reps 274.6 / 258.1 / 274.0 |
| sift100M_auto_g1_spott1 | 386.77 | 316.3 | 100.9 | 162.3 | 48.6 | 22.7 | 2 / 2 / 0 | shared mode |

## Ideas backlog (O iterations, after P4)

- **O12 no-event overhead (FIRST)** — SP4: `_spotnone` vs plain (bucket2 + prune) G=1 +1.1 %, G=2 +3.1 %, **G=4 +19.8 %** (prune 47.7 → 75.1 s, Step 6 52.6 → 61.5 s; G=4 none reps 218.9 / 250.2 / 252.7 bimodal). The overhead gate requires ≤ +3 %. G=2 prune log: the on-demand worker finishes its half at 31 s, steals 22k buckets from the spot worker's tail, the stolen lease cold-starts (plan 2.65 s, empty pool, 4.50 vs 4.26 × N rows loaded) and ends at 47.9 s, later than the victim would have. Combines O4 / O10 / O11: steal only when the remaining cost of the victim ≫ the thief's per-lease setup, steal from the victim's tail by *time* (EWMA rate) not by unit cost, prune: truncate a plan at `end_of(L)` instead of stopping. Even G=1 (no steals) has prune 42.0 → 47.0 s: check the spot path's `write_rows_at` (pwrite, not counted) vs the sequential writer, and the per-lease setup.

- **O1 Step 4 commit granularity** — in spot mode commit sub-batches (smaller Nv for the search loop, or split one search batch into k commits) so a preemption loses ≤ ~1 s instead of a whole ~8 s batch (SIFT100M).
- **O2 hand off at notice** — at NOTICE, give `[issued, end)` back to the pool immediately (not at Down) so another worker starts while the noticed worker drains its in-flight units.
- **O3 cold start** — after a join, the new instance rebuilds CAGRA index / uint8 copy / bucket caches; keep host-side copies (pinned) of per-device setup inputs, start setup on join before a segment is available (overlap with waiting).
- **O4 locality-aware stealing** — Step 6 / prune: steal the range adjacent to what the thief just finished (bucket order neighbours share pool buckets → warmer GPU cache, fewer reloads); minimum steal size by cost to avoid ping-pong.
- **O5 on-demand share** — tune `BUCKET2_SPOT_OD_SHARE` / priority (dedicated mode); shared-mode rescue delay.
- **O6 Step 6 per-lease setup** — `future_access` / Belady tables, slot buffers re-created per lease: keep them per epoch, rebuild only the future table (or incrementally) when a lease starts.
- **O7 prune per-lease plan** — re-plan only the remaining range; keep the bucket pool across leases of the same epoch.
- **O9 rejected-units accounting** — `commit` counts a rejected commit as `new_cursor - s.cursor`; after a reap + re-grant that is the distance from the *new* owner's cursor (SP1: G=1 rejected_units=9.98M for one ~2.5M batch). Pass the unit start `u0` to `commit` so lost work = `new_cursor - u0`. Needed before P4 / loss-cut judgements.
- **O10 tail ping-pong** — SP2 logs: at the stage end both workers steal 3–29 positions from each other repeatedly, each Step 6 lease paying a full setup. Minimum steal size by cost (e.g. ≥ 2 × the per-lease setup time × the victim's unit rate) or no stealing below a threshold.
- **O11 prune re-plan per lease** — SP3 quality run: spot prune loaded 2.98 × N rows vs 1.93 × N plain (each lease plans its own chunks and refills the bucket pool from scratch; chunks that cross a shrunken `end` are not claimed at all → the lease ends early and the rest is re-acquired). Truncate the last chunk at `end_of(L)` instead of stopping, keep the pool across leases of one epoch (with O7).
- **O8 reap latency** — acquire() is the only place that reaps; reap from the Client monitor thread on Down so idle workers wake immediately.

## Awaiting approval

(none)

## Tried & rejected

(none)

## Iterations

### SP1 — P1: Step 4 on the SegmentScheduler (2026-11-21) — **done**

**What.**
- `spot_sched.hpp`: `run_worker_epochs(w, body)` — one `body(L, next)` call per instance epoch, for stages whose setup / work / teardown live in one function (Step 4's `top2_worker`): body sets up lazily, processes `L`, calls `next(L)` (release + acquire the next lease; false = epoch over: preempted / stage done / rescue worker idle) and tears down. `acquire(w, L, block=false)` (single try). Shared mode: `may_start` makes the rescue worker yield as soon as a spot worker is Up, and its idle `next` ends the epoch (frees device memory before the spot instance's setup). Exceptions release the active lease. `note_setup(w, s)`.
- `bucket.cu` Step 4: spot ON (and not PQ / GPU-resident loader) → workers from `make_workers(G4)` (G=1 goes through the threaded path too), initial cuts `N_nc*g/G4` reserved for the device's spot worker (dedicated: device 0 = on-demand worker), one thread per worker calling `top2_worker` per epoch under a per-device mutex (same-device workers never hold memory at once). In `top2_worker`: `claim(bs)` = `may_start` before every batch, *before* its prefetch read (only claimed batches are read), batch ends clipped to `end_of(L)`; `commit(L, batch_end, memcpy into h_top2_all)`; rejected = dropped. Spot mode sizes `Nv` by `N_nc` and the prefetch buffers by the longest `[lo, hi]` of any `batch_len`-row window (OpenMP scan, bounds-checked). Per-worker stats accumulate across epochs. Spot OFF: same code path (claim = old `batch_end_of`, static ranges).
- Tests: `test_spot_sched` +2 cases (`epochs_dedicated_g4`, `epochs_shared_g1`, with a per-device overlap check) — ALL PASS, TSan clean.
- New dense correctness traces `traces/d2_g{1,2,4}.csv` (`gen --duration-s 4800 --mtti-s 150 --down-s 60 --min-down-s 20 --notice-s 30 --hard-frac 0.3 --correlated 0.3 --seed 71/72/74`): SIFT10M Step 4 is only ~2 s per round, d1 rarely lands in it.

**Correctness (deterministic, `spot_det.sh sift10M G d2_gG`)** — all 5 artifacts byte-identical, step-4 checksums equal:

| G | GPUs | result | step4 [spot] round 1 / round 2 | bucket2 plain → spot (s) |
|---|---|---|---|---|
| 2 | 2,3 | **EQUAL** | reaped 1, preempts 1, setups 3 / reaped 1, preempts 1, setups 2 | 21.87 → 23.04 |
| 1 | 2 (shared) | **EQUAL** | reaped 1, preempts 1, setups 2 / reaped 2, preempts 2, setups 2 | 28.54 → 31.59 |
| 4 | 2,3,0,1 | **EQUAL** | segments 10, steals 1, reaped 3, preempts 5, setups 6 / segments 7, steals 1, reaped 1, preempts 3 | 20.28 → 21.24 |

(rejected_units > 0 in every run, but over-counted — see O9.)

**Quality (normal mode, `SPOT=1 d2_g2 MGPU=2 sift10M auto`, 1 rep):** recall@10 itopk 32/64/128 = 0.8336 / 0.9213 / 0.9695 vs best `sift10M_auto` 0.8315 / 0.9208 / 0.9703; knn recall@10 0.7211 vs 0.7227; 0 bad ids / self loops / dup edges. Pass.

### SP2 — P2: Step 6 on the SegmentScheduler (2026-11-21) — **done**

**What.**
- `run_step6` (bucket.cu): spot ON → always the threaded path (G=1 too), workers from `make_workers(G6)`, initial cuts = the S3 cost cuts reserved for each device's spot worker (dedicated: device 0 = on-demand), `unit_cost = wcost` (steals split by cost), one arrival lane per worker (`set_lanes(workers)`, lane = worker index). `run_worker` with no-op setup / teardown; `proc(L)` = one `build_vector_knn_with_tensorcore(pos_begin = L.begin, pos_end = end_of(L), lane = w, &S, &L)` call per lease under a per-device mutex (shared mode: rescue + spot worker on device 0 never overlap). So the per-call setup (bucket order slice, future_access / Belady tables, slot buffers, caches) is redone **per lease** for now (→ O6).
- `build_vector_knn_with_tensorcore`: new trailing args `spot_S`, `spot_L`. Main loop: `may_start(pos)` before every position (empty ones too); false → stop issuing, `stage_abort` the producer (it may wait for a slot that is never consumed), drain slots / encoders / scatter, return. Scatter order is *not* position order (slot rotation + empty buckets skipped + tail flush by slot index), so in spot mode the scatter thread holds encoded jobs by position and commits the contiguous completed prefix: the main loop publishes per position `has_job` + `spot_resolved = pos + 1` (release) once it knows whether the bucket produces a job (empty / `actual_M <= 0` → none); the scatter thread (and the main thread after the scatter join, for trailing empty positions) runs `commit(L, pos_begin + p, append_encoded of the held jobs)`; rejected → the jobs are dropped (counted, printed). Bounds checks on job positions; leftover held jobs throw. Spot OFF: unchanged path (direct append, no holding).

**Correctness (deterministic, `spot_det.sh sift10M G d2_gG`)** — all 5 artifacts byte-identical:

| G | GPUs | result | step6 [spot] round 1 / round 2 | bucket2 plain → spot (s) |
|---|---|---|---|---|
| 2 | 2,3 | **EQUAL** (1st run MISMATCH-VACUOUS: no preemption landed in a spot stage; rerun) | steals 7, drains 2, preempts 3, setups 5 / steals 2, drains 2, preempts 3, setups 4 | 21.64 → 31.36 |
| 1 | 2 (shared) | **EQUAL** | drains 2, reaped 3, preempts 5, rejected 0, setups 7 / drains 2, reaped 4, preempts 6, rejected 17, setups 7 | 28.14 → 40.64 |
| 4 | 2,3,0,1 | **EQUAL** | steals 1, drains 2, reaped 2, preempts 8, rejected 12, setups 9 / steals 4, drains 6, reaped 2, preempts 12, rejected 112, setups 16 | 19.66 → 36.12 |

**Quality (normal mode, `SPOT=1 d2_g2 MGPU=2 sift10M auto`, 1 rep, logs kept):** recall@10 itopk 32/64/128 = 0.8314 / 0.9218 / 0.9688 (best 0.8315 / 0.9208 / 0.9703), knn recall@10 0.7235; 0 defects. The log shows step-4 reap + on-demand steal, step-6 preemptions mid-lease (`committed [6583,12126)`), rejected commits 5/18, steals of the tail by both workers.

**Note.** Under the dense d2 traces spot bucket2 is much slower than plain (G=4 +84 %, G=1 +44 %): instance down time + the per-lease Step 6 setup (16 setups in one G=4 round) + tail ping-pong (steals of 3–29 positions). O6 (per-epoch setup) and O4 (minimum steal size) address that after P4.

### SP3 — P3: prune on the SegmentScheduler (2026-11-21) — **done**

**What.**
- `prune_windowed_gpu::ChunkHooks {claim(p0, p1), done(p0, p1)}` + trailing `hooks` arg of `run_chunked`, in both planning modes (pool mode: `claim` before chunk c, `finish(last_started)` after a stop, `done` at the end of `finish(c)` = after the chunk's rows went to the sink, in chunk order; legacy mode: `claim` after the chunk's union is planned, `done` after its D2H + sink).
- `prune_windowed.cpp` chunked branch: spot ON → always the scheduler path (G=1 too): initial cuts = the existing row-count cuts reserved per device's spot worker (dedicated: device 0 = on-demand), unit cost = rows per bucket + 1, `run_worker` with `proc(L)` = one `run_chunked([L.begin, end_of(L)))` per lease under a per-device mutex; `claim = may_start`, `done = commit(L, p1)`. Rows are pwritten at their offsets (`write_rows_at(..., count=false)`); a rejected chunk's bytes equal the redo's (each point only sees its own window), so they are written anyway and only **accepted** commits add their rows to `rows_written` (`StreamingGraphWriter::add_rows_written`) → the `rows_written == N` check stays exact.
- Only the chunked prune path is spot-integrated; the batched (window data fits the host budget, e.g. SIFT10M at the default 16 GB) path stays on the on-demand device. Correctness runs therefore use `PRUNE_CACHE_MB=4000` (honoured by params.sh) to force the chunked path at SIFT10M; SIFT100M uses it anyway.

**Correctness (deterministic, `PRUNE_CACHE_MB=4000 spot_det.sh sift10M G <trace>`)** — all 5 artifacts byte-identical:

| G | trace | result | prune [spot] | prune plain → spot (s) |
|---|---|---|---|---|
| 2 | d2_g2 | **EQUAL** | segments 4, steals 2, drains 1, preempts 3, setups 5 | 4.70 → 11.58 |
| 1 | d2_g1 | EQUAL but prune not hit (step 4/6: 16 preempts) | — | 4.86 → 5.11 |
| 1 | d1_g1 (3rd try; d1, d2 tries also EQUAL with 0 prune preempts) | **EQUAL** | segments 5, steals 1, drains 2, reaped 2, preempts 5, rejected 877, setups 6 | — → 38.4 wall |
| 4 | d2_g4 | **EQUAL** | segments 8, steals 3, drains 3, reaped 3, preempts 8, setups 10 | 7.95 → 16.72 |

**Quality (normal mode, G=2, `PRUNE_CACHE_MB=4000`, 1 rep):** spot d2: recall@10 itopk 32/64/128 = 0.8078 / 0.9075 / 0.9618 (prune preempts 2); same-setting plain reference = 0.8101 / 0.9080 / 0.9615 → spot within normal-mode noise; 0 defects. Both are below the default-path best (0.8315 / 0.9208): that gap is the chunked prune path itself vs the batched one at SIFT10M (pre-existing on main, not spot-related; noted for the main loops).

### SP4 — P4: SIFT100M baseline (2026-11-21) — **done**

`runs/spot4`, chain G = 2 → 4 → 1 (all four GPUs were free for G=4), each config `run_pipeline -d sift100M -m auto` (3 reps, search eval on the last rep), plain = `MGPU=G` (optimize_chunked multi-GPU), spot = `SPOT=1` (optimize_chunked `--num-gpus 1` on the on-demand device, user decision). Recorded with `JUDGE_BEST=state/best_spot.json judge.py --update --add-new` (9 keys: `sift100M_auto_g{2,4}`, `sift100M_auto` (G=1 plain), `_g{1,2,4}_spotnone`, `_g{1,2,4}_spott1`).

| G | config | total (s) (reps) | bucket2 | Step 4 | Step 6 | prune | optimize | bucket2 + prune | recall@10 itopk 32/64/128 | defects |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | plain | 252.23 (250.2 / 252.2 / 252.6) | 191.9 | 53.9 | 83.7 | 42.7 | 17.6 | 234.6 | 0.7265 / 0.8336 / 0.9110 | 0 |
| 2 | spot none | 264.24 (262.4 / 264.2 / 270.7) | 192.8 | 53.6 | 85.8 | 49.1 | 22.9 | 241.9 (**+3.1 %**) | 0.7275 / 0.8355 / 0.9112 | 0 |
| 2 | spot t1 | 280.84 (282.7 / 280.8 / 279.4) | 209.3 | 53.8 | 102.1 | 48.6 | 22.9 | 257.9 (+9.9 %) | 0.7275 / 0.8340 / 0.9110 | 0 |
| 4 | plain | 205.56 (203.4 / 206.3 / 205.6) | 141.6 | 33.7 | 52.6 | 47.7 | 16.1 | 189.3 | 0.7278 / 0.8335 / 0.9116 | 0 |
| 4 | spot none | 250.15 (218.9 / 250.2 / 252.7) | 151.7 | 34.2 | 61.5 | 75.1 | 22.6 | 226.8 (**+19.8 %**) | 0.7256 / 0.8330 / 0.9110 | 0 |
| 4 | spot t1 | 274.00 (274.6 / 258.1 / 274.0) | 175.2 | 63.9 | 58.8 | 75.4 | 23.2 | 250.6 (+32.4 %) | 0.7248 / 0.8315 / 0.9106 | 0 |
| 1 | plain | 370.03 (370.0 / 371.1 / 368.3) | 305.4 | 93.2 | 158.7 | 42.0 | 23.7 | 347.4 | 0.7248 / 0.8319 / 0.9105 | 0 |
| 1 | spot none | 374.43 (374.4 / 377.4 / 371.0) | 304.1 | 94.0 | 158.3 | 47.0 | 22.3 | 351.1 (+1.1 %) | 0.7274 / 0.8344 / 0.9106 | 0 |
| 1 | spot t1 | 386.77 (386.8 / 395.8 / 386.0) | 316.3 | 100.9 | 162.3 | 48.6 | 22.7 | 364.9 (+5.0 %) | 0.7242 / 0.8319 / 0.9097 | 0 |

**Overhead gate** (`_spotnone` ≤ plain + 3 %). The gate is applied to **bucket2 + prune** (the spot stages), not the total: in spot mode optimize_chunked deliberately runs on 1 GPU (user decision) while plain uses G (+5–6 s at G=2/4). G=1 passes (+1.1 %), G=2 is borderline (+3.1 %), **G=4 fails (+19.8 %)** → O12 is the first O iteration. Search quality is unchanged everywhere (spot vs plain within normal-mode noise, 0 defects).

**`[spot]` per rep under t1** (steals / drains / reaped / preempts / setups):
- G=2: step4 0/0/0/0/4 every rep; step6 3/2/1/2/5, 4/1/1/2/5, 3/1/0/1/5; prune 0/1/0/1/3, 2/1/0/1/3, 0/1/0/1/3.
- G=4: step4 1/1/0/6/12, 23/1/0/6/13, 22/1/1/6/13; step6 13/3/0/5/10, 3/3/1/4/9, 3/3/0/4/9; prune 13/3/0/5/7, 3/3/0/4/7, 10/4/0/5/7. Step-4 rejected_units 54–84 M per rep are the O9 over-count (a rejected commit is counted from the new owner's cursor).
- G=1: step4 0/2/1/2/6, 1/1/1/2/6, 0/2/1/2/6; step6 0/1/1/2/4, 0/3/3/3/5, 1/1/0/2/4; prune never preempted under t1 (sparse trace, ~4 trace hours per build).

**Slowdown under t1 vs spot none:** G=2 +6.3 % total (Step 6 +16 s: 2 preemptions / round of the spot device), G=4 +9.5 % (Step 4 +30 s: the 8 s search batches lost at each preemption + reaps → O1), G=1 +3.3 %.

