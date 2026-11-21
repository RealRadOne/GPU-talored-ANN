# bucket2 spot-instance loop — log (`/optimize-bucket2-spot`, branch `spot/bucket2`, worktree `/home/lanlu/microBenchmark_spot`)

Design: `bucketDemo/SPOT_DESIGN.md` (user decisions 2026-11-20/21). Skill: `.claude/skills/optimize-bucket2-spot/SKILL.md`.

## Implementation plan (phases — do them in order, one per iteration)

| phase | what | status |
|---|---|---|
| P0 | infrastructure: `src/spot_sched.hpp` (Client + SegmentScheduler), unit test (`src/tests/test_spot_sched.cpp`, TSan clean), cross-process e2e (`src/tests/test_spot_e2e.cpp`), `spot_sim.py` (gen / run / show), run_pipeline `SPOT=1` hooks, summarize / judge `_spot<TAG>` keys, `spot_build.sh`, `spot_det.sh`, traces `t1_g{1,2,4}`, `d1_g{1,2,4}`; `Client::init()` at the top of bucket2 / prune_windowed `main()` | **done 2026-11-21** (set up interactively, see commit) |
| P1 | Step 4 on the scheduler (SPOT_DESIGN §4.1) + `spot_det.sh sift10M {1,2,4} d1_g*` EQUAL | **done 2026-11-21** (SP1; det with denser `d2_g*`) |
| P2 | Step 6 on the scheduler (§4.2) + det EQUAL | **done 2026-11-21** (SP2) |
| P3 | prune on the scheduler (§4.3) + det EQUAL | todo |
| P4 | baseline: SIFT100M G=1/2/4 × {plain (spot off), SPOT none, SPOT t1} → `state/best_spot.json`, **Current best** | todo |
| O* | optimization iterations (Ideas backlog) | — |

## Current best

(empty until P4)

| config | total (s) | Step 4 | Step 6 | prune | lost (rejected units) | preempts | notes |
|---|---|---|---|---|---|---|---|

## Ideas backlog (O iterations, after P4)

- **O1 Step 4 commit granularity** — in spot mode commit sub-batches (smaller Nv for the search loop, or split one search batch into k commits) so a preemption loses ≤ ~1 s instead of a whole ~8 s batch (SIFT100M).
- **O2 hand off at notice** — at NOTICE, give `[issued, end)` back to the pool immediately (not at Down) so another worker starts while the noticed worker drains its in-flight units.
- **O3 cold start** — after a join, the new instance rebuilds CAGRA index / uint8 copy / bucket caches; keep host-side copies (pinned) of per-device setup inputs, start setup on join before a segment is available (overlap with waiting).
- **O4 locality-aware stealing** — Step 6 / prune: steal the range adjacent to what the thief just finished (bucket order neighbours share pool buckets → warmer GPU cache, fewer reloads); minimum steal size by cost to avoid ping-pong.
- **O5 on-demand share** — tune `BUCKET2_SPOT_OD_SHARE` / priority (dedicated mode); shared-mode rescue delay.
- **O6 Step 6 per-lease setup** — `future_access` / Belady tables, slot buffers re-created per lease: keep them per epoch, rebuild only the future table (or incrementally) when a lease starts.
- **O7 prune per-lease plan** — re-plan only the remaining range; keep the bucket pool across leases of the same epoch.
- **O9 rejected-units accounting** — `commit` counts a rejected commit as `new_cursor - s.cursor`; after a reap + re-grant that is the distance from the *new* owner's cursor (SP1: G=1 rejected_units=9.98M for one ~2.5M batch). Pass the unit start `u0` to `commit` so lost work = `new_cursor - u0`. Needed before P4 / loss-cut judgements.
- **O10 tail ping-pong** — SP2 logs: at the stage end both workers steal 3–29 positions from each other repeatedly, each Step 6 lease paying a full setup. Minimum steal size by cost (e.g. ≥ 2 × the per-lease setup time × the victim's unit rate) or no stealing below a threshold.
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

