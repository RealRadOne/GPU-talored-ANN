# bucket2 spot-instance loop — log (`/optimize-bucket2-spot`, branch `spot/bucket2`, worktree `/home/lanlu/microBenchmark_spot`)

Design: `bucketDemo/SPOT_DESIGN.md` (user decisions 2026-11-20/21). Skill: `.claude/skills/optimize-bucket2-spot/SKILL.md`.

## Implementation plan (phases — do them in order, one per iteration)

| phase | what | status |
|---|---|---|
| P0 | infrastructure: `src/spot_sched.hpp` (Client + SegmentScheduler), unit test (`src/tests/test_spot_sched.cpp`, TSan clean), cross-process e2e (`src/tests/test_spot_e2e.cpp`), `spot_sim.py` (gen / run / show), run_pipeline `SPOT=1` hooks, summarize / judge `_spot<TAG>` keys, `spot_build.sh`, `spot_det.sh`, traces `t1_g{1,2,4}`, `d1_g{1,2,4}`; `Client::init()` at the top of bucket2 / prune_windowed `main()` | **done 2026-11-21** (set up interactively, see commit) |
| P1 | Step 4 on the scheduler (SPOT_DESIGN §4.1) + `spot_det.sh sift10M {1,2,4} d1_g*` EQUAL | **done 2026-11-21** (SP1; det with denser `d2_g*`) |
| P2 | Step 6 on the scheduler (§4.2) + det EQUAL | todo |
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

