# bucket2 spot-instance loop — log (`/optimize-bucket2-spot`, branch `spot/bucket2`, worktree `/home/lanlu/microBenchmark_spot`)

Design: `bucketDemo/SPOT_DESIGN.md` (user decisions 2026-11-20/21). Skill: `.claude/skills/optimize-bucket2-spot/SKILL.md`.

## Implementation plan (phases — do them in order, one per iteration)

| phase | what | status |
|---|---|---|
| P0 | infrastructure: `src/spot_sched.hpp` (Client + SegmentScheduler), unit test (`src/tests/test_spot_sched.cpp`, TSan clean), cross-process e2e (`src/tests/test_spot_e2e.cpp`), `spot_sim.py` (gen / run / show), run_pipeline `SPOT=1` hooks, summarize / judge `_spot<TAG>` keys, `spot_build.sh`, `spot_det.sh`, traces `t1_g{1,2,4}`, `d1_g{1,2,4}`; `Client::init()` at the top of bucket2 / prune_windowed `main()` | **done 2026-11-21** (set up interactively, see commit) |
| P1 | Step 4 on the scheduler (SPOT_DESIGN §4.1) + `spot_det.sh sift10M {1,2,4} d1_g*` EQUAL | todo |
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
- **O8 reap latency** — acquire() is the only place that reaps; reap from the Client monitor thread on Down so idle workers wake immediately.

## Awaiting approval

(none)

## Tried & rejected

(none)

## Iterations
