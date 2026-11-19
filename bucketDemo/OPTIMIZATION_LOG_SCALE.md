# bucket2 multi-GPU scalability log (`/optimize-bucket2-scale`)

Goal: SIFT100M build faster with 2 GPUs (always) and 4 GPUs (when all four are free), search quality
unchanged, single-GPU path not slower. Skill: `.claude/skills/optimize-bucket2-scale/SKILL.md`.
Best: `.claude/skills/optimize-bucket2-scale/state/best_scale.json` (judge keys `sift100M_auto_g2`, `sift100M_auto_g4`).

## Current best

**G = 2: 307.50 s** (S1, 2026-11-19, GPUs 0,1, reps 315.50 / 307.50 / 307.01) — bucket2 229.49 (Step 4 62.3,
Step 6 99.4, Wait 25.1, Step 4.5 15.2), prune 42.9, optimize 37.4; recall@10 L 32 / 64 / 128 = 0.7254 / 0.8351 /
0.9115, 0 defects; bucket2 peak RSS 80.4 GB. Run `runs/scale1/sift100M_auto_g2`. Baseline was 465.78 s (`runs/scale0_base`).
**G = 4: 259.46 s** (S2, 2026-11-19, GPUs 0–3, reps 258.90 / 259.46 / 259.76) — bucket2 171.94 (Step 4 42.4,
Step 6 63.3, Wait 25.9), prune 50.0, optimize 37.6; recall@10 0.7257 / 0.8324 / 0.9105; peak RSS 83.3 GB.
Run `runs/scale2/sift100M_auto_g4`. (S1 HEAD G = 4: 294.57 s, `runs/scale1m/sift100M_auto_g4`.)
**msTuring100M G = 2 (milestone, S1 HEAD, ×1): 607.54 s** (1 GPU best 741.0) — bucket2 453.8 (Step 4 151.3, Step 6
127.3, Wait 29.2), prune 109.3, optimize 44.4; recall@10 L 32 / 64 / 128 = 0.6108 / 0.7186 / 0.8024 (1 GPU 0.6117 /
0.7151 / 0.8020); peak RSS 102.7 GB. Run `runs/scale1m/msTuring100M_auto_g2`.

Single-GPU reference (`optimize-bucket2/state/best.json`, #96): SIFT100M 429.56 s — Step 4 101.6, Step 6 159.8,
Wait 33.8, Step 4.5 14.9, prune 55.8, optimize 39.5; recall@10 L 32 / 64 / 128 = 0.7246 / 0.8306 / 0.9097.

Pre-loop measurements (OPTIMIZATION_LOG.md #97, −r 1): 2 GPUs 317.70 s (Step 4 62.5, Step 6 109.9, Wait 25.2,
prune 44.7, optimize 35.8); 4 GPUs 347.79 s (Step 4 42.3, Step 6 121.8, Wait 50.7, prune 45.8, optimize 46.5).

## Ideas backlog (largest multi-GPU gap first)

Item names below are backlog labels, not iteration numbers (iterations are logged as `### S<N>` under Iterations).

- **S1 — done** (arrival records in RAM). S1b (bigger budget, no round-2 spill) tried as S5: no gain (rejected).
- **S2 — done** (per-GPU arrival lanes).
- **S3 / S4 — load balance (Step 6 cost cut + optimize segment cut):** real per-stage gains, rejected because the disk-bound
  Wait absorbs them at G = 2 (see Tried & rejected). Re-try `bucketDemo/patches/scale4_balance.patch` after S1b, and at G = 4.
- **S8 — per-GPU setup.** `gpu_buffer_alloc` 0.2–0.9 s per GPU per round, `future_access` 0.16–0.68 s; Step 4
  per-GPU convert_upload doubling at G = 4 (shared H2D, NUMA). Pin host threads / staging to the GPU's NUMA node;
  allocate once for both rounds.
- **S5 — prune multi-GPU.** 55.8 → 42.6 (G = 2) / 50.0 s (G = 4): per-GPU `load` and `d2h+sink` dominate; the sink
  (forward graph write, 12.8 GB, disk-bound) is shared. Look at per-GPU sinks / output ranges.
- **S6 — Wait (final merge) 25–26 s at G = 2 and 4:** reads ~37 GB in RAM + 16.7 GB spilled + writes vector_knn
  (25.6 GB, disk-bound). Tied to S1b; could stream the final rows straight to prune instead of the file (bigger change).
- **S9 — compact vector_knn file (measured, not implemented).** sift10M vector_knn: varint(nbr − gid) 3.03 B/id, sorted deltas + order
  permutation 2.67 B/id vs 4 B fixed → 25.6 GB → ~17–19 GB at 100M. Needs a block-indexed format read by prune_windowed (range loads)
  and the eval script (knn_similarity); saving ~7 GB of a dirty-throttled write. Pattern from S3–S6: at G = 2 savings in one stage
  move to another (disk / memory bandwidth) — only fewer total bytes help.
- **S7 — optimize_chunked source-row split** (was S2). `GpuReverseMerge` re-uploads the whole forward graph per GPU
  per dst segment; but at G = 2 Phase 2 is 35.8 s of which GPU work ~20.7 s and the rest the in-process translate +
  12.8 GB final write (disk-bound) → at most a few seconds. Source: cuVS `mg/snmg.cuh` sharded build.

## Awaiting approval

_(none)_

## Tried & rejected

- **S6 (2026-11-19) — data_reordered (12.8 GB) written in the background in blob mode** (`write_reordered_async`, join before round-2
  Step 4 and at exit) instead of synchronously in Step 4.5. Step 4.5 15.3 → 4.6 s, but round-1 Step 6 99 → 110.6 s (the background
  write competes for memory bandwidth / page cache while Step 6 reads the host blob) → G = 2 323.11 s (−5.1 % vs best), same-period
  HEAD ref 314.95 s (bucket2 231.2 vs 235.8). Quality unchanged (file intact). Guard: sift1M_large −3.5 % (noise-level; Step 4.5 is
  0.04 s there), sift10M +2.2 %. det 2 EQUAL. G = 4 skipped. Patch `bucketDemo/patches/scale6_async_reorder_write.patch`.
- **S5 (2026-11-19) — S1b: arrival budget 51.2 → 75 GB** (`dataset_env.py` ARRIVAL_MEM_MB = 3/8 HOST_BUDGET) so round 2 does not
  spill (33.0 of 33.0 GB in RAM, 70.7 GB held). G = 2 311.63 s (−1.3 % vs best; reps 311.3 / 316.1 / 311.6), same-period ref with
  51.2 GB: bucket2 230.95 vs 229.61 s → no gain; Step 6 99.2 s and Wait 27.6 s unchanged; peak RSS 80.4 → 92.8 GB. **Finding:** the
  final merge (Wait) is ~7 s merge CPU + ~15 s `pwrite` of vector_knn (25.6 GB) throttled by the dirty-page limit — the round-2
  spill is not what it waits for. det 2 EQUAL, guard ACCEPT, G = 4 skipped (wangcy07). Patch `bucketDemo/patches/scale5_arrival_budget_75g.patch`.
- **S4 (2026-11-19) — multi-GPU load balance = S3 cost cut (κ = 0.3) + optimize_chunked segment cut.** The segment fix:
  dst segments were capped at ceil(E/G) edges and stopped one row short → a tiny extra segment went to GPU 0, which re-uploaded
  the whole forward graph for a few rows (G = 2 passes 3 → 2, fill 9.3 → 5.3 s; byte-identical). det 2 EQUAL, guard ACCEPT.
  Judged G = 2: 304.42 s (+1.0 % vs best 307.50; reps 304.4 / 299.4 / 365.0), same-period HEAD ref 325.6 s. Interleaved A/B
  (`runs/scale4/ab.out`, HEAD / S4 total s): 311.0 / 312.1, 361.1 (optimize 82.7, disk) / 304.5, 306.5 / 302.1 → S4 wins 2 of 3,
  median −2.1 % (bucket2 −3.6 %). **Why the gain does not reach the total:** Step 6 is ~14 s faster every time, but Wait grows by
  about the same (A1 38.1 vs B1 25.4 s): finishing Step 6 earlier makes the final merge's 25.6 GB vector_knn write hit the
  not-yet-flushed dirty pages (data_reordered 12.8 GB + round-2 arrival spill 16.7 GB) → G = 2 end-to-end is bound by bytes
  written. optimize's GPU saving is hidden the same way (its tail is the 12.8 GB final write). G = 4 skipped (wangcy07).
  Patch `bucketDemo/patches/scale4_balance.patch` (bucket.cu + optimize_chunked.cu). Re-try after fewer bytes go to disk (S1b).
- **S3 (2026-11-19) — Step 6 cut by estimated cost** (s·pool over the nprobe-expanded graph + κ·mean, instead of point
  count). Balances the GPUs: G = 2 round-1 loops 50.2 / 34.6 s (points) → 42.4 / 42.6 s (κ = 0.3); Step 6 99.3 → 85.1 s.
  κ sweep (same period, timing only, `runs/scale3/ab.out`): HEAD 312.9, κ = 0.5 303.0, κ = 0 299.0 s (κ = 0 overshoots,
  0.5 undershoots). Judged runs: κ = 0.5 308.22 s (−0.2 %, GPUs 2,3, `runs/scale3`), κ = 0.3 **302.20 s (+1.7 % vs
  best 307.50)** (`runs/scale3k`), same-period HEAD reference 311.3 s (−2.9 % total, bucket2 232.0 → 215.1). Below the
  3 % bar because the untouched stages were slower in these runs (prune 47–48 vs 42.6, optimize 40 vs 34.9 s). det 2
  EQUAL (κ 0.5 and 0.3); single-GPU guard ACCEPT (sift10M +3.5 %, sift1M +3.1 %). G = 4 skipped twice (wangcy07).
  Patch `bucketDemo/patches/scale3_cost_cut_k03.patch` (κ = 0.3 default; env `BUCKET2_S6_CUT=points`, `BUCKET2_S6_CUT_FIXED`).
- **S0 (2026-11-19) — Step 6 per-GPU arrival lanes** (`ChunkedKnnAccumulator::Lane`, lock per chunk file instead of global `append_mtx`). SIFT100M G = 2 A/B, same period, alternating (`runs/mgpu/ab_lanes.out`), bucket2 s: A1 lanes 334.39 (Step 6 112.6 + 72.1, Wait 45.5), B1 lock 289.34 (53.5 + 103.8, Wait 25.9), A2 lanes 346.79 (77.5 + 122.6, Wait 45.3), B2 lock 348.17 (75.0 + 144.1, Wait 27.2). Lanes are not faster (mean 340.6 vs 318.8; noisy disk, B2 slow too), and Wait is consistently +18 s: the parallel writes fill the dirty-page budget (`nr_dirty` ~33 GB vs `nr_dirty_threshold` ~38.7 GB) so Step 6 turns disk-bound (/ ~0.29 GB/s) instead of overlapping the flush with the next round. Patch `bucketDemo/patches/scale0_lanes.patch`.

## Iterations

### Baseline (2026-11-19) — HEAD 6192de7, SIFT100M MGPU=2 ×3; G=4 skipped (GPU 2,3 used by wangcy07)

| stage | 1 GPU (best #96) | 2 GPUs | speedup |
|---|---|---|---|
| Step 4 | 101.6 | 63.1 | 1.61x |
| Step 6 | 159.8 | 233.1 | 0.69x |
| Wait | 33.8 | 29.3 | 1.16x |
| prune | 55.8 | 48.5 | 1.15x |
| optimize | 39.5 | 43.9 | 0.90x |
| total | 429.6 | 465.8 | 0.92x |

Much slower than #97's single run (317.7 s, 2026-11-16) and the same-day lock A/B runs (289.3 / 348.2): Step 6
rounds 134.8 + 110.2 / 82.9 + 150.2 / 77.1 + 136.1 s, per-GPU `function total` up to 149 s in round 2 (vs ~80 s per
round at 1 GPU). Step 6 is disk-bound (arrival records ~35 GB / round through one appender, / ~0.29 GB/s, 94 % full);
no other heavy writer or memory user seen after the run (`nr_dirty_threshold` 46.7 GB). **Noise is ±5–10 % at this
scale right now → confirm any accept candidate with an alternating A/B against HEAD (`mgpu_run.sh`, `build_wip/`).**
Quality same as 1 GPU (0.7276 / 0.8348 / 0.9112 vs 0.7246 / 0.8306 / 0.9097). Side fix: `summarize.py` `gpus`
now lists every device of the reps (was the first one only).

### S1 — Step 6 arrival records in host memory (size-gated, disk fallback)

- **Idea:** out-of-core Step 6 wrote every round's arrival records (SIFT100M ~37 + 33 GB) through the page cache to
  `/` (~0.29 GB/s); round 1's files stay until the deferred final merge, so dirty pages hit `nr_dirty_threshold` and
  the single appender (all GPUs' scatter threads behind `append_mtx`) was throttled by the kernel. Now a full
  256 KB write buffer is handed to `ChunkedKnnAccumulator::mem_arrival[iter % 3][chunk]` (pointer move, no copy)
  while the held bytes fit `--arrival-mem-mb`; past the budget the flush writes the chunk file as before.
  `merge_iteration` decodes file part + memory blocks; frees blocks unless kept for the deferred final merge.
- **Source:** TAGORE `large_index.cu` keeps sub-graph results in host buffers and only spills what does not fit
  (GPU–CPU–disk pipeline); our code today: every arrival record goes to `chunk_<c>_iter<t>.bin`.
- **Class:** implementation (merge result independent of where / in which order records are stored). Same for 1 and
  G GPUs. Scripts: `dataset_env.py` `ARRIVAL_MEM_MB` = HOST_BUDGET_MB / 4 (51200); `run_pipeline.sh` / `mgpu_run.sh`
  pass `--arrival-mem-mb` only when > 0 (`ARRIVAL_MEM_MB=0` = old behaviour, also for pre-S1 binaries in `build_wip/`).
- **Correctness:** deterministic sift10M G = 1 with `ARRIVAL_MEM_MB` 0 / 300 (partial spill, 0.31 GB in RAM) / default
  (6.8 GB in RAM): `vector_knn`, forward graph, final graph byte-identical; no leftover chunk files.
  `det_check.sh sift10M 2`: **EQUAL** (Step 4 checksums equal).
- **Single-GPU guard** (`--quick --no-mem`): ACCEPT — sift10M_auto 35.52 s (+3.3 %), sift1M_large 7.29 s (+0.7 %);
  sift10M bucket2 peak RSS 9.65 → 12.94 GB (memory guard off, recorded).
- **SIFT100M G = 2** (judge vs baseline): 307.50 s vs 465.78 (**+34.0 %**, ACCEPT); Step 6 233.1 → 99.4 s (rounds
  53.0 / 45.0 s; round 1 37.3 GB all in RAM, round 2 16.2 of 33.0 GB in RAM). Quality 0.7254 / 0.8351 / 0.9115.
  Peak RSS 33.6 → 80.4 GB. **G = 4 skipped** (GPU 3 used by wangcy07).
- **Same-period A/B** (`runs/scale1/ab.out`, timing only): HEAD bucket2 289.78 / total 371.95 s vs S1 229.41 / 309.63 s
  → **−20.8 % bucket2, −16.8 % total** (the +34 % vs best is inflated by the slow baseline period).

| stage | 1 GPU (best #96) | 2 GPUs (S1) | speedup |
|---|---|---|---|
| Step 4 | 101.6 | 62.3 | 1.63x |
| Step 6 | 159.8 | 99.4 | 1.61x |
| Wait | 33.8 | 25.1 | 1.35x |
| prune | 55.8 | 42.9 | 1.30x |
| optimize | 39.5 | 37.4 | 1.05x |
| total | 429.6 | 307.5 | 1.40x |

### S2 — per-GPU arrival lanes (lock only per chunk on flush)

- **Idea:** all GPUs' scatter threads appended through one `append_mtx`. After S1, G = 4 showed 3 of 4 GPUs with
  `scatter_pending` 19.6–25.9 s / round (waiting for the appender) and only 4–8 s `stream_sync`; Step 6 at G = 4
  (100.8 s) was no faster than G = 2 (99.4 s). Now each GPU has its own lane of per-chunk write buffers (no lock on
  append); a full buffer goes to `mem_arrival` / the chunk file under `chunk_mtx[c]` only; memory budget claimed
  with a CAS loop. Same idea as the rejected scale0 lanes — **what is different:** after S1 the flush hands the
  buffer to RAM instead of the page cache, so parallel flushes no longer hit the dirty-page throttle.
- **Source:** cuVS `mg/snmg.cuh` (one rank = one device with its own resources, results merged afterwards); TAGORE
  per-device threads with their own host buffers. Class: implementation (record order only; merge order-independent).
- **Correctness:** `det_check.sh sift10M 2` **EQUAL**, `det_check.sh sift10M 4` **EQUAL**.
- **Single-GPU guard** (`--quick --no-mem`): ACCEPT — sift10M_auto 36.15 s (+1.6 %), sift1M_large 7.49 s (−2.0 %).
- **SIFT100M:** G = 2 309.42 s vs 307.50 (−0.6 %, neutral: `scatter_pending` 17 → 5 s but GPU 0's loop is now
  GPU-bound, 50.2 vs 34.5 s → S3); **G = 4 259.46 s vs 294.57 (+11.9 %)**, Step 6 100.8 → 63.3 s (rounds 32.9 / 28.2,
  per-GPU loops 18.6–30.3 s). Untouched stages match the G = 4 best (Step 4 42.4 vs 41.8, optimize 37.6 vs 38.0), so no
  separate A/B. Quality G = 2 0.7275 / 0.8326 / 0.9109, G = 4 0.7257 / 0.8324 / 0.9105, 0 defects.

| stage | 1 GPU (best #96) | 2 GPUs (S2) | 4 GPUs (S2) |
|---|---|---|---|
| Step 4 | 101.6 | 61.1 (1.66x) | 42.4 (2.39x) |
| Step 6 | 159.8 | 99.3 (1.61x) | 63.3 (2.52x) |
| Wait | 33.8 | 25.8 (1.31x) | 25.9 (1.30x) |
| prune | 55.8 | 42.6 (1.31x) | 50.0 (1.12x) |
| optimize | 39.5 | 34.9 (1.13x) | 37.6 (1.05x) |
| total | 429.6 | 309.4 (1.39x) | 259.5 (1.66x) |

