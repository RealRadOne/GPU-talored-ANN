# bucket2 multi-GPU scalability log (`/optimize-bucket2-scale`)

Goal: SIFT100M build faster with 2 GPUs (always) and 4 GPUs (when all four are free), search quality
unchanged, single-GPU path not slower. Skill: `.claude/skills/optimize-bucket2-scale/SKILL.md`.
Best: `.claude/skills/optimize-bucket2-scale/state/best_scale.json` (judge keys `sift100M_auto_g2`, `sift100M_auto_g4`).

## Current best

**G = 2: 307.50 s** (S1, 2026-11-19, GPUs 0,1, reps 315.50 / 307.50 / 307.01) — bucket2 229.49 (Step 4 62.3,
Step 6 99.4, Wait 25.1, Step 4.5 15.2), prune 42.9, optimize 37.4; recall@10 L 32 / 64 / 128 = 0.7254 / 0.8351 /
0.9115, 0 defects; bucket2 peak RSS 80.4 GB. Run `runs/scale1/sift100M_auto_g2`. Baseline was 465.78 s (`runs/scale0_base`).
**G = 4:** not measured yet (GPUs 2,3 held by wangcy07 at baseline time).

Single-GPU reference (`optimize-bucket2/state/best.json`, #96): SIFT100M 429.56 s — Step 4 101.6, Step 6 159.8,
Wait 33.8, Step 4.5 14.9, prune 55.8, optimize 39.5; recall@10 L 32 / 64 / 128 = 0.7246 / 0.8306 / 0.9097.

Pre-loop measurements (OPTIMIZATION_LOG.md #97, −r 1): 2 GPUs 317.70 s (Step 4 62.5, Step 6 109.9, Wait 25.2,
prune 44.7, optimize 35.8); 4 GPUs 347.79 s (Step 4 42.3, Step 6 121.8, Wait 50.7, prune 45.8, optimize 46.5).

## Ideas backlog (largest multi-GPU gap first)

- **S1 — done (S1, accepted):** arrival records in host memory up to `--arrival-mem-mb` (51.2 GB). Left over:
  round 2 still spills ~16.7 GB (budget full with round 1's 37.3 GB kept for the deferred final merge) →
  **S1b**: round-2 Step 6 46 s vs round-1 53 s; try a larger budget only if RAM allows (shared machine), or
  give round 2 priority by spilling round-1 records after the deferred kth pass (they are read only once more).
  Also still one appender thread (`append_mtx`) — now cheap (pointer swap per 256 KB), re-check its wait share.
- **S2 — optimize_chunked source-row split.** `GpuReverseMerge` re-uploads the whole forward graph per GPU per
  dst segment (H2D 3.8 / 7.9 / 12.1 s at G = 1 / 2 / 4). Each GPU uploads only its source rows once, scatters
  edges to the dst owner over NVLink (peer copies), owner selects + merges. Byte-identical to CPU (top-K is
  decomposable, keys distinct). Source: cuVS `mg/snmg.cuh` sharded build + `sharded_search_with_direct_merge`.
- **S3 — Step 6 load balance.** Cuts by point count; per-GPU loop times 31 / 41 / 50 / 59 s at G = 4 (part of it
  is lock waiting). Cut by estimated cost (bucket size × pool size) instead.
- **S4 — per-GPU setup.** `gpu_buffer_alloc` 0.26–1.26 s per GPU per round, `future_access` 0.16–0.68 s;
  Step 4 per-GPU index setup + convert_upload doubling at G = 4 (shared H2D, NUMA). Pin host threads / staging
  to the GPU's NUMA node; allocate once for both rounds.
- **S5 — prune multi-GPU.** 55.8 → 44.7 / 45.8 s: per-GPU `load` 4.9–8.8 s and `d2h+sink` 15.8–22 s dominate;
  the sink (forward graph write) is shared. Look at per-GPU sinks / output ranges.
- **S6 — Wait (background merge) grows with G** (25 → 51 s at G = 4): more dirty data to flush when the merge
  runs. Tied to S1.

## Awaiting approval

_(none)_

## Tried & rejected

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

