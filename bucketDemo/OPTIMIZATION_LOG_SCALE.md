# bucket2 multi-GPU scalability log (`/optimize-bucket2-scale`)

Goal: SIFT100M build faster with 2 GPUs (always) and 4 GPUs (when all four are free), search quality
unchanged, single-GPU path not slower. Skill: `.claude/skills/optimize-bucket2-scale/SKILL.md`.
Best: `.claude/skills/optimize-bucket2-scale/state/best_scale.json` (judge keys `sift100M_auto_g2`, `sift100M_auto_g4`).

## Current best

_Not recorded yet — the first `/optimize-bucket2-scale` invocation is the baseline (HEAD, MGPU=2 ×3, MGPU=4 ×3 if free)._

Single-GPU reference (`optimize-bucket2/state/best.json`, #96): SIFT100M 429.56 s — Step 4 101.6, Step 6 159.8,
Wait 33.8, Step 4.5 14.9, prune 55.8, optimize 39.5; recall@10 L 32 / 64 / 128 = 0.7246 / 0.8306 / 0.9097.

Pre-loop measurements (OPTIMIZATION_LOG.md #97, −r 1): 2 GPUs 317.70 s (Step 4 62.5, Step 6 109.9, Wait 25.2,
prune 44.7, optimize 35.8); 4 GPUs 347.79 s (Step 4 42.3, Step 6 121.8, Wait 50.7, prune 45.8, optimize 46.5).

## Ideas backlog (largest multi-GPU gap first)

- **S1 — Step 6 arrival appender.** One shared `ChunkedKnnAccumulator`; #97 (global `append_mtx`): one thread
  writes ~35 GB / round, the other GPUs' threads wait 33–50 s on the lock. Per-GPU write lanes were tried and
  rejected (see **Tried & rejected**, `patches/scale0_lanes.patch`): parallel writes only fill the kernel
  dirty-page budget faster; the disk (~0.29 GB/s) is the limit, not the lock. Real fix = **fewer bytes to disk**:
  keep arrival records in host memory when they fit (size-gated, disk fallback for 1B), or merge per round in
  memory; or overlap the flush with the next round's Steps 1–4 (rate-shape writes so the final Wait shrinks).
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

