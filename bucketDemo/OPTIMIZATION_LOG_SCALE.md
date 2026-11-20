# bucket2 multi-GPU scalability log (`/optimize-bucket2-scale`)

Goal: SIFT100M build faster with 2 GPUs (always) and 4 GPUs (when all four are free), search quality
unchanged, single-GPU path not slower. Skill: `.claude/skills/optimize-bucket2-scale/SKILL.md`.
Best: `.claude/skills/optimize-bucket2-scale/state/best_scale.json` (judge keys `sift100M_auto_g2`, `sift100M_auto_g4`).

**GPU agreement (user, 2026-11-19):** wangcy07 uses GPUs 0,1, ours are 2,3 (0,1 usable while free). Selection order
`2 3 0 1`; 2-GPU runs prefer the pair 2,3. G = 4 only when all four are free (never blocks a verdict); the G = 4 best is kept
as reference.

## Current best

**G = 2: 268.24 s** (S19 = S17 + S18, 2026-11-20, GPUs 2,3, `runs/scale19/sift100M_auto_g2`) — bucket2 197.0 (Step 4 53.7,
Step 4.5 4.7, Step 6 83.6, Wait 26.0), prune 42.1, optimize 28.9; recall@10 0.7272 / 0.8343 / 0.9114. Speedup vs 1 GPU 1.60×.
Previous **G = 2: 285.33 s** (S16 run, 2026-11-19, GPUs 2,3, `runs/scale16/sift100M_auto_g2`; S16 itself is inactive at G = 2 — faster
period) — bucket2 215.1, prune 41.6, optimize 28.7; recall@10 0.7241 / 0.8322 / 0.9108. Previous **290.97 s** (S10, GPUs 2,3, `runs/scale10/sift100M_auto_g2`) — bucket2 214.1 (Step 4 61.4, Step 6 84.9,
Wait 26.4), prune 46.3, optimize 30.0; recall@10 0.7267 / 0.8340 / 0.9109 (same-period HEAD ref 293.27 s — faster disk period).
Previous: **304.42 s** (S4 re-judged 2026-11-19, "accepted (compute gain, disk-bound)"; reps 304.4 / 299.4 / 365.0, GPUs 0,1;
`runs/scale4/sift100M_auto_g2`) — Step 6 85.2, Wait 33.5, optimize 35.0; recall@10 0.7272 / 0.8334 / 0.9120. Previous best
307.50 (S1). S9 (optimize src-split) accepted under the compute-gain rule without a new best (311.50 s, slow-disk period). Compute metrics for the fast-disk re-check are in the S3 / S4 re-judge entry below.
**G = 4: 222.82 s** (S19, 2026-11-20, GPUs 0–3, `runs/scale19g4/sift100M_auto_g4`) — bucket2 145.1 (Step 4 34.0, Step 4.5 6.3,
Step 6 51.7, Wait 26.3), prune 49.0, optimize 27.2; recall@10 0.7251 / 0.8332 / 0.9100. Speedup vs 1 GPU 1.93×.
Previous **G = 4: 237.60 s** (S16, 2026-11-19, GPUs 2,3,0,1, `runs/scale16/sift100M_auto_g4`) — bucket2 161.0, prune 49.4, optimize 26.6;
recall@10 0.7261 / 0.8338 / 0.9104. Previous **G = 4: 238.74 s** (S14 measurement of HEAD incl. S9–S11, 2026-11-19, GPUs 0–3, `runs/scale14/sift100M_auto_g4`) — bucket2
162.8 (Step 4 43.0, Step 6 52.1, Wait 25.9), prune 47.3, optimize 28.6; recall@10 0.7245 / 0.8326 / 0.9111.
Previous **G = 4: 259.46 s** (S2, 2026-11-19, GPUs 0–3, reps 258.90 / 259.46 / 259.76) — bucket2 171.94 (Step 4 42.4,
Step 6 63.3, Wait 25.9), prune 50.0, optimize 37.6; recall@10 0.7257 / 0.8324 / 0.9105; peak RSS 83.3 GB.
Run `runs/scale2/sift100M_auto_g4`. (S1 HEAD G = 4: 294.57 s, `runs/scale1m/sift100M_auto_g4`.)
**msTuring100M G = 2 (milestone, after S11, ×1): 561.61 s** (2026-11-19, `runs/scale12/msTuring100M_auto_g2`) — bucket2 434.8,
prune 95.2, optimize 31.6 (was 44.4); recall@10 L 32 / 64 / 128 = 0.6135 / 0.7174 / 0.8023; peak RSS 103.6 GB.
Previous milestone **(S1 HEAD, ×1): 607.54 s** (1 GPU best 741.0) — bucket2 453.8 (Step 4 151.3, Step 6
127.3, Wait 29.2), prune 109.3, optimize 44.4; recall@10 L 32 / 64 / 128 = 0.6108 / 0.7186 / 0.8024 (1 GPU 0.6117 /
0.7151 / 0.8020); peak RSS 102.7 GB. Run `runs/scale1m/msTuring100M_auto_g2`.

Single-GPU reference (`optimize-bucket2/state/best.json`, #96): SIFT100M 429.56 s — Step 4 101.6, Step 6 159.8,
Wait 33.8, Step 4.5 14.9, prune 55.8, optimize 39.5; recall@10 L 32 / 64 / 128 = 0.7246 / 0.8306 / 0.9097.

Pre-loop measurements (OPTIMIZATION_LOG.md #97, −r 1): 2 GPUs 317.70 s (Step 4 62.5, Step 6 109.9, Wait 25.2,
prune 44.7, optimize 35.8); 4 GPUs 347.79 s (Step 4 42.3, Step 6 121.8, Wait 50.7, prune 45.8, optimize 46.5).

## Ideas backlog (largest multi-GPU gap first)

Item names below are backlog labels, not iteration numbers (iterations are logged as `### S<N>` under Iterations).

- **S10 — done (partly: warm-up + staging threads; GPU-side count reduction not done).** optimize count phase (4.2 s at G = 2): each GPU D2Hs a full N×4 B count array and the host sums G arrays +
  serial prefix over N; reduce on GPU (peer add into GPU 0, or per-GPU dst-range counts) + parallel prefix.
- **S14 — analysed (S12 entry): shared host link, not fixable at G = 2.** ~~GPU 0 H2D 2× slower than GPU 1 in optimize count / fill~~ (same bytes; NUMA placement of `h_fwd_all` pages read by
  `parallel_pread_file`, or PCIe topology). Check `numactl`/`move_pages` of the first half; interleave or bind the pread threads.
- **S12 — checked 2026-11-19:** prune already caps per-GPU load threads (8); Step 6 scatter uses explicit threads; Step 4
  read is prefetched (float path). No oversubscription found besides optimize staging (S10). Was: per-GPU OpenMP oversubscription elsewhere: prune_windowed per-GPU load / sink and bucket2 per-GPU host loops
  run `omp parallel` from G threads at once (as optimize staging did before S10); cap team sizes per GPU thread.
- **S15 (backlog name) — done as S16 (accepted).** ~~keep each GPU's source rows resident between count and fill~~
- **S11 — done as S15: S8 re-judged at G = 4, rejected.**

**User priority (2026-11-19): after S7 (compact vector_knn), do these two next, in this order. Both implementation-level;
outputs must stay byte-identical to today's.**

- **P1 — prune multi-GPU: per-GPU parallel plan / load + per-GPU direct output.** Today at G = 4 the per-GPU plan
  (0.7 / 1.4 / 2.9 / 5.6 s) and load (4.4 / 5.8 / 9.0 / 13.1 s) queue up one after another, and all rows go through one shared
  sink (writer-thread time summed 67.8 s at G = 4 vs 30.6 s at G = 2). Make each GPU plan / load concurrently and `pwrite` its
  own rows straight to their offsets in the output file (no shared sink).
- **P2 — done (S9, accepted compute gain).** ~~optimize_chunked source-row split~~ (was S7 below). Each GPU uploads only its own source rows once and sends the
  edges to the GPU that owns the destination rows over NVLink (peer copies), instead of every GPU re-uploading the whole
  forward graph on every pass (G = 4: passes 5, h2d 11.4 s vs G = 2: passes 3, 6.6 s). Source: cuVS `mg/snmg.cuh`.

- **S1 — done** (arrival records in RAM). S1b (bigger budget, no round-2 spill) tried as S5: no gain (rejected).
- **S2 — done** (per-GPU arrival lanes).
- **S3 / S4 — done:** accepted on re-judge under the compute-gain rule (2026-11-19).
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

- **User decisions 2026-11-20 (later):** (a) memory: in-memory fast paths only within one user-given host budget (reuse / unify
  HOST_BUDGET_MB → --host-budget-mb / --bucket-mem-mb / --arrival-mem-mb / --cache-mb), never "if it fits / RAM is free";
  otherwise the existing disk / out-of-core fallback; default peak memory must not rise noticeably; judge memory guard ON again.
  (b) correctness: byte identity (paths, G vs 1) no longer required; gate = search quality (recall@10 per L within judge gates, QPS
  not lower); fast path and fallback each get ≥ 1 build with search eval; det_check is optional. Everything must scale to
  1B / degree 512. Order after S19: 1 optimize id translation + final write → 2 prune parallel plan / load (S8 part without the
  async writer, compute rule) → 3 Step 4 (serial init, round-2 upload OpenMP threads, read waits) → 4 Step 6 load balance at
  G = 4 → A incremental final merge (keep the disk spill path). B not now.

- **User decision 2026-11-20:** Steps 1, 2, 3, 4.5, 5 stay on one GPU but may now be CPU-parallelised or moved to background
  threads overlapping GPU stages (identical output, det EQUAL, quality unchanged, 1-GPU not slower). SKILL.md invariant 2
  updated. First target: Step 4.5 (mind S6: background data_reordered write vs Step 6).

- **S13 — approved 2026-11-19, tried, rejected (see Tried & rejected).**
## Tried & rejected

- **S19 — accepted (see Iterations), was pending:** det 2 EQUAL (plain + forced S18 path), guard ACCEPT; the SIFT100M
  part did not start: `/` 198 GB < 200 GB before the same-period ref (chain order: ref first, then G = 2 ×3). Patch
  `bucketDemo/patches/scale19_s17_s18_combined.patch`; resume (`runs/scale19/chain.sh` minus the done parts) when disk ≥ 200 GB.
- **S18 (2026-11-20, first idea under the relaxed single-GPU-stage rule) — Step 4.5: data_reordered written in the background
  with bounded dirty pages** (64 MB pieces, `sync_file_range` write + wait 4 pieces back → ≤ ~256 MB dirty; `BUCKET2_REORDER_BG`),
  round 1 Step 6 no longer joins (blob mode), round 2 Step 4 + bucket write take `r1_blob` as host_raw (no file read; blob
  kept until then). Test switch `BUCKET2_HOST_RAW_BUDGET_MB` forces this path on SIFT10M: data_reordered / vector_knn / forward /
  final graph byte-identical to the synchronous write; det 2 EQUAL (plain + forced path), guard ACCEPT. SIFT100M G = 2 ×3:
  **280.01 s (−1.9 % vs 285.33)**; Step 4.5 15.5 → 4.5 s, Step 6 unchanged 84.5 s (the S6 penalty is gone), Step 4 58.0 s,
  bucket2 204.0 vs ~215 s, but prune 47.0 s (vs 41.6 — disk period or page-cache effect, unresolved: the same-period HEAD ref
  was not run because `/` dropped to 197 GB < 200 GB). Below the 3 % gate, I/O overlap → rejected for now; **re-judge with a
  same-period ref when disk space allows** (candidate together with S17). Patch `bucketDemo/patches/scale18_bg_reorder_write.patch`.
- **S17 (2026-11-20) — Step 4 prefetch path: batches of Nv / 4** (`BUCKET2_STEP4_PF_DIV`; v1 shrank only the first batch to
  Nv / 8 → no gain, the wait moved to the 2nd full batch: G = 4 237.48 s, read still 3.5 s). v2: every prefetch batch Nv / 4 so
  each read + H2D hides behind the previous search. det 4 / det 2 EQUAL, guard ACCEPT. Per-GPU Step 4 read 3.5 → 1.0 s (G = 4),
  2.8 → 0.8 s (G = 2); Step 4 G = 4 42.0 → 36.8 s (same-period ref), G = 2 61.4 → 54.8 s. Totals: **G = 2 277.74 s (−2.7 % vs
  285.33), G = 4 236.76 s (−0.4 %; ref 237.21)** — G = 4 saving absorbed by Step 4.5 / Step 6 (+2 s each, disk). Below the 3 %
  gate and an I/O overlap (not a compute saving) → rejected. Strongest near-miss so far; worth a re-judge on the fast disk.
  Patch `bucketDemo/patches/scale17_step4_small_prefetch_batches.patch`.
- **S13 (2026-11-19, user-approved) — optimize_chunked unlinks forward_graph_reordered.bin once read** (`--keep-forward` /
  `OC_KEEP_FORWARD=1` to keep; det_check exported it). Final graph byte-identical (keep vs drop, both GPU-rev and GPU-phase-2
  paths), det 2 EQUAL, guard ACCEPT. SIFT100M G = 2 ×3: 295.62 s (+1.6 % vs best 290.97); same-period ref (keep) optimize
  28.83 s vs S13 32.45 s: the synchronous ext4 unlink of the 12.8 GB file costs ~2.9 s (Phase 1 1.39 → 4.2 s), and Phase 2
  (final-graph stream) is unchanged (27.7 vs 27.0 s) — no timed stage waits for the forward graph's flush, the final graph
  itself goes to the page cache. An async unlink would at best be neutral → rejected. Patch
  `bucketDemo/patches/scale13_unlink_forward.patch` (incl. det_check change); still useful for disk space at 1B scale.
- **S15 (2026-11-19) — S8 re-judged at G = 4** (scale8 patch on HEAD 05f2bd1, all four GPUs free): det 4 EQUAL; SIFT100M
  G = 4 ×3 239.03 s vs best 238.74 (same-period HEAD, `runs/scale14`) → no gain; prune 46.4 / 46.5 / 53.2 s vs HEAD 47.3 / 47.9 /
  45.6 — prune stays bound by the 12.8 GB forward-graph write at G = 4 too. Pure I/O overlap → not a compute gain. **S8 finally
  rejected** (chain stopped before the G = 2 part; nothing to gain there either, see S8 below).
- **S8 (2026-11-19) — P1 prune multi-GPU: per-GPU async writers + parallel plan / load** (each GPU plans / loads concurrently and
  writes its own rows through its own writer thread with `pwrite` to their offsets; byte-identical, det 2 EQUAL, guard ACCEPT
  sift10M / sift1M_large). SIFT100M G = 2 ×3: total 304.06 s (+0.1 % vs best 304.42); prune 40.9 / 40.7 / 41.0 s (best 44.64,
  same-period HEAD 42.22 → −2.8 %); prune visible write 11.8–18.5 s (HEAD 15.9). Not eligible for the compute-gain rule:
  prune `compute(parallel)` median 25.3 s = HEAD 25.3 s (pure I/O overlap). Real gain expected at G = 4 (shared-sink contention,
  writer time 67.8 s vs 30.6 s at G = 2), but G = 4 skipped (GPU 0 held by wangcy07). Patch kept for a G = 4 re-judge:
  `bucketDemo/patches/scale8_prune_async_writers.patch`.
- **S7 (2026-11-19) — compact vector_knn.bin** (zigzag varint of nbr − gid, 256-row groups + offset table; prune / optimize /
  knn_similarity read both formats; deterministic in-order offsets). Lossless: forward + final graph byte-identical with the
  compact input, 50 004 sampled rows decode identical, knn_similarity identical, det 2 EQUAL, guard ACCEPT (+1.9 / +1.2 %). At
  100M only 21.0 vs 25.6 GB (−18 %; sift10M −24 %). G = 2 (GPUs 2,3) **335.37 s (−9.1 %)**: final merge slightly faster (Wait
  24.8 s) but prune `load` 8–13 → 18–46 s per GPU — prune re-reads candidate rows ~4.3 × N and every re-read now pays the
  varint decode (prune 43 → 69 s). Not a compute saving. Patch `bucketDemo/patches/scale7_compact_vector_knn.patch`
  (includes the new `knn_compact.hpp` and the knn_similarity decoder).
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

### S3 + S4 re-judged — accepted (compute gain, disk-bound) (2026-11-19, user rule of the same day)

S3 (Step 6 cut by estimated cost, κ = 0.3) and S4 (= S3 + optimize_chunked segment cut at k·E/G) had real compute savings that
the slow disk absorbed (originally rejected, see Tried & rejected). Re-judged from their existing runs (same code as today's HEAD
+ `scale4_balance.patch`, which is now committed): (a) attributable compute drop ≥ 5 s / ≥ 10 %; (b) total not > 3 % slower than
best (S4 304.42 s, S3 302.20 s vs best 307.50 → faster; same-period HEAD refs 325.6 / 311.3 / A/B 311.0 s); (c) det 2 EQUAL,
guard ACCEPT, quality unchanged. `judge.py --update --compute-gain` → G = 2 best 307.50 → 304.42 s. G = 4 never measured
(GPUs busy). **Compute metrics for the fast-disk re-check** (SIFT100M G = 2, median of 3; loop = max over GPUs of the Step 6
`bucket loop` per round; opt GPU = count + fill + select + merge+emit of the GPU reverse merge):

| set | loop r1 | loop r2 | loop sum | Step 6 | Wait | opt GPU | fill | h2d | passes | total |
|---|---|---|---|---|---|---|---|---|---|---|
| HEAD (S2 judged, `runs/scale2`) | 50.2 | 44.2 | **94.4** | 99.3 | 25.8 | **21.9** | 8.5 | 7.4 | 3 | 309.4 |
| HEAD A/B B1–3 (`runs/scale4/ab`; optimize here was already the S4 binary) | 50.3 | 43.8 | 94.0 | 98.5 | 26.3 | 18.0 | 5.6 | 4.0 | 2 | 311.0 |
| S3 κ = 0.5 (`runs/scale3`, GPUs 2,3) | 45.2 | 39.3 | 84.7 | 89.4 | 26.0 | 24.2 | 9.4 | 8.2 | 3 | 308.2 |
| S3 κ = 0.3 (`runs/scale3k`) | 42.6 | 37.3 | **80.0** | 85.1 | 25.8 | 22.3 | 8.9 | 7.7 | 3 | 302.2 |
| S4 judged (`runs/scale4`) | 42.5 | 37.5 | **80.1** | 85.2 | 33.5 | **17.0** | 5.5 | 3.8 | 2 | 304.4 |
| S4 A/B A1–3 (`runs/scale4/ab`) | 42.6 | 37.5 | 79.9 | 85.4 | 28.6 | 18.7 | 5.7 | 4.4 | 2 | 304.5 |

- Step 6 (S3): loop sum 94.4 → 80.0 s (**−14.4 s, −15 %**); absorbed by the final merge (Wait 25.8 → 28.6–33.5 s: the earlier
  end of Step 6 meets un-flushed dirty pages when the 25.6 GB vector_knn is written).
- optimize (S4 segment cut): GPU reverse+merge 21.9 → 17.0 s (**−4.9 s, −22 %**; fill 8.5 → 5.5, h2d 7.4 → 3.8, passes 3 → 2);
  absorbed by the 12.8 GB final-graph write (optimize_chunked total ~35 s in both).
- Expected on a fast disk: up to ~19 s off the G = 2 total; more at G = 4 (per-GPU loops 18.6–30.3 s / round after S2).

### S9 — P2 optimize_chunked source-row split (NVLink peer fill) — accepted (compute gain, disk-bound) (2026-11-19)

**Idea.** `GpuReverseMerge` fill: before, every GPU streamed the *whole* forward graph over PCIe for each dst segment it
owned (GPU 2,3 share one PCIe switch). Now segments go in rounds (round k: GPU g owns segment k·G+g, all resident at once);
each GPU streams only its own source rows [N·g/G, N·(g+1)/G) once per round and `kern_rev_fill_multi` writes every edge
straight into the owning GPU's `keys` (NVLink peer store + peer `atomicAdd`; `off_rel` replicated read-only per GPU). Needs
P2P + native peer atomics between all pairs, else (and for G = 1, or `OC_GPU_REV_SPLIT=0`) the old path. Budget charges
8·(G−1) B/row for the replicas. Select / merge / emit unchanged. Source: cuVS `mg/snmg.cuh` sharded build (each rank works on
its own shard). Class: implementation (same result; det byte-identical). New test switch `OC_GPU_PHASE2=0` forces the
GpuReverseMerge path on small data (SIFT10M normally uses single-chunk GPU Phase 2 and would not exercise it).

**Correctness.** det 2 EQUAL in plain mode, with `OC_GPU_PHASE2=0` (new path, passes=2 src-split) and with
`OC_GPU_PHASE2=0 OC_GPU_REV_BUDGET_MB=300` (passes=6 → 3 rounds). Single-GPU guard ACCEPT. SIFT10M probe (G = 2):
fill 1.20 → 0.44 s, h2d 0.95 → 0.24 s. G = 4 skipped (GPU 0 held by wangcy07).

**SIFT100M G = 2 ×3 (GPUs 2,3).** Total 311.50 s (+2.3 % vs best 304.42 → within the 3 % gate; best unchanged), reps:
optimize 39.6 / 38.7 / 32.1, prune 57.3 / 53.3 / 45.6 (prune code unchanged; same-period HEAD ref 42.1 → slow-disk reps).
bucket2 219.9 (Step 4 61.6, Step 6 85.6, Wait 28.6). Recall@10 0.7254 / 0.8338 / 0.9116 (unchanged).

| compute metric (optimize GPU reverse + merge) | same-period HEAD ref (`OC_GPU_REV_SPLIT=0`) | S9 median of 3 |
|---|---|---|
| fill | 5.52 s | 4.60 s (−17 %) |
| h2d inside | 4.29 s | 2.54 s (−41 %) |
| count + fill + select | 10.78 s | 9.50 s (−11.9 %) |
| merge + emit (incl. disk-bound emit) | 6.30 s | 7.32 s |

Absorbed by: optimize Phase 2 stream write of the final graph (Phase 2 37.8 / 37.0 / 30.3 s vs ref 31.0, tracks the slow-disk
reps that also slowed the unchanged prune). Expected larger at G = 4 (each GPU used to stream the whole graph per pass).

| speedup vs 1 GPU (429.56 s) | Step 4 | Step 6 | Wait | prune | optimize | total |
|---|---|---|---|---|---|---|
| G = 2 (S9) | 1.65× | 1.87× | 1.18× | 1.05× | 1.02× | 1.38× |

### S10 — optimize: GPU context warm-up + per-GPU staging threads — accepted (2026-11-19)

**Idea.** Breakdown of the GpuReverseMerge count phase at SIFT100M G = 2 (new `[Timing] GPU reverse count detail` line):
count 4.31 s = per-GPU loop 2.25–2.63 s (H2D only ~2.4 GB/s per GPU) + D2H 0.6–0.7 s + `off` alloc ~0.65 s + prefix 0.16 s;
the first touch of GPU g ≠ 0 also paid its CUDA context (SIFT10M: setup 0.48 s). Two implementation fixes:
(1) while the forward graph is read from disk, background threads create every GPU's context and enable P2P
(`OC_GPU_WARMUP=0` disables); (2) `PinnedStaging` memcpy used one 80-thread OpenMP team *per GPU thread* → G × 80 threads
fight for cores / memory bandwidth; microbenchmark: 2 concurrent stagings reach 4.5–9 GB/s per GPU with 4–16 threads.
For G > 1 each staging now uses max(4, omp_max / (4G)) threads (G = 1 unchanged; `OC_STAGE_THREADS` overrides).
Sources: TAGORE `large_index.cu` per-device threads set up before the pipeline; cuVS `cagra_build.cuh::build_knn_graph`
pinned double buffers. Class: implementation (det byte-identical).

**Correctness.** det 2 EQUAL plain / `OC_GPU_PHASE2=0` / `+ OC_GPU_REV_BUDGET_MB=300` (3 rounds). Guard ACCEPT.
SIFT10M G = 2 probes: count 1.01 → 0.68 s (warm-up), count H2D 0.23 → 0.10 s (8 / 4 threads); G = 1: no change.
G = 4 skipped (GPU 0,1 held by wangcy07).

**SIFT100M G = 2 ×3 (GPUs 2,3).** **290.97 s, −4.4 % vs best 304.42 → ACCEPT (new best).** Reps bucket2 214.1 / 217.9 /
214.0, prune 43.6 / 46.3 / 47.0, optimize 32.1 / 29.4 / 30.0. Recall@10 0.7267 / 0.8340 / 0.9109. Note: same-period HEAD
ref (`OC_GPU_WARMUP=0 OC_STAGE_THREADS=80`) total 293.27 s (bucket2 215.2, prune 46.1, optimize 31.98) — most of the −4.4 %
vs best is the faster disk period; the attributable gain is optimize −2.0 s (−6 %):

| optimize GPU reverse + merge | same-period ref | S10 median of 3 |
|---|---|---|
| count | 4.33 s | 3.32 s (−23 %) |
| fill | 4.51 s | 3.74 s (−17 %) |
| count + fill + select | 9.68 s | 7.56 s (−22 %) |
| count H2D (slow / fast GPU) | 2.26 / 1.54 s | 1.83 / 0.88 s |

| speedup vs 1 GPU (429.56 s) | Step 4 | Step 6 | Wait | prune | optimize | total |
|---|---|---|---|---|---|---|
| G = 2 (S10) | 1.66× | 1.88× | 1.28× | 1.21× | 1.32× | 1.48× |

Follow-up: the same per-GPU-thread OpenMP oversubscription may exist in prune_windowed's per-GPU loaders and bucket2's
per-GPU Step 4 / Step 6 host loops (backlog S12).

### S11 — optimize: GPU-side count reduction + uint64 scan — accepted (compute gain, disk-bound) (2026-11-19)

**Idea.** The rest of the GpuReverseMerge count phase after S10 was host work: G × N·4 B count D2H, host sum of G arrays,
serial prefix over N and a serially zeroed 800 MB `off` vector. Now GPU g ≠ 0 keeps its partial counts in device memory, GPU 0
adds them over NVLink (`kern_add_u32`, peer reads), `cub::DeviceScan::InclusiveSum` (uint32 → uint64 via transform iterator)
produces the row offsets, and one staged D2H fills an uninitialised `off` buffer (parallel first touch). Also used for G = 1.
Fallback (no P2P, N > INT32_MAX, `OC_GPU_REV_GPUSCAN=0`) = old host path. Class: implementation (integers, identical).
Source: cuVS `mg/snmg.cuh` (reduce on one rank over NVLink instead of host).

**Correctness.** det 2 EQUAL plain / `OC_GPU_PHASE2=0` / rounds; SIFT10M G = 2 final graph byte-identical with
`OC_GPU_REV_GPUSCAN` 1 vs 0; guard ACCEPT. G = 4 skipped (GPU 0,1 held by wangcy07).

**SIFT100M G = 2 ×3.** 293.01 s (+0.7 % vs best 290.97 → within gate, best unchanged); same-period ref
(`OC_GPU_REV_GPUSCAN=0`) 289.61 s (bucket2 212.9, prune 44.5, optimize 32.15). Recall@10 0.7253 / 0.8330 / 0.9109.

| optimize count phase | same-period ref | S11 median of 3 |
|---|---|---|
| count | 3.35 s | 2.51 s (−25 %) |
| host part (D2H + sum + prefix + off alloc) → GPU reduce+scan+D2H | 0.39 + 0.05 + 0.65 + ~0.8 s | 0.26 s |

Absorbed by: optimize merge + emit (disk-bound final-graph stream, 6.0 / 8.5 / 7.4 s vs ref 6.3). Total gain is small
(~0.8 s); the rule accepts it as a real compute saving for the fast-disk re-test.

Observation for S14: count H2D of GPU 0 (device 2, first half of the rows) is consistently ~2× slower than GPU 1
(1.85–2.05 s vs 0.84–0.98 s, also in S10 / ref) for the same bytes.

### S12 — milestone msTuring100M G = 2 ×1 after S9–S11 + S14 analysis (2026-11-19)

No code change. S9–S11 changed `optimize_chunked` (also the float32 path) → milestone run: **561.61 s vs 607.54 (S1 HEAD),
−7.6 %**, optimize 44.4 → 31.6 s; recall@10 0.6135 / 0.7174 / 0.8023 vs 0.6108 / 0.7186 / 0.8024 (unchanged, ≥ own best −0.05
and CAGRA ACE −0.05); judge ACCEPT, written to best_scale.json. Speedup vs 1 GPU (741.0 s): 1.32×.

S14 (GPU 0 H2D 2× slower than GPU 1 in count / fill): not a code / NUMA effect but the shared host link of GPUs 2,3 —
microbenchmark: concurrent direct DMA to 2 + 3 aggregates 8.9 GB/s (single GPU 11.4), staged 6.4 GB/s, and which GPU gets
the larger share varies run to run; count's 2 × 6.4 GB finish in ~2.0 s = 6.4 GB/s aggregate. Only fewer bytes (keep the
source rows resident between count and fill, fits at G = 4 only) or pinned direct DMA (`cudaHostRegister` costs 1.6 s + 0.7 s
unregister for 12.8 GB, net ≈ +0) would help. Not pursued at G = 2.

### S14 — G = 4 measurement of HEAD (S9–S11) while all four GPUs were free — new G = 4 best (2026-11-19)

No code change. det 4 EQUAL (plain and `OC_GPU_PHASE2=0`, i.e. the S9 src-split GpuReverseMerge path with 4 segments / 1
round). SIFT100M G = 4 ×3: **238.74 s vs 259.46 (S2) → −8.0 %, judge ACCEPT, written to best_scale.json.** Stages: bucket2
162.8 (Step 4 43.0, Step 6 52.1 vs 63.3 at S2 — S3/S4 load balance, Wait 25.9), prune 47.3 (50.0), optimize 28.6 (37.6;
GPU reverse count 2.5 / fill 3.0 / select 0.43 / merge+emit 6.2 s, passes=4 src-split). Recall@10 0.7245 / 0.8326 / 0.9111.

| speedup vs 1 GPU (429.56 s) | Step 4 | Step 6 | Wait | prune | optimize | total |
|---|---|---|---|---|---|---|
| G = 2 (best, S10) | 1.66× | 1.88× | 1.28× | 1.21× | 1.32× | 1.48× |
| G = 4 (S14) | 2.36× | 3.07× | 1.31× | 1.18× | 1.38× | 1.80× |

### S16 — optimize: source rows resident between count and fill (split, size-gated) — accepted (compute gain, disk-bound) (2026-11-19)

**Idea.** With the S9 source-row split each GPU uploads its rows [N·g/G, N·(g+1)/G) twice (count, then fill). If they fit
(≤ 30 % of free device memory on every GPU; SIFT100M: G = 4 3.2 GB yes, G = 2 6.4 GB no → old path), count uploads them
batch by batch into one resident buffer and fill / later rounds launch `kern_rev_fill_multi` straight on it — half the H2D
bytes of the reverse phase. `OC_GPU_REV_RESIDENT=0` disables. Class: implementation (same edges, det identical). Source: cuVS
`cagra_build.cuh::build_knn_graph` (dataset kept on device across passes when it fits).

**Correctness.** SIFT10M G = 2 (resident there): final graph byte-identical with `OC_GPU_REV_RESIDENT` 1 vs 0; det 2 EQUAL
(`OC_GPU_PHASE2=0`, rounds, plain); det 4 EQUAL (`OC_GPU_PHASE2=0`); guard ACCEPT.

**SIFT100M (all four GPUs free).** G = 4 ×3: **237.60 s** (−0.5 % vs best 238.74); same-period ref G = 4 (`OC_GPU_REV_RESIDENT=0`)
239.97 s (bucket2 162.7, prune 48.5, optimize 28.74). G = 2 ×3 (not resident, unchanged path): 285.33 s (−1.9 % vs 290.97, faster
period). Recall unchanged (G = 4 0.7261 / 0.8338 / 0.9104, G = 2 0.7241 / 0.8322 / 0.9108). `--update --compute-gain` → both bests.

| optimize GPU reverse, G = 4 | same-period ref | S16 median of 3 |
|---|---|---|
| fill | 3.68 s | 1.93 s (−48 %) |
| count + fill + select | 6.75 s | 4.54 s (−33 %) |
| optimize stage | 28.74 s | 26.58 s |

Absorbed by: optimize merge + emit / final-graph stream (disk-bound, 5.5–6.0 s). G = 4 speedup vs 1 GPU: total 1.81×, optimize 1.49×.

### M1 — no-disk-limit measurement on /dev/shm (user request 2026-11-20; measurement only, best unchanged)

HEAD 74060c5 binaries built separately (git worktree + niced build in the scratchpad; `mgpu_run.sh` now takes `BIN=` like
`B2BIN=`), OUTDIR (work/, tmp, all intermediates, final graph) on `/dev/shm/bucket2_shm/<run>`, timing only, each run's shm dir
removed right after it (safety: MemAvailable ≥ 170 GB and /dev/shm ≥ 100 GB before each run; min MemAvailable seen 111 GB, shm
peak 55–61 GB). G = 4 skipped in both chains (GPU 0,1 held by other users). Logs: `runs/shm/`.

| SIFT100M | total | Step 4 | Step 4.5 | Step 6 | Wait | Steps 1–3 + 5 | prune | optimize | disk best |
|---|---|---|---|---|---|---|---|---|---|
| G = 1 shm | **406.0** | 99.2 | 12.0 | 159.9 | 19.5 | 18.7 | 53.0 | 37.8 | 429.56 (#96) |
| G = 2 shm #1 | **263.4** | 60.8 | 12.2 | 85.6 | 21.4 | 18.7 | 33.5 | 24.6 | 285.33 |
| G = 2 shm #2 | **266.9** | 60.8 | 11.7 | 85.6 | 19.4 | 18.3 | 38.3 | 25.8 | |
| G = 2 disk (S16 rep1, ref) | 285.3 | 59.5 | 16.0 | 84.1 | 25.2 | 18.8 | 41.6 | 28.7 | |

Speedup 1 → 2 GPUs on shm: 406.0 / 265.2 = **1.53×** (disk 1.51×); Step 4 1.63×, Step 6 1.87×, prune 1.47×, optimize 1.50×,
Wait / Steps 1–3, 4.5, 5 ≈ 1×. shm vs disk at G = 2: −7 % (Wait −5 s, Step 4.5 −4 s, prune −5 s, optimize −3.5 s); **Step 6 and
Step 4 are not disk-bound** (unchanged on shm).

Value of the compute changes without the disk (G = 2, shm, A/B):
- S9 + S11 off (`OC_GPU_REV_SPLIT=0 OC_GPU_REV_GPUSCAN=0`): optimize 28.72 s vs 24.6 / 25.8 s → **−3.5 s (−12 %)**; count
  3.48 → 2.26 s, fill 4.92 → 3.50 s. (S10 is not switchable here.)
- S17 (patch build): Step 4 60.8 → **54.6 s (−6.2 s)**; per-GPU read 2.75 → 0.67 s even on shm (the original dataset is still
  read from `/`); bucket2 total 202.1 vs 205.3 / 202.8 s — the gain is partly absorbed by the other steps' noise.
- S16: G = 4 only → not measured (G = 4 skipped).

Remaining bottlenecks on shm at G = 2 (265 s): Step 6 85.6 s (GPU-bound, 1.87×), Step 4 60.8 s (per-GPU search ~20 s × 2
rounds + ~3 s serial step0/init per round + read waits, cf. S17), the single-GPU / serial parts Steps 1–3 + 4.5 + 5 + Wait
≈ 50 s (now allowed to be CPU-parallel / overlapped, user 2026-11-20 → S18…), prune 33–38 s, optimize 25 s.

### S19 — S17 + S18 combined: Step 4 prefetch batches Nv / 4 + Step 4.5 background reorder write (bounded dirty) + round-2 blob — accepted (2026-11-20)

**Idea.** The two near-misses touch different phases: S17 hides the per-batch dataset read behind the previous search in
round-1 Step 4 (prefetch batches Nv / 4); S18 (first idea under the user's 2026-11-20 relaxation: single-GPU steps may overlap
GPU stages) writes data_reordered in a background thread with ≤ ~256 MB dirty pages (`sync_file_range`), round-1 Step 6 does not
wait for it, and round-2 Step 4 + bucket write read the in-memory `r1_blob` instead of the file. Same results (S17: top-2 per
point is batch-independent; S18: data_reordered byte-identical). Sources: TAGORE `large_index.cu` GPU–CPU–disk pipeline; cuVS
`cagra_ace.cuh::ace_check_use_disk_mode` (memory mode when it fits). Switches: `BUCKET2_STEP4_PF_DIV`, `BUCKET2_REORDER_BG`;
test switch `BUCKET2_HOST_RAW_BUDGET_MB`.

**Correctness.** det 2 EQUAL plain and with the S18 path forced on SIFT10M (`BUCKET2_HOST_RAW_BUDGET_MB=2000`); SIFT10M
data_reordered / vector_knn / forward / final graph byte-identical with the background write on vs off; guard ACCEPT.
G = 4 skipped (GPU 0 held by wangcy07). The SIFT100M part ran on a second attempt (first stopped at `/` 198 GB < 200 GB).

**SIFT100M G = 2 ×3 (GPUs 2,3): 268.24 s, −6.0 % vs best 285.33 → judge ACCEPT, new best.** Same-period HEAD ref (HEAD
binary): 291.32 s (bucket2 217.2, prune 45.5, optimize 28.7).

| stage | same-period HEAD ref | S19 median |
|---|---|---|
| Step 4 (2 rounds) | 60.6 s | 53.7 s |
| Step 4.5 | 16.7 s | 4.7 s |
| Step 6 | 85.8 s | 83.6 s (no S6 penalty) |
| bucket2 | 217.2 s | 197.0 s |
| prune / optimize | 45.5 / 28.7 s | 42.1 / 28.9 s |

| speedup vs 1 GPU (429.56 s) | Step 4 | Step 6 | Wait | prune | optimize | total |
|---|---|---|---|---|---|---|
| G = 2 (S19) | 1.89× | 1.91× | 1.30× | 1.32× | 1.37× | 1.60× |

**S19 at G = 4 (2026-11-20, all four GPUs free):** det 4 EQUAL (plain + forced S18 path). SIFT100M G = 4 ×3: **222.82 s (reps
222.8 / 220.2 / 223.8), −6.2 % vs best 237.60 → judge ACCEPT (memory guard on: peak RSS 95.6 GB ≤ 1.2 × 82.2), new G = 4 best.**
Step 4 43.0 → 34.0 s, Step 4.5 15.6 → 6.3 s, Step 6 51.7 s, prune 49.0, optimize 27.2. Same-period ref (pre-S19 HEAD 74060c5,
×1): 222.98 s = bucket2 163.7 + prune **32.6** + optimize 26.7 — bucket2 −18.6 s for S19, but the ref's prune is an outlier
(every other G = 4 prune 46–49 s: S14 47.3, S15 46.5, S16 49.4, S17 48.6) → caveat noted; a second ref would settle it.
Peak RSS note: S19 keeps the 12.8 GB `r1_blob` into round 2 (G = 2 91.7 vs 80.4 GB, G = 4 95.6 vs 82.2 GB) — within the guard,
but under rule (a) this retention must be counted against the host budget (to do with item 1/3).

| speedup vs 1 GPU (429.56 s) | Step 4 | Step 6 | Wait | prune | optimize | total |
|---|---|---|---|---|---|---|
| G = 2 (S19) | 1.89× | 1.91× | 1.30× | 1.32× | 1.37× | 1.60× |
| G = 4 (S19) | 2.99× | 3.09× | 1.29× | 1.14× | 1.45× | 1.93× |
