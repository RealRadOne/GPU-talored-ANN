#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bucketed kNN graph builder for BIGANN-style binaries (.fbin/.ibin/.i8bin).

Steps:
1) Load vectors (N, D) from .fbin/.ibin/.i8bin (BIGANN header: int32 N, int32 D, then data).
2) Sample N/32 centers (indices), use them as bucket centers.
3) Assign each point to its nearest center -> buckets.
4) Compute bucket-bucket distances (center-to-center), find k nearest buckets per bucket.
5) For each bucket, compute distances from its points to points in its k neighbor buckets,
   using Tensor Core-accelerated matmul on CUDA (float16/bfloat16), then pick m nearest per point.
6) Save graph index: neighbors.npy (N,m) and neighbors_dist.npy (N,m)
"""

import os
import argparse
import math
import struct
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from load import load_bigann_bin

# ----------------------------
# Utilities: device / precision
# ----------------------------

def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tensorcore_ready_dtype(device: torch.device) -> torch.dtype:
    """
    Prefer float16; if BF16 matmul has good support you may choose bfloat16.
    """
    if device.type == "cuda":
        # Enable TF32 for speed on Ampere+ (safe for L2 distances)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.float16
    return torch.float32

# ----------------------------
# Math: distance via matmul
# ----------------------------

@torch.inference_mode()
def pairwise_l2_min_assign(
    X: torch.Tensor, C: torch.Tensor, batch: int, device: torch.device, matmul_chunk: int, prec: torch.dtype
) -> torch.Tensor:
    """
    Assign each row in X to argmin L2 distance center in C.
    Memory-efficient chunking:
      Process X in batches of size `batch`; within each batch, process centers in chunks of size `matmul_chunk`.
    Returns LongTensor of shape (X.shape[0],) with center indices [0..C-1].
    """
    N, D = X.shape
    K = C.shape[0]
    idx = torch.empty(N, dtype=torch.long)

    # Precompute center norms on device once
    C_dev = C.to(device, non_blocking=True)
    C_dev_h = C_dev.to(prec) if device.type == "cuda" else C_dev
    Cn = (C_dev_h.float() ** 2).sum(dim=1)  # keep norms in fp32 for stability

    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb = X[s:e].to(device, non_blocking=True)
        xb_h = xb.to(prec) if device.type == "cuda" else xb
        xb2 = (xb_h.float() ** 2).sum(dim=1, keepdim=True)  # (b,1) fp32

        best = None
        best_j = None
        # chunk over centers
        for c0 in range(0, K, matmul_chunk):
            c1 = min(c0 + matmul_chunk, K)
            cc = C_dev_h[c0:c1]  # (c, D)
            # dist = ||x||^2 + ||c||^2 - 2 x c^T
            dots = xb_h @ cc.T  # Tensor Core matmul if on CUDA+fp16/bf16
            dist = xb2 + Cn[c0:c1].unsqueeze(0) - 2.0 * dots.float()  # (b, c) fp32
            cand, arg = torch.min(dist, dim=1)  # (b,)
            if best is None:
                best = cand
                best_j = arg + c0
            else:
                mask = cand < best
                best[mask] = cand[mask]
                best_j[mask] = (arg + c0)[mask]

        idx[s:e] = best_j.cpu()

    return idx


@torch.inference_mode()
def topk_from_A_to_B(
    A: torch.Tensor, B: torch.Tensor, topk: int,
    device: torch.device, prec: torch.dtype,
    query_chunk: int, base_chunk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each a in A, find topk nearest in B (L2). Chunked Tensor-Core matmul.
    Returns (indices, dists):
      indices: LongTensor (A.shape[0], topk) with indices in [0..B-1]
      dists:   FloatTensor (A.shape[0], topk)
    """
    NA, D = A.shape
    NB = B.shape[0]
    out_idx = torch.empty(NA, topk, dtype=torch.long)
    out_dist = torch.empty(NA, topk, dtype=torch.float32)

    # Precompute B norm on device
    B_dev = B.to(device, non_blocking=True)
    B_h = B_dev.to(prec) if device.type == "cuda" else B_dev
    Bn = (B_h.float() ** 2).sum(dim=1)  # (NB,)

    for s in range(0, NA, query_chunk):
        e = min(s + query_chunk, NA)
        Qa = A[s:e].to(device, non_blocking=True)
        Qa_h = Qa.to(prec) if device.type == "cuda" else Qa
        Qa2 = (Qa_h.float() ** 2).sum(dim=1, keepdim=True)  # (b,1)

        # maintain running topk over chunks of B
        best_dist = None
        best_idx = None
        offset = 0

        for b0 in range(0, NB, base_chunk):
            b1 = min(b0 + base_chunk, NB)
            Bb = B_h[b0:b1]  # (c, D)
            dots = Qa_h @ Bb.T
            dist = Qa2 + Bn[b0:b1].unsqueeze(0) - 2.0 * dots.float()  # (b,c)

            # topk within this block
            blk_dist, blk_idx = torch.topk(dist, k=min(topk, b1 - b0), dim=1, largest=False)
            blk_idx = blk_idx + b0  # map to global indices

            if best_dist is None:
                # if first block smaller than topk, pad by inf to unify logic
                if blk_dist.size(1) < topk:
                    pad = topk - blk_dist.size(1)
                    best_dist = torch.cat([blk_dist, torch.full((blk_dist.size(0), pad), float("inf"), device=dist.device)], dim=1)
                    best_idx = torch.cat([blk_idx, torch.full((blk_idx.size(0), pad), -1, device=blk_idx.device, dtype=torch.long)], dim=1)
                else:
                    best_dist = blk_dist
                    best_idx = blk_idx
            else:
                # merge current block topk with running topk -> then take topk
                merged_dist = torch.cat([best_dist, blk_dist], dim=1)
                merged_idx = torch.cat([best_idx, blk_idx], dim=1)
                best_dist, sel = torch.topk(merged_dist, k=topk, dim=1, largest=False)
                best_idx = torch.gather(merged_idx, 1, sel)

            offset += (b1 - b0)

        out_idx[s:e] = best_idx.cpu()
        out_dist[s:e] = best_dist.cpu()

    return out_idx, out_dist


# ----------------------------
# Pipeline
# ----------------------------

def _dist_one_to_all(
    c: torch.Tensor,    # (1, D) CPU
    X: torch.Tensor,    # (N, D) CPU
    device: torch.device,
    prec: torch.dtype,
    batch: int,
) -> torch.Tensor:      # (N,) CPU float32, squared L2
    """Compute squared L2 distance from single center c to every row of X."""
    N = X.shape[0]
    out = torch.empty(N)
    c_dev = c.to(device)
    c_h = c_dev.to(prec) if device.type == "cuda" else c_dev
    cn2 = (c_h.float() ** 2).sum()
    for s in range(0, N, batch):
        e = min(s + batch, N)
        xb = X[s:e].to(device)
        xb_h = xb.to(prec) if device.type == "cuda" else xb
        xb2 = (xb_h.float() ** 2).sum(dim=1)
        dots = (xb_h @ c_h.T).float().squeeze(1)
        out[s:e] = (xb2 + cn2 - 2.0 * dots).cpu().clamp(min=0.0)
    return out


def kmeans_pp_centers(
    X: torch.Tensor,
    B: int,
    device: torch.device,
    prec: torch.dtype,
    assign_batch: int,
    matmul_chunk: int,
    seed: int,
    pool_ratio: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-Means++ 初始化：在大小为 min(N, pool_ratio*B) 的候选池上执行 D² 加权采样，
    使得初始中心在空间上分散、覆盖更均匀。
    Returns: (center_ids (B,), centers (B, D))
    """
    torch.manual_seed(seed)
    N = X.shape[0]
    pool_n = min(N, pool_ratio * B)
    pool_ids = torch.randperm(N)[:pool_n]
    Xp = X[pool_ids]  # (pool_n, D) stays on CPU

    chosen_local: List[int] = [int(torch.randint(pool_n, (1,)).item())]
    min_d2 = torch.full((pool_n,), float("inf"))

    for _ in range(1, B):
        last_c = Xp[chosen_local[-1]].unsqueeze(0)  # (1, D)
        d2 = _dist_one_to_all(last_c, Xp, device, prec, assign_batch)
        torch.minimum(min_d2, d2, out=min_d2)

        probs = min_d2.clamp(min=0.0)
        total = probs.sum()
        if total > 0:
            nxt = int(torch.multinomial(probs / total, 1).item())
        else:
            nxt = int(torch.randint(pool_n, (1,)).item())
        chosen_local.append(nxt)

    chosen_t = torch.tensor(chosen_local, dtype=torch.long)
    center_ids = pool_ids[chosen_t]
    centers = X[center_ids].clone()
    return center_ids, centers


def refine_centers_kmeans(
    X: torch.Tensor,
    centers: torch.Tensor,
    n_iters: int,
    assign_batch: int,
    matmul_chunk: int,
    device: torch.device,
    prec: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    执行 n_iters 轮 Lloyd's K-Means：将每个 centroid 更新为所属点的均值。
    多轮后 Voronoi 分区更均匀，centroid 也更贴近数据分布。
    Returns: (refined_centers (B, D), final_assign (N,))
    """
    N, D = X.shape
    B = centers.shape[0]
    assign = torch.empty(N, dtype=torch.long)
    for _ in range(n_iters):
        assign = pairwise_l2_min_assign(
            X=X, C=centers,
            batch=assign_batch, device=device,
            matmul_chunk=matmul_chunk, prec=prec
        )
        new_centers = torch.zeros(B, D, dtype=torch.float32)
        new_centers.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), X.float())
        counts = torch.bincount(assign, minlength=B)
        nonempty = counts > 0
        new_centers[nonempty] /= counts[nonempty].unsqueeze(1).float()
        new_centers[~nonempty] = centers[~nonempty]  # 空 bucket 保留旧中心
        centers = new_centers
    return centers, assign


def rebalance_assign(
    assign: torch.Tensor,    # (N,) 当前分配
    top2_idx: torch.Tensor,  # (N, 2) 每点最近和第2近的 centroid 索引
    top2_dist: torch.Tensor, # (N, 2) 对应距离
    B: int,
    target: int = 32,
    slack: int = 4,          # 允许 ±slack 的偏差
    max_iters: int = 30,
) -> torch.Tensor:
    """
    迭代地把超载 bucket 中"移走代价最小"的边界点重分配到它们的第2近 centroid，
    直到所有 bucket 大小在 [target-slack, target+slack] 范围内。
    代价 = dist_to_2nd - dist_to_1st（越小说明该点越接近边界，越容易移走）。
    """
    for _ in range(max_iters):
        counts = torch.bincount(assign, minlength=B)
        over_buckets = (counts > target + slack).nonzero().flatten()
        if over_buckets.numel() == 0:
            break
        new_assign = assign.clone()
        for b in over_buckets.tolist():
            excess = int(counts[b].item()) - target
            pts = (assign == b).nonzero().flatten()
            # 优先移向当前未过载的 2nd choice
            second = top2_idx[pts, 1]
            not_over = counts[second] <= target + slack
            movable = pts[not_over]
            if movable.numel() == 0:
                movable = pts  # 放宽约束
            if movable.numel() == 0:
                continue
            cost = top2_dist[movable, 1] - top2_dist[movable, 0]
            to_move = movable[cost.argsort()[:excess]]
            new_assign[to_move] = top2_idx[to_move, 1]
        assign = new_assign
    return assign


def _print_size_stats(assign: torch.Tensor, B: int, target: int, label: str) -> None:
    """打印当前 assign 的 bucket size 分布，用于各步骤后的质量追踪。"""
    counts = torch.bincount(assign, minlength=B).float()
    sizes = counts.tolist()
    n = len(sizes)
    mn, mx = int(min(sizes)), int(max(sizes))
    avg = sum(sizes) / n
    std = (sum((s - avg) ** 2 for s in sizes) / n) ** 0.5
    empty = sum(1 for s in sizes if s == 0)
    w4  = sum(1 for s in sizes if abs(s - target) <= 4)
    w8  = sum(1 for s in sizes if abs(s - target) <= 8)
    w16 = sum(1 for s in sizes if abs(s - target) <= 16)
    print(f"  [{label}] min={mn:4d}  max={mx:4d}  avg={avg:6.1f}  std={std:5.1f}  "
          f"empty={empty}  "
          f"±4={w4/n*100:5.1f}%  ±8={w8/n*100:5.1f}%  ±16={w16/n*100:5.1f}%")


def analyze_bucket_quality(buckets: List[torch.Tensor], target: int = 32) -> None:
    """
    详细打印 bucket size 质量报告：
      - 基础统计量
      - 百分位数
      - 文本直方图（按 size 区间统计桶数）
      - 离群桶列举（最大的 5 个、最小的 5 个非空桶）
    """
    sizes = [b.numel() for b in buckets]
    n = len(sizes)
    if n == 0:
        print("  [analyze] No buckets.")
        return
    arr = sorted(sizes)
    mn, mx = arr[0], arr[-1]
    avg = sum(arr) / n
    std = (sum((s - avg) ** 2 for s in arr) / n) ** 0.5
    median = arr[n // 2]
    p10 = arr[int(n * 0.10)]
    p25 = arr[int(n * 0.25)]
    p75 = arr[int(n * 0.75)]
    p90 = arr[int(n * 0.90)]
    p99 = arr[int(n * 0.99)]
    empty = sum(1 for s in arr if s == 0)
    w4   = sum(1 for s in arr if abs(s - target) <= 4)
    w8   = sum(1 for s in arr if abs(s - target) <= 8)
    w16  = sum(1 for s in arr if abs(s - target) <= 16)

    print(f"\n{'='*60}")
    print(f"  Bucket Quality Report  (B={n}, target={target})")
    print(f"{'='*60}")
    print(f"  Basic stats : min={mn}  max={mx}  avg={avg:.1f}  std={std:.1f}  median={median}")
    print(f"  Percentiles : p10={p10}  p25={p25}  p75={p75}  p90={p90}  p99={p99}")
    print(f"  Empty buckets: {empty} ({empty/n*100:.1f}%)")
    print(f"  Within ±4  of {target}: {w4}/{n} = {w4/n*100:.1f}%")
    print(f"  Within ±8  of {target}: {w8}/{n} = {w8/n*100:.1f}%")
    print(f"  Within ±16 of {target}: {w16}/{n} = {w16/n*100:.1f}%")

    # 文本直方图：按区间 [0,8), [8,16), [16,24), [24,32), [32,40), [40,48), [48,64), [64,+)
    bins = [0, 8, 16, 24, 32, 40, 48, 64, float("inf")]
    labels = ["[0,8)", "[8,16)", "[16,24)", "[24,32)", "[32,40)", "[40,48)", "[48,64)", "[64,+)"]
    hist = [0] * (len(bins) - 1)
    for s in arr:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                hist[i] += 1
                break
    bar_width = 30
    max_h = max(hist) if max(hist) > 0 else 1
    print(f"\n  Size distribution histogram:")
    for i, (lbl, h) in enumerate(zip(labels, hist)):
        bar = "#" * int(h / max_h * bar_width)
        marker = " <-- target" if lbl == "[32,40)" or lbl == "[24,32)" else ""
        print(f"  {lbl:8s} | {bar:<{bar_width}} {h:6d}{marker}")

    # 最大 / 最小离群桶
    indexed = sorted(enumerate(sizes), key=lambda x: x[1])
    nonempty = [(i, s) for i, s in indexed if s > 0]
    print(f"\n  Largest  5 buckets: {[(i, s) for i, s in indexed[-5:]][::-1]}")
    print(f"  Smallest 5 non-empty: {nonempty[:5]}")
    print(f"{'='*60}\n")


def build_buckets(
    X: torch.Tensor,
    seed: int,
    assign_batch: int,
    matmul_chunk: int,
    device: torch.device,
    prec: torch.dtype,
    kmeans_iters: int = 5,
    balance: bool = True,
    balance_slack: int = 4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
    """
    构建 buckets，三步流程：
      1) K-Means++ 初始化 → 让初始 centroid 在空间上均匀分散
      2) K-Means 迭代精化 → centroid 移向真实 cluster 均值，使分区更均匀
      3) Top-2 rebalancing → 把边界点从超载 bucket 移到第2近 centroid，平衡大小
    Returns:
      centers    : (B, D) 精化后的 centroid（均值，不一定是原始数据点）
      center_ids : (B,) 初始采样的全局索引（精化后为 None）
      buckets    : List[LongTensor]
    """
    N, D = X.shape
    B = max(1, N // 32)
    target = 32

    # Step 1: K-Means++ 初始化
    print("  [bucket] K-Means++ init ...")
    center_ids, centers = kmeans_pp_centers(
        X, B, device, prec, assign_batch, matmul_chunk, seed
    )

    # Step 2: K-Means 精化
    if kmeans_iters > 0:
        print(f"  [bucket] Running {kmeans_iters} K-Means iterations ...")
        centers, assign = refine_centers_kmeans(
            X, centers, kmeans_iters, assign_batch, matmul_chunk, device, prec
        )
        center_ids = None  # centroid 已变为均值，不再对应原始索引
    else:
        assign = pairwise_l2_min_assign(
            X=X, C=centers, batch=assign_batch,
            device=device, matmul_chunk=matmul_chunk, prec=prec
        )
    assign_t: torch.Tensor = assign  # explicit type for Pylance (no torch stubs)
    _print_size_stats(assign_t, B, target, "after K-Means")

    # Step 3: 均衡重分配
    if balance:
        print("  [bucket] Computing top-2 assignments for rebalancing ...")
        top2_idx, top2_dist = topk_from_A_to_B(
            A=X, B=centers, topk=2,
            device=device, prec=prec,
            query_chunk=assign_batch, base_chunk=matmul_chunk
        )
        print("  [bucket] Rebalancing buckets ...")
        assign_t = rebalance_assign(
            assign_t, top2_idx, top2_dist, B,
            target=target, slack=balance_slack
        )
        _print_size_stats(assign_t, B, target, "after rebalance")

    buckets: List[torch.Tensor] = [
        (assign_t == b).nonzero(as_tuple=False).flatten() for b in range(B)
    ]
    return centers, center_ids, buckets


def k_nearest_buckets(centers: torch.Tensor, k: int, device: torch.device, prec: torch.dtype, chunk: int) -> torch.Tensor:
    """
    计算每个 center 到其他 center 的距离，返回 (B, k) 的近邻桶索引（排除自身）。
    """
    B, D = centers.shape
    C = centers  # alias
    # 我们用相同接口 topk_from_A_to_B(A=C, B=C)
    idx, _ = topk_from_A_to_B(
        A=C, B=C, topk=k+1, device=device, prec=prec, query_chunk=chunk, base_chunk=chunk
    )
    # 去掉自身
    # 每行 idx 第一列理论上是自身；稳妥起见过滤 self==row 的项，再取前 k
    rows = torch.arange(B).unsqueeze(1).expand_as(idx)
    mask_self = (idx == rows)
    # 将自环替换为后续元素：简单办法是排序后取前k个非自环
    cleaned = []
    for r in range(B):
        row = idx[r]
        row = row[row != r]
        cleaned.append(row[:k])
    return torch.stack(cleaned, dim=0)


def _apply_reverse_edges(all_neighbors: torch.Tensor, all_dists: torch.Tensor, m: int) -> None:
    """
    反向边处理（in-place）：
    若 vector a 将 vector b 放在了 a 的 neighbor list 中（距离 d），
    则也尝试将 a 加入 b 的 neighbor list（如果 d 小于 b 当前最远邻居的距离）。
    """
    N = all_neighbors.shape[0]

    # 收集所有正向边 (source=a, target=b, dist=d)
    src_all = torch.arange(N).unsqueeze(1).expand(-1, m).reshape(-1)
    tgt_all = all_neighbors.reshape(-1)
    dist_all = all_dists.reshape(-1)

    # 过滤无效边 (target == -1)
    valid = tgt_all >= 0
    src_v = src_all[valid]
    tgt_v = tgt_all[valid]
    dist_v = dist_all[valid]

    # 按 target 排序，便于分组处理
    order = tgt_v.argsort()
    src_v = src_v[order]
    tgt_v = tgt_v[order]
    dist_v = dist_v[order]

    # 找出每个 target 的分组边界
    unique_tgts, counts = torch.unique_consecutive(tgt_v, return_counts=True)

    updated = 0
    offset = 0
    for i in range(unique_tgts.shape[0]):
        b = unique_tgts[i].item()
        cnt = counts[i].item()

        # b 的所有反向候选（即正向边中 target==b 的 source）
        cand_src = src_v[offset:offset + cnt]
        cand_dist = dist_v[offset:offset + cnt]
        offset += cnt

        cur_ids = all_neighbors[b]    # (m,) view
        cur_dists = all_dists[b]      # (m,) view

        # 快速跳过：如果所有候选距离都 >= b 的当前最远邻居距离，无需处理
        worst_dist = cur_dists.max().item()
        if cand_dist.min().item() >= worst_dist:
            continue

        # 过滤：去掉已在 b 的 neighbor list 中的候选
        in_list = (cand_src.unsqueeze(1) == cur_ids.unsqueeze(0)).any(dim=1)
        new_src = cand_src[~in_list]
        new_dist = cand_dist[~in_list]

        if new_src.numel() == 0:
            continue

        # 合并 current + new，取 top-m（距离最小的 m 个）
        merged_ids = torch.cat([cur_ids, new_src])
        merged_dists = torch.cat([cur_dists, new_dist])

        k = min(m, merged_ids.shape[0])
        _, sel = merged_dists.topk(k, largest=False)

        all_neighbors[b, :k] = merged_ids[sel]
        all_dists[b, :k] = merged_dists[sel]
        if k < m:
            all_neighbors[b, k:] = -1
            all_dists[b, k:] = float("inf")

        updated += 1

    print(f"  [reverse edges] Updated neighbor lists for {updated}/{N} vectors")


def neighbors_within_knn_buckets(
    X: torch.Tensor,
    buckets: List[torch.Tensor],
    knb: torch.Tensor,  # (B,k) 每个桶的近邻桶ID
    m: int,
    device: torch.device,
    prec: torch.dtype,
    query_chunk: int,
    base_chunk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对每个桶中每个点，在其 k 个近邻桶（含自身桶）联合的点集中找 m 个最近邻。
    返回：
      all_neighbors: (N,m) 全局索引
      all_dists    : (N,m) 距离
    """
    N = X.shape[0]
    all_neighbors = torch.full((N, m), -1, dtype=torch.long)
    all_dists = torch.full((N, m), float("inf"), dtype=torch.float32)

    # 先把每个桶对应的“候选点全集合”缓存好，减少重复拼接
    candidate_map: Dict[int, torch.Tensor] = {}
    B = len(buckets)
    for b in range(B):
        neigh_bs = knb[b].tolist()
        # 原先：只拼 k 个近邻桶
        # cand = torch.cat([buckets[n] for n in neigh_bs if len(buckets[n]) > 0], dim=0) if len(neigh_bs) > 0 else torch.empty(0, dtype=torch.long)

        # 改为：把自身桶 b 也一起拼进去
        parts = []
        if len(buckets[b]) > 0:
            parts.append(buckets[b])
        for n in neigh_bs:
            if len(buckets[n]) > 0:
                parts.append(buckets[n])
        cand = torch.cat(parts, dim=0) if len(parts) > 0 else torch.empty(0, dtype=torch.long)

        # 去重，避免重复索引
        candidate_map[b] = cand.unique() if cand.numel() > 0 else cand

    # 逐桶处理
    for b in range(B):
        q_idx = buckets[b]
        if q_idx.numel() == 0:
            continue
        cand_idx = candidate_map[b]
        if cand_idx.numel() == 0:
            continue

        Qa = X[q_idx]      # (nq, D)
        Bb = X[cand_idx]   # (nb, D)

        # 原先：topk=m
        # neigh_idx_local, neigh_dist = topk_from_A_to_B(A=Qa, B=Bb, topk=m, ...)

        # 改为：topk=m+1，给“自环”留出一个名额用于过滤
        neigh_idx_local, neigh_dist = topk_from_A_to_B(
            A=Qa, B=Bb, topk=m+1,
            device=device, prec=prec,
            query_chunk=query_chunk, base_chunk=base_chunk
        )
        neigh_idx_global = cand_idx[neigh_idx_local]  # (nq, m+1)

        # 逐行去掉自身 id，再截断为 m 个
        # （注意：如果该行没有自环，仍然会保留前 m 个）
        nq = q_idx.shape[0]
        keep_idx = torch.empty((nq, m), dtype=torch.long)
        keep_dist = torch.empty((nq, m), dtype=torch.float32)
        for i in range(nq):
            qi = q_idx[i].item()
            row_ids = neigh_idx_global[i]
            row_ds  = neigh_dist[i]
            mask = (row_ids != qi)
            # 压缩后取前 m 个
            filtered_ids = row_ids[mask][:m]
            filtered_ds  = row_ds[mask][:m]
            # 若过滤后不足 m（极少见，除非候选非常小），用 -1 / inf 填充
            if filtered_ids.numel() < m:
                pad = m - filtered_ids.numel()
                filtered_ids = torch.cat([filtered_ids, torch.full((pad,), -1, dtype=torch.long)])
                filtered_ds  = torch.cat([filtered_ds,  torch.full((pad,), float("inf"))])
            keep_idx[i] = filtered_ids
            keep_dist[i] = filtered_ds

        all_neighbors[q_idx] = keep_idx
        all_dists[q_idx] = keep_dist

    # 反向边处理：若 a→b，则尝试将 a 加入 b 的 neighbor list
    print("  [reverse edges] Applying reverse edge logic ...")
    _apply_reverse_edges(all_neighbors, all_dists, m)

    return all_neighbors, all_dists


def _merge_dedup_neighbors(
    stacked_ids: torch.Tensor,    # (N, t*m)
    stacked_dists: torch.Tensor,  # (N, t*m)
    m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    合并 t 轮的 neighbor list 并去重，每个 vector 取距离最近的 m 个。
    对每行：按 dist 排序 → 去重(跳过重复 id 和 -1) → 取前 m 个。
    """
    N, total = stacked_ids.shape
    final_ids = torch.full((N, m), -1, dtype=torch.long)
    final_dists = torch.full((N, m), float("inf"), dtype=torch.float32)

    # 按 dist 排序
    sorted_dists, order = stacked_dists.sort(dim=1)  # ascending
    sorted_ids = stacked_ids.gather(1, order)

    for i in range(N):
        seen = set()
        cnt = 0
        for j in range(total):
            vid = sorted_ids[i, j].item()
            if vid < 0 or vid in seen:
                continue
            seen.add(vid)
            final_ids[i, cnt] = vid
            final_dists[i, cnt] = sorted_dists[i, j]
            cnt += 1
            if cnt >= m:
                break

    return final_ids, final_dists


def build_index_multi_round(
    X: torch.Tensor,
    k: int, m: int, t: int,
    device: torch.device, prec: torch.dtype,
    assign_batch: int, matmul_chunk: int,
    query_chunk: int, base_chunk: int,
    seed: int,
    kmeans_iters: int = 5,
    balance: bool = True,
    balance_slack: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    多轮采样建图：重复 t 次 (采样 centroid → 建 bucket → 建 index)，
    然后合并 t 轮的 neighbor list，每个 vector 取距离最近的 m 个。
    """
    N = X.shape[0]
    all_neighbors_list: List[torch.Tensor] = []
    all_dists_list: List[torch.Tensor] = []

    for round_i in range(t):
        round_seed = seed + round_i
        print(f"\n{'='*60}")
        print(f"  Round {round_i + 1}/{t}  (seed={round_seed})")
        print(f"{'='*60}")

        print(f"  [round {round_i+1}] Building buckets ...")
        centers, center_ids, buckets = build_buckets(
            X=X, seed=round_seed,
            assign_batch=assign_batch, matmul_chunk=matmul_chunk,
            device=device, prec=prec,
            kmeans_iters=kmeans_iters,
            balance=balance, balance_slack=balance_slack,
        )

        print(f"  [round {round_i+1}] Finding k={k} nearest buckets ...")
        knb = k_nearest_buckets(
            centers=centers, k=k,
            device=device, prec=prec, chunk=matmul_chunk,
        )

        print(f"  [round {round_i+1}] Computing m={m} nearest neighbors per point ...")
        neighbors_i, dists_i = neighbors_within_knn_buckets(
            X=X, buckets=buckets, knb=knb, m=m,
            device=device, prec=prec,
            query_chunk=query_chunk, base_chunk=base_chunk,
        )
        all_neighbors_list.append(neighbors_i)
        all_dists_list.append(dists_i)

        # 统计本轮有效邻居数
        valid_count = (neighbors_i >= 0).sum().item()
        print(f"  [round {round_i+1}] Valid edges: {valid_count}/{N * m}")

    # 合并 t 轮结果
    print(f"\n{'='*60}")
    print(f"  Merging {t} rounds ...")
    print(f"{'='*60}")
    stacked_ids = torch.cat(all_neighbors_list, dim=1)    # (N, t*m)
    stacked_dists = torch.cat(all_dists_list, dim=1)      # (N, t*m)

    final_neighbors, final_dists = _merge_dedup_neighbors(stacked_ids, stacked_dists, m)

    # 统计合并效果
    valid_final = (final_neighbors >= 0).sum().item()
    print(f"  [merge] Final valid edges: {valid_final}/{N * m}")

    return final_neighbors, final_dists


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to .fbin/.ibin/.i8bin")
    ap.add_argument("--k", type=int, required=True, help="每个bucket的近邻桶个数")
    ap.add_argument("--m", type=int, required=True, help="每个样本在近邻桶中的最近邻个数")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录，将写出 neighbors.npy 和 neighbors_dist.npy")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（用于采样中心）")
    ap.add_argument("--assign-batch", type=int, default=65536, help="分配到中心时 X 的batch大小")
    ap.add_argument("--matmul-chunk", type=int, default=262144, help="中心分配/中心间距离的 chunk 大小")
    ap.add_argument("--query-chunk", type=int, default=32768, help="跨桶查询时 A 的分块")
    ap.add_argument("--base-chunk", type=int, default=262144, help="跨桶查询时 B 的分块")
    ap.add_argument("--force-cpu", action="store_true", help="即使有GPU也强制CPU跑（调试用）")
    ap.add_argument("--use-bf16", action="store_true", help="在CUDA上使用bfloat16（默认float16）")
    ap.add_argument("--prune_max_degree", type=int, default=64, help="prune后最大保留的领居数量")
    ap.add_argument("--kmeans-iters", type=int, default=5, help="K-Means精化迭代次数（0=跳过）")
    ap.add_argument("--no-balance", action="store_true", help="禁用top-2均衡重分配")
    ap.add_argument("--balance-slack", type=int, default=4, help="均衡允许偏差（bucket大小在32±slack内视为合格）")
    ap.add_argument("--t", type=int, default=1, help="多轮采样次数（t>1 时对 t 次独立 bucket 建图结果合并取 top-m）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1] Loading data from {args.data} ...")
    X = load_bigann_bin(args.data)  # CPU float32
    N, D = X.shape
    print(f"Loaded X: shape=({N}, {D}), dtype={X.dtype}, device=CPU")

    device = torch.device("cpu") if args.force_cpu else pick_device()
    prec = tensorcore_ready_dtype(device)
    if args.use_bf16 and device.type == "cuda":
        prec = torch.bfloat16
    print(f"Device: {device}, matmul precision: {prec}")

    if args.t > 1:
        # ---- 多轮模式 ----
        print(f"\n[2] Multi-round mode: t={args.t}, k={args.k}, m={args.m}")
        neighbors, neighbors_dist = build_index_multi_round(
            X=X, k=args.k, m=args.m, t=args.t,
            device=device, prec=prec,
            assign_batch=args.assign_batch, matmul_chunk=args.matmul_chunk,
            query_chunk=args.query_chunk, base_chunk=args.base_chunk,
            seed=args.seed,
            kmeans_iters=args.kmeans_iters,
            balance=not args.no_balance,
            balance_slack=args.balance_slack,
        )
    else:
        # ---- 原有单轮模式 ----
        print("[2/6] Building buckets (K-Means++ init → K-Means refine → rebalance) ...")
        centers, center_ids, buckets = build_buckets(
            X=X,
            seed=args.seed,
            assign_batch=args.assign_batch,
            matmul_chunk=args.matmul_chunk,
            device=device,
            prec=prec,
            kmeans_iters=args.kmeans_iters,
            balance=not args.no_balance,
            balance_slack=args.balance_slack,
        )
        B = centers.shape[0]
        sizes = [len(b) for b in buckets]
        avg = sum(sizes) / len(sizes)
        variance = sum((s - avg) ** 2 for s in sizes) / len(sizes)
        std = variance ** 0.5
        within = sum(1 for s in sizes if abs(s - 32) <= 4)
        print(f"Centers: B={B} (~N/32). Bucket size stats: "
              f"min={min(sizes)}, max={max(sizes)}, avg={avg:.1f}, std={std:.1f}, "
              f"within±4 of 32: {within/len(sizes)*100:.1f}%")

        print("[3/6] Finding k nearest buckets by center-center distance ...")
        knb = k_nearest_buckets(
            centers=centers,
            k=args.k,
            device=device,
            prec=prec,
            chunk=args.matmul_chunk
        )

        print("[4/6] For each bucket, using Tensor Core matmul to compute distances to its k neighbor buckets ...")

        print("[5/6] Selecting m nearest neighbors per point within its k neighbor buckets ...")
        neighbors, neighbors_dist = neighbors_within_knn_buckets(
            X=X,
            buckets=buckets,
            knb=knb,
            m=args.m,
            device=device,
            prec=prec,
            query_chunk=args.query_chunk,
            base_chunk=args.base_chunk
        )

    assert neighbors.shape == (N, args.m)
    assert neighbors_dist.shape == (N, args.m)

    print("[Save] Saving graph index ...")
    suffix = f"_k{args.k}m{args.m}" + (f"t{args.t}" if args.t > 1 else "")
    f_neighbors = f"neighbors{suffix}.npy"
    f_dists = f"neighbors_dist{suffix}.npy"
    np.save(os.path.join(args.out_dir, f_neighbors), neighbors.numpy().astype(np.int64))
    np.save(os.path.join(args.out_dir, f_dists), neighbors_dist.numpy().astype(np.float32))

    meta = {
        "N": int(N), "D": int(D),
        "k": int(args.k), "m": int(args.m), "t": int(args.t),
    }
    np.save(os.path.join(args.out_dir, f"meta{suffix}.npy"), meta, allow_pickle=True)
    print(f"Done. Files written to: {args.out_dir}  ({f_neighbors}, {f_dists})")

if __name__ == "__main__":
    main()