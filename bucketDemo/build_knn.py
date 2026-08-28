#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import struct
import numpy as np
from typing import Tuple, Optional
from load import get_ext, load_bigann_to_f32, load_ibin, load_neighbors
from load import save_npy

# =========================
# Math helpers
# =========================

# Lan: todo: add GPU acceleration
def _block_l2_topk_merge(
    A: np.ndarray, B: np.ndarray,
    a2: np.ndarray, b2: np.ndarray,
    k: int,
    A_global_idx: np.ndarray,  # shape [a], 全局 id
    B_global_idx: np.ndarray   # shape [b], 全局 id
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 dist(A, B) 的局部块并与“当前最优”合并。
    这里实现“从零开始”的一次块处理：返回 A 每行在 B 中的 topk (indices, dists)。
    上层会把多个 B 块的结果反复 merge。
    """
    # dist = ||a||^2 + ||b||^2 - 2 A B^T
    # (a,b)
    dots = A @ B.T  # float32
    dist = a2[:, None] + b2[None, :] - 2.0 * dots

    # 自环屏蔽（A、B 可能是同一个矩阵的不同/相同切片），把同 id 的距离设成 +inf
    # 用广播比对：A_global_idx[:,None] vs B_global_idx[None,:]
    same = (A_global_idx[:, None] == B_global_idx[None, :])
    if np.any(same):
        dist[same] = np.inf

    # 在该块内取 topk
    if k < dist.shape[1]:
        part = np.argpartition(dist, kth=k-1, axis=1)[:, :k]  # (a,k) 未排序
    else:
        # 候选不足 k，全拿
        part = np.tile(np.arange(dist.shape[1], dtype=np.int64), (dist.shape[0], 1))

    row = np.arange(dist.shape[0])[:, None]
    cand_d = dist[row, part]
    cand_i = B_global_idx[part]

    # 对每行排序
    order = np.argsort(cand_d, axis=1)
    topk_d = np.take_along_axis(cand_d, order, axis=1)
    topk_i = np.take_along_axis(cand_i, order, axis=1)
    # 若不足 k，用 inf/-1 填充（保持上层合并一致）
    if topk_d.shape[1] < k:
        pad = k - topk_d.shape[1]
        topk_d = np.hstack([topk_d, np.full((topk_d.shape[0], pad), np.inf, dtype=np.float32)])
        topk_i = np.hstack([topk_i, np.full((topk_i.shape[0], pad), -1, dtype=np.int64)])
    return topk_i, topk_d

def _merge_topk(
    cur_idx: np.ndarray, cur_dist: np.ndarray,
    add_idx: np.ndarray, add_dist: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 (add_idx, add_dist) 合并进 (cur_idx, cur_dist)，都为 (a,k)。
    返回合并后的 (a,k)。
    """
    a = cur_idx.shape[0]
    # 拼接成 (a, 2k)
    idx_cat = np.hstack([cur_idx, add_idx])
    dist_cat = np.hstack([cur_dist, add_dist])

    # 选出最小的 k 个
    if idx_cat.shape[1] > k:
        sel = np.argpartition(dist_cat, kth=k-1, axis=1)[:, :k]
        row = np.arange(a)[:, None]
        idx_k = idx_cat[row, sel]
        dist_k = dist_cat[row, sel]
        # 再排序
        ord2 = np.argsort(dist_k, axis=1)
        idx_k = np.take_along_axis(idx_k, ord2, axis=1)
        dist_k = np.take_along_axis(dist_k, ord2, axis=1)
    else:
        # 直接排序再截断
        ord_all = np.argsort(dist_cat, axis=1)
        row = np.arange(a)[:, None]
        idx_k = idx_cat[row, ord_all[:, :k]]
        dist_k = dist_cat[row, ord_all[:, :k]]
    return idx_k, dist_k

# =========================
# (1) 精确 kNN 构建（分块）
# =========================

def build_exact_knn(
    data_path: str, k: int, out_dir: str,
    query_chunk: int = 8192,
    base_chunk: int = 262144
) -> Tuple[str, str]:
    """
    为整个数据集（自查询、自库；排除自环）计算精确 kNN。
    返回写出文件路径 (idx_path, dist_path)。
    """
    X, N, D = load_bigann_to_f32(data_path)
    print(f"[exact] Loaded X: {X.shape}, dtype={X.dtype}")

    # 预计算范数
    norms = np.sum(X.astype(np.float32) ** 2, axis=1).astype(np.float32)  # (N,)

    idx_all = np.full((N, k), -1, dtype=np.int64)
    dist_all = np.full((N, k), np.inf, dtype=np.float32)

    # 遍历查询块
    for a0 in range(0, N, query_chunk):
        a1 = min(N, a0 + query_chunk)
        A = X[a0:a1]                              # (a, D)
        a2 = norms[a0:a1]                         # (a,)
        A_g = np.arange(a0, a1, dtype=np.int64)   # 全局 id

        # 对该 A 块的临时 topk
        cur_idx = np.full((a1 - a0, k), -1, dtype=np.int64)
        cur_dist = np.full((a1 - a0, k), np.inf, dtype=np.float32)

        # 遍历库块
        for b0 in range(0, N, base_chunk):
            b1 = min(N, b0 + base_chunk)
            B = X[b0:b1]                          # (b, D)
            b2 = norms[b0:b1]
            B_g = np.arange(b0, b1, dtype=np.int64)

            add_idx, add_dist = _block_l2_topk_merge(A, B, a2, b2, k, A_g, B_g)
            cur_idx, cur_dist = _merge_topk(cur_idx, cur_dist, add_idx, add_dist, k)

        idx_all[a0:a1] = cur_idx
        dist_all[a0:a1] = cur_dist
        if (a0 // query_chunk) % 10 == 0:
            print(f"[exact] progress: {a1}/{N}")

    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, "exact_knn_idx.npy")
    dist_path = os.path.join(out_dir, "exact_knn_dist.npy")
    save_npy(idx_path, idx_all)
    save_npy(dist_path, dist_all)
    print(f"[exact] saved: {idx_path}")
    print(f"[exact] saved: {dist_path}")
    return idx_path, dist_path

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_b = sub.add_parser("build", help="Build exact kNN for dataset")
    ap_b.add_argument("--data", type=str, required=True)
    ap_b.add_argument("--k", type=int, required=True)
    ap_b.add_argument("--out_dir", type=str, required=True)
    ap_b.add_argument("--query-chunk", type=int, default=8192)
    ap_b.add_argument("--base-chunk", type=int, default=262144)

    args = ap.parse_args()

    if args.cmd == "build":
        build_exact_knn(
            data_path=args.data, k=args.k, out_dir=args.out_dir,
            query_chunk=args.query_chunk, base_chunk=args.base_chunk
        )

if __name__ == "__main__":
    main()
