import argparse
import os
import struct
import numpy as np
from typing import Tuple, Optional, List
from load import get_ext, load_bigann_to_f32, load_ibin, load_neighbors
from prun import prune_graph


# =========================
# NN-Descent 构图
# =========================

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple


def _nndescent_update_block(
    X: np.ndarray,
    neighbors: np.ndarray,
    k: int,
    max_cand: int,
    n_probes: int,
    start: int,
    end: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """
    单个 block 的更新：处理节点 [start, end) 的邻居列表更新。
    返回 (new_neighbors_block, updated_count)：
      - new_neighbors_block: shape (end-start, k)
      - updated_count: 这个 block 里有多少个节点发生了更新
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape

    block_size = end - start
    new_block = np.empty((block_size, k), dtype=np.int64)
    updated = 0

    for local_idx, i in enumerate(range(start, end)):
        ni = neighbors[i]
        # 只从前 n_probes 个最近邻居展开（防止候选爆炸）
        probes = ni[: min(n_probes, k)]

        cand_set = set(ni.tolist())
        for u in probes:
            if 0 <= u < N:
                cand_set.update(neighbors[u].tolist())
        cand_set.discard(i)

        if not cand_set:
            new_block[local_idx] = ni
            continue

        cand = np.fromiter(cand_set, dtype=np.int64)
        # 候选过多时随机截断到 max_cand
        if cand.size > max_cand:
            sel = rng.choice(cand.size, size=max_cand, replace=False)
            cand = cand[sel]

        Xi = X[i]
        Xcand = X[cand]                # (C, D)
        diff = Xcand - Xi
        dists = np.einsum("ij,ij->i", diff, diff)  # (C,)

        if cand.size <= k:
            new_idx = cand
            new_dist = dists
        else:
            part = np.argpartition(dists, kth=k-1)[:k]
            new_idx = cand[part]
            new_dist = dists[part]
            # 再按距离排序，方便下一轮 probes 选最近的
            srt = np.argsort(new_dist)
            new_idx = new_idx[srt]

        if not np.array_equal(neighbors[i], new_idx):
            updated += 1

        # 写入 block
        # 如果不足 k（理论上不会），用 -1 填充
        if new_idx.size < k:
            pad = -np.ones(k - new_idx.size, dtype=np.int64)
            new_block[local_idx] = np.concatenate([new_idx, pad], axis=0)
        else:
            new_block[local_idx] = new_idx

    return new_block, updated


def nndescent_build_mt(
    X: np.ndarray,
    k: int,
    iters: int,
    rng: np.random.Generator,
    max_cand: int | None = None,
    n_probes: int = 8,
    n_threads: int = 8,
) -> np.ndarray:
    """
    多线程 NN-Descent（带限流版）：

    参数：
      X         : (N, D) float32 数据集
      k         : 每个点的邻居数
      iters     : 迭代轮数
      max_cand  : 候选集合最大大小（默认 8*k）
      n_probes  : 每个点只从前 n_probes 个邻居的“邻居”中扩展候选
      n_threads : 线程数

    返回：
      neighbors: (N, k) int64 邻居索引
    """
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    k = min(k, N - 1)
    if max_cand is None or max_cand <= 0:
        max_cand = 8 * k

    neighbors = np.empty((N, k), dtype=np.int64)

    # 初始化随机邻接
    for i in range(N):
        cand = rng.choice(N - 1, size=k, replace=False)
        cand = np.where(cand >= i, cand + 1, cand)
        neighbors[i] = cand

    # 迭代
    for it in range(iters):
        print(f"[NN-Descent-mt] iter {it+1}/{iters}")
        new_neighbors = np.empty_like(neighbors)
        updated_total = 0

        # 分块：每个线程处理一段 [start, end)
        # 简单均分
        print(f"Number of threads working {n_threads}")
        block_size = (N + n_threads - 1) // n_threads
        tasks = []
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for t in range(n_threads):
                start = t * block_size
                end = min(N, (t + 1) * block_size)
                if start >= end:
                    continue
                # 给每个 block 一个不同 seed，保证随机性
                seed_block = int(rng.integers(0, 2**31 - 1))
                fut = ex.submit(
                    _nndescent_update_block,
                    X, neighbors, k, max_cand, n_probes,
                    start, end, seed_block
                )
                tasks.append((start, end, fut))

            # 收集结果
            for start, end, fut in tasks:
                block, upd = fut.result()
                new_neighbors[start:end] = block
                updated_total += upd

        neighbors = new_neighbors
        print(f"[NN-Descent-mt] updated nodes: {updated_total}/{N}")
        if updated_total == 0:
            print("[NN-Descent-mt] converged early.")
            break

    return neighbors


def nndescent_build(
    X: np.ndarray,
    k: int,
    iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    非高配版 NN-Descent（简化实现，便于阅读与调试）：

    - 初始化：每个点随机选 k 个不同邻居
    - 迭代 iters 轮：
      对于每个点 i，收集 neighbors[i] 及“邻居的邻居”的并集为候选，
      计算到所有候选的距离，选 k 个最近更新 neighbors[i]。
    - 返回 neighbors[int64][N,k]

    复杂度大约 O(iters * N * k^2 * D)，大数据集可能会慢，调试/中小规模用足够。
    """
    N, D = X.shape
    k = min(k, N - 1)
    neighbors = np.empty((N, k), dtype=np.int64)

    # 初始化随机邻接
    for i in range(N):
        # 随机选 k 个 != i
        cand = rng.choice(N - 1, size=k, replace=False)
        cand = np.where(cand >= i, cand + 1, cand)
        neighbors[i] = cand

    for it in range(iters):
        print(f"[NN-Descent] iter {it+1}/{iters}")
        updated = 0

        for i in range(N):
            ni = neighbors[i]
            cand = set(ni.tolist())
            for u in ni:
                if 0 <= u < N:
                    cand.update(neighbors[u].tolist())
            cand.discard(i)
            if not cand:
                continue
            cand = np.fromiter(cand, dtype=np.int64)
            # 距离计算
            Xi = X[i]
            Xcand = X[cand]
            diff = Xcand - Xi
            dists = np.einsum("ij,ij->i", diff, diff)  # (len(cand),)

            if cand.size <= k:
                # 候选不足 k，全保留
                new_idx = cand
                new_dist = dists
            else:
                order = np.argpartition(dists, kth=k-1)[:k]
                new_idx = cand[order]
                new_dist = dists[order]
                # 再排序
                srt = np.argsort(new_dist)
                new_idx = new_idx[srt]

            if not np.array_equal(neighbors[i], new_idx):
                neighbors[i] = new_idx
                updated += 1

        print(f"[NN-Descent] updated nodes: {updated}/{N}")
        if updated == 0:
            print("[NN-Descent] converged early.")
            break

    return neighbors


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build NN-Descent index
    ap_b = sub.add_parser("build", help="Build NN-Descent KNN graph index")
    ap_b.add_argument("--data", type=str, required=True, help="base dataset (.fbin/.bin/.u8bin/.i8bin)")
    ap_b.add_argument("--k", type=int, required=True, help="K in KNN graph")
    ap_b.add_argument("--iters", type=int, default=8, help="NN-Descent iterations")
    ap_b.add_argument("--out", type=str, required=True, help="output neighbors_nnd.npy")
    ap_b.add_argument("--seed", type=int, default=42)
    ap_b.add_argument("--normalize", action="store_true", help="L2-normalize data before NN-Descent")
    ap_b.add_argument("--n_threads", type=int, default=80,
                  help="number of threads for NN-Descent")

    # build NN-Descent index with pruning
    ap_p = sub.add_parser("buildWithPrune", help="Build NN-Descent KNN graph index with pruning")
    ap_p.add_argument("--data", type=str, required=True, help="base dataset (.fbin/.bin/.u8bin/.i8bin)")
    ap_p.add_argument("--k", type=int, required=True, help="K in KNN graph")
    ap_p.add_argument("--iters", type=int, default=8, help="NN-Descent iterations")
    ap_p.add_argument("--out", type=str, required=True, help="output neighbors_nnd.npy")
    ap_p.add_argument("--seed", type=int, default=42)
    ap_p.add_argument("--normalize", action="store_true", help="L2-normalize data before NN-Descent")
    ap_p.add_argument("--prune-max-degree", type=int, required=True, help="max degree in prunned KNN graph")
    ap_p.add_argument("--n_threads", type=int, default=80,
                  help="number of threads for NN-Descent")


    args = ap.parse_args()

    if args.cmd == "build":
        X, N, D = load_bigann_to_f32(args.data)
        print(f"[build] loaded data: {X.shape}, dtype={X.dtype}")
        if args.normalize:
            print("[build] L2-normalizing data")
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X /= n

        rng = np.random.default_rng(args.seed)
        neighbors = nndescent_build_mt(
            X,
            k=args.k,
            iters=args.iters,
            rng=rng,
            max_cand=args.k * args.k,   # 你也可以做成参数
            n_probes=args.k,
            n_threads=args.n_threads,           # 或者改成机器 CPU 核数
        )
        np.save(args.out, neighbors.astype(np.int64))
        print(f"[build] NN-Descent index saved to: {args.out}")
    
    elif args.cmd == "buildWithPrune":
        X, N, D = load_bigann_to_f32(args.data)
        print(f"[build] loaded data: {X.shape}, dtype={X.dtype}")
        if args.normalize:
            print("[build] L2-normalizing data")
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X /= n

        rng = np.random.default_rng(args.seed)
        neighbors = nndescent_build_mt(
            X,
            k=args.k,
            iters=args.iters,
            rng=rng,
            max_cand=args.k * args.k,   # 你也可以做成参数
            n_probes=args.k,
            n_threads=args.n_threads,           # 或者改成机器 CPU 核数
        )

        # from prune import prune_graph 
        max_deg = args.k  # 或者额外加一个 --prune-max-degree 参数
        print(f"[build] pruning NN-Descent graph with max_degree={max_deg}")
        neighbors = prune_graph(X, neighbors, max_degree=max_deg)

        np.save(args.out, neighbors.astype(np.int64))
        print(f"[build] NN-Descent index saved to: {args.out}")


if __name__ == "__main__":
    main()
