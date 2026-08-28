import argparse
import os
import struct
import numpy as np
from typing import Tuple, List, Optional
from load import get_ext, load_bigann_to_f32, load_ibin, load_neighbors

# --------------------------
# Metric helpers
# --------------------------

def l2sqr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.dot(diff))

# --------------------------
# Graph Search (efSearch style)
# --------------------------

def graph_search_single(
    q: np.ndarray,
    base: np.ndarray,
    graph: np.ndarray,  # [N, M] uint32/int64
    topk: int,
    ef: int,
    rng: np.random.Generator,
    num_entry: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单 efSearch 风格图检索：从 num_entry 个随机起点出发，
    用 best-list + candidate min-heap 拓展，返回 topk (indices,dists)。
    """
    N, M = graph.shape
    visited = np.zeros(N, dtype=bool)

    import heapq
    cand_heap: List[Tuple[float, int]] = []
    best_idx = np.empty(0, dtype=np.int64)
    best_dst = np.empty(0, dtype=np.float32)

    entries = rng.integers(0, N, size=max(1, num_entry), endpoint=False)
    entries = np.unique(entries)
    for s in entries:
        if not visited[s]:
            ds = l2sqr(q, base[s])
            visited[s] = True
            heapq.heappush(cand_heap, (ds, int(s)))
            if best_idx.size < ef:
                best_idx = np.append(best_idx, s)
                best_dst = np.append(best_dst, ds)
            else:
                worst_pos = int(np.argmax(best_dst))
                if ds < best_dst[worst_pos]:
                    best_idx[worst_pos] = s
                    best_dst[worst_pos] = ds

    while cand_heap:
        d_u, u = heapq.heappop(cand_heap)
        if best_idx.size >= ef:
            worst = float(np.max(best_dst))
            if d_u > worst:
                break

        nbrs = graph[u]
        for v_int in nbrs:
            v = int(v_int)
            if v < 0 or v >= N:
                continue
            if visited[v]:
                continue
            visited[v] = True
            dv = l2sqr(q, base[v])
            heapq.heappush(cand_heap, (dv, v))

            if best_idx.size < ef:
                best_idx = np.append(best_idx, v)
                best_dst = np.append(best_dst, dv)
            else:
                worst_pos = int(np.argmax(best_dst))
                if dv < best_dst[worst_pos]:
                    best_idx[worst_pos] = v
                    best_dst[worst_pos] = dv

    if best_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    order = np.argsort(best_dst)
    order = order[:topk]
    return best_idx[order], best_dst[order]


# =========================
# Recall vs groundtruth(.ibin)
# =========================

def recall_vs_gt(pred: np.ndarray, gt: np.ndarray, topk: int) -> Tuple[float, float]:
    """
    pred: [nq, topk] int64
    gt:   [nq, k_gt] int32
    指标：
      - Recall@topk(命中首真近邻)
      - Recall@topk(命中前 k_gt 任一)
    """
    nq = pred.shape[0]
    hit_first = 0
    hit_any = 0
    for i in range(nq):
        if gt[i, 0] in pred[i]:
            hit_first += 1
        if np.intersect1d(pred[i], gt[i], assume_unique=False).size > 0:
            hit_any += 1
    r_first = hit_first / float(nq) if nq > 0 else 0.0
    r_any = hit_any / float(nq) if nq > 0 else 0.0

    k = min(topk, pred.shape[1], gt.shape[1])
    total = 0.0
    for i in range(nq):
        a = pred[i, :k]
        b = gt[i, :k]
        # 过滤 -1 padding
        a = a[a >= 0]
        b = b[b >= 0]
        # 不看顺序：用集合/去重
        hit = len(set(a.tolist()).intersection(set(b.tolist())))
        total += hit / float(k)
    recall = total / float(nq)
    
    return r_first, r_any, recall


# --------------------------
# CLI functions
# --------------------------

def search_graph_recall(
    base_path: str,
    query_path: str,
    index_path: str,
    gt_path: str,
    topk: int,
    ef: int,
    num_entry: int,
    normalize: bool = False,
):
    base, N, D = load_bigann_to_f32(base_path)
    Q, nq, dq = load_bigann_to_f32(query_path)
    if dq != D:
        raise ValueError(f"Dim mismatch: base D={D}, query D={dq}")
    neighbors = load_neighbors(index_path)
    if neighbors.shape[0] != N:
        raise ValueError("Index N mismatch with base")
    GT, gt_nq, k_gt = load_ibin(gt_path)
    if gt_nq != nq:
        raise ValueError("GT nq mismatch with query")

    if normalize:
        print("[eval] L2-normalizing base & queries")
        def norm_rows(X):
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X /= n
        norm_rows(base)
        norm_rows(Q)

    rng = np.random.default_rng(42)
    ef = max(ef, topk)
    preds = np.empty((nq, topk), dtype=np.int64)
    print("[eval] start searching ...")
    for i in range(nq):
        if i % 1000 == 0:
            print(f"  query {i}/{nq}")
        idx, _ = graph_search_single(
            q=Q[i], base=base, graph=neighbors, topk=topk, ef=ef, rng=rng, num_entry=num_entry
        )
        if idx.size < topk:
            pad = -np.ones(topk - idx.size, dtype=np.int64)
            idx = np.concatenate([idx, pad], axis=0)
        preds[i] = idx[:topk]

    r_first, r_any, recall = recall_vs_gt(preds, GT, topk)
    print("\n======== Recall vs GT ========")
    print(f"Recall@{topk} (hit GT[:,0])         : {r_first:.6f}")
    print(f"Recall@{topk} (hit any of GT[:,:]) : {r_any:.6f}")
    print(f"Recall@{topk} (recall mean overlap): {recall:.6f}")



"""
比较两个索引的前 k 个邻居（忽略顺序），输出：
    - MeanOverlap@k: mean(|A∩B|/k)
    - HitsAny@k: 至少有1个共同邻居的比例
    - SetEqual@k: 两个集合完全一致的比例
这可以用来：
    - NN-Descent 索引 vs 精确 KNN 索引
    - NN-Descent 索引 vs 自己构建的图索引
    - 自己构建的图索引 vs 精确 KNN 索引
"""
def compare_indices_setwise(
    index_a_path: str,
    index_b_path: str,
    k_cmp: Optional[int] = None,
    save_csv: Optional[str] = None,
):
    A = load_neighbors(index_a_path)
    B = load_neighbors(index_b_path)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("indices must be 2D")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"N mismatch: A={A.shape[0]}, B={B.shape[0]}")
    N = A.shape[0]
    Ma = A.shape[1]
    Mb = B.shape[1]

    if k_cmp is None:
        k = min(Ma, Mb)
    else:
        k = min(int(k_cmp), Ma, Mb)
    if k <= 0:
        raise ValueError("k_cmp must be > 0")

    Ak = A[:, :k].astype(np.int64, copy=False)
    Bk = B[:, :k].astype(np.int64, copy=False)

    overlaps = np.empty(N, dtype=np.float32)
    hits_any = 0
    set_equal = 0
    rows = []

    for i in range(N):
        a = Ak[i]
        b = Bk[i]
        a = a[a >= 0]
        b = b[b >= 0]
        SA = set(a.tolist())
        SB = set(b.tolist())
        inter_sz = len(SA.intersection(SB))
        overlaps[i] = inter_sz / float(k)

        if inter_sz >= 1:
            hits_any += 1
        if len(SA) == len(SB) == k and inter_sz == k:
            set_equal += 1

        if save_csv is not None:
            rows.append((i, inter_sz, len(SA), len(SB), overlaps[i]))

    mean_overlap = float(overlaps.mean()) if N > 0 else 0.0
    hits_any_rate = hits_any / float(N) if N > 0 else 0.0
    set_equal_rate = set_equal / float(N) if N > 0 else 0.0

    print("\n======= Index Comparison (set-wise) =======")
    print(f"N={N}, k={k} (A:{Ma}, B:{Mb})")
    print(f"MeanOverlap@{k} (|A∩B|/k) : {mean_overlap:.6f}")
    print(f"HitsAny@{k}               : {hits_any_rate:.6f}")
    print(f"SetEqual@{k}              : {set_equal_rate:.6f}")

    if save_csv is not None:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w") as f:
            f.write("node,intersect_size,deg_A,deg_B,overlap\n")
            for (i, inter_sz, la, lb, ov) in rows:
                f.write(f"{i},{inter_sz},{la},{lb},{ov:.6f}\n")
        print(f"[compare] per-node stats saved to: {save_csv}")




# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # search an existing index
    ap_s = sub.add_parser("search", help="Run graph search")
    ap_s.add_argument("--base", type=str, required=True, help="Path to base vectors (.fbin/.bin/.u8bin/.i8bin)")
    ap_s.add_argument("--query", type=str, required=True, help="Path to query vectors (.fbin/.bin/.u8bin/.i8bin)")
    ap_s.add_argument("--index", type=str, required=True, help="neighbors.npy (N, M)")
    ap_s.add_argument("--gt", type=str, required=True, help="groundtruth .ibin (nq, k_gt)")
    ap_s.add_argument("--topk", type=int, default=10, help="return top-k results")
    ap_s.add_argument("--ef", type=int, default=200, help="efSearch breadth (>= topk)")
    ap_s.add_argument("--num-entry", type=int, default=4, help="number of random entry points per query")
    # ap_s.add_argument("--seed", type=int, default=42)
    ap_s.add_argument("--normalize", action="store_true", help="L2-normalize rows for cosine search")
    # ap_s.add_argument("--metric", type=str, default="l2", choices=["l2"])

    # compare two indices
    ap_c = sub.add_parser("compare", help="Compare two indices (set-wise)")
    ap_c.add_argument("--index1", type=str, required=True, help="first index npy")
    ap_c.add_argument("--index2", type=str, required=True, help="second index npy")
    ap_c.add_argument("--k_cmp", type=int, default=None, help="use top-k for comparison")
    ap_c.add_argument("--save_csv", type=str, default=None)


    args = ap.parse_args()

    if args.cmd == "search":
        search_graph_recall(
            base_path=args.base,
            query_path=args.query,
            index_path=args.index,
            gt_path=args.gt,
            topk=args.topk,
            ef=args.ef,
            num_entry=args.num_entry,
            normalize=args.normalize,
        )

    elif args.cmd == "compare":
        compare_indices_setwise(
            index_a_path=args.index1,
            index_b_path=args.index2,
            k_cmp=args.k_cmp,
            save_csv=args.save_csv,
        )


if __name__ == "__main__":
    main()
