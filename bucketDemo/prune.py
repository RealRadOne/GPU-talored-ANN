import numpy as np
from load import get_ext, load_bigann_to_f32, load_ibin, load_neighbors

def _l2sqr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.dot(diff))

def prune_graph(
    X: np.ndarray,
    neighbors: np.ndarray,
    max_degree: int | None = None,
) -> np.ndarray:
    """
    对图索引做 pruning。
    X: (N, D) float32，数据集向量
    neighbors: (N, M) int，邻接表（如 NN-Descent 或你构建的 neighbors.npy）
      - 可包含 -1 占位
    max_degree: 最多保留的邻居数 y；如果为 None 或 <=0，则不做上限裁剪（只做 occlusion pruning）

    返回 pruned_neighbors: (N, M) int64，未用位置填 -1
    """

    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    nb = np.asarray(neighbors).copy()
    if nb.ndim != 2:
        raise ValueError("neighbors must be 2D")
    if nb.shape[0] != N:
        raise ValueError(f"N mismatch: X has {N}, neighbors has {nb.shape[0]}")
    if nb.dtype.kind not in "iu":
        raise ValueError("neighbors must be integer dtype")

    M = nb.shape[1]
    pruned = -np.ones((N, M), dtype=np.int64)

    for v in range(N):
        # 取出 v 的邻居，过滤掉 -1、自身，以及越界 id
        nbrs = nb[v]
        mask = (nbrs >= 0) & (nbrs < N) & (nbrs != v)
        nbrs = nbrs[mask]
        if nbrs.size == 0:
            continue

        x_v = X[v]

        # 先按 dist(v, ·) 从近到远排序
        diff = X[nbrs] - x_v
        dist_v = np.einsum("ij,ij->i", diff, diff)  # (k,)
        order = np.argsort(dist_v)
        nbrs = nbrs[order]
        dist_v = dist_v[order]

        k = nbrs.size
        removed = np.zeros(k, dtype=bool)

        # occlusion pruning
        for i in range(k):
            if removed[i]:
                continue
            u = int(nbrs[i])
            x_u = X[u]
            for j in range(i + 1, k):
                if removed[j]:
                    continue
                z = int(nbrs[j])
                # 比较 dist(u, z) 和 dist(v, z)
                d_uz = _l2sqr(x_u, X[z])
                if d_uz < dist_v[j]:
                    # 把 z 从 v 的邻居中移除
                    removed[j] = True

        kept = nbrs[~removed]
        kept_dist = dist_v[~removed]

        # 如果有 max_degree，上限 y
        if max_degree is not None and max_degree > 0 and kept.size > max_degree:
            kept = kept[:max_degree]
            kept_dist = kept_dist[:max_degree]

        # 写回 pruned[v]，其余位置保持 -1
        pruned[v, :kept.size] = kept

    return pruned


# =========================
# Prune Existing
# =========================
def prune_existing_index(
    data_path: str,
    index_in: str,
    index_out: str,
    max_degree: int,
):
    """
    读出 NN-descent 直接 build（未 prune）的 index，
    在内存里做 prune，然后再存成新的 npy。
    """
    # 读 base 数据（和 build 时的 base 要一致）
    X, N, D = load_bigann_to_f32(data_path)
    print(f"[prune_existing] loaded base: {X.shape}, dtype={X.dtype}")

    # 读未 prune 的 index
    nb = load_neighbors(index_in)
    print(f"[prune_existing] loaded index: {nb.shape}, dtype={nb.dtype}")
    if nb.shape[0] != N:
        raise ValueError(f"N mismatch: base={N}, index={nb.shape[0]}")

    # 做 pruning
    print(f"[prune_existing] pruning with max_degree={max_degree} ...")
    pruned = prune_graph(X, nb, max_degree=max_degree)

    # 存结果
    np.save(index_out, pruned.astype(np.int64))
    print(f"[prune_existing] pruned index saved to: {index_out}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prune an existing index
    ap_pe = sub.add_parser("prune_existing", help="Prune an existing NN-descent index and save")
    ap_pe.add_argument("--data", type=str, required=True,
                    help="base dataset used when building the index (.fbin/.u8bin/...)")
    ap_pe.add_argument("--index_in", type=str, required=True,
                    help="NN-descent raw index (not pruned) npy")
    ap_pe.add_argument("--index_out", type=str, required=True,
                    help="output pruned index npy")
    ap_pe.add_argument("--max_degree", type=int, required=True,
                    help="max neighbors per node after pruning")


    args = ap.parse_args()

    
    if args.cmd == "prune":
        prune_existing_index(
            data_path=args.data,
            index_in=args.index_in,
            index_out=args.index_out,
            max_degree=args.max_degree,
        )


if __name__ == "__main__":
    main()
