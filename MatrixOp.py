import argparse
import torch
import math
import triton
import triton.language as tl

@torch.inference_mode()
def measure_cuda_time(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        torch.cuda.synchronize()
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / repeat
    return ms

def get_dtype_info(dtype_name):
    if dtype_name == "fp32":
        return torch.float32, 4
    elif dtype_name == "int32":
        return torch.int32, 4
    elif dtype_name == "int8":
        return torch.int8, 1
    else:
        raise ValueError("unsupported dtype")


@triton.jit
def l2sq_pairs_kernel(
    X_ptr, Y_ptr, O_ptr,
    IX_ptr, IY_ptr,
    n: tl.constexpr, m: tl.constexpr, d: tl.constexpr,
    stride_xn, stride_xd,
    stride_ym, stride_yd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)   # 当前线程 = 第 pid 个 pair
    i = tl.load(IX_ptr + pid)     # X 的行
    j = tl.load(IY_ptr + pid)     # Y 的行

    # 可选越界保护
    cond = (i >= 0) & (i < n) & (j >= 0) & (j < m)
    if ~cond:
        tl.store(O_ptr + pid, 0.0)
        return

    x_row_ptr = X_ptr + i * stride_xn
    y_row_ptr = Y_ptr + j * stride_ym

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, d, BLOCK_D):
        offs = k + tl.arange(0, BLOCK_D)
        mask = offs < d
        x = tl.load(x_row_ptr + offs * stride_xd, mask=mask, other=0.0)
        y = tl.load(y_row_ptr + offs * stride_yd, mask=mask, other=0.0)
        diff = x - y
        acc += tl.sum(diff * diff, axis=0)

    tl.store(O_ptr + pid, acc)

@torch.inference_mode()
def l2sq_pairwise(A, B, ix, iy, block_d=128, num_warps=4, out=None):
    """
    A:(n,d), B:(m,d) 都在 CUDA；ix,iy:(P,) 为 CUDA long 索引。
    每个线程只计算 1 个 pair 的 L2^2，返回 (P,) float32（或写入 out）。
    整型输入会在函数里上浮到 float32 计算，避免溢出与类型限制。
    """
    assert A.dim() == 2 and B.dim() == 2 and A.size(1) == B.size(1)
    assert ix.shape == iy.shape and ix.device.type == "cuda" and iy.device.type == "cuda"
    n, d = A.shape
    m = B.shape[0]
    dev = A.device
    # 计算用 float32（输入若是 int32/int8，会临时上浮；不复制回原 tensor）
    Af = A.float() if not A.is_floating_point() else A
    Bf = B.float() if not B.is_floating_point() else B

    P = ix.numel()
    if out is None:
        out = torch.empty(P, device=dev, dtype=torch.float32)

    grid = (triton.cdiv(P, 1),)
    l2sq_pairs_kernel[grid](
        Af, Bf, out,
        ix, iy,
        n, m, d,
        Af.stride(0), Af.stride(1),
        Bf.stride(0), Bf.stride(1),
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out

# def l2sq_pairwise_shard(X, Y, pair_idx):


def l2sq_pairwise_expand(X, Y, pair_idx):
    x = X.index_select(0, pair_idx[:,0])
    y = Y.index_select(0, pair_idx[:,1])
    print(f"pairwise computation: shape of matrix x: {x.shape}, shape of matrxt y: {y.shape}")
    # 若为整型，则提升为float计算距离
    if not x.is_floating_point():
        x = x.float(); y = y.float()
    return ((x - y) ** 2).sum(dim=-1)



def l2sq_vec_mat(X, Y):
    if not X.is_floating_point():
        X = X.float(); Y = Y.float()
    dot = X @ Y.t()
    xn = (X**2).sum(dim=1, keepdim=True)
    yn = (Y**2).sum(dim=1).unsqueeze(0)
    return xn + yn - 2*dot


def l2sq_single_vec(x, Y):
    """单个向量 x 与矩阵 Y 的 L2^2 距离"""
    if not x.is_floating_point():
        x = x.float(); Y = Y.float()
    dot = (x.unsqueeze(0) @ Y.t())  # (1,m)
    xn = (x**2).sum()
    yn = (Y**2).sum(dim=1)
    return xn + yn - 2*dot.squeeze(0)

def multi_vecmat_individual(X, Y):
    """对 X 中的每个 x_i 分别计算 l2sq(x_i, Y)"""
    results = []
    for i in range(X.size(0)):
        dists = l2sq_single_vec(X[i], Y)
        results.append(dists)
    return results


def l2sq_mat_mat(X, Y):
    if not X.is_floating_point():
        X = X.float(); Y = Y.float()
    dot = X @ Y.t()
    print(f"matrix computation: shape of matrix X: {X.shape}, shape of matrxt Y: {Y.shape}")
    xn = (X**2).sum(dim=1, keepdim=True)
    yn = (Y**2).sum(dim=1).unsqueeze(0)
    return xn + yn - 2*dot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32","int32","int8"])
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=5_000_000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    dtype, dtype_bytes = get_dtype_info(args.dtype)
    print(f"[Device] {torch.cuda.get_device_name(0)}  dtype={args.dtype}")

    X = torch.randn(args.n, args.d, device="cuda", dtype=torch.float32)
    Y = torch.randn(args.m, args.d, device="cuda", dtype=torch.float32)
    if not torch.is_floating_point(torch.tensor([], dtype=dtype)):
        # 随机整数，范围防止溢出
        X = (X * 100).to(dtype)
        Y = (Y * 100).to(dtype)
    else:
        X = X.to(dtype); Y = Y.to(dtype)

    # 1) Pairwise
    P = args.pairs
    pair_idx = torch.randint(0, args.n, (P,2), device="cuda", dtype=torch.long)
    def run_pair_expand():
        _ = l2sq_pairwise_expand(X, Y, pair_idx)
    ms_pair_expand_as_matrix = measure_cuda_time(run_pair_expand, repeat=args.repeat)
    print(f"[Pairwise_Expanded_Matrix] {P} pairs -> {ms_pair_expand_as_matrix:.3f} ms")

    ix = torch.randint(0, args.n, (P,), device="cuda", dtype=torch.long)
    iy = torch.randint(0, args.m, (P,), device="cuda", dtype=torch.long)
    # warmup
    # l2sq_pairswise(A, B, ix[:1024], iy[:1024])
    # torch.cuda.synchronize()
    def run_pair():
        _ = l2sq_pairwise(X, Y, ix, iy, block_d=128, num_warps=4)
    t1 = measure_cuda_time(run_pair, repeat=args.repeat)
    print(f"[Pairwise] {P} pairs -> {t1:.3f} ms")

    # 2) Vec-Mat
    XB = X[:args.batch]
    def run_vm():
        _ = l2sq_vec_mat(XB, Y)
    ms_vecs_as_matrix = measure_cuda_time(run_vm, repeat=args.repeat)
    print(f"[Vec-Mat] {args.batch}x{args.m} -> {ms_vecs_as_matrix:.3f} ms")

    def run_multi_vecmat():
        _ = multi_vecmat_individual(X[:args.batch], Y)
    t2 = measure_cuda_time(run_multi_vecmat, args.repeat)
    print(f"Multi-VecMat: {args.batch} independent (x,Y) -> {t2:.3f} ms")

    # 3) Mat-Mat
    n_small = min(args.n, 8192)
    m_small = min(args.m, 8192)
    def run_mm():
        _ = l2sq_mat_mat(X[:n_small], Y[:m_small])
    t3 = measure_cuda_time(run_mm, repeat=args.repeat)
    print(f"[Mat-Mat] {n_small}x{m_small} -> {t3:.3f} ms")

    print("-----")
    print(f"Overall time (smaller=better):")
    print(f"Pairwise: {t1:.3f} ms,  Vec-Mat: {t2:.3f} ms,  Mat-Mat: {t3:.3f} ms")
    print(f"Relative Throughput (smaller=better):")
    base = min(t1, t2, t3)
    print(f"Pairwise: {t1/base:.1f}×,  Vec-Mat: {t2/base:.1f}×,  Mat-Mat: {t3/base:.1f}×")


if __name__ == "__main__":
    main()
