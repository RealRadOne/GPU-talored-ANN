#pragma once
// ============== Phase 12: Per-Vector KNN via Tensor Core Bucket MatMul ==============

// 计算 X 中每行的 L2 范数平方: norms[i] = sum_d X[i,d]^2
// 模板：DataT 是输入元素类型 (uint8 / int8 / int32 / uint32 / float / half)，
//       内部累加和输出仍用 float (norms 始终 fp32)。
template <typename DataT>
__global__ void compute_row_norms_kernel(
    const DataT* __restrict__ X,
    float* __restrict__ norms,
    int64_t N, int64_t D)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const DataT* row = X + i * D;
    float s = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float v = static_cast<float>(row[d]);
        s += v * v;
    }
    norms[i] = s;
}

// 按 int32 索引从 src (DataT) 中 gather 行，并转换成 float 写入 dst
//   dst[k, :] = float(src[idx[k], :])
// 这样 d_X_full 可以以 DataT (省 GPU 显存)，per-bucket A/B 仍是 float (走 cuBLAS fp32)。
template <typename DataT>
__global__ void gather_rows_int32(
    const DataT* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst,
    int64_t n_dst_rows, int64_t D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_dst_rows * D;
    if (tid >= total) return;
    int64_t row = tid / D;
    int64_t col = tid % D;
    dst[tid] = static_cast<float>(src[static_cast<int64_t>(idx[row]) * D + col]);
}

// === Stage 2: INT8 IMMA path helpers ===
// gather rows 不做类型转换：dst 与 src 同 DataT，cuBLAS INT8 GEMM 直接吃。
template <typename DataT>
__global__ void gather_rows_raw(
    const DataT* __restrict__ src,
    const int32_t* __restrict__ idx,
    DataT* __restrict__ dst,
    int64_t n_dst_rows, int64_t D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_dst_rows * D;
    if (tid >= total) return;
    int64_t row = tid / D;
    int64_t col = tid % D;
    dst[tid] = src[static_cast<int64_t>(idx[row]) * D + col];
}

// uint8 → int8 in-place 平移：每个 byte 减 128，结果按 int8 解释。
//   uint8 0   → -128  (0x80 unchanged byte 模式)
//   uint8 128 →  0
//   uint8 255 →  127
// 调用方在 shift 之后用 reinterpret_cast<int8_t*> 拿同一块显存的 int8 视图。
// L2 距离平移不变，只要 norms 和 dot 都从同一个 (shifted) 视图算出来即可。
__global__ void shift_uint8_to_int8_inplace(
    uint8_t* __restrict__ data, int64_t total)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    // (int)data - 128 然后回写 uint8：byte pattern 与 (int8)(int - 128) 一致
    int v = static_cast<int>(data[tid]) - 128;
    data[tid] = static_cast<uint8_t>(v);
}

// ====================================================================
// RAFT select_k path: dots → dist 原地转换 + RAFT select_k + gather global id
// ====================================================================

// 原地把 dots（int32 或 float）改写成 float dist = norms_pool[j] - 2*dots[i,j]，
// 顺便把 self-loop（cand_gid == my_gid）置 +inf 让 select_k 自动跳过。
// 复用 d_dots 的同一段显存（4 字节宽度相同）→ 零额外 buffer。
template<typename DotsT>
__global__ void dots_to_dist_inplace_kernel(
    DotsT* dots,                                 // (bucket_size, pool_size)
    const float* __restrict__ norms_pool,
    const int32_t* __restrict__ global_ids_bucket,
    const int32_t* __restrict__ global_ids_pool,
    int bucket_size, int pool_size)
{
    static_assert(sizeof(DotsT) == sizeof(float),
                  "DotsT must be 4 bytes for in-place reuse");
    int64_t tid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(bucket_size) * pool_size;
    if (tid >= total) return;

    int i = static_cast<int>(tid / pool_size);
    int j = static_cast<int>(tid - static_cast<int64_t>(i) * pool_size);

    float dot = static_cast<float>(dots[tid]);
    float dist = norms_pool[j] - 2.0f * dot;
    if (global_ids_pool[j] == global_ids_bucket[i]) dist = INFINITY;

    reinterpret_cast<float*>(dots)[tid] = dist;
}

// RAFT select_k 输出的是 pool 内的 local index（0..pool_size-1）；
// 这个 kernel 把它映射成 global id 写到最终 out_neighbors。
__global__ void gather_global_ids_kernel(
    const int32_t* __restrict__ out_local_idx,    // (bucket_size, K)
    const int32_t* __restrict__ global_ids_pool,  // (pool_size,)
    int32_t* __restrict__ out_neighbors,          // (bucket_size, K)
    int64_t total)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int32_t local = out_local_idx[tid];
    out_neighbors[tid] = (local >= 0) ? global_ids_pool[local] : -1;
}

// RAFT-based dispatcher：复用现有 d_dots buffer 当 dist 输入。
// 比自己写的 warp kernel 性能更好（RAFT 自动在 radix-select / warp-sort 之间选）。
// 唯一额外开销：dots→dist 转换（每 bucket ~10 MB DRAM 写）+ select_k 内部读
// + global_id gather（K × bucket_size 个 int32）。
//
// 调用方需要分配两个额外 per-slot buffer：
//   - d_select_idx : (max_bucket_size × M) int32 — RAFT 输出的 local pool idx
//   - d_select_dist: (max_bucket_size × M) float — RAFT 必填距离输出（我们不用）
template<typename DotsT>
static inline void launch_extract_topM_raft(
    raft::resources& res,
    cudaStream_t stream,
    DotsT*         d_dots,                  // 原地改写为 float dist
    const float*   d_norms_pool,
    const int32_t* d_ids_bucket,
    const int32_t* d_ids_pool,
    int32_t* d_select_idx,                  // RAFT idx 输出 scratch
    float*   d_select_dist,                 // RAFT dist 输出 scratch (也是最终距离输出)
    int32_t* d_out_neighbors,               // 最终输出 (邻居 global id)
    float*   d_out_distances,               // 可选, nullptr 表示不导出距离
    int bucket_size, int pool_size, int actual_M)
{
    // 1) dots → dist (in-place, 含 self-loop +inf)
    {
        constexpr int threads = 256;
        int64_t total = static_cast<int64_t>(bucket_size) * pool_size;
        int64_t blocks = (total + threads - 1) / threads;
        dots_to_dist_inplace_kernel<DotsT><<<blocks, threads, 0, stream>>>(
            d_dots, d_norms_pool, d_ids_bucket, d_ids_pool,
            bucket_size, pool_size);
    }

    // 2) RAFT select_k: kAuto 会按 (batch, len, k) 自动选 radix-select / warp-sort
    raft::resource::set_cuda_stream(res, stream);

    auto in_val = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        reinterpret_cast<const float*>(d_dots),
        static_cast<int64_t>(bucket_size),
        static_cast<int64_t>(pool_size));
    auto out_val = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        d_select_dist,
        static_cast<int64_t>(bucket_size),
        static_cast<int64_t>(actual_M));
    auto out_idx = raft::make_device_matrix_view<int32_t, int64_t, raft::row_major>(
        d_select_idx,
        static_cast<int64_t>(bucket_size),
        static_cast<int64_t>(actual_M));

    raft::matrix::select_k<float, int32_t>(
        res,
        in_val,
        std::nullopt,        // in_idx: 隐式 0..pool_size-1
        out_val,
        out_idx,
        /*select_min=*/ true,
        /*sorted=*/    true);  // 输出按距离升序，跟原 kernel 行为一致

    // 3) local pool idx → global id
    {
        constexpr int threads = 256;
        int64_t total = static_cast<int64_t>(bucket_size) * actual_M;
        int64_t blocks = (total + threads - 1) / threads;
        gather_global_ids_kernel<<<blocks, threads, 0, stream>>>(
            d_select_idx, d_ids_pool, d_out_neighbors, total);
    }

    // 4) (可选) 把 RAFT 选出的距离拷到调用方提供的 d_out_distances
    if (d_out_distances != nullptr) {
        size_t nbytes = static_cast<size_t>(bucket_size) * actual_M * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_out_distances, d_select_dist,
                                   nbytes, cudaMemcpyDeviceToDevice, stream));
    }
}

// ====================================================================
// Warp-cooperative top-K kernel
// ====================================================================
// 一个 warp（32 lane）共同处理一个 query。每 lane 在寄存器里持有
// K_PER_LANE = K_OUT/32 个元素，全展开排序——彻底干掉 local memory。
//
// Invariants:
//   - lane k 的 reg_dist[0..K_PER_LANE-1] 升序
//   - 跨 lane: lane 0 持最小的 K_PER_LANE 个，lane 31 持最大的 K_PER_LANE 个
//   - threshold = lane 31 的 reg_dist[K_PER_LANE-1]，broadcast 给所有 lane
//
// 主循环每次取 32 个 candidate（每 lane 一个），先 ballot 快速过滤
// （绝大多数 batch 0 个 qualified，整批跳过）；少数 qualified 的按
// cascade 模式插入：找到目标 lane → 该 lane 全展开 bubble → 被挤掉的
// max 通过 __shfl_up_sync 传给下一 lane → 一路到 lane 31 把全局 max 丢掉。
//
// 模板参数 K_OUT 必须是 32 的倍数（K_OUT ∈ {32,64,128,256,512,1024}）。
// K_PER_LANE = K_OUT/32, 最大 32（K_OUT=1024）→ ~94 reg/lane, V100 占用 ~50%。
// DotsT = int32_t（INT8 IMMA 路径）或 float（FP32 TF32 路径）。
template<int K_OUT, typename DotsT>
__global__ void warp_topK_from_dots_kernel(
    const DotsT* __restrict__ dots,
    const float* __restrict__ norms_pool,
    const int32_t* __restrict__ global_ids_bucket,
    const int32_t* __restrict__ global_ids_pool,
    int32_t* __restrict__ out_neighbors,
    float*   __restrict__ out_dists,   // 可选, NULL 表示不写距离
    int bucket_size, int pool_size, int K)
{
    static_assert(K_OUT % 32 == 0 && K_OUT >= 32,
                  "K_OUT must be a multiple of 32 (>=32)");
    constexpr int LANES = 32;
    constexpr int K_PER_LANE = K_OUT / LANES;
    constexpr unsigned ALL = 0xFFFFFFFFu;

    const int warps_per_block = blockDim.x / LANES;
    const int warp_in_block   = threadIdx.x / LANES;
    const int lane            = threadIdx.x & (LANES - 1);
    const int query_idx       = blockIdx.x * warps_per_block + warp_in_block;
    if (query_idx >= bucket_size) return;

    const int32_t my_gid = global_ids_bucket[query_idx];

    // 每 lane 的 K_PER_LANE 个 slot 进寄存器（K_PER_LANE 编译期已知，
    // 索引在全展开后都是常量）。ptxas 报告里这部分应当 STACK:0。
    float   reg_dist[K_PER_LANE];
    int32_t reg_id  [K_PER_LANE];

    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        reg_dist[p] = INFINITY;
        reg_id  [p] = -1;
    }

    float threshold = INFINITY;

    const DotsT* dots_row = dots + static_cast<int64_t>(query_idx) * pool_size;

    for (int j_base = 0; j_base < pool_size; j_base += LANES) {
        int  j     = j_base + lane;
        bool valid = (j < pool_size);

        float   my_dist;
        int32_t my_id;
        if (valid) {
            float d = static_cast<float>(dots_row[j]);
            my_dist = norms_pool[j] - 2.0f * d;
            my_id   = global_ids_pool[j];
            if (my_id == my_gid) my_dist = INFINITY;
        } else {
            my_dist = INFINITY;
            my_id   = -1;
        }

        unsigned mask = __ballot_sync(ALL, my_dist < threshold);
        if (mask == 0) continue;

        while (mask) {
            int src = __ffs(mask) - 1;
            mask &= ~(1u << src);

            float   c_dist = __shfl_sync(ALL, my_dist, src);
            int32_t c_id   = __shfl_sync(ALL, my_id,   src);
            if (c_dist >= threshold) continue;

            unsigned accept = __ballot_sync(ALL, c_dist <= reg_dist[K_PER_LANE - 1]);
            int target = __ffs(accept) - 1;

            float   old_max    = reg_dist[K_PER_LANE - 1];
            int32_t old_max_id = reg_id  [K_PER_LANE - 1];

            float   in_dist = __shfl_up_sync(ALL, old_max,    1);
            int32_t in_id   = __shfl_up_sync(ALL, old_max_id, 1);

            if (lane > target) {
                #pragma unroll
                for (int p = K_PER_LANE - 1; p > 0; --p) {
                    reg_dist[p] = reg_dist[p - 1];
                    reg_id  [p] = reg_id  [p - 1];
                }
                reg_dist[0] = in_dist;
                reg_id  [0] = in_id;
            } else if (lane == target) {
                reg_dist[K_PER_LANE - 1] = c_dist;
                reg_id  [K_PER_LANE - 1] = c_id;
                #pragma unroll
                for (int p = K_PER_LANE - 1; p > 0; --p) {
                    bool sw = reg_dist[p] < reg_dist[p - 1];
                    float   td  = sw ? reg_dist[p - 1] : reg_dist[p];
                    float   te  = sw ? reg_dist[p]     : reg_dist[p - 1];
                    int32_t tdi = sw ? reg_id  [p - 1] : reg_id  [p];
                    int32_t tei = sw ? reg_id  [p]     : reg_id  [p - 1];
                    reg_dist[p]     = td;
                    reg_dist[p - 1] = te;
                    reg_id  [p]     = tdi;
                    reg_id  [p - 1] = tei;
                }
            }

            threshold = __shfl_sync(ALL, reg_dist[K_PER_LANE - 1], LANES - 1);
        }
    }

    int base = lane * K_PER_LANE;
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        if (base + p < K) {
            out_neighbors[static_cast<int64_t>(query_idx) * K + base + p] = reg_id[p];
            if (out_dists != nullptr) {
                out_dists[static_cast<int64_t>(query_idx) * K + base + p] = reg_dist[p];
            }
        }
    }
}

// 按 int32 索引从 src 中 gather 标量: dst[k] = src[idx[k]]
__global__ void gather_floats_int32(
    const float* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst,
    int64_t n)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = src[static_cast<int64_t>(idx[tid])];
}

// Host dispatcher: 按 actual_M 选最小的 K_OUT ∈ {32,64,128,256,512}。
// 用 warp-cooperative kernel: 4 warps/block → 4 queries/block，
// blocks = ceil(bucket_size/4)。avg bucket=100 → 25 blocks，比原 per-thread
// 的 1 block 高 25×，SM 利用率从 1/80 → ~25/80。
template<typename DotsT>
static inline void launch_extract_topM(
    cudaStream_t stream,
    const DotsT*   d_dots,
    const float*   d_norms_pool,
    const int32_t* d_ids_bucket,
    const int32_t* d_ids_pool,
    int32_t* d_out_neighbors,
    float*   d_out_distances,   // 可选, nullptr 表示不写距离
    int bucket_size, int pool_size, int actual_M)
{
    constexpr int LANES = 32;
    // 按 bucket 大小自适应 warps_per_block：
    //   小 bucket → 1 warp/block，让 block 散到尽量多 SM 上（单 query 也独立调度）
    //   大 bucket → 多 warp/block，减少 launch overhead，单 SM occupancy 更高
    int warps_per_block;
    if      (bucket_size <= 64)    warps_per_block = 1;
    else if (bucket_size <= 256)   warps_per_block = 2;
    else if (bucket_size <= 1024)  warps_per_block = 4;
    else                            warps_per_block = 8;
    int threads_per_block = LANES * warps_per_block;
    int blocks = (bucket_size + warps_per_block - 1) / warps_per_block;

#define LAUNCH_KOUT(KV) do {                                                                  \
    if constexpr (std::is_same<DotsT, int32_t>::value) {                                      \
        warp_topK_from_dots_kernel<(KV), int32_t>                                             \
            <<<blocks, threads_per_block, 0, stream>>>(                                       \
            reinterpret_cast<const int32_t*>(d_dots), d_norms_pool,                           \
            d_ids_bucket, d_ids_pool,                                                         \
            d_out_neighbors, d_out_distances,                                                 \
            bucket_size, pool_size, actual_M);                                                \
    } else {                                                                                  \
        warp_topK_from_dots_kernel<(KV), float>                                               \
            <<<blocks, threads_per_block, 0, stream>>>(                                       \
            reinterpret_cast<const float*>(d_dots), d_norms_pool,                             \
            d_ids_bucket, d_ids_pool,                                                         \
            d_out_neighbors, d_out_distances,                                                 \
            bucket_size, pool_size, actual_M);                                                \
    }                                                                                         \
} while (0)

    if      (actual_M <= 32)   LAUNCH_KOUT(32);
    else if (actual_M <= 64)   LAUNCH_KOUT(64);
    else if (actual_M <= 128)  LAUNCH_KOUT(128);
    else if (actual_M <= 256)  LAUNCH_KOUT(256);
    else if (actual_M <= 512)  LAUNCH_KOUT(512);
    else                       LAUNCH_KOUT(1024);

#undef LAUNCH_KOUT
}

// ============== Disk-backed running per-vector KNN (row read-merge-write) ==============
//
// running_vector_knn/dists 不再是常驻内存的 (N, M) 数组。build_vector_knn_with_
// tensorcore 每算完一个 bucket 的结果，就对该 bucket 里的每个点：读它在磁盘上
// 现有的一行 -> 跟这一轮新算出来的候选合并去重 -> 写回同一行。文件在第一轮开始
// 前用 sentinel (-1 / +inf) 填满，所以"这个点还没有旧结果"不需要特殊处理——
// 跟全 sentinel 的一行合并，等价于直接取新候选，第 0 轮和后续轮走同一条代码路径。
//
// 代价：原来跨 iteration 的合并是整体一次性、且用 std::async 跟下一轮 GPU 计算
// 重叠；现在合并变成了每个 bucket 结束时同步做的小块磁盘 I/O（读+写各 M*8 字节/
// 点），发生在 scatter_pending 里，不再和"下一轮"重叠，而是跟同一轮里其他 bucket
// 的 GPU 计算竞争 CPU 时间。多数情况下这点 I/O 应该远小于 GEMM+topM 的耗时、能被
// 现有的 double-buffer 流水线掩盖掉，但如果磁盘慢（非 NVMe）或 bucket 很小、M 很
// 大，可能会看到吞吐下降——这是用内存换来的一个真实的性能取舍，如果 profiling
// 发现这里成为瓶颈，可以再把每个 bucket 的 merge 扔进后台线程池重新做重叠。
struct RunningKnnFile {
    std::fstream neighbors_f;
    std::fstream dists_f;
    int M = 0;

    static constexpr size_t header_bytes() { return sizeof(int64_t) + sizeof(int32_t); }

    // 建文件 + 写 header + 分块用 sentinel 填满 body（只在第 0 轮之前调用一次）。
    static RunningKnnFile create(const std::string& knn_path, const std::string& dist_path,
                                 int64_t N, int M, size_t chunk_bytes_budget) {
        RunningKnnFile f;
        f.M = M;
        int32_t M32 = static_cast<int32_t>(M);

        for (const auto& path : {knn_path, dist_path}) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("RunningKnnFile: cannot create " + path);
            out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
            out.write(reinterpret_cast<const char*>(&M32), sizeof(int32_t));
        }

        f.neighbors_f.open(knn_path, std::ios::binary | std::ios::in | std::ios::out);
        f.dists_f.open(dist_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f.neighbors_f.is_open() || !f.dists_f.is_open())
            throw std::runtime_error("RunningKnnFile: cannot reopen for read/write");

        int64_t chunk_rows = std::max<int64_t>(1,
            static_cast<int64_t>(chunk_bytes_budget / (static_cast<size_t>(M) * sizeof(int32_t))));
        chunk_rows = std::min(chunk_rows, N);
        std::vector<int32_t> nbuf(static_cast<size_t>(chunk_rows) * M, -1);
        std::vector<float>   dbuf(static_cast<size_t>(chunk_rows) * M,
                                  std::numeric_limits<float>::infinity());

        f.neighbors_f.seekp(header_bytes());
        f.dists_f.seekp(header_bytes());
        for (int64_t start = 0; start < N; start += chunk_rows) {
            int64_t cur = std::min(chunk_rows, N - start);
            f.neighbors_f.write(reinterpret_cast<const char*>(nbuf.data()),
                                static_cast<std::streamsize>(cur * M * sizeof(int32_t)));
            f.dists_f.write(reinterpret_cast<const char*>(dbuf.data()),
                            static_cast<std::streamsize>(cur * M * sizeof(float)));
        }
        if (!f.neighbors_f.good() || !f.dists_f.good())
            throw std::runtime_error("RunningKnnFile: failed sentinel-filling " + knn_path);
        return f;
    }

    void read_row(int64_t gid, int32_t* n_out, float* d_out) {
        neighbors_f.seekg(static_cast<std::streamoff>(header_bytes())
                          + static_cast<std::streamoff>(gid) * M * sizeof(int32_t));
        neighbors_f.read(reinterpret_cast<char*>(n_out), static_cast<std::streamsize>(M * sizeof(int32_t)));
        dists_f.seekg(static_cast<std::streamoff>(header_bytes())
                      + static_cast<std::streamoff>(gid) * M * sizeof(float));
        dists_f.read(reinterpret_cast<char*>(d_out), static_cast<std::streamsize>(M * sizeof(float)));
    }

    void write_row(int64_t gid, const int32_t* n_in, const float* d_in) {
        neighbors_f.seekp(static_cast<std::streamoff>(header_bytes())
                          + static_cast<std::streamoff>(gid) * M * sizeof(int32_t));
        neighbors_f.write(reinterpret_cast<const char*>(n_in), static_cast<std::streamsize>(M * sizeof(int32_t)));
        dists_f.seekp(static_cast<std::streamoff>(header_bytes())
                      + static_cast<std::streamoff>(gid) * M * sizeof(float));
        dists_f.write(reinterpret_cast<const char*>(d_in), static_cast<std::streamsize>(M * sizeof(float)));
    }
};

// 读一个点现有的一行、跟新算出来的候选合并去重、写回同一行。逻辑跟
// merge_two_per_vector_knn 完全一样，只是作用范围是单独一行而不是整块 (N,M)
// 数组，因为 running 状态现在活在磁盘上而不是内存里。
inline void merge_row_into_disk(RunningKnnFile& f, int64_t gid, int M,
                                const int32_t* new_n, const float* new_d) {
    std::vector<int32_t> old_n(M);
    std::vector<float>   old_d(M);
    f.read_row(gid, old_n.data(), old_d.data());

    std::vector<std::pair<float, int32_t>> cand;
    cand.reserve(static_cast<size_t>(2) * M);
    for (int m = 0; m < M; ++m) if (old_n[m] >= 0) cand.push_back({old_d[m], old_n[m]});
    for (int m = 0; m < M; ++m) if (new_n[m] >= 0) cand.push_back({new_d[m], new_n[m]});

    std::vector<int32_t> merged_n(M, -1);
    std::vector<float>   merged_d(M, std::numeric_limits<float>::infinity());
    if (!cand.empty()) {
        std::sort(cand.begin(), cand.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        std::unordered_set<int32_t> seen;
        seen.reserve(static_cast<size_t>(M) * 2);
        int written = 0;
        for (const auto& [dist, id] : cand) {
            if (seen.insert(id).second) {
                merged_n[written] = id;
                merged_d[written] = dist;
                if (++written >= M) break;
            }
        }
    }
    f.write_row(gid, merged_n.data(), merged_d.data());
}

// 把 vector_knn.bin (RunningKnnFile 的 int64 N + int32 M header, flat int32
// body) 顺序分块转换成 neighbors.npy (int64)，给 Python 端评测用。纯顺序拷贝
// + 类型转换，不需要整份常驻内存。只在全部 iteration 跑完、running 文件已经
// close 之后调用一次。
inline void convert_vector_knn_to_npy(const std::string& knn_path, const std::string& npy_path,
                                      int64_t N, int M, size_t chunk_bytes_budget) {
    std::ifstream in(knn_path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open: " + knn_path);
    in.seekg(static_cast<std::streamoff>(RunningKnnFile::header_bytes()));

    size_t header_bytes = load::create_npy_int64_2d(npy_path, N, M);
    std::fstream out(npy_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out.is_open()) throw std::runtime_error("Cannot open for write: " + npy_path);
    out.seekp(static_cast<std::streamoff>(header_bytes));

    int64_t chunk_rows = std::max<int64_t>(1,
        static_cast<int64_t>(chunk_bytes_budget / (static_cast<size_t>(M) * sizeof(int64_t))));
    chunk_rows = std::min(chunk_rows, N);
    std::vector<int32_t> buf32(static_cast<size_t>(chunk_rows) * M);
    std::vector<int64_t> buf64(static_cast<size_t>(chunk_rows) * M);

    for (int64_t start = 0; start < N; start += chunk_rows) {
        int64_t cur = std::min(chunk_rows, N - start);
        size_t cnt = static_cast<size_t>(cur) * M;
        in.read(reinterpret_cast<char*>(buf32.data()), static_cast<std::streamsize>(cnt * sizeof(int32_t)));
        if (!in.good()) throw std::runtime_error("Failed reading " + knn_path);
        for (size_t i = 0; i < cnt; ++i) buf64[i] = static_cast<int64_t>(buf32[i]);
        out.write(reinterpret_cast<const char*>(buf64.data()),
                 static_cast<std::streamsize>(cnt * sizeof(int64_t)));
        if (!out.good()) throw std::runtime_error("Failed writing " + npy_path);
    }
}

/**
 * 为每个向量在其 bucket 及 K 个最近邻 bucket 内，用 Tensor Core 矩阵乘法
 * 计算距离并找到 M 个最近邻。
 *
 * 流程 (per bucket c):
 *   1) 收集 search pool: bucket c 自身 + K 个最近邻 bucket 的所有点
 *   2) 将 bucket c 的向量 (A) 和 pool 向量 (B) 上传 GPU
 *   3) cuBLAS GEMM: dots = A * B^T (Tensor Core, FP16 compute)
 *   4) CUDA kernel: 从 dots 矩阵中为 bucket 内每个点提取 top-M 最近邻
 *   5) 下载结果
 *
 * @param X_full              完整数据集 (CPU, float32, N*D)
 * @param N                   数据点总数
 * @param D                   向量维度
 * @param assignments         (N,) 每个点的 bucket (local centroid index)
 * @param centroid_knn_graph  (n_centroids, K) centroid KNN 图 (CPU, uint32)
 * @param n_centroids         centroid / bucket 数量
 * @param K                   centroid KNN 图度数
 * @param M                   每个向量要找的邻居数
 *
 * @param running             磁盘上的 running per-vector KNN 文件（read_row/write_row）。
 *                            每算完一个 bucket，就把该 bucket 里每个点的新候选跟
 *                            running 里现有的一行合并写回——不再攒进内存里的
 *                            (N, M) 数组，也不再单独返回结果，见文件顶部
 *                            RunningKnnFile 的说明。
 *
 * 距离对合并是必需的 (要按距离排序去重)，所以内部始终按 want_distances=true
 * 的路径跑；不再对外暴露"不算距离"这个选项。
 */
template <typename DataT>
void build_vector_knn_with_tensorcore(
    const DataT* X_full,
    int64_t N,
    int64_t D,
    const std::vector<int64_t>& assignments,
    const std::vector<int64_t>& centroid_global_indices,
    const uint32_t* centroid_knn_graph,  // (n_centroids, K) row-major
    int64_t n_centroids,
    uint32_t K,
    int M,
    RunningKnnFile& running,
    const std::string& output_dir = "")
{

    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    cudaEventRecord(start_event,streams[0]);
    CUDA_CHECK(cudaMemcpyAsync(d_X_full, X_reordered.data(),bytes_X, cudaMemcpyHostToDevice,streams[0]));
    cudaEventSynchronize(start_event);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event, end_event);
    printf("[step6_full_upload] bytes=%zu ms=%.3f gbps=%.2f\n", bytes_X, ms, gb_per_s);

    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);

    constexpr bool want_distances = true;  // merge_row_into_disk 总是需要距离
    // ============= Stage 2 路径选择（INT8 IMMA / fp32 fallback）=============
    // - int8/uint8 → 走 INT8 IMMA Tensor Core（uint8 入口先减 128 转 int8）
    // - 其它 (float/half/uint32/int32 等) → 走 fp32 cuBLAS GEMM（原行为）
    constexpr bool kIsInt8Path =
        std::is_same<DataT, int8_t>::value || std::is_same<DataT, uint8_t>::value;
    using GemmInT  = typename std::conditional<kIsInt8Path, int8_t, float>::type;
    using GemmOutT = typename std::conditional<kIsInt8Path, int32_t, float>::type;

    std::cout << "[VectorKNN] Building per-vector KNN with Tensor Core matmul\n"
              << "  N=" << N << ", D=" << D
              << ", n_centroids=" << n_centroids
              << ", K=" << K << ", M=" << M
              << ", path=" << (kIsInt8Path ? "INT8 IMMA" : "FP32 TF32") << "\n";

    // launch_extract_topM 最大特化档位是 K_OUT=1024；超过会跑错。
    if (M > 1024) {
        throw std::runtime_error(
            "build_vector_knn_with_tensorcore: M=" + std::to_string(M)
            + " exceeds kernel max K_OUT=1024. Lower M or add a larger K_OUT case in launch_extract_topM.");
    }

    // ================================================================
    // Step 0: Build in-memory bucket lists & ensure centroid is in its bucket
    // ================================================================
    std::vector<std::vector<int32_t>> buckets(n_centroids);
    for (int64_t i = 0; i < N; ++i) {
        int64_t c = assignments[i];
        if (c >= 0 && c < n_centroids) {
            buckets[c].push_back(static_cast<int32_t>(i));
        }
    }

    // 确认每个 centroid 在自己的 bucket 中
    for (int64_t c = 0; c < n_centroids; ++c) {
        int32_t centroid_gid = static_cast<int32_t>(centroid_global_indices[c]);
        bool found = false;
        for (int32_t pid : buckets[c]) {
            if (pid == centroid_gid) { found = true; break; }
        }
        if (!found) {
            std::cout << "  [WARN] Centroid " << c << " (global=" << centroid_gid
                      << ") not in its bucket, inserting.\n";
            buckets[c].push_back(centroid_gid);
        }
    }

    // ================================================================
    // [Cache-Opt] Step 0.5: Compute Contiguous Bucket Offsets & Permutations
    // ================================================================
    std::vector<int64_t> bucket_offsets(n_centroids + 1, 0);
    std::vector<int32_t> perm_order(N);
    std::vector<int32_t> inverse_perm(N);
    int64_t cur_ofs = 0;
    for (int64_t c = 0; c < n_centroids; ++c) {
        bucket_offsets[c] = cur_ofs;
        for (int32_t gid : buckets[c]) {
            perm_order[cur_ofs] = gid;
            inverse_perm[gid] = static_cast<int32_t>(cur_ofs);
            cur_ofs++;
        }
    }
    bucket_offsets[n_centroids] = cur_ofs;

    // ================================================================
    // Step 1: cuBLAS handle 初始化
    // ================================================================
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create cuBLAS handle");

    // 启用 Tensor Core (TF32 for FP32 inputs — 自动利用 Tensor Core)
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // RAFT 资源句柄；每次调 select_k 前 set_cuda_stream 切到对应 slot stream
    raft::resources raft_res;

    // 查询可用 GPU 显存来确定处理策略
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t usable_bytes = static_cast<size_t>(free_bytes * 0.85);

    std::cout << "  GPU memory available: " << free_bytes / 1e9 << " GB\n";

    // ================================================================
    // Step 2: 计算所有 bucket 的尺寸上界，一次性预分配 GPU/CPU buffer
    // ================================================================
    // 关键优化:
    //   1) 取消每 bucket cudaMalloc/cudaFree (原本每 bucket 7 次同步分配)
    //   2) X_full 整体常驻 GPU，A/B 改为 GPU gather (省去 CPU memcpy + 大块 H2D)
    //   3) ‖x‖² 全局只算一次
    //   4) Double-buffer + 双 stream: bucket c 的 GEMM/topM/D2H 与 bucket c+1 的
    //      CPU prep + H2D + gather 并行；CPU 的 sort/unique 与 GPU 工作完全重叠。
    //
    // 关于 (2) 的一个已知权衡（暂不改，先记录）：
    // 之所以要求 X_full/d_X_full 整份常驻 CPU+GPU，是因为 gather A/B 这一步是
    // GPU kernel 直接按 id 去 d_X_full 里抠数据（gather_rows_int32/gather_rows_
    // raw），CUDA kernel 只能解引用显存指针，没法在 kernel 内部临时去读磁盘或
    // CPU 内存，而任意一个 bucket 的近邻桶都可能覆盖数据集里的任意点，没法只常
    // 驻一部分。理论上可以换成 bucket_build.cu 那种做法：需要哪个 bucket 就现读
    // 磁盘、CPU 端拼好向量再整块 H2D，完全不需要 d_X_full 常驻——但一个 bucket
    // 里的点在原文件里是随机散布的（分桶本身就打乱了顺序），这样读会退化成"每个
    // 点一次 seek"，而不是几次大块顺序读；把这种随机 I/O 插进现在这条为吞吐量
    // 设计的热循环（tensor core + 多 slot 流水线），可能会让 I/O 耗时超过 GPU
    // 计算耗时，反而拖慢整体。要让它划算，需要先把数据集按桶重排到磁盘上一份
    // （assign 阶段顺便生成，或者复用现成的 reorder 逻辑），这样每个 bucket 和它
    // 的近邻桶就变成几段连续区间，读起来才是大块顺序读而不是随机 seek。这块目前
    // 没有实现，先维持 d_X_full 整份常驻的现状，把这个方案记在这里，之后要做的
    // 话再单独展开。
    size_t max_bucket_size = 0;
    size_t max_pool_size_ub = 0;  // 上界(未去重)
    for (int64_t c = 0; c < n_centroids; ++c) {
        max_bucket_size = std::max(max_bucket_size, buckets[c].size());
        size_t ps = buckets[c].size();
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t nb_c = centroid_knn_graph[c * K + k];
            if (nb_c < static_cast<uint32_t>(n_centroids)) {
                ps += buckets[nb_c].size();
            }
        }
        max_pool_size_ub = std::max(max_pool_size_ub, ps);
    }
    // bucket 平均大小由 centroid 率直接得出: assignment 把 N 点均分到 n_centroids 个 bucket
    size_t avg_bucket_size = (n_centroids > 0)
        ? static_cast<size_t>(N / n_centroids) : size_t{1};
    if (avg_bucket_size == 0) avg_bucket_size = 1;

    if (max_bucket_size == 0 || max_pool_size_ub == 0) {
        cublasDestroy(cublas_handle);
        std::cout << "[VectorKNN] All buckets empty, nothing to do.\n";
        return;
    }

    // ---- 显存预算分两块: 全局常驻 + per-slot ----
    // d_X_full 以 DataT 存（uint8/int8 时省 4×）；per-bucket A/B 用 GemmInT (int8 时也省 4×)
    size_t bytes_X            = static_cast<size_t>(N) * D * sizeof(DataT);
    size_t bytes_norms_full   = static_cast<size_t>(N) * sizeof(float);
    size_t bytes_A            = max_bucket_size * D * sizeof(GemmInT);
    size_t bytes_B            = max_pool_size_ub * D * sizeof(GemmInT);
    size_t bytes_dots         = max_bucket_size * max_pool_size_ub * sizeof(GemmOutT);
    size_t bytes_norms_pool   = max_pool_size_ub * sizeof(float);
    size_t bytes_ids_bucket   = max_bucket_size * sizeof(int32_t);
    size_t bytes_ids_pool     = max_pool_size_ub * sizeof(int32_t);
    size_t bytes_out          = max_bucket_size * static_cast<size_t>(M) * sizeof(int32_t);
    // 距离输出 buffer 仅在 want_distances 时分配 (multi-iteration merge 需要)
    size_t bytes_out_dists    = want_distances ? max_bucket_size * static_cast<size_t>(M) * sizeof(float) : 0;

    size_t bytes_global   = bytes_X + bytes_norms_full;
    size_t bytes_per_slot = bytes_A + bytes_B + bytes_dots + bytes_norms_pool
                          + bytes_ids_bucket + bytes_ids_pool + bytes_out + bytes_out_dists;

    // ---- 动态 num_slots ----
    // 目标：让 num_slots × avg_blocks_per_kernel ≈ num_sm × OVERSUB，正好填满 SM 并留点
    // 富余 hide launch overhead。bucket 大单 kernel 已塞满 SM → num_slots 小；
    // bucket 小单 kernel 占 SM 少 → num_slots 大。
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev_id);
    const int num_sm = prop.multiProcessorCount;

    // 用 avg bucket 估算 warps_per_block (跟 launch_extract_topM 里那段一致)
    int est_wpb;
    if      (avg_bucket_size <= 64)   est_wpb = 1;
    else if (avg_bucket_size <= 256)  est_wpb = 2;
    else if (avg_bucket_size <= 1024) est_wpb = 4;
    else                              est_wpb = 8;
    int est_blocks_per_kernel = std::max(1,
        static_cast<int>((avg_bucket_size + est_wpb - 1) / est_wpb));

    constexpr int MAX_SLOTS = 8;
    constexpr double OVERSUB = 1.5;  // 1.5× SM 利用率目标，hide launch overhead
    int target_streams = std::max(2,
        static_cast<int>(std::ceil(num_sm * OVERSUB / est_blocks_per_kernel)));
    target_streams = std::min(target_streams, MAX_SLOTS);

    // 内存上限：从 target_streams 往下试，挑能装下的最大值
    int num_slots = 1;
    for (int try_slots = target_streams; try_slots >= 1; --try_slots) {
        if (bytes_global + try_slots * bytes_per_slot <= usable_bytes) {
            num_slots = try_slots;
            break;
        }
    }
    if (bytes_global + bytes_per_slot > usable_bytes) {
        throw std::runtime_error(
            "[VectorKNN] Need at least " + std::to_string((bytes_global + bytes_per_slot) / 1e9)
            + " GB GPU mem but only " + std::to_string(usable_bytes / 1e9)
            + " GB usable. Increase n_centroids (smaller buckets) or lower K.");
    }

    std::cout << "  [VectorKNN] buffer plan: X_full=" << bytes_X/1e9
              << "GB, dots(max)=" << bytes_dots/1e9 << "GB, "
              << "per-slot=" << bytes_per_slot/1e9 << "GB × " << num_slots
              << " + global=" << bytes_global/1e9 << "GB / usable="
              << usable_bytes/1e9 << "GB"
              << " [streams=" << num_slots
              << ", num_sm=" << num_sm
              << ", avg_bucket=" << avg_bucket_size
              << ", est_blocks/kernel=" << est_blocks_per_kernel
              << ", target_streams(pre-mem)=" << target_streams << "]\n";

    // ---- 全局常驻 GPU buffer ----
    DataT* d_X_full     = nullptr;          // 数据集本体保留原始 element type
    float* d_norms_full = nullptr;          // 范数始终 fp32
    CUDA_CHECK(cudaMalloc(&d_X_full,     bytes_X));
    CUDA_CHECK(cudaMalloc(&d_norms_full, bytes_norms_full));

    // ---- per-slot 持久化 buffer (最多 8 slot 轮转, 实际用 num_slots 个) ----
    // d_A/d_B/d_dots 类型由 GemmInT/GemmOutT 决定（INT8 路径 / FP32 路径不同）
    // C++ partial init: {nullptr, nullptr} 后面的 slot 会被零初始化为 nullptr
    GemmInT*     d_A[MAX_SLOTS]             = {nullptr};
    GemmInT*     d_B[MAX_SLOTS]             = {nullptr};
    GemmOutT*    d_dots[MAX_SLOTS]          = {nullptr};
    float*       d_norms_pool[MAX_SLOTS]    = {nullptr};
    int32_t*     d_ids_bucket[MAX_SLOTS]    = {nullptr};
    int32_t*     d_ids_pool[MAX_SLOTS]      = {nullptr};
    int32_t*     d_out_neighbors[MAX_SLOTS] = {nullptr};
    float*       d_out_dists[MAX_SLOTS]     = {nullptr};  // 仅 want_distances 时使用
    int32_t*     d_select_idx[MAX_SLOTS]    = {nullptr};  // RAFT idx 输出
    float*       d_select_dist[MAX_SLOTS]   = {nullptr};  // RAFT dist 输出 scratch
    int32_t*     h_ids_bucket[MAX_SLOTS]    = {nullptr};
    int32_t*     h_ids_pool[MAX_SLOTS]      = {nullptr};
    int32_t*     h_out[MAX_SLOTS]           = {nullptr};
    float*       h_out_dists[MAX_SLOTS]     = {nullptr};  // 仅 want_distances 时使用
    cudaStream_t streams[MAX_SLOTS]         = {nullptr};

    // RAFT path 的额外 buffer 大小：(max_bucket_size × M) × 4 bytes，K=1024 时 ~4MB/slot，可忽略
    size_t bytes_select_idx  = max_bucket_size * static_cast<size_t>(M) * sizeof(int32_t);
    size_t bytes_select_dist = max_bucket_size * static_cast<size_t>(M) * sizeof(float);

    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaMalloc(&d_A[s],             bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B[s],             bytes_B));
        CUDA_CHECK(cudaMalloc(&d_dots[s],          bytes_dots));
        CUDA_CHECK(cudaMalloc(&d_norms_pool[s],    bytes_norms_pool));
        CUDA_CHECK(cudaMalloc(&d_ids_bucket[s],    bytes_ids_bucket));
        CUDA_CHECK(cudaMalloc(&d_ids_pool[s],      bytes_ids_pool));
        CUDA_CHECK(cudaMalloc(&d_out_neighbors[s], bytes_out));
        CUDA_CHECK(cudaMalloc(&d_select_idx[s],    bytes_select_idx));
        CUDA_CHECK(cudaMalloc(&d_select_dist[s],   bytes_select_dist));
        CUDA_CHECK(cudaMallocHost(&h_ids_bucket[s], bytes_ids_bucket));
        CUDA_CHECK(cudaMallocHost(&h_ids_pool[s],   bytes_ids_pool));
        CUDA_CHECK(cudaMallocHost(&h_out[s],        bytes_out));
        if (want_distances) {
            CUDA_CHECK(cudaMalloc(&d_out_dists[s],  bytes_out_dists));
            CUDA_CHECK(cudaMallocHost(&h_out_dists[s], bytes_out_dists));
        }
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    // ---- 一次性上传 X_full + (uint8 时) shift → int8 + 计算所有点的 ‖x‖² (用 stream 0) ----
    // norms 必须从 GEMM 实际看到的数据视图算出来（uint8 路径下要在 shift 之后用 int8 视图算）。
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<DataT> X_reordered(static_cast<size_t>(N) * D);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            int32_t old_id = perm_order[i];
            std::memcpy(&X_reordered[i * D], &X_full[static_cast<int64_t>(old_id) * D], D * sizeof(DataT));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_X_full, X_reordered.data(), bytes_X,
                                   cudaMemcpyHostToDevice, streams[0]));
        int threads = 256;

        // uint8 → int8 in-place 平移 (仅 uint8 路径)
        if constexpr (std::is_same<DataT, uint8_t>::value) {
            int64_t total = static_cast<int64_t>(N) * D;
            int64_t shift_blocks = (total + threads - 1) / threads;
            shift_uint8_to_int8_inplace<<<shift_blocks, threads, 0, streams[0]>>>(
                reinterpret_cast<uint8_t*>(d_X_full), total);
            CUDA_CHECK(cudaGetLastError());
        }

        int64_t blocks = (N + threads - 1) / threads;
        if constexpr (kIsInt8Path) {
            // 用 int8 视图算 norms（uint8 已 shift；int8 直接）
            compute_row_norms_kernel<int8_t><<<blocks, threads, 0, streams[0]>>>(
                reinterpret_cast<const int8_t*>(d_X_full), d_norms_full, N, D);
        } else {
            compute_row_norms_kernel<DataT><<<blocks, threads, 0, streams[0]>>>(
                d_X_full, d_norms_full, N, D);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        auto t1 = std::chrono::high_resolution_clock::now();
        double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  [VectorKNN] X_full upload + norms: " << upload_ms << " ms\n";
    }

    // ================================================================
    // Step 3: 逐 bucket ping-pong 处理
    // ================================================================
    // 每个 slot 上有未消费 D2H 时记一笔 pending；下一次该 slot 被复用前必须:
    //   1) cudaStreamSynchronize (drain D2H)
    //   2) 把 h_out[slot] 中的结果逐点 merge 进磁盘上的 running 文件
    // 之后才能安全覆写 h_out[slot] / d_*[slot]。
    struct Pending {
        int64_t c           = -1;   // bucket index in this slot's last submitted iteration; -1 = empty
        int     bucket_size = 0;
        int     actual_M    = 0;
    };
    Pending pending[MAX_SLOTS]{};   // 全部默认初始化为 {-1, 0, 0}

    // ---- GEMM 纯计算时间统计 (GPU side, via cudaEvent on stream) ----
    cudaEvent_t ev_gemm_start[MAX_SLOTS] = {nullptr};
    cudaEvent_t ev_gemm_stop[MAX_SLOTS]  = {nullptr};
    bool gemm_evt_valid[MAX_SLOTS] = {false};
    double gemm_total_ms = 0.0;
    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaEventCreate(&ev_gemm_start[s]));
        CUDA_CHECK(cudaEventCreate(&ev_gemm_stop[s]));
    }
    auto consume_gemm_timing = [&](int slot) {
        if (!gemm_evt_valid[slot]) return;
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_gemm_start[slot], ev_gemm_stop[slot]);
        gemm_total_ms += ms;
        gemm_evt_valid[slot] = false;
    };

    // 每个点的新候选先铺成 M 宽（不足 aM 的部分补 -1/inf），再跟磁盘上现有的
    // 一行合并写回 —— 复用同一块 scratch buffer，避免每个点都分配一次。
    std::vector<int32_t> scatter_new_n(M);
    std::vector<float>   scatter_new_d(M);

    auto scatter_pending = [&](int slot) {
        if (pending[slot].c < 0) return;
        const auto& bk_prev = buckets[pending[slot].c];
        int bs = pending[slot].bucket_size;
        int aM = pending[slot].actual_M;
        const int32_t* hp = h_out[slot];
        const float*   hd = h_out_dists[slot];  // 始终已分配 (want_distances 内部恒为 true)
        for (int i = 0; i < bs; ++i) {
            int32_t gid = bk_prev[i];
            std::fill(scatter_new_n.begin(), scatter_new_n.end(), -1);
            std::fill(scatter_new_d.begin(), scatter_new_d.end(),
                     std::numeric_limits<float>::infinity());
            for (int m = 0; m < aM && m < M; ++m) {
                scatter_new_n[m] = hp[static_cast<size_t>(i) * aM + m];
                scatter_new_d[m] = hd[static_cast<size_t>(i) * aM + m];
            }
            merge_row_into_disk(running, gid, M, scatter_new_n.data(), scatter_new_d.data());
        }
        pending[slot].c = -1;
    };

    int64_t processed_buckets = 0;
    auto loop_t0 = std::chrono::high_resolution_clock::now();

    // Compute task ordering for Step 6 cache-locality
    auto bucket_process_order = bucket_order::compute_bucket_processing_order(
        bucket_order::adjacency_from_flat_graph(
            centroid_knn_graph, n_centroids, static_cast<int32_t>(K)),
        64);

    // Algorithmic Bucket Cache Tracker (Capacity = 64 buckets)
    const size_t cache_capacity = 64;
    std::unordered_map<int32_t, int64_t> cache_map;
    int64_t cache_access_time = 0;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;
    std::vector<std::pair<double, double>> cache_trace;

    auto access_bucket = [&](int32_t b_id) {
        cache_access_time++;
        auto it = cache_map.find(b_id);
        if (it != cache_map.end()) {
            cache_hits++;
            it->second = cache_access_time;
        } else {
            cache_misses++;
            if (cache_map.size() >= cache_capacity) {
                int32_t lru_id = -1;
                int64_t min_t = std::numeric_limits<int64_t>::max();
                for (const auto& kv : cache_map) {
                    if (kv.second < min_t) {
                        min_t = kv.second;
                        lru_id = kv.first;
                    }
                }
                if (lru_id != -1) cache_map.erase(lru_id);
            }
            cache_map[b_id] = cache_access_time;
        }
    };

    for (int64_t step_idx = 0; step_idx < n_centroids; ++step_idx) {
        int64_t c = (step_idx < static_cast<int64_t>(bucket_process_order.size()))
            ? static_cast<int64_t>(bucket_process_order[step_idx]) : step_idx;
        if (buckets[c].empty()) continue;

        access_bucket(static_cast<int32_t>(c));
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t nb_c = centroid_knn_graph[c * K + k];
            if (nb_c < static_cast<uint32_t>(n_centroids) && !buckets[nb_c].empty()) {
                access_bucket(static_cast<int32_t>(nb_c));
            }
        }
        if (n_centroids > 0 && (step_idx % std::max<int64_t>(1, n_centroids / 100) == 0 || step_idx == n_centroids - 1)) {
            double prog = (static_cast<double>(step_idx + 1) / n_centroids) * 100.0;
            double mr = (cache_hits + cache_misses > 0)
                ? (static_cast<double>(cache_misses) / (cache_hits + cache_misses) * 100.0) : 100.0;
            cache_trace.emplace_back(prog, mr);
        }

        int slot = static_cast<int>(c % num_slots);

        // ---- 3a: 等本 slot 上一轮 D2H 落地，scatter 老结果，腾出 buffer ----
        CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
        consume_gemm_timing(slot);
        scatter_pending(slot);

        // ---- 3b: CPU 端拼 pool_ids ----
        // BatchAssign 里 line 791 强制 assignments[centroid_global_indices[c]] = c,
        // 加上每个非 centroid 点只在 assignments[] 里有一个值 → buckets 互不相交,
        // 拼出来的 pool 不会重复，无需 sort+unique。
        const auto& bk = buckets[c];
        size_t ofs = 0;
        std::memcpy(h_ids_pool[slot] + ofs, bk.data(), bk.size() * sizeof(int32_t));
        ofs += bk.size();
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t nb_c = centroid_knn_graph[c * K + k];
            if (nb_c < static_cast<uint32_t>(n_centroids) && !buckets[nb_c].empty()) {
                std::memcpy(h_ids_pool[slot] + ofs, buckets[nb_c].data(),
                            buckets[nb_c].size() * sizeof(int32_t));
                ofs += buckets[nb_c].size();
            }
        }
        int pool_size   = static_cast<int>(ofs);
        int bucket_size = static_cast<int>(bk.size());
        int actual_M    = std::min(M, pool_size - 1);
        if (actual_M <= 0) continue;

        std::memcpy(h_ids_bucket[slot], bk.data(), bk.size() * sizeof(int32_t));

        // ---- 3c: 上传 id 列表 (async on slot stream) ----
        CUDA_CHECK(cudaMemcpyAsync(d_ids_bucket[slot], h_ids_bucket[slot],
                                   bucket_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(d_ids_pool[slot], h_ids_pool[slot],
                                   pool_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams[slot]));

        // ---- 3d: GPU gather A / B / norms_pool ----
        {
            size_t cur_pool_offset = 0;
            auto copy_bucket_slice = [&](int64_t b_idx) {
                int64_t b_start = bucket_offsets[b_idx];
                int64_t b_len = bucket_offsets[b_idx + 1] - b_start;
                if (b_len <= 0) return;

                size_t byte_count = static_cast<size_t>(b_len) * D * sizeof(GemmInT);
                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<uint8_t*>(d_B[slot]) + cur_pool_offset * D * sizeof(GemmInT),
                    reinterpret_cast<const uint8_t*>(d_X_full) + b_start * D * sizeof(GemmInT),
                    byte_count,
                    cudaMemcpyDeviceToDevice, streams[slot]));

                CUDA_CHECK(cudaMemcpyAsync(
                    d_norms_pool[slot] + cur_pool_offset,
                    d_norms_full + b_start,
                    b_len * sizeof(float),
                    cudaMemcpyDeviceToDevice, streams[slot]));

                cur_pool_offset += b_len;
            };

            // 1. A 矩阵直接连续拷贝当前 bucket
            int64_t c_start = bucket_offsets[c];
            CUDA_CHECK(cudaMemcpyAsync(
                d_A[slot],
                reinterpret_cast<const uint8_t*>(d_X_full) + c_start * D * sizeof(GemmInT),
                static_cast<size_t>(bucket_size) * D * sizeof(GemmInT),
                cudaMemcpyDeviceToDevice, streams[slot]));

            // 2. B 矩阵按邻居 bucket 拼接连续切片
            copy_bucket_slice(c);
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t nb_c = centroid_knn_graph[c * K + k];
                if (nb_c < static_cast<uint32_t>(n_centroids) && nb_c != static_cast<uint32_t>(c)) {
                    copy_bucket_slice(nb_c);
                }
            }
        }

        // ---- 3e: cuBLAS GEMM 在 slot stream 上 (event 包夹做纯 GEMM 计时) ----
        cublasSetStream(cublas_handle, streams[slot]);
        CUDA_CHECK(cudaEventRecord(ev_gemm_start[slot], streams[slot]));
        {
            cublasStatus_t stat;
            if constexpr (kIsInt8Path) {
                // INT8 IMMA: int8 in × int8 in → int32 out, Tensor Core
                int alpha_i = 1, beta_i = 0;
                stat = cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, bucket_size, D,
                    &alpha_i,
                    d_B[slot], CUDA_R_8I, D,
                    d_A[slot], CUDA_R_8I, D,
                    &beta_i,
                    d_dots[slot], CUDA_R_32I, pool_size,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error(
                        "cublasGemmEx (INT8 IMMA) failed, status=" + std::to_string(stat));
            } else {
                float alpha = 1.0f, beta = 0.0f;
                stat = cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, bucket_size, D,
                    &alpha,
                    reinterpret_cast<const float*>(d_B[slot]), D,
                    reinterpret_cast<const float*>(d_A[slot]), D,
                    &beta,
                    reinterpret_cast<float*>(d_dots[slot]), pool_size);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("cublasSgemm failed, status=" + std::to_string(stat));
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_gemm_stop[slot], streams[slot]));
        gemm_evt_valid[slot] = true;

        // ---- 3f: top-M ----
        // 切换实现：把下面 USE_RAFT_SELECT_K 改 0 即用自写 warp-cooperative kernel，
        //          改 1 即用 RAFT select_k (radix / warp-sort auto)。
        // 两个 dispatcher 都是模板化 GemmOutT (int32 INT8 路径 / float TF32 路径)。
        #ifndef USE_RAFT_SELECT_K
        #define USE_RAFT_SELECT_K 1
        #endif
        {
        #if USE_RAFT_SELECT_K
            // d_dots 被 dots_to_dist_inplace_kernel 原地改写为 float dist,
            // RAFT 输出落 d_select_idx/dist scratch, 最后 gather 到 d_out_neighbors
            launch_extract_topM_raft<GemmOutT>(
                raft_res,
                streams[slot],
                d_dots[slot],
                d_norms_pool[slot],
                d_ids_bucket[slot], d_ids_pool[slot],
                d_select_idx[slot], d_select_dist[slot],
                d_out_neighbors[slot],
                want_distances ? d_out_dists[slot] : nullptr,
                bucket_size, pool_size, actual_M);
        #else
            // 自写 warp-cooperative kernel: 一 warp 处理一 query,
            // K_PER_LANE = K_OUT/32 个 slot 全在寄存器
            launch_extract_topM<GemmOutT>(
                streams[slot],
                d_dots[slot],
                d_norms_pool[slot],
                d_ids_bucket[slot], d_ids_pool[slot],
                d_out_neighbors[slot],
                want_distances ? d_out_dists[slot] : nullptr,
                bucket_size, pool_size, actual_M);
        #endif
            CUDA_CHECK(cudaGetLastError());
        }

        // ---- 3g: 异步 D2H, 不 sync ----
        CUDA_CHECK(cudaMemcpyAsync(h_out[slot], d_out_neighbors[slot],
                                   static_cast<size_t>(bucket_size) * actual_M * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, streams[slot]));
        if (want_distances) {
            CUDA_CHECK(cudaMemcpyAsync(h_out_dists[slot], d_out_dists[slot],
                                       static_cast<size_t>(bucket_size) * actual_M * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[slot]));
        }

        pending[slot] = {c, bucket_size, actual_M};
        ++processed_buckets;
    }

    // ---- Tail flush: 处理两 slot 上的最后未消费结果 ----
    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        consume_gemm_timing(s);
        scatter_pending(s);
    }

    auto loop_t1 = std::chrono::high_resolution_clock::now();
    double loop_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();
    std::cout << "  [VectorKNN] bucket loop: " << loop_ms << " ms over "
              << processed_buckets << " buckets ("
              << (loop_ms / std::max<int64_t>(1, processed_buckets))
              << " ms/bucket avg, slots=" << num_slots << ")\n";
    std::cout << "  [VectorKNN] pure GEMM: " << gemm_total_ms << " ms total, "
              << (gemm_total_ms / std::max<int64_t>(1, processed_buckets))
              << " ms/bucket avg, "
              << (gemm_total_ms / std::max(1e-6, loop_ms) * 100.0)
              << "% of loop wall-time\n";

    double final_mr = (cache_hits + cache_misses > 0)
        ? (static_cast<double>(cache_misses) / (cache_hits + cache_misses) * 100.0) : 0.0;
    std::cout << "  [CacheTrack] Capacity=" << cache_capacity
              << " | Hits=" << cache_hits
              << " | Misses=" << cache_misses
              << " | Final Miss Rate: " << std::fixed << std::setprecision(2) << final_mr << "%\n";

    if (!output_dir.empty()) {
        std::string trace_path = output_dir + "/cache_miss_trace.csv";
        std::ofstream trace_out(trace_path);
        if (trace_out.is_open()) {
            trace_out << "progress_pct,miss_rate_pct\n";
            for (const auto& pt : cache_trace) {
                trace_out << std::fixed << std::setprecision(2) << pt.first << ","
                          << std::fixed << std::setprecision(2) << pt.second << "\n";
            }
            trace_out.close();
            std::cout << "  [CacheTrack] Saved " << trace_path << "\n";
        }
    }

    // ---- 释放 ----
    cudaFree(d_X_full);
    cudaFree(d_norms_full);
    for (int s = 0; s < num_slots; ++s) {
        cudaFree(d_A[s]);
        cudaFree(d_B[s]);
        cudaFree(d_dots[s]);
        cudaFree(d_norms_pool[s]);
        cudaFree(d_ids_bucket[s]);
        cudaFree(d_ids_pool[s]);
        cudaFree(d_out_neighbors[s]);
        cudaFree(d_select_idx[s]);
        cudaFree(d_select_dist[s]);
        if (d_out_dists[s])  cudaFree(d_out_dists[s]);
        cudaFreeHost(h_ids_bucket[s]);
        cudaFreeHost(h_ids_pool[s]);
        cudaFreeHost(h_out[s]);
        if (h_out_dists[s])  cudaFreeHost(h_out_dists[s]);
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(ev_gemm_start[s]);
        cudaEventDestroy(ev_gemm_stop[s]);
    }
    cublasDestroy(cublas_handle);

    std::cout << "[VectorKNN] Done. Output shape: (" << N << ", " << M << ")"
              << " merged into running KNN file on disk\n";
}

/**
 * 增量合并: 把一轮新的 per-vector KNN 结果 (new_n/new_d) 并入"运行中"的累积
 * 结果 (running_n/running_d, 原地更新)。
 *
 * 语义上等价于对 {running 之前已含的所有轮, new 这一轮} 调 dedupe+top-M，
 * 但每次只处理 2*M 个候选/点 (而不是 T*M)，所以可以在每轮 iteration 结束后
 * 立刻调用一次，逐轮把结果并进来 —— 不需要等全部 T 轮跑完再一次性合并。
 * 这让 CPU 侧的 merge 可以和下一轮 iteration 的 GPU 计算 (Step 2-6) 重叠:
 * 主线程 std::async 出去后立刻继续发起下一轮的 GPU 工作，merge 在后台线程
 * 用 CPU 核心跑，两者互不阻塞（只要不同时读写同一份 running_* 缓冲区）。
 *
 * 输入: running_n/d 和 new_n/d 都是 (N, M) row-major，且各自已经是"每行内
 *       按距离升序、已去重"的结果 (build_vector_knn_with_tensorcore /
 *       上一次 merge_two_per_vector_knn 的输出都满足这个不变式)。
 * 输出: running_n/d 原地更新为二者按距离升序 dedupe 后的 top-M。
 *
 * 注: 相同 (query, neighbor) pair 在不同轮的距离是确定性复现的 (公式相同 +
 *     浮点可复现)，dedupe 时按先遇到的为准即可，不需要比较取 min。
 */
inline void merge_two_per_vector_knn(
    std::vector<int32_t>& running_n, std::vector<float>& running_d,
    const std::vector<int32_t>& new_n, const std::vector<float>& new_d,
    int64_t N, int M)
{
    if (running_n.size() != static_cast<size_t>(N) * M ||
        running_d.size() != static_cast<size_t>(N) * M ||
        new_n.size()     != static_cast<size_t>(N) * M ||
        new_d.size()     != static_cast<size_t>(N) * M) {
        throw std::runtime_error("merge_two_per_vector_knn: shape mismatch");
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        // 候选/去重缓冲区按线程复用，避免每行都重新分配
        std::vector<std::pair<float, int32_t>> cand;
        cand.reserve(static_cast<size_t>(2) * M);
        std::unordered_set<int32_t> seen;
        seen.reserve(static_cast<size_t>(M) * 2);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            cand.clear();
            seen.clear();

            int32_t* rn = running_n.data() + i * M;
            float*   rd = running_d.data() + i * M;
            const int32_t* nn = new_n.data() + i * M;
            const float*   nd = new_d.data() + i * M;

            for (int m = 0; m < M; ++m) if (rn[m] >= 0) cand.push_back({rd[m], rn[m]});
            for (int m = 0; m < M; ++m) if (nn[m] >= 0) cand.push_back({nd[m], nn[m]});
            if (cand.empty()) continue;

            // 只有 2*M 个候选 (M 一般 <= 1024)，直接排序足够快
            std::sort(cand.begin(), cand.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            int written = 0;
            for (const auto& [dist, id] : cand) {
                if (seen.insert(id).second) {
                    rn[written] = id;
                    rd[written] = dist;
                    if (++written >= M) break;
                }
            }
            for (; written < M; ++written) {
                rn[written] = -1;
                rd[written] = std::numeric_limits<float>::infinity();
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[MergeKNN] Incremental merge into (N=" << N << ", M=" << M
              << ") in " << ms << " ms\n";
}

/**
 * 将 per-vector KNN 结果写入磁盘。
 *
 * 格式: 二进制文件
 *   header:
 *     int64_t  N
 *     int32_t  M
 *   body:
 *     int32_t  neighbors[N * M]   // row-major, 每行 M 个邻居 global id
 */
void write_vector_knn_to_disk(
    const std::string& output_path,
    const std::vector<int32_t>& neighbors,
    int64_t N, int M)
{
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + output_path);

    int32_t M32 = static_cast<int32_t>(M);
    out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&M32), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(neighbors.data()),
              static_cast<size_t>(N) * M * sizeof(int32_t));
    out.close();

    std::cout << "[WriteVectorKNN] Written to " << output_path
              << " (" << static_cast<size_t>(N) * M * sizeof(int32_t) / 1e6 << " MB)\n";
}

