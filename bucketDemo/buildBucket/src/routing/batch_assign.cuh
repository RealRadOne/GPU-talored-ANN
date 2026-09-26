#pragma once

// ============== Phase 7: Greedy Graph Search Kernel (template, 支持 uint8/float 等) ==============

cudaEvent_t begin(cudaStream_t stream){
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    return event;
}

template <typename T>
__device__ float device_l2_dist(const T* a, const T* b, int D)
{
    float dist = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = static_cast<float>(a[d]) - static_cast<float>(b[d]);
        dist += diff * diff;
    }
    return dist;
}

/**
 * 每个线程处理一个 query，在 KNN 图上做贪心搜索，返回 top-2 最近 centroid。
 * 支持任意数值类型 T (uint8_t, float, uint32_t, ...)
 */
template <typename T>
__global__ void greedy_graph_search_top2_kernel(
    const T*        dataset,     // (n, D) centroid data on GPU
    const uint32_t* graph,       // (n, K) KNN graph on GPU
    const T*        queries,     // (batch, D) query data
    uint32_t*       out_top2,    // (batch, 2) output: [top1, top2] per query
    int n, int D, int K, int batch, int max_iters)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= batch) return;

    const T* q = queries + static_cast<int64_t>(qid) * D;

    // 用伪随机的 entry point 开始搜索
    uint32_t cur = static_cast<uint32_t>(static_cast<uint64_t>(qid) * 7919ULL
                                         % static_cast<uint64_t>(n));

    float best1 = device_l2_dist(q, dataset + static_cast<int64_t>(cur) * D, D);
    uint32_t top1 = cur;
    float best2 = 1e30f;
    uint32_t top2 = cur;

    for (int iter = 0; iter < max_iters; ++iter) {
        bool improved = false;
        for (int k = 0; k < K; ++k) {
            uint32_t nb = graph[static_cast<int64_t>(cur) * K + k];
            if (nb >= static_cast<uint32_t>(n)) continue;

            float d = device_l2_dist(q, dataset + static_cast<int64_t>(nb) * D, D);
            if (d < best1) {
                best2 = best1; top2 = top1;
                best1 = d;     top1 = nb;
                improved = true;
            } else if (d < best2 && nb != top1) {
                best2 = d; top2 = nb;
            }
        }
        if (!improved) break;
        cur = top1;
    }

    out_top2[static_cast<int64_t>(qid) * 2]     = top1;
    out_top2[static_cast<int64_t>(qid) * 2 + 1] = top2;
}

// CAGRA-style beam width 估算:
//   - 参考 raft::neighbors::cagra::search_params 默认 itopk_size = 64
//   - floor 64, 上限 1024, 至少 max(K, 5)
//   - 大 K 时让 beam 适当宽于 K (1.5x)
//   - 例: K=2  → 64;  K=10 → 64;  K=64 → 96;  K=512 → 768;  K≥683 → 1024(cap)
inline uint32_t compute_beam_width(uint32_t k) {
    constexpr uint32_t MIN_BEAM    = 5;
    constexpr uint32_t MAX_BEAM    = 1024;
    constexpr uint32_t CAGRA_ITOPK = 64;
    uint32_t beam = std::max(CAGRA_ITOPK, (3u * k + 1u) / 2u);
    beam = std::max(beam, std::max(k, MIN_BEAM));
    beam = std::min(beam, MAX_BEAM);
    return beam;
}

/**
 * Greedy best-first 图搜索 (CAGRA search 的 single-thread-per-query 简化版),
 * 在 navigable graph 上为每个 query 找 top-K_out 近邻。
 *
 * 算法 (per qid):
 *   1) 维护 sorted candidate buffer 大小 = beam_width (≥ K_out, 余量提升 recall)
 *   2) 起始: self-search 用 graph[qid][0]; query 模式用 hash 落到 db
 *   3) 每轮: 取 buffer 里最近的未访问候选 → 扩展其 K_graph 邻居 → try_insert
 *   4) 直到 max_iters 或 buffer 全部访问
 *   5) self-search 时排除 idx == qid (distance=0 自环)
 *   6) 输出 buffer 中前 K_out 个作为最终结果
 *
 * 工作 buffer (idx/dist/visited) 全部由 caller 在 GPU global memory 预分配,
 * 大小均为 n_q * beam_width.
 *
 * 两种调用模式:
 *   self-search:  queries == dataset, n_q == n_db, exclude_self_idx == true
 *   query mode:   queries 与 dataset 不同, exclude_self_idx == false
 */
template <typename T>
__global__ void greedy_graph_search_topK_kernel(
    const T*        dataset,       // (n_db, D)
    const uint32_t* graph,         // (n_db, K_graph)
    const T*        queries,       // (n_q, D); self-search 时传 dataset
    uint32_t*       out_topK,      // (n_q, K_out)  最终输出 idx
    uint32_t*       buf_idx,       // (n_q, beam_width)  candidate idx
    float*          buf_dist,      // (n_q, beam_width)  candidate dist
    uint8_t*        buf_visited,   // (n_q, beam_width)  visited flags (0/1)
    int n_db, int n_q, int D, int K_graph, int K_out, int beam_width, int max_iters,
    bool exclude_self_idx)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= n_q) return;

    const T* q = queries + static_cast<int64_t>(qid) * D;
    uint32_t* idx_buf = buf_idx     + static_cast<int64_t>(qid) * beam_width;
    float*    d_buf   = buf_dist    + static_cast<int64_t>(qid) * beam_width;
    uint8_t*  vis_buf = buf_visited + static_cast<int64_t>(qid) * beam_width;

    // 初始化 visited
    for (int i = 0; i < beam_width; ++i) vis_buf[i] = 0;

    int filled = 0;

    // 在 sorted candidate buffer 里插入 (idx, d); idx/dist/visited 三数组同步移位
    auto try_insert = [&](uint32_t idx, float d) {
        if (exclude_self_idx && idx == static_cast<uint32_t>(qid)) return;
        // 去重 (线性扫描 buffer)
        for (int i = 0; i < filled; ++i) {
            if (idx_buf[i] == idx) return;
        }
        if (filled < beam_width) {
            int pos = filled;
            while (pos > 0 && d_buf[pos-1] > d) {
                d_buf[pos]   = d_buf[pos-1];
                idx_buf[pos] = idx_buf[pos-1];
                vis_buf[pos] = vis_buf[pos-1];
                --pos;
            }
            d_buf[pos]   = d;
            idx_buf[pos] = idx;
            vis_buf[pos] = 0;
            ++filled;
        } else if (d < d_buf[beam_width - 1]) {
            int pos = beam_width - 1;
            while (pos > 0 && d_buf[pos-1] > d) {
                d_buf[pos]   = d_buf[pos-1];
                idx_buf[pos] = idx_buf[pos-1];
                vis_buf[pos] = vis_buf[pos-1];
                --pos;
            }
            d_buf[pos]   = d;
            idx_buf[pos] = idx;
            vis_buf[pos] = 0;
        }
    };

    // 起始点
    uint32_t cur;
    if (exclude_self_idx) {
        // self-search: qid 既是 query 也是 db 索引, 用 qid 的图邻居作种
        cur = graph[static_cast<int64_t>(qid) * K_graph];
        if (cur >= static_cast<uint32_t>(n_db) || cur == static_cast<uint32_t>(qid)) {
            cur = static_cast<uint32_t>((qid + 1) % n_db);
        }
    } else {
        // query mode: query 不在 db 里, hash 选一个 db 节点作种
        cur = static_cast<uint32_t>(static_cast<uint64_t>(qid) * 7919ULL
                                    % static_cast<uint64_t>(n_db));
    }
    float d_cur = device_l2_dist(q, dataset + static_cast<int64_t>(cur) * D, D);
    try_insert(cur, d_cur);

    // Greedy best-first 扩展
    for (int iter = 0; iter < max_iters; ++iter) {
        int next_pos = -1;
        for (int i = 0; i < filled; ++i) {
            if (!vis_buf[i]) { next_pos = i; break; }
        }
        if (next_pos < 0) break;  // buffer 全部访问完 → 收敛

        uint32_t expand_node = idx_buf[next_pos];
        vis_buf[next_pos] = 1;

        for (int k = 0; k < K_graph; ++k) {
            uint32_t nb = graph[static_cast<int64_t>(expand_node) * K_graph + k];
            if (nb >= static_cast<uint32_t>(n_db)) continue;
            float d = device_l2_dist(q, dataset + static_cast<int64_t>(nb) * D, D);
            try_insert(nb, d);
        }
    }

    // 输出 buffer 中前 K_out 个 (sorted by L2)
    uint32_t* out_buf = out_topK + static_cast<int64_t>(qid) * K_out;
    for (int i = 0; i < K_out; ++i) {
        out_buf[i] = (i < filled) ? idx_buf[i] : 0xFFFFFFFFu;
    }
}

// ============== Phase 8: Batch Assignment via Graph ANNS ==============

/**
 * 分批处理原始数据集，在 GPU 上基于 centroid KNN 图做 ANNS 检索，
 * 为每个非 centroid 数据点找到最近的 2 个 centroid，在 CPU 上做均衡分配。
 *
 * centroid 数据和 KNN 图由 caller 提前上传至 GPU，本函数直接复用，不再重复上传。
 *
 * @tparam T                     原始数据元素类型 (float, uint8_t, uint32_t 等)
 * @tparam CentroidT             centroid 在 GPU 上的数据类型 (float 或 uint8_t)
 * @param X_full                 完整原始数据集 (N * D, T, row-major, CPU)
 * @param N                      数据点总数
 * @param D                      向量维度
 * @param n_centroids            centroid 数量
 * @param centroid_global_indices 每个 centroid 在原始数据中的全局索引
 * @param d_centroids            centroid 数据 (n_centroids * D, CentroidT, 已在 GPU)
 * @param centroid_gpu_bytes     centroid 数据在 GPU 上占用的字节数
 * @param d_graph                centroid KNN 图 (n_centroids * K, uint32, 已在 GPU)
 * @param graph_bytes            KNN 图在 GPU 上占用的字节数
 * @param K                      KNN 图度数
 * @param quantizer              量化器（仅 CentroidT=uint8_t 时使用）
 * @param Mbatch_bytes           GPU 剩余可用空间 (bytes), 默认 10GB
 * @param search_max_iters       图搜索最大迭代次数, 默认 64
 *
 * @return  (N,) 每个点分配到的 local centroid index [0, n_centroids)
 */
template <typename T, typename CentroidT>
std::vector<int64_t> batch_assign_with_cagra_anns(
    const T* X_full,
    int64_t N,
    int64_t D,
    int64_t n_centroids,
    const std::vector<int64_t>& centroid_global_indices,
    CentroidT* d_centroids,
    size_t centroid_gpu_bytes,
    uint32_t* d_graph,
    size_t graph_bytes,
    uint32_t K,
    const SimpleQuantizer& quantizer,
    size_t Mbatch_bytes = 10ULL * 1024 * 1024 * 1024,
    int search_max_iters = 64)
{
    constexpr bool is_pq = std::is_same<CentroidT, uint8_t>::value;

    // ================================================================
    // Step 0: 建立 centroid 集合，收集非 centroid 点索引
    // ================================================================
    std::unordered_set<int64_t> centroid_set(
        centroid_global_indices.begin(), centroid_global_indices.end());

    std::vector<int64_t> non_centroid_indices;
    non_centroid_indices.reserve(N - n_centroids);
    for (int64_t i = 0; i < N; ++i) {
        if (centroid_set.find(i) == centroid_set.end()) {
            non_centroid_indices.push_back(i);
        }
    }
    int64_t N_nc = static_cast<int64_t>(non_centroid_indices.size());

    std::cout << "[BatchAssign] N=" << N
              << ", centroids=" << n_centroids
              << ", non-centroid=" << N_nc
              << ", is_pq=" << is_pq << "\n";

    // ================================================================
    // Step 1: 估算每批大小 Nv
    //         centroid 和 graph 已在 GPU，扣除其占用后计算剩余空间
    // ================================================================
    size_t constant_gpu = graph_bytes + centroid_gpu_bytes;
    size_t remaining = (Mbatch_bytes > constant_gpu) ? (Mbatch_bytes - constant_gpu) : 0;

    std::cout << "[BatchAssign] GPU constant: centroids=" << centroid_gpu_bytes / 1e6
              << "MB, graph=" << graph_bytes / 1e6
              << "MB, remaining=" << remaining / 1e9 << "GB\n";

    // 用 cagra::search 替代手写 kernel; queries != centroids, 不需 self-exclusion.
    constexpr uint32_t TOP_K = 2;
    // CAGRA itopk_size 至少 TOP_K, 32 对齐, floor 64
    const uint32_t itopk = std::max<uint32_t>(64, ((TOP_K + 31) / 32) * 32);

    // Per-point GPU (CAGRA 不再需要 buf_idx/dist/visited):
    //   query:       D * sizeof(CentroidT)
    //   neighbors:   TOP_K * 4
    //   distances:   TOP_K * 4
    size_t per_point_bytes = static_cast<size_t>(D) * sizeof(CentroidT)
                           + TOP_K * (sizeof(uint32_t) + sizeof(float));

    int64_t Nv = static_cast<int64_t>(remaining / per_point_bytes);
    Nv = std::max(static_cast<int64_t>(1), std::min(Nv, N_nc));

    std::cout << "[BatchAssign] per_point=" << per_point_bytes << "B"
              << ", itopk=" << itopk
              << ", Nv=" << Nv
              << ", batches=" << (N_nc + Nv - 1) / Nv << "\n";

    // ================================================================
    // Step 3: 初始化分配结果和聚类大小
    // ================================================================
    std::vector<int64_t> assignments(N, -1);

    // centroid 点预分配给自身
    for (int64_t c = 0; c < n_centroids; ++c) {
        assignments[centroid_global_indices[c]] = c;
    }

    // 聚类大小：每个 centroid 初始大小为 1
    std::vector<std::atomic<int64_t>> cluster_sizes(n_centroids);
    for (auto& s : cluster_sizes) s.store(1, std::memory_order_relaxed);

    // ================================================================
    // Step 4: 构造 cagra::index<CentroidT, uint32_t> (zero-copy view, 只构造一次)
    // ================================================================
    raft::resources res;
    auto dataset_view = raft::make_device_matrix_view<const CentroidT, int64_t>(
        d_centroids, n_centroids, static_cast<int64_t>(D));
    auto graph_view = raft::make_device_matrix_view<const uint32_t, int64_t>(
        d_graph, n_centroids, static_cast<int64_t>(K));
    raft::neighbors::cagra::index<CentroidT, uint32_t> cagra_idx(
        res, raft::distance::DistanceType::L2Expanded,
        dataset_view, graph_view);

    raft::neighbors::cagra::search_params sp;
    sp.itopk_size     = itopk;
    sp.search_width   = 1;
    sp.max_iterations = 0;
    sp.algo           = raft::neighbors::cagra::search_algo::SINGLE_CTA;
    (void)search_max_iters;  // CAGRA 自己决定 iter, 参数保留为接口兼容

    // ================================================================
    // Step 5: 分批处理
    // ================================================================
    // 分配 per-batch GPU buffer (复用)
    CentroidT* d_queries    = nullptr;
    uint32_t*  d_top2       = nullptr;
    float*     d_distances  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_queries,    Nv * D * sizeof(CentroidT)));
    CUDA_CHECK(cudaMalloc(&d_top2,       Nv * TOP_K * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_distances,  Nv * TOP_K * sizeof(float)));

    std::vector<uint32_t> h_top2(Nv * TOP_K);

    for (int64_t batch_start = 0; batch_start < N_nc; batch_start += Nv) {
        int64_t batch_end = std::min(batch_start + Nv, N_nc);
        int64_t batch_size = batch_end - batch_start;

        // ---- 4a: 读取 batch 数据，转换为 CentroidT，上传到 GPU ----
        if constexpr (is_pq) {
            // PQ 模式：读取原始 T 数据 -> 转为 float32 -> encode uint8 -> 上传
            std::vector<float> batch_f32(batch_size * D);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t gi = non_centroid_indices[batch_start + i];
                for (int64_t d = 0; d < D; ++d) {
                    batch_f32[i * D + d] = static_cast<float>(X_full[gi * D + d]);
                }
            }

            auto codes = simple_encode(batch_f32, batch_size, D, quantizer);

            CUDA_CHECK(cudaMemcpy(d_queries, codes.data(),
                                  batch_size * D * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice));
        } else {
            // 非 PQ 模式：读取原始 T 数据 -> 转为 CentroidT -> 上传
            std::vector<CentroidT> batch_data(batch_size * D);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t gi = non_centroid_indices[batch_start + i];
                for (int64_t d = 0; d < D; ++d) {
                    batch_data[i * D + d] = static_cast<CentroidT>(X_full[gi * D + d]);
                }
            }

            CUDA_CHECK(cudaMemcpy(d_queries, batch_data.data(),
                                  batch_size * D * sizeof(CentroidT),
                                  cudaMemcpyHostToDevice));
        }

        // ---- 4b: cagra::search (k=2, queries 不在 dataset 里, 无 self-exclusion) ----
        auto queries_view = raft::make_device_matrix_view<const CentroidT, int64_t>(
            d_queries, batch_size, static_cast<int64_t>(D));
        auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
            d_top2, batch_size, static_cast<int64_t>(TOP_K));
        auto distances_view = raft::make_device_matrix_view<float, int64_t>(
            d_distances, batch_size, static_cast<int64_t>(TOP_K));

        raft::neighbors::cagra::search(res, sp, cagra_idx,
            queries_view, neighbors_view, distances_view);

        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- 4c: 下载 top-2 local centroid indices ----
        CUDA_CHECK(cudaMemcpy(h_top2.data(), d_top2,
                              batch_size * TOP_K * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        // ---- 4d: CPU 并行: local idx -> global idx 映射 + 均衡分配 ----
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < batch_size; ++i) {
            uint32_t local_idx1 = h_top2[i * TOP_K];      // 最近
            uint32_t local_idx2 = h_top2[i * TOP_K + 1];  // 第二近

            // 映射: local centroid idx -> global dataset idx
            // (caller 可通过 centroid_global_indices[local_idx] 得到全局)

            // 比较两个聚类大小，做均衡分配
            int64_t s1 = cluster_sizes[local_idx1].load(std::memory_order_relaxed);
            int64_t s2 = cluster_sizes[local_idx2].load(std::memory_order_relaxed);

            uint32_t chosen;
            if (static_cast<double>(s2) < 0.8 * static_cast<double>(s1)) {
                chosen = local_idx2;  // 聚类 2 太小，分配到聚类 2
            } else {
                chosen = local_idx1;  // 分配到最近的聚类 1
            }

            int64_t gi = non_centroid_indices[batch_start + i];
            assignments[gi] = chosen;
            cluster_sizes[chosen].fetch_add(1, std::memory_order_relaxed);
        }

        if ((batch_start / Nv) % 10 == 0 || batch_end == N_nc) {
            std::cout << "[BatchAssign] " << batch_end << "/" << N_nc << "\n";
        }
    }

    // ================================================================
    // Step 6: 释放 per-batch GPU buffer（centroid 和 graph 保留）
    // ================================================================
    if (d_queries)    CUDA_CHECK(cudaFree(d_queries));
    if (d_top2)       CUDA_CHECK(cudaFree(d_top2));
    if (d_distances)  CUDA_CHECK(cudaFree(d_distances));
    // NOTE: d_centroids 和 d_graph 由 caller 管理，本函数不释放

    // 聚类大小统计
    int64_t min_sz = std::numeric_limits<int64_t>::max(), max_sz = 0;
    double avg_sz = 0;
    for (int64_t c = 0; c < n_centroids; ++c) {
        int64_t sz = cluster_sizes[c].load();
        min_sz = std::min(min_sz, sz);
        max_sz = std::max(max_sz, sz);
        avg_sz += sz;
    }
    avg_sz /= n_centroids;
    std::cout << "[BatchAssign] Done. Cluster size: min=" << min_sz
              << " max=" << max_sz
              << " avg=" << std::fixed << std::setprecision(1) << avg_sz << "\n";

    return assignments;
}

