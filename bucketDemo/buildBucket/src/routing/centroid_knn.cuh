#pragma once

// ============== Phase 10: GPU-Resident KNN Graph Building ==============

/**
 * 在 GPU 上构建 centroid KNN 图，结果直接保留在 GPU 上。
 *
 * @param d_centroids_f32   centroid 数据 (已在 GPU, float32, n_centroids*D)
 * @param n_centroids       centroid 数量
 * @param D                 向量维度
 * @param K                 KNN 图度数
 * @param d_graph           [out] KNN 图在 GPU 上的指针 (uint32, n_centroids*K)
 * @param max_iterations    NN Descent 最大迭代次数
 * @param termination_threshold 收敛阈值
 */
void build_centroid_knn_on_gpu(
    const float* d_centroids_f32,
    int64_t n_centroids,
    int64_t D,
    uint32_t K,
    uint32_t** d_graph,
    size_t max_iterations = 20,
    double termination_threshold = 0.0001)
{
    if (K == 0 || K > LoadConfig::MAX_KNN_K)
        throw std::runtime_error("K must be in [1, " + std::to_string(LoadConfig::MAX_KNN_K) + "]");
    if (static_cast<int64_t>(K) >= n_centroids)
        throw std::runtime_error("K must be < n_centroids");

    uint32_t intermediate_graph_degree = std::max(K * 2, (uint32_t)64);
    if (static_cast<int64_t>(intermediate_graph_degree) >= n_centroids)
        intermediate_graph_degree = static_cast<uint32_t>(n_centroids) - 1;
    if (intermediate_graph_degree < K)
        intermediate_graph_degree = K;

    std::cout << "[CAGRA KNN - GPU Resident] n=" << n_centroids
              << ", K=" << K
              << ", intermediate=" << intermediate_graph_degree << "\n";

    // Copy centroid data from GPU to CPU (NN Descent requires host data)
    std::vector<float> centroids_host(n_centroids * D);
    CUDA_CHECK(cudaMemcpy(centroids_host.data(), d_centroids_f32,
                          n_centroids * D * sizeof(float), cudaMemcpyDeviceToHost));

    raft::neighbors::experimental::nn_descent::index_params params;
    params.graph_degree              = intermediate_graph_degree;
    params.intermediate_graph_degree = static_cast<size_t>(1.5 * intermediate_graph_degree);
    params.max_iterations            = max_iterations;
    params.termination_threshold     = termination_threshold;

    auto dataset = raft::make_host_matrix_view<const float, int64_t>(
        centroids_host.data(), n_centroids, D);

    raft::resources res;

    std::cout << "  Step 1/3: NN Descent...\n";
    auto idx = raft::neighbors::experimental::nn_descent::detail::build<float, uint32_t>(
        res, params, dataset);

    std::cout << "  Step 2/3: Sort by L2...\n";
    raft::neighbors::cagra::detail::graph::sort_knn_graph(res, dataset, idx.graph());

    std::cout << "  Step 3/3: CAGRA prune to K=" << K << "...\n";
    auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(n_centroids, K);
    raft::neighbors::cagra::detail::graph::optimize(
        res, idx.graph(), cagra_graph.view());

    // Upload pruned graph to GPU
    size_t graph_bytes = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(d_graph, graph_bytes));
    CUDA_CHECK(cudaMemcpy(*d_graph, cagra_graph.data_handle(),
                          graph_bytes, cudaMemcpyHostToDevice));

    std::cout << "  KNN graph on GPU: " << graph_bytes / 1e6 << " MB\n";
}


// ============== Phase 11.5: Centroid Neighbor Expansion (K → nprobe) ==============

/**
 * GPU greedy graph search: 为每个 centroid 找 top-nprobe 个最近 centroid。
 *
 * 流程:
 *   1) 上传 centroids + graph 到 GPU
 *   2) 启动 greedy_graph_search_topK_kernel (每个 centroid 一个线程)
 *   3) 下载结果到 CPU
 *
 * 注意: 输入的 graph_host 是 CAGRA-pruned navigable graph (不是 sorted KNN),
 *       所以即使 nprobe <= K 也必须在图上 search, 不能直接截前 nprobe 个。
 *
 * @param centroids_host  (n_centroids, D) CPU centroid 坐标
 * @param graph_host      (n_centroids, K) CPU 上的 navigable graph
 * @param n_centroids     centroid 数
 * @param D               维度
 * @param K               图度数
 * @param nprobe          目标邻居数
 * @param max_iters       图搜索最大迭代轮数
 * @return                (n_centroids, nprobe) 真实近似 KNN 邻居 idx
 */
std::vector<uint32_t> expand_centroid_neighbors_gpu(
    const std::vector<float>& centroids_host,
    const std::vector<uint32_t>& graph_host,
    int64_t n_centroids,
    int D,
    uint32_t K,
    uint32_t nprobe,
    int /*max_iters*/)
{
    constexpr uint32_t MAX_NPROBE = 1024;
    if (nprobe > MAX_NPROBE) {
        throw std::runtime_error(
            "nprobe (" + std::to_string(nprobe) +
            ") exceeds MAX_NPROBE=" + std::to_string(MAX_NPROBE));
    }
    if (static_cast<int64_t>(nprobe) >= n_centroids) {
        throw std::runtime_error(
            "nprobe (" + std::to_string(nprobe) +
            ") must be < n_centroids=" + std::to_string(n_centroids));
    }

    // CAGRA self-search: 多搜 1 个候选, 用于 host 端剥掉自环 (验证 idx==qid 才剥)
    const uint32_t k_search = nprobe + 1;
    // itopk_size 至少 k_search, 按 32 对齐, floor 64
    uint32_t itopk = std::max<uint32_t>(64, ((k_search + 31) / 32) * 32);

    std::cout << "[CentroidTopK] CAGRA search (self): K=" << K
              << " → nprobe=" << nprobe
              << ", k_search=" << k_search
              << ", itopk_size=" << itopk << "\n";

    // 1) 上传 centroids 和 graph 到 GPU
    float*    d_dataset    = nullptr;
    uint32_t* d_graph      = nullptr;
    uint32_t* d_neighbors  = nullptr;
    float*    d_distances  = nullptr;
    if (centroids_host.empty()) {
        throw std::runtime_error("expand_centroid_neighbors_gpu: centroids_host is empty");
    }

    size_t dataset_bytes    = static_cast<size_t>(n_centroids) * D * sizeof(float);
    size_t graph_bytes      = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
    size_t neighbors_bytes  = static_cast<size_t>(n_centroids) * k_search * sizeof(uint32_t);
    size_t distances_bytes  = static_cast<size_t>(n_centroids) * k_search * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_dataset,    dataset_bytes));
    CUDA_CHECK(cudaMalloc(&d_graph,      graph_bytes));
    CUDA_CHECK(cudaMalloc(&d_neighbors,  neighbors_bytes));
    CUDA_CHECK(cudaMalloc(&d_distances,  distances_bytes));

    CUDA_CHECK(cudaMemcpy(d_dataset, centroids_host.data(),
                          dataset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_graph, graph_host.data(),
                          graph_bytes, cudaMemcpyHostToDevice));

    // 2) 构造 cagra::index (zero-copy view, 不重建图)
    raft::resources res;
    auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
        d_dataset, n_centroids, static_cast<int64_t>(D));
    auto graph_view   = raft::make_device_matrix_view<const uint32_t, int64_t>(
        d_graph, n_centroids, static_cast<int64_t>(K));
    raft::neighbors::cagra::index<float, uint32_t> idx(
        res, raft::distance::DistanceType::L2Expanded,
        dataset_view, graph_view);

    // 3) cagra::search (queries == dataset, k = nprobe+1)
    raft::neighbors::cagra::search_params sp;
    sp.itopk_size     = itopk;
    sp.search_width   = 1;
    sp.max_iterations = 0;  // 0 = auto
    sp.algo           = raft::neighbors::cagra::search_algo::SINGLE_CTA;

    auto queries_view = raft::make_device_matrix_view<const float, int64_t>(
        d_dataset, n_centroids, static_cast<int64_t>(D));
    auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        d_neighbors, n_centroids, static_cast<int64_t>(k_search));
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
        d_distances, n_centroids, static_cast<int64_t>(k_search));

    raft::neighbors::cagra::search(res, sp, idx,
        queries_view, neighbors_view, distances_view);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4) 下载 (n_centroids, k_search) 邻居
    std::vector<uint32_t> raw(static_cast<size_t>(n_centroids) * k_search);
    CUDA_CHECK(cudaMemcpy(raw.data(), d_neighbors,
                          neighbors_bytes, cudaMemcpyDeviceToHost));

    // 5) Host 端剥自环: 扫描每行, 找到 idx == qid 才 skip 一次, 取剩下 nprobe 个
    std::vector<uint32_t> result(static_cast<size_t>(n_centroids) * nprobe,
                                 0xFFFFFFFFu);
    #pragma omp parallel for schedule(static)
    for (int64_t qid = 0; qid < n_centroids; ++qid) {
        const uint32_t* row_in  = raw.data() + qid * k_search;
        uint32_t*       row_out = result.data() + qid * nprobe;
        uint32_t self_qid = static_cast<uint32_t>(qid);
        bool self_skipped = false;
        uint32_t written = 0;
        for (uint32_t i = 0; i < k_search && written < nprobe; ++i) {
            uint32_t nb = row_in[i];
            if (!self_skipped && nb == self_qid) {
                self_skipped = true;   // 验证 idx == qid 才剥
                continue;
            }
            row_out[written++] = nb;
        }
    }

    // 释放
    cudaFree(d_dataset);
    cudaFree(d_graph);
    cudaFree(d_neighbors);
    cudaFree(d_distances);

    return result;
}

