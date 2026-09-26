#pragma once

// ============== Phase 9: GPU helper kernels ==============

// Functor: 按 stride 访问 min_dist 数组，用于子采样估计 sum
struct StridedAccessOp {
    const float* d_min_dist;
    int64_t stride;
    __host__ __device__ float operator()(int64_t i) const {
        return d_min_dist[i * stride];
    }
};

/**
 * CUDA kernel: 计算两个 flag 数组的差异。
 *   diff[i] = 1 if curr[i] && !prev[i], else 0
 * 用于在 GPU 上找出本轮新增的候选点，避免将 flags 拷回 CPU。
 */
__global__ void compute_flag_diff(
    const uint8_t* __restrict__ curr,
    const uint8_t* __restrict__ prev,
    uint8_t* __restrict__ diff,
    int64_t N)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    diff[i] = (curr[i] && !prev[i]) ? 1 : 0;
}

/**
 * CUDA kernel: 按索引从源矩阵中 gather 行到目标矩阵。
 *   dst[k, :] = src[idx[k], :]   for k in [0, n_dst_rows)
 */
__global__ void gather_rows_by_index(
    const float* __restrict__ src,
    const int64_t* __restrict__ idx,
    float* __restrict__ dst,
    int64_t n_dst_rows, int64_t D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_dst_rows * D;
    if (tid >= total) return;

    int64_t row = tid / D;
    int64_t col = tid % D;
    dst[tid] = src[idx[row] * D + col];
}

// ============== Phase 9b: KMeans|| (Scalable KMeans++) Centroid Selection ==============

/**
 * 在 GPU 上用 Scalable KMeans++ (KMeans||, Bahmani et al. 2012) 选取 centroid。
 * 结果保留在 GPU 上不拷贝回 CPU。
 *
 * 调用方负责采样和 index 管理; 本函数只处理 GPU 上的 centroid 选取。
 *
 * 三层索引关系 (由调用方维护):
 *   raw_idx  ←→ sample_idx   : sampled_indices[sample_idx] = raw_idx
 *   sample_idx ←→ centroid_idx: 本函数返回值 [centroid_idx] = sample_idx
 *
 * @param X_sampled       采样后的数据 (CPU, float32, working_N × D)，可以是全量或子集
 * @param sampled_indices 采样索引表: sampled_indices[i] = raw dataset index
 * @param working_N       采样后的行数
 * @param D               向量维度
 * @param config          配置
 * @param mem_est         内存估算
 * @param out_quantizer   [out] PQ 量化器 (如果启用)
 * @param d_centroids_f32 [out] GPU 上的 centroid 数据 (caller 管理)
 * @return  (n_centroids,) 每个 centroid 的 sample_idx
 */
std::vector<int64_t> select_centroids_on_gpu_kmeans_parallel(
    const float* X_sampled,
    const std::vector<int64_t>& sampled_indices,
    int64_t working_N, int D,
    const LoadConfig& config,
    const MemoryEstimate& mem_est,
    SimpleQuantizer* out_quantizer,
    float** d_centroids_f32)
{
    std::cout << "[GPU Centroid Selection - KMeans|| + KMeans++]\n";

    float* allocated_centroids = nullptr;

    try {
        // Stage 1: PQ quantization if needed
        const float* upload_data = X_sampled;   // 直接指向调用方数据，零拷贝
        std::vector<float> X_pq_buf;            // 仅 PQ 时持有数据
        SimpleQuantizer quantizer{};
        if (mem_est.need_pq) {
            std::cout << "  Training PQ quantizer (" << mem_est.final_pq_bits << " bits)...\n";
            std::vector<float> pq_input(upload_data, upload_data + static_cast<size_t>(working_N) * D);
            quantizer = train_simple_quantizer(pq_input, working_N, D, mem_est.final_pq_bits);
            auto codes = simple_encode(pq_input, working_N, D, quantizer);
            X_pq_buf = simple_decode(codes, working_N, D, quantizer);
            upload_data = X_pq_buf.data();
            if (out_quantizer) *out_quantizer = quantizer;
        }

        int64_t n_centroids = mem_est.centroid_rows;
        std::cout << "  N_work=" << working_N << ", D=" << D
                  << ", n_centroids=" << n_centroids << "\n";

        // Stage 3: Upload working data to GPU & precompute L2 norms
        //          CPU 侧只有一个指针 upload_data，无额外副本
        raft::resources res;
        cudaStream_t stream = raft::resource::get_cuda_stream(res);

        auto dataset_gpu = raft::make_device_matrix<float, int64_t>(res, working_N, D);
        raft::copy(dataset_gpu.data_handle(), upload_data,
                   working_N * D, stream);
        raft::resource::sync_stream(res);

        auto X_view = raft::make_device_matrix_view<const float, int64_t>(
            dataset_gpu.data_handle(), working_N, static_cast<int64_t>(D));

        // L2 norms required by minClusterDistanceCompute (fused L2 path)
        auto L2NormX = raft::make_device_vector<float, int64_t>(res, working_N);
        raft::linalg::rowNorm(L2NormX.data_handle(),
                              dataset_gpu.data_handle(),
                              static_cast<int64_t>(D), working_N,
                              raft::linalg::L2Norm, true, stream);

        // ================================================================
        // Stage 4: KMeans|| oversampling
        // ================================================================
        std::cout << "  Running KMeans|| oversampling...\n";
        raft::random::RngState rng(config.seed);
        std::mt19937 gen(config.seed);

        // 优化策略 (针对 ~1% centroid_ratio, 仅需粗略分组):
        //   - niter = 3 轮 (而非 RAFT 默认的 8 轮)
        //   - 每轮期望选出 l*k 个候选, niter 轮总共 ≈ niter * l * k 个
        //   - 目标: 总候选 ≈ 1.5 * n_centroids (轻度过采样, 减少 GPU 工作量)
        //   - 因此 l = 1.5 / niter ≈ 0.5
        constexpr int FIXED_NITER = 3;
        constexpr double TOTAL_OVERSHOOT = 1.5;  // 总候选数倍数
        double oversampling_factor = TOTAL_OVERSHOOT / FIXED_NITER;

        // Step 1: Pick first centroid uniformly at random
        std::uniform_int_distribution<int64_t> uniform(0, working_N - 1);
        int64_t cIdx = uniform(gen);

        // isSampleCentroid[i] = 1 if point i is a candidate
        auto isSampleCentroid = raft::make_device_vector<uint8_t, int64_t>(res, working_N);
        CUDA_CHECK(cudaMemsetAsync(isSampleCentroid.data_handle(), 0,
                                   working_N * sizeof(uint8_t), stream));
        uint8_t one_val = 1;
        CUDA_CHECK(cudaMemcpyAsync(
            isSampleCentroid.data_handle() + cIdx, &one_val, 1,
            cudaMemcpyHostToDevice, stream));

        // Growing candidate data buffer on GPU
        rmm::device_uvector<float> centroidsBuf(D, stream);
        raft::copy(centroidsBuf.data(),
                   dataset_gpu.data_handle() + cIdx * D, D, stream);

        auto potentialCentroids = raft::make_device_matrix_view<float, int64_t>(
            centroidsBuf.data(), static_cast<int64_t>(1), static_cast<int64_t>(D));

        // Track candidate local indices — order matches centroidsBuf rows
        std::vector<int64_t> candidate_local_indices;
        candidate_local_indices.push_back(cIdx);

        // GPU-side prev_flags for round-by-round diffing (避免每轮 D2H)
        auto d_prev_flags = raft::make_device_vector<uint8_t, int64_t>(res, working_N);
        CUDA_CHECK(cudaMemsetAsync(d_prev_flags.data_handle(), 0,
                                   working_N * sizeof(uint8_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_prev_flags.data_handle() + cIdx, &one_val, 1,
            cudaMemcpyHostToDevice, stream));
        auto d_diff_flags = raft::make_device_vector<uint8_t, int64_t>(res, working_N);

        // Buffers for RAFT distance computation
        rmm::device_uvector<float> L2NormBuf_OR_DistBuf(0, stream);
        rmm::device_uvector<char> workspace(0, stream);
        auto minClusterDistVec = raft::make_device_vector<float, int64_t>(res, working_N);
        auto uniformRands = raft::make_device_vector<float, int64_t>(res, working_N);
        rmm::device_scalar<float> clusterCost(stream);

        // Step 2: Compute initial cost psi = phi_X(C)
        raft::cluster::detail::minClusterDistanceCompute<float, int64_t>(
            res, X_view, potentialCentroids, minClusterDistVec.view(),
            L2NormX.view(), L2NormBuf_OR_DistBuf,
            raft::distance::DistanceType::L2Expanded,
            1 << 15, 0, workspace);

        raft::cluster::detail::computeClusterCost(
            res, minClusterDistVec.view(), workspace,
            raft::make_device_scalar_view(clusterCost.data()),
            raft::identity_op{}, raft::add_op{});

        float psi = clusterCost.value(stream);
        raft::resource::sync_stream(res, stream);

        // 写死 niter (而非 log(psi)), 避免对 ~1% centroid_ratio 的过度迭代
        int niter = FIXED_NITER;
        std::cout << "  KMeans||: psi=" << psi << ", niter=" << niter
                  << ", l=" << oversampling_factor << "\n";

        // 子采样估计 psi 的参数
        // 对 minClusterDistVec 做 strided 采样 (~1%)，估计 sum 而不做全 N reduction
        constexpr int64_t PSI_SUBSAMPLE_TARGET = 16384;  // 目标采样点数
        int64_t psi_stride = std::max<int64_t>(1, working_N / PSI_SUBSAMPLE_TARGET);
        int64_t psi_sub_n = working_N / psi_stride;     // 实际采样点数
        float psi_scale = static_cast<float>(working_N) / static_cast<float>(psi_sub_n);

        // Step 3-6: Oversampling rounds
        for (int iter = 0; iter < niter; ++iter) {
            // Recompute min distances to ALL accumulated candidates (batched GEMM)
            raft::cluster::detail::minClusterDistanceCompute<float, int64_t>(
                res, X_view, potentialCentroids, minClusterDistVec.view(),
                L2NormX.view(), L2NormBuf_OR_DistBuf,
                raft::distance::DistanceType::L2Expanded,
                1 << 15, 0, workspace);

            // 子采样估计 psi: 仅对 ~1% 的点求和，再放大回估计值
            // strided iterator 访问 min_dist[0], min_dist[stride], min_dist[2*stride], ...
            StridedAccessOp stride_op{minClusterDistVec.data_handle(), psi_stride};
            auto strided_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator<int64_t>(0), stride_op);
            float sub_sum = thrust::reduce(
                thrust::cuda::par.on(stream),
                strided_iter, strided_iter + psi_sub_n,
                0.0f, thrust::plus<float>());
            psi = sub_sum * psi_scale;

            // Independent D² sampling: prob(x) = l * k * d²(x,C) / psi
            raft::random::uniform(
                res, rng, uniformRands.data_handle(), working_N, 0.0f, 1.0f);

            raft::cluster::detail::SamplingOp<float, int64_t> select_op(
                psi, oversampling_factor, n_centroids,
                uniformRands.data_handle(),
                isSampleCentroid.data_handle());

            // CUB DeviceSelect::If — parallel filtering, one kernel, no per-point sync
            rmm::device_uvector<float> CpRaw(0, stream);
            raft::cluster::detail::sampleCentroids<float, int64_t>(
                res, X_view, minClusterDistVec.view(),
                isSampleCentroid.view(), select_op, CpRaw, workspace);

            int64_t n_new = CpRaw.size() / D;
            if (n_new == 0) {
                std::cout << "  Round " << iter << ": no new candidates, stopping early\n";
                break;
            }

            // Append new candidate data to growing buffer
            size_t old_size = centroidsBuf.size();
            centroidsBuf.resize(old_size + CpRaw.size(), stream);
            raft::copy(centroidsBuf.data() + old_size,
                       CpRaw.data(), CpRaw.size(), stream);

            int64_t tot = potentialCentroids.extent(0) + n_new;
            potentialCentroids = raft::make_device_matrix_view<float, int64_t>(
                centroidsBuf.data(), tot, static_cast<int64_t>(D));

            // GPU-side flag diff + ordered compact — 避免 working_N 字节 D2H
            // sampleCentroids 标记新点 → isSampleCentroid; CUB 保序
            {
                int diff_threads = 256;
                int diff_blocks = static_cast<int>((working_N + diff_threads - 1) / diff_threads);
                compute_flag_diff<<<diff_blocks, diff_threads, 0, stream>>>(
                    isSampleCentroid.data_handle(),
                    d_prev_flags.data_handle(),
                    d_diff_flags.data_handle(),
                    working_N);
                CUDA_CHECK(cudaGetLastError());

                // thrust::copy_if 保序提取新增索引 (输出 n_new 个 int64)
                rmm::device_uvector<int64_t> d_new_idx(n_new, stream);
                auto cnt_begin = thrust::make_counting_iterator<int64_t>(0);
                thrust::copy_if(
                    thrust::cuda::par.on(stream),
                    cnt_begin, cnt_begin + working_N,
                    thrust::device_pointer_cast(d_diff_flags.data_handle()),
                    d_new_idx.begin(),
                    [] __device__ (uint8_t f) { return f > 0; });

                // 仅传回 n_new 个 int64 (几 KB)，而非 working_N 字节
                std::vector<int64_t> h_new_idx(n_new);
                raft::copy(h_new_idx.data(), d_new_idx.data(), n_new, stream);
                raft::resource::sync_stream(res, stream);
                candidate_local_indices.insert(
                    candidate_local_indices.end(), h_new_idx.begin(), h_new_idx.end());

                // prev_flags = curr_flags (GPU D2D，无 CPU 参与)
                raft::copy(d_prev_flags.data_handle(),
                           isSampleCentroid.data_handle(), working_N, stream);
            }

            std::cout << "  Round " << iter << ": +" << n_new
                      << " candidates (total=" << tot << "), psi=" << psi << "\n";
        }

        int64_t n_candidates = potentialCentroids.extent(0);
        std::cout << "  KMeans|| done: " << n_candidates << " candidates oversampled\n";

        // ================================================================
        // Stage 5: 从候选集中选 n_centroids 个实际数据点
        // 候选集已经是 D² 加权过采样的结果，直接 uniform subsample 即可
        // ================================================================
        size_t centroid_bytes = static_cast<size_t>(n_centroids) * D * sizeof(float);
        CUDA_CHECK(cudaMalloc(d_centroids_f32, centroid_bytes));
        allocated_centroids = *d_centroids_f32;

        std::vector<int64_t> sub_indices;

        if (n_candidates <= n_centroids) {
            std::cout << "  [WARN] Only " << n_candidates << " candidates <= "
                      << n_centroids << " requested. Using all.\n";
            CUDA_CHECK(cudaMemcpyAsync(
                *d_centroids_f32, centroidsBuf.data(),
                n_candidates * D * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sub_indices.resize(n_candidates);
            std::iota(sub_indices.begin(), sub_indices.end(), 0LL);
        } else {
            // 候选集已由 KMeans|| D² 加权采样产生，分布良好
            // 直接 uniform 随机选 n_centroids 个，O(n_centroids)，无 GPU sync
            std::cout << "  Selecting " << n_centroids << " from "
                      << n_candidates << " candidates (uniform subsample)...\n";

            // Fisher-Yates 前 n_centroids 步，生成不重复的候选索引
            std::vector<int64_t> perm(n_candidates);
            std::iota(perm.begin(), perm.end(), 0LL);
            std::mt19937 sub_gen(config.seed + 42);
            for (int64_t i = 0; i < n_centroids; ++i) {
                std::uniform_int_distribution<int64_t> d(i, n_candidates - 1);
                std::swap(perm[i], perm[d(sub_gen)]);
            }
            sub_indices.assign(perm.begin(), perm.begin() + n_centroids);

            // GPU gather: 按选中的行号从 centroidsBuf 拷贝到 d_centroids_f32
            // 上传 sub_indices 到 GPU，用 gather kernel 一次完成
            rmm::device_uvector<int64_t> d_sub_indices(n_centroids, stream);
            raft::copy(d_sub_indices.data(), sub_indices.data(), n_centroids, stream);

            int64_t total_elems = n_centroids * static_cast<int64_t>(D);
            int gthreads = 256;
            int gblocks = static_cast<int>((total_elems + gthreads - 1) / gthreads);
            gather_rows_by_index<<<gblocks, gthreads, 0, stream>>>(
                centroidsBuf.data(), d_sub_indices.data(),
                *d_centroids_f32, n_centroids, static_cast<int64_t>(D));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Release working data — only caller-managed centroids remain
        { auto _drop = std::move(dataset_gpu); }

        // Stage 6: centroid_idx → candidate_idx → sample_idx
        //   sub_indices[k]                        : centroid k 在 candidate 集中的行号
        //   candidate_local_indices[sub_indices[k]]: 该候选点的 sample_idx
        //   调用方可用 sampled_indices[sample_idx] 得到 raw_idx
        int64_t n_selected = static_cast<int64_t>(sub_indices.size());
        std::vector<int64_t> centroid_sample_indices(n_selected);
        for (int64_t k = 0; k < n_selected; ++k) {
            centroid_sample_indices[k] = candidate_local_indices[sub_indices[k]];
        }

        std::cout << "  Centroids on GPU: " << centroid_bytes / 1e6 << " MB\n";
        std::cout << "  Generated " << n_selected << " centroids\n";

        return centroid_sample_indices;

    } catch (const std::exception& e) {
        std::cerr << "[select_centroids_on_gpu_kmeans_parallel] Error: " << e.what() << "\n";

        if (allocated_centroids) {
            cudaFree(allocated_centroids);
            *d_centroids_f32 = nullptr;
        }

        throw;
    }
}

