#include "core/common_types.cuh"
#include "core/memory_budget.hpp"
#include "core/sampling.hpp"
#include "clustering/pq_quantizer.hpp"
#include "core/cli.hpp"
#include "routing/batch_assign.cuh"
#include "clustering/kmeans_gpu.cuh"
#include "routing/centroid_knn.cuh"
#include "merge/disk_writer.hpp"
#include "gemm/tensorcore_knn.cuh"
#include "merge/async_merge.hpp"

int run_pipeline_impl(
    const std::string& input_path,
    const std::string& output_root,
    int search_max_iters,
    int neighbors_m,
    uint32_t nprobe,
    bool do_reorder,
    int iterations,
    LoadConfig config,
    const std::string& ext,
    int32_t order_window_arg)
{
    try {
        init_gpu_limit_if_needed(config);

        // 在 output_root 下建子目录: k<knn_k>p<nprobe>m<neighbors_m>
        // knn_k = 图度数 K; nprobe = Step 6 邻居扩展数 (0 时回退到 knn_k)
        uint32_t output_nprobe = (nprobe > 0) ? nprobe : config.knn_k;
        std::string output_dir = output_root
                               + "/k" + std::to_string(config.knn_k)
                               + "p" + std::to_string(output_nprobe)
                               + "m" + std::to_string(neighbors_m);
        std::filesystem::create_directories(output_dir);
        std::cout << "Output subdir: " << output_dir << "\n";

        using Clock = std::chrono::high_resolution_clock;
        auto t_total_start = Clock::now();
        double elapsed_step1 = 0, elapsed_step2 = 0, elapsed_step3 = 0;
        double elapsed_step3p5 = 0, elapsed_step4 = 0, elapsed_step5 = 0;
        double elapsed_step6 = 0, elapsed_step7 = 0;
        double elapsed_write_knn = 0;

        // ================================================================
        // Step 1: Read header + prepare sampled data for centroid selection
        // ================================================================
        std::cout << "=== Step 1: Preparing data from " << input_path << " ===\n";
        auto t1 = Clock::now();
        int32_t N = 0, D = 0;

        // ext 由 caller 传入；此处仅校验
        if (ext != ".fbin" && ext != ".bin" && ext != ".u8bin" && ext != ".i8bin"
            && ext != ".ibin" && ext != ".ubin") {
            throw std::runtime_error("Unsupported file format: " + ext);
        }

        // 只读 header 获取 N, D
        auto [hdr_n, hdr_d] = load::read_fbin_header(input_path);
        N = hdr_n; D = hdr_d;
        std::cout << "  Header: N=" << N << ", D=" << D << "\n";

        validate_load_config(config, N);

        // Memory estimation
        static constexpr int64_t MAX_CENTROIDS = 1000000;
        int64_t n_centroids = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
        if (n_centroids > MAX_CENTROIDS) {
            std::cout << "  Capping n_centroids to " << MAX_CENTROIDS << "\n";
            n_centroids = MAX_CENTROIDS;
        }
        MemoryEstimate mem_est = estimate_memory_requirement(N, D, n_centroids, config);
        print_memory_estimate(mem_est);

        // 采样索引: sampled_indices[sample_idx] = raw_idx
        std::vector<int64_t> sampled_indices;
        std::vector<float> X_sampled;
        int64_t working_N;

        if (!mem_est.fits_in_gpu) {
            // 只从 disk 读取采样行，不加载完整数据集
            working_N = mem_est.sampled_data_rows;
            sampled_indices = sample_without_replacement(N, working_N, config.seed);
            // 排序以实现顺序磁盘读取 (SSD/NVMe 友好)
            auto sorted_order = sampled_indices;
            std::sort(sorted_order.begin(), sorted_order.end());

            if (ext == ".fbin" || ext == ".bin") {
                int32_t n_tmp, d_tmp;
                load::read_fbin_sampled(input_path, sorted_order, X_sampled, n_tmp, d_tmp);
            } else {
                // u8bin/ibin: fallback 到全量读取再采样
                // TODO: 为其他格式实现 sampled reader
                std::vector<float> X_full_tmp;
                if (ext == ".u8bin" || ext == ".i8bin") {
                    int32_t n2, d2;
                    load::read_u8bin_to_f32(input_path, X_full_tmp, n2, d2);
                } else {
                    std::vector<int32_t> itmp;
                    int32_t n2, d2;
                    load::read_ibin_i32(input_path, itmp, n2, d2);
                    X_full_tmp.resize(itmp.size());
                    for (size_t i = 0; i < itmp.size(); ++i) X_full_tmp[i] = static_cast<float>(itmp[i]);
                }
                X_sampled.resize(static_cast<size_t>(working_N) * D);
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < working_N; ++i) {
                    std::memcpy(X_sampled.data() + i * D,
                                X_full_tmp.data() + sorted_order[i] * D,
                                D * sizeof(float));
                }
            }

            // sampled_indices 需要和 X_sampled 行顺序一致 (sorted_order)
            sampled_indices = std::move(sorted_order);
            std::cout << "  Sampled " << working_N << " rows from disk ("
                      << X_sampled.size() * sizeof(float) / 1e9 << " GB)\n";
        } else {
            // 全量数据可放入 GPU — 此处仍需全量读取 (后续 step 也需要)
            working_N = N;
            sampled_indices.resize(N);
            std::iota(sampled_indices.begin(), sampled_indices.end(), 0LL);

            if (ext == ".fbin" || ext == ".bin") {
                load::read_fbin_f32(input_path, X_sampled, N, D);
            } else if (ext == ".u8bin" || ext == ".i8bin") {
                load::read_u8bin_to_f32(input_path, X_sampled, N, D);
            } else {
                std::vector<int32_t> tmp;
                load::read_ibin_i32(input_path, tmp, N, D);
                X_sampled.resize(tmp.size());
                for (size_t i = 0; i < tmp.size(); ++i) X_sampled[i] = static_cast<float>(tmp[i]);
            }
            std::cout << "  Loaded full dataset: " << X_sampled.size() * sizeof(float) / 1e9 << " GB\n";
        }

        elapsed_step1 = std::chrono::duration<double>(Clock::now() - t1).count();
        std::cout << "  Step 1 done [" << std::fixed << std::setprecision(3) << elapsed_step1 << "s]\n";

        // ================================================================
        // Per-iteration outer loop (Steps 2-6 may repeat with varying seed)
        // ================================================================
        // running per-vector KNN 现在活在磁盘上 (见 RunningKnnFile)，不再是
        // 内存里的 (N,M) 数组，也不再需要 merge_future 这个跨 iteration 的
        // 后台任务 —— 合并已经下沉到 build_vector_knn_with_tensorcore 内部,
        // 逐 bucket 同步做掉了 (scatter_pending -> merge_row_into_disk)。
        std::unique_ptr<RunningKnnFile> running_knn_file;
        if (neighbors_m > 0) {
            running_knn_file = std::make_unique<RunningKnnFile>(
                RunningKnnFile::create(
                    output_dir + "/vector_knn.bin",
                    output_dir + "/vector_dists.bin",
                    N, neighbors_m, config.cpu_limit_bytes / 4));
        }

        // 这些值由最后一次 iteration 决定 (用于 Step 5/7)
        std::vector<int64_t> assignments;
        std::vector<int64_t> centroid_global_indices;
        std::vector<uint32_t> centroid_knn_graph_host;  // (n_centroids, K), row-major
        uint32_t             K = config.knn_k;

        // X_full 在 Step 3.5 加载, 跨 iteration 复用 (大数据)
        // 在第一次 iteration 的 Step 4 之前加载
        std::vector<DataT> X_full;
        bool X_full_loaded = false;

        for (int iter = 0; iter < iterations; ++iter) {
            // 每次 iteration 用不同 seed (centroid 选取多样化)
            uint32_t iter_seed = config.seed + static_cast<uint32_t>(iter);
            LoadConfig iter_config = config;
            iter_config.seed = iter_seed;

            if (iterations > 1) {
                std::cout << "\n========== Iteration " << (iter + 1) << " / " << iterations
                          << " (seed=" << iter_seed << ") ==========\n";
            }

        // ================================================================
        // Step 2: Select centroids — results stay on GPU
        // ================================================================
        std::cout << "=== Step 2: Selecting centroids (GPU-resident) ===\n";
        auto t2 = Clock::now();
        float* d_centroids_f32 = nullptr;
        SimpleQuantizer quantizer{};

        // 返回 centroid 的 sample_idx
        std::vector<int64_t> centroid_sample_indices = select_centroids_on_gpu_kmeans_parallel(
            X_sampled.data(), sampled_indices,
            working_N, D, iter_config, mem_est, &quantizer, &d_centroids_f32);

        // sample_idx → raw_idx
        centroid_global_indices.assign(centroid_sample_indices.size(), 0);
        for (size_t k = 0; k < centroid_sample_indices.size(); ++k) {
            centroid_global_indices[k] = sampled_indices[centroid_sample_indices[k]];
        }

        n_centroids = static_cast<int64_t>(centroid_global_indices.size());
        cudaDeviceSynchronize();
        double iter_step2 = std::chrono::duration<double>(Clock::now() - t2).count();
        elapsed_step2 += iter_step2;
        std::cout << "  Step 2 done [" << std::fixed << std::setprecision(3) << iter_step2 << "s]\n";

        // ================================================================
        // Step 3: Build KNN graph on centroids — results stay on GPU
        // ================================================================
        std::cout << "=== Step 3: Building centroid KNN graph (GPU-resident) ===\n";
        auto t3 = Clock::now();
        uint32_t* d_graph = nullptr;
        K = config.knn_k;

        build_centroid_knn_on_gpu(
            d_centroids_f32, n_centroids, D, K, &d_graph);

        size_t centroid_gpu_bytes = static_cast<size_t>(n_centroids) * D * sizeof(float);
        size_t graph_bytes = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
        cudaDeviceSynchronize();
        double iter_step3 = std::chrono::duration<double>(Clock::now() - t3).count();
        elapsed_step3 += iter_step3;
        std::cout << "  Step 3 done [" << std::fixed << std::setprecision(3) << iter_step3 << "s]\n";

        // ================================================================
        // Step 3.5: Lazy-load full dataset if not already loaded
        //           (采样模式下 X_sampled 只有子集，assignment/KNN 需要全量数据)
        // ================================================================
        // X_full 以 DataT 存（保留原始 element type，省 host RAM/PCIe，u8 时 4×）
        // 多 iteration 时只在第一次 iteration 加载, 后续复用
        if (!X_full_loaded) {
            auto t_load = Clock::now();
            if (!mem_est.fits_in_gpu) {
                std::cout << "=== Step 3.5: Loading full dataset for assignment ===\n";
                // 不做类型转换，直接以 DataT 读 BIGANN payload
                int32_t fullN = 0, fullD = 0;
                load::read_bigann_raw<DataT>(input_path, X_full, fullN, fullD);
                if (fullN != N || fullD != D)
                    throw std::runtime_error("Full-load header mismatches sampled header");
                std::cout << "  Full data loaded: " << X_full.size() * sizeof(DataT) / 1e9 << " GB"
                          << " [" << std::chrono::duration<double>(Clock::now() - t_load).count() << "s]\n";
            } else {
                // fits_in_gpu: X_sampled (fp32) 是全量；转成 DataT 给 Step 4/6 用
                X_full.resize(X_sampled.size());
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < X_sampled.size(); ++i)
                    X_full[i] = static_cast<DataT>(X_sampled[i]);
            }
            // 多 iteration: X_sampled 不再需要 (centroid 选取从 iter 1 起会重复用 X_sampled,
            // 但我们改为始终用 X_full → 跨 iteration 复用)。
            // 不过 select_centroids_on_gpu_kmeans_parallel 需要 fp32 X_sampled,
            // 多 iter 时需要保留 X_sampled. 单 iter 时可以 drop.
            if (iterations <= 1) {
                auto _drop = std::move(X_sampled);
            }
            X_full_loaded = true;
            elapsed_step3p5 = std::chrono::duration<double>(Clock::now() - t_load).count();
        }

        // ================================================================
        // Step 4: Batch assign — GPU centroids + graph reused, no re-upload
        // ================================================================
        std::cout << "=== Step 4: Batch assignment ===\n";
        auto t4 = Clock::now();

        // Step 6 需要 centroid 坐标做邻居扩展; 在 Step 4 释放 GPU centroids 前先存 CPU
        std::vector<float> centroids_host_for_step6;
        bool need_centroids_for_step6 = (neighbors_m > 0);
        if (need_centroids_for_step6) {
            centroids_host_for_step6.resize(static_cast<size_t>(n_centroids) * D);
            CUDA_CHECK(cudaMemcpy(centroids_host_for_step6.data(), d_centroids_f32,
                                  centroid_gpu_bytes, cudaMemcpyDeviceToHost));
        }

        // Compute available GPU memory for batch processing
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        size_t Mbatch_bytes = static_cast<size_t>(free_bytes * 0.9);

        assignments.clear();
        if (mem_est.need_pq) {
            // PQ mode: centroids need to be encoded to uint8 on GPU
            // Encode centroid f32 -> uint8 on CPU, then upload
            std::vector<float> centroids_host(n_centroids * D);
            CUDA_CHECK(cudaMemcpy(centroids_host.data(), d_centroids_f32,
                                  centroid_gpu_bytes, cudaMemcpyDeviceToHost));

            auto centroid_codes = simple_encode(centroids_host, n_centroids, D, quantizer);

            uint8_t* d_centroids_u8 = nullptr;
            size_t u8_bytes = static_cast<size_t>(n_centroids) * D * sizeof(uint8_t);
            CUDA_CHECK(cudaMalloc(&d_centroids_u8, u8_bytes));
            CUDA_CHECK(cudaMemcpy(d_centroids_u8, centroid_codes.data(),
                                  u8_bytes, cudaMemcpyHostToDevice));

            // Free f32 centroids from GPU — no longer needed in PQ mode
            CUDA_CHECK(cudaFree(d_centroids_f32));
            d_centroids_f32 = nullptr;

            assignments = batch_assign_with_cagra_anns<DataT, uint8_t>(
                X_full.data(), N, D, n_centroids,
                centroid_global_indices,
                d_centroids_u8, u8_bytes,
                d_graph, graph_bytes,
                K, quantizer, Mbatch_bytes, search_max_iters);

            CUDA_CHECK(cudaFree(d_centroids_u8));
        } else {
            // Non-PQ mode: use float32 centroids directly
            assignments = batch_assign_with_cagra_anns<DataT, float>(
                X_full.data(), N, D, n_centroids,
                centroid_global_indices,
                d_centroids_f32, centroid_gpu_bytes,
                d_graph, graph_bytes,
                K, quantizer, Mbatch_bytes, search_max_iters);

            CUDA_CHECK(cudaFree(d_centroids_f32));
        }

        // Download centroid KNN graph to CPU before freeing (needed for Step 6,
        // and for Step 7's bucket processing order when --reorder is set).
        if (neighbors_m > 0 || do_reorder) {
            centroid_knn_graph_host.resize(static_cast<size_t>(n_centroids) * K);
            CUDA_CHECK(cudaMemcpy(centroid_knn_graph_host.data(), d_graph,
                                  graph_bytes, cudaMemcpyDeviceToHost));
        }

        // Free KNN graph from GPU
        CUDA_CHECK(cudaFree(d_graph));
        cudaDeviceSynchronize();
        double iter_step4 = std::chrono::duration<double>(Clock::now() - t4).count();
        elapsed_step4 += iter_step4;
        std::cout << "  Step 4 done [" << std::fixed << std::setprecision(3) << iter_step4 << "s]\n";

        // ================================================================
        // Step 5: Write bucket assignments to disk (last iteration only,
        //         多 iter 时仅最后一次的 bucket 结构落盘)
        // ================================================================
        if (iter == iterations - 1) {
            std::cout << "=== Step 5: Writing buckets to disk ===\n";
            auto t5 = Clock::now();
            write_buckets_to_disk(output_dir, assignments, N, n_centroids);

            // Also save centroid global indices for later reference
            std::string centroid_idx_path = output_dir + "/centroid_global_indices.bin";
            {
                std::ofstream out(centroid_idx_path, std::ios::binary);
                int64_t nc = n_centroids;
                out.write(reinterpret_cast<const char*>(&nc), sizeof(int64_t));
                out.write(reinterpret_cast<const char*>(centroid_global_indices.data()),
                          n_centroids * sizeof(int64_t));
            }

            // Save the centroid KNN graph too, so reorder.cpp (the standalone,
            // post-hoc equivalent of Step 7) can recompute the same DiskJoin-
            // style bucket processing order without rebuilding it from scratch.
            if (!centroid_knn_graph_host.empty()) {
                std::string p = output_dir + "/centroid_knn.bin";
                std::ofstream out(p, std::ios::binary);
                int64_t nc = n_centroids;
                int32_t Kw = static_cast<int32_t>(K);
                out.write(reinterpret_cast<const char*>(&nc), sizeof(int64_t));
                out.write(reinterpret_cast<const char*>(&Kw), sizeof(int32_t));
                out.write(reinterpret_cast<const char*>(centroid_knn_graph_host.data()),
                          static_cast<std::streamsize>(centroid_knn_graph_host.size() * sizeof(uint32_t)));
                std::cout << "  Wrote " << p << " (K=" << K << ")\n";
            }

            elapsed_step5 = std::chrono::duration<double>(Clock::now() - t5).count();
            std::cout << "  Step 5 done [" << std::fixed << std::setprecision(3) << elapsed_step5 << "s]\n";
        }

        // ================================================================
        // Step 6: Per-vector KNN via Tensor Core bucket matmul (optional)
        // ================================================================
        if (neighbors_m > 0) {
            std::cout << "=== Step 6: Building per-vector KNN (M=" << neighbors_m << ") ===\n";
            auto t6 = Clock::now();

            // 解析 nprobe: 若用户没指定 (=0)，回退到图度数 K
            uint32_t effective_nprobe = (nprobe > 0) ? nprobe : K;

            // CAGRA-pruned graph 不是 sorted KNN, 必须用 greedy graph search
            // 在导航图上为每个 centroid 找真实的 top-nprobe 近邻 (含 nprobe == K)
            std::vector<uint32_t> centroid_topK = expand_centroid_neighbors_gpu(
                centroids_host_for_step6,
                centroid_knn_graph_host,
                n_centroids, D, K, effective_nprobe,
                search_max_iters);
            const uint32_t* graph_ptr = centroid_topK.data();
            uint32_t graph_K = effective_nprobe;

            // 结果直接 merge 进 running_knn_file (磁盘上)，不再返回整块数组；
            // 不管 iterations 是 1 还是多轮，都是同一条代码路径 —— 第 0 轮
            // 跟全 sentinel 的文件合并，等价于直接写入,不需要为它单独分支。
            build_vector_knn_with_tensorcore(
                X_full.data(), N, D,
                assignments,
                centroid_global_indices,
                graph_ptr,
                n_centroids, graph_K, neighbors_m,
                *running_knn_file,
                output_dir);

            cudaDeviceSynchronize();
            double iter_step6 = std::chrono::duration<double>(Clock::now() - t6).count();
            elapsed_step6 += iter_step6;
            std::cout << "  Step 6 done [" << std::fixed << std::setprecision(3) << iter_step6 << "s]\n";
        }

        }   // end of per-iteration loop

        // vector_knn.bin / vector_dists.bin 已经在每一轮 Step 6 里被增量写完了
        // (running_knn_file)，这里不需要再整块写一次——只要把文件关掉 (flush)。
        if (neighbors_m > 0) {
            auto t_write = Clock::now();
            running_knn_file->neighbors_f.close();
            running_knn_file->dists_f.close();

            // 分块把 vector_knn.bin 转成 neighbors.npy，给 Python 端评测用
            // (纯顺序拷贝+类型转换，不需要整份常驻内存)
            convert_vector_knn_to_npy(
                output_dir + "/vector_knn.bin", output_dir + "/neighbors.npy",
                N, neighbors_m, config.cpu_limit_bytes / 4);
            std::cout << "[WriteVectorKNN] Saved neighbors.npy\n";
            elapsed_write_knn = std::chrono::duration<double>(Clock::now() - t_write).count();
        }

        // ================================================================
        // Step 7: Bucket-aligned ID reorder (optional, --reorder)
        //
        // 已知限制: assignments/centroid_global_indices (以及 bucket_index.bin/
        // bucket_data.bin) 都只保留"最后一轮" iteration 的分桶结果 (见上面
        // "这些值由最后一次 iteration 决定" 的注释)。当 iterations > 1 时，
        // 最终合并出的 vector_knn 里的边可能来自任意一轮的分桶，并不都落在
        // "最后一轮"划出的桶边界内——所以这里按最后一轮的桶结构重排，只是让
        // 那一轮产生的边对齐，其余轮的边不保证对齐，reorder 的"bucket-aligned"
        // 效果在多轮场景下是打了折扣的近似，不是严格保证。暂时按这个近似做，
        // 没有为多轮场景单独设计一套对齐方案。
        // ================================================================
        if (do_reorder) {
            std::cout << "=== Step 7: Bucket-aligned reorder ===\n";
            auto t7 = Clock::now();

            // Order buckets so ranges that end up adjacent in the new ID space
            // are also adjacent in feature space (DiskJoin's task ordering,
            // Algorithm 2 - see bucket_order.hpp), using the centroid KNN
            // graph already built in Step 3 as the bucket dependency graph.
            int32_t order_window = (order_window_arg > 0)
                ? order_window_arg
                : std::max<int32_t>(4 * static_cast<int32_t>(K), 16);
            auto bucket_process_order = bucket_order::compute_bucket_processing_order(
                bucket_order::adjacency_from_flat_graph(
                    centroid_knn_graph_host.data(), n_centroids, static_cast<int32_t>(K)),
                order_window);
            std::cout << "  order_window=" << order_window << "\n";

            auto reorder_info = compute_bucket_reorder(assignments, N, n_centroids,
                                                        bucket_process_order);
            std::cout << "  N=" << N
                      << " total_in_buckets=" << reorder_info.total_in_buckets
                      << " (unassigned=" << (N - reorder_info.total_in_buckets) << ")\n";

            // write_reordered_outputs 需要整份 vector_knn 在内存里做按 id 重排；
            // 现在它活在磁盘上，这里读回来（顺序读，一次性，只有 --reorder 时
            // 才会触发）。
            std::vector<int32_t> vector_knn;
            if (neighbors_m > 0) {
                std::ifstream in(output_dir + "/vector_knn.bin", std::ios::binary);
                if (!in.is_open())
                    throw std::runtime_error("Cannot open vector_knn.bin for reorder");
                in.seekg(static_cast<std::streamoff>(RunningKnnFile::header_bytes()));
                vector_knn.resize(static_cast<size_t>(N) * neighbors_m);
                in.read(reinterpret_cast<char*>(vector_knn.data()),
                       static_cast<std::streamsize>(vector_knn.size() * sizeof(int32_t)));
                if (!in.good())
                    throw std::runtime_error("Failed reading vector_knn.bin for reorder");
            }

            write_reordered_outputs(output_dir, ext, X_full, N, D,
                                    vector_knn, neighbors_m,
                                    reorder_info, n_centroids);

            elapsed_step7 = std::chrono::duration<double>(Clock::now() - t7).count();
            std::cout << "  Step 7 done [" << std::fixed << std::setprecision(3)
                      << elapsed_step7 << "s]\n";
        }

        // X_full no longer needed
        { auto _drop = std::move(X_full); }

        double elapsed_total = std::chrono::duration<double>(Clock::now() - t_total_start).count();

        std::cout << "\n=== All done! ===\n";
        std::cout << "\n=== Timing Summary"
                  << (iterations > 1 ? (" (Step 2-6 sums over T=" + std::to_string(iterations) + " iterations)") : "")
                  << " ===\n";
        std::cout << "  Step 1   (Load sampled data):  " << std::fixed << std::setprecision(3) << elapsed_step1 << "s\n";
        std::cout << "  Step 2   (Select centroids):    " << elapsed_step2 << "s\n";
        std::cout << "  Step 3   (Centroid KNN graph):  " << elapsed_step3 << "s\n";
        std::cout << "  Step 3.5 (Load full dataset):   " << elapsed_step3p5 << "s\n";
        std::cout << "  Step 4   (Batch assignment):    " << elapsed_step4 << "s\n";
        std::cout << "  Step 5   (Write buckets):       " << elapsed_step5 << "s\n";
        if (neighbors_m > 0) {
            std::cout << "  Step 6   (Per-vector KNN, merge included):      " << elapsed_step6 << "s\n";
            std::cout << "  Write    (close + .npy convert):    " << elapsed_write_knn << "s\n";
        }
        if (do_reorder)
            std::cout << "  Step 7   (Bucket reorder):      " << elapsed_step7 << "s\n";
        std::cout << "  --------------------------------\n";
        std::cout << "  Total:                        " << elapsed_total << "s\n";
        std::cout << "  Output: " << output_dir << "/\n";
        std::cout << "    bucket_index.bin  — offset/count table for each centroid\n";
        std::cout << "    bucket_data.bin   — packed int32 point IDs per bucket\n";
        std::cout << "    centroid_global_indices.bin — centroid-to-original-data mapping\n";
        if (!centroid_knn_graph_host.empty()) {
            std::cout << "    centroid_knn.bin  — centroid KNN graph (K=" << K
                      << "), lets reorder.cpp reproduce Step 7's bucket order\n";
        }
        if (neighbors_m > 0) {
            std::cout << "    vector_knn.bin    — per-vector " << neighbors_m << "-NN index\n";
            std::cout << "    neighbors.npy     — same as above in NumPy format (for search.py)\n";
        }
        if (do_reorder) {
            std::cout << "    data_reordered" << ext
                      << "       — bucket-aligned dataset (preserves input format)\n";
            if (neighbors_m > 0)
                std::cout << "    vector_knn_reordered.bin  — KNN graph in new ID space\n";
            std::cout << "    bucket_offsets.bin        — bucket boundaries (new ID space)\n";
            std::cout << "    perm.bin / inverse_perm.bin — ID translation tables\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}

// ============== Real main: CLI parse + DataT dispatch ==============
//
// 根据输入文件扩展名选择合适的 DataT，调用对应的 run_pipeline_impl<DataT> 实例。
// 支持的格式 → DataT:
//   .fbin/.bin → float
//   .u8bin     → uint8_t
//   .i8bin     → int8_t
//   .ibin      → int32_t
//   .ubin      → uint32_t
int main(int argc, char** argv) {
    try {
        // ---- CLI parsing ----
        po::options_description desc("Bucket Builder Options");
        desc.add_options()
            ("help,h",        "Show help")
            ("input,i",       po::value<std::string>()->required(),
                "Input data file (.fbin/.bin/.u8bin/.i8bin/.ibin/.ubin)")
            ("output,o",      po::value<std::string>()->required(),
                "Output directory for bucket files")
            ("cpu-limit",     po::value<size_t>(),     "CPU memory limit (bytes)")
            ("gpu-limit",     po::value<size_t>(),     "GPU memory limit (bytes, 0=auto)")
            ("sample-rate",   po::value<float>(),      "Sampling ratio (default 0.1)")
            ("centroid-ratio", po::value<float>(),     "Centroid ratio (default 0.01)")
            ("use-pq",        po::value<bool>(),       "Force PQ quantization")
            ("pq-bits-start", po::value<uint32_t>(),   "PQ starting bits (default 8)")
            ("pq-bits-min",   po::value<uint32_t>(),   "PQ minimum bits (default 4)")
            ("seed",          po::value<uint32_t>(),   "Random seed (default 42)")
            ("knn-k",         po::value<uint32_t>(),   "KNN graph degree (default 32)")
            ("nprobe",        po::value<uint32_t>()->default_value(0),
                "Per-bucket neighbor count for Step 6 (FAISS-style nprobe; 0=use knn-k; max 256)")
            ("search-iters",  po::value<int>()->default_value(64),
                "Graph search max iterations")
            ("neighbors-m",   po::value<int>()->default_value(0),
                "Per-vector KNN neighbor count M (0=skip)")
            ("iterations,t",  po::value<int>()->default_value(1),
                "Run Step 2-6 multiple times with seed+iter, dedupe-merge per-vector KNN. "
                "Bucket files (Step 5) reflect the last iteration only. Default 1.")
            ("reorder",       po::bool_switch()->default_value(false),
                "Also output bucket-aligned reordered files (data_reordered.<ext>, "
                "vector_knn_reordered.bin, bucket_offsets.bin, perm.bin, inverse_perm.bin) "
                "for optimize_chunked --method C/D")
            ("order-window",  po::value<int32_t>()->default_value(0),
                "Sliding-window size for the bucket processing order used by --reorder "
                "(DiskJoin-style task ordering over the centroid KNN graph; 0 = auto: 4*knn-k)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);

        std::string input_path  = vm["input"].as<std::string>();
        std::string output_root = vm["output"].as<std::string>();
        int search_max_iters    = vm["search-iters"].as<int>();
        int neighbors_m         = vm["neighbors-m"].as<int>();
        uint32_t nprobe         = vm["nprobe"].as<uint32_t>();
        bool do_reorder         = vm["reorder"].as<bool>();
        int iterations          = vm["iterations"].as<int>();
        int32_t order_window_arg = vm["order-window"].as<int32_t>();

        constexpr uint32_t MAX_NPROBE = 256;
        if (nprobe > MAX_NPROBE) {
            throw std::runtime_error(
                "nprobe (" + std::to_string(nprobe) + ") exceeds MAX_NPROBE=" +
                std::to_string(MAX_NPROBE) +
                ". Use --nprobe <= " + std::to_string(MAX_NPROBE) + ".");
        }

        if (iterations < 1) {
            throw std::runtime_error(
                "iterations (" + std::to_string(iterations) + ") must be >= 1");
        }
        if (iterations > 1 && neighbors_m <= 0) {
            std::cerr << "[Warn] iterations=" << iterations
                      << " but neighbors-m=0 (Step 6 skipped) → no merge will happen, "
                         "reverting to iterations=1\n";
            iterations = 1;
        }

        LoadConfig config = parse_load_config(vm);

        // ---- 据 ext 分派到对应 DataT 的 run_pipeline_impl ----
        std::string ext = std::filesystem::path(input_path).extension().string();
        std::cout << "[main] Dispatching DataT for ext=" << ext << "\n";

        if (ext == ".fbin" || ext == ".bin") {
            return run_pipeline_impl<float>(input_path, output_root, search_max_iters,
                                            neighbors_m, nprobe, do_reorder, iterations,
                                            config, ext, order_window_arg);
        } else if (ext == ".u8bin") {
            return run_pipeline_impl<uint8_t>(input_path, output_root, search_max_iters,
                                              neighbors_m, nprobe, do_reorder, iterations,
                                              config, ext, order_window_arg);
        } else if (ext == ".i8bin") {
            return run_pipeline_impl<int8_t>(input_path, output_root, search_max_iters,
                                             neighbors_m, nprobe, do_reorder, iterations,
                                             config, ext, order_window_arg);
        } else if (ext == ".ibin") {
            return run_pipeline_impl<int32_t>(input_path, output_root, search_max_iters,
                                              neighbors_m, nprobe, do_reorder, iterations,
                                              config, ext, order_window_arg);
        } else if (ext == ".ubin") {
            return run_pipeline_impl<uint32_t>(input_path, output_root, search_max_iters,
                                               neighbors_m, nprobe, do_reorder, iterations,
                                               config, ext, order_window_arg);
        } else {
            throw std::runtime_error("Unsupported file extension: " + ext +
                ". Supported: .fbin/.bin (float), .u8bin (uint8), .i8bin (int8), "
                ".ibin (int32), .ubin (uint32).");
        }
