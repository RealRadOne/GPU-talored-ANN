// CAGRA build pipeline with per-stage timing.
// Mirrors the timing pattern used in buildBucket/src/bucket.cu.
//
// Stages measured:
//   Step 1: Load .u8bin dataset from disk
//   Step 2: NN-Descent build (intermediate kNN graph, on GPU)
//   Step 3: sort_knn_graph (re-sort neighbors by true L2 distance)
//   Step 4: CAGRA optimize (2-hop detour pruning + reverse-edge replacement)
//   Step 5: Write final graph to .npy
//
// GPU stages (2/3/4) sync the RAFT stream before stopping their timer so that
// the wall-clock includes async kernel work, matching how bucket.cu calls
// cudaDeviceSynchronize() before reading its step timers.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/cagra/cagra_build.cuh>
#include <raft/neighbors/detail/cagra/graph_core.cuh>
#include <raft/neighbors/detail/nn_descent.cuh>
#include <raft/neighbors/nn_descent_types.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv)
{
  const std::string data_path =
    (argc > 1) ? argv[1]
               : "/home/lanlu/microBenchmark/dataset/sift1M/base.1M.u8bin";
  const uint32_t graph_degree =
    (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 32;
  const uint32_t intermediate_graph_degree =
    (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 64;
  const size_t max_iterations =
    (argc > 4) ? static_cast<size_t>(std::stoul(argv[4])) : 20;
  const double termination_threshold =
    (argc > 5) ? std::stod(argv[5]) : 0.0001;
  const bool return_distances =
    (argc > 6) ? static_cast<bool>(std::stoi(argv[6])) : false;

  const std::string folder = std::filesystem::path(data_path).parent_path().string();
  const std::string graph_save_path = folder + "/neighbors_nnCAGRA_prune"
    + std::to_string(graph_degree) + "-"
    + std::to_string(intermediate_graph_degree) + ".npy";

  using Clock = std::chrono::high_resolution_clock;
  auto t_total_start = Clock::now();
  double elapsed_step1 = 0, elapsed_step2 = 0, elapsed_step3 = 0;
  double elapsed_step4 = 0, elapsed_step5 = 0;

  // ================================================================
  // Step 1: Load u8bin dataset from disk
  // ================================================================
  std::cout << "=== Step 1: Loading dataset from " << data_path << " ===\n";
  auto t1 = Clock::now();

  int64_t N = 0, D = 0;
  std::vector<uint8_t> h_data = read_u8bin_raw(data_path, N, D);
  auto dataset = raft::make_host_matrix_view<const uint8_t, int64_t>(h_data.data(), N, D);

  elapsed_step1 = std::chrono::duration<double>(Clock::now() - t1).count();
  std::cout << "  Loaded N=" << N << " D=" << D
            << " (" << h_data.size() / 1e9 << " GB, uint8)\n";
  std::cout << "  Step 1 done [" << std::fixed << std::setprecision(3)
            << elapsed_step1 << "s]\n";

  // -------- RAFT resources / NN-Descent params --------
  // Match raft-ann-bench's runtime setup so NN-Descent's many internal
  // allocations don't fall back to plain cudaMalloc:
  //   1. 1 GB RMM device pool, set as the current device resource
  //   2. managed_memory_resource as raft's "large workspace" resource
  // Without these, repeated cudaMalloc inside NN-Descent dominates wall time.
  using pool_mr_t = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  auto* orig_mr = rmm::mr::get_current_device_resource();
  pool_mr_t pool_mr(orig_mr, 1024ull * 1024ull * 1024ull);  // 1 GB initial pool
  rmm::mr::set_current_device_resource(&pool_mr);

  auto large_mr = std::make_shared<rmm::mr::managed_memory_resource>();

  raft::resources res;
  raft::resource::set_large_workspace_resource(res, large_mr);
  raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(8));

  raft::neighbors::experimental::nn_descent::index_params params;
  params.graph_degree              = intermediate_graph_degree;
  params.intermediate_graph_degree = static_cast<size_t>(1.5 * intermediate_graph_degree);
  params.max_iterations            = max_iterations;
  params.termination_threshold     = termination_threshold;
  params.return_distances          = return_distances;

  // ================================================================
  // Step 2: NN-Descent build (GPU)
  // ================================================================
  std::cout << "=== Step 2: NN-Descent build "
            << "(intermediate_graph_degree=" << intermediate_graph_degree
            << ", max_iter=" << max_iterations << ") ===\n";
  auto t2 = Clock::now();

  auto idx = raft::neighbors::experimental::nn_descent::detail::build<uint8_t, uint32_t>(
    res, params, dataset);

  raft::resource::sync_stream(res);
  cudaDeviceSynchronize();
  elapsed_step2 = std::chrono::duration<double>(Clock::now() - t2).count();
  std::cout << "  Step 2 done [" << std::fixed << std::setprecision(3)
            << elapsed_step2 << "s]\n";

  // ================================================================
  // Step 3: sort_knn_graph — re-rank neighbors by true L2
  // ================================================================
  std::cout << "=== Step 3: sort_knn_graph ===\n";
  auto t3 = Clock::now();

  raft::neighbors::cagra::detail::graph::sort_knn_graph(res, dataset, idx.graph());

  raft::resource::sync_stream(res);
  cudaDeviceSynchronize();
  elapsed_step3 = std::chrono::duration<double>(Clock::now() - t3).count();
  std::cout << "  Step 3 done [" << std::fixed << std::setprecision(3)
            << elapsed_step3 << "s]\n";

  // ================================================================
  // Step 4: CAGRA optimize — 2-hop detour pruning + reverse edges
  // ================================================================
  std::cout << "=== Step 4: CAGRA optimize "
            << "(graph_degree=" << graph_degree << ") ===\n";
  auto t4 = Clock::now();

  auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(N, graph_degree);
  raft::neighbors::cagra::detail::graph::optimize(res, idx.graph(), cagra_graph.view());

  raft::resource::sync_stream(res);
  cudaDeviceSynchronize();
  elapsed_step4 = std::chrono::duration<double>(Clock::now() - t4).count();
  std::cout << "  Step 4 done [" << std::fixed << std::setprecision(3)
            << elapsed_step4 << "s]\n";

  // ================================================================
  // Step 5: Save final graph as .npy
  // ================================================================
  std::cout << "=== Step 5: Saving graph to " << graph_save_path << " ===\n";
  auto t5 = Clock::now();

  write_npy_2d<uint32_t>(graph_save_path,
                         cagra_graph.data_handle(),
                         cagra_graph.extent(0),
                         cagra_graph.extent(1));

  elapsed_step5 = std::chrono::duration<double>(Clock::now() - t5).count();
  std::cout << "  Step 5 done [" << std::fixed << std::setprecision(3)
            << elapsed_step5 << "s]\n";

  double elapsed_total = std::chrono::duration<double>(Clock::now() - t_total_start).count();

  std::cout << "\n=== All done! ===\n";
  std::cout << "\n=== Timing Summary ===\n";
  std::cout << "  Step 1 (Load data):          " << std::fixed
            << std::setprecision(3) << elapsed_step1 << "s\n";
  std::cout << "  Step 2 (NN-Descent build):    " << elapsed_step2 << "s\n";
  std::cout << "  Step 3 (sort_knn_graph):      " << elapsed_step3 << "s\n";
  std::cout << "  Step 4 (CAGRA optimize):      " << elapsed_step4 << "s\n";
  std::cout << "  Step 5 (Write npy):           " << elapsed_step5 << "s\n";
  std::cout << "  --------------------------------\n";
  std::cout << "  Total:                        " << elapsed_total << "s\n";
  std::cout << "  Output: " << graph_save_path << "\n";

  rmm::mr::set_current_device_resource(orig_mr);
  return 0;
}
