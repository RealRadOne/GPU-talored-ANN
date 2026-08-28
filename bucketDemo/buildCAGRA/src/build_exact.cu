// CAGRA build via the EXACT same path as raft-ann-bench's RaftCagra wrapper.
// Used to confirm whether ann-bench's ~12s on V100 is reproducible from a
// minimal driver, separate from build_timing.cu (which times each NND/sort/
// optimize stage individually but goes via nn_descent::detail::build).
//
// Differences from build_timing.cu:
//   - Calls raft::neighbors::cagra::detail::build (NND + sort + optimize fused)
//   - Uses uint8_t end-to-end (no float conversion)
//   - Same RMM pool + managed large-workspace + stream pool setup as
//     ann-bench's `shared_raft_resources` / `configured_raft_resources`

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <raft/core/copy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/cagra/cagra_build.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "utils.hpp"

int main(int argc, char** argv)
{
  const std::string data_path =
    (argc > 1) ? argv[1]
               : "/home/lanlu/microBenchmark/dataset/sift1M/base.1M.u8bin";
  const uint32_t graph_degree =
    (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 32;
  const uint32_t intermediate_graph_degree =
    (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 64;
  const size_t nn_descent_niter =
    (argc > 4) ? static_cast<size_t>(std::stoul(argv[4])) : 20;

  const std::string folder = std::filesystem::path(data_path).parent_path().string();
  const std::string graph_save_path = folder + "/neighbors_nnCAGRA_exact"
    + std::to_string(graph_degree) + "-"
    + std::to_string(intermediate_graph_degree) + ".npy";

  using Clock = std::chrono::high_resolution_clock;
  auto t_total_start = Clock::now();

  // ---- Step 1: load dataset (uint8) ----
  std::cout << "=== Step 1: Loading " << data_path << " ===\n";
  auto t1 = Clock::now();
  int64_t N = 0, D = 0;
  std::vector<uint8_t> h_data = read_u8bin_raw(data_path, N, D);
  auto dataset = raft::make_host_matrix_view<const uint8_t, int64_t>(h_data.data(), N, D);
  double elapsed_step1 = std::chrono::duration<double>(Clock::now() - t1).count();
  std::cout << "  Loaded N=" << N << " D=" << D << " (" << h_data.size() / 1e9 << " GB, uint8)\n";
  std::cout << "  Step 1 done [" << std::fixed << std::setprecision(3) << elapsed_step1 << "s]\n";

  // ---- Resource setup mirroring ann-bench ----
  using pool_mr_t = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  auto* orig_mr = rmm::mr::get_current_device_resource();
  pool_mr_t pool_mr(orig_mr, 1024ull * 1024ull * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);
  auto large_mr = std::make_shared<rmm::mr::managed_memory_resource>();

  raft::resources res;
  raft::resource::set_large_workspace_resource(res, large_mr);
  raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(8));

  // ---- CAGRA build params (match ann-bench yaml) ----
  raft::neighbors::cagra::index_params cagra_params;
  cagra_params.graph_degree              = graph_degree;
  cagra_params.intermediate_graph_degree = intermediate_graph_degree;
  cagra_params.build_algo                = raft::neighbors::cagra::graph_build_algo::NN_DESCENT;
  cagra_params.nn_descent_niter          = nn_descent_niter;
  cagra_params.metric                    = raft::distance::DistanceType::L2Expanded;

  // ---- Step 2: cagra::detail::build (NND + sort + optimize, fused) ----
  std::cout << "=== Step 2: cagra::detail::build (NND + sort + optimize) "
            << "graph_degree=" << graph_degree
            << ", intermediate=" << intermediate_graph_degree
            << ", nnd_niter=" << nn_descent_niter << " ===\n";
  auto t2 = Clock::now();

  auto cagra_idx = raft::neighbors::cagra::detail::build<uint8_t, uint32_t>(
    res, cagra_params, dataset);

  raft::resource::sync_stream(res);
  cudaDeviceSynchronize();
  double elapsed_step2 = std::chrono::duration<double>(Clock::now() - t2).count();
  std::cout << "  Step 2 done [" << std::fixed << std::setprecision(3) << elapsed_step2 << "s]\n";

  // ---- Step 3: copy graph to host + write npy ----
  std::cout << "=== Step 3: Copy graph + save to " << graph_save_path << " ===\n";
  auto t3 = Clock::now();
  auto h_graph = raft::make_host_matrix<uint32_t, int64_t>(N, graph_degree);
  raft::copy(h_graph.data_handle(),
             cagra_idx.graph().data_handle(),
             static_cast<size_t>(N) * graph_degree,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  write_npy_2d<uint32_t>(graph_save_path, h_graph.data_handle(), N, graph_degree);
  double elapsed_step3 = std::chrono::duration<double>(Clock::now() - t3).count();
  std::cout << "  Step 3 done [" << std::fixed << std::setprecision(3) << elapsed_step3 << "s]\n";

  double elapsed_total = std::chrono::duration<double>(Clock::now() - t_total_start).count();

  std::cout << "\n=== Timing Summary (exact ann-bench path) ===\n";
  std::cout << "  Step 1 (Load):                        " << elapsed_step1 << "s\n";
  std::cout << "  Step 2 (cagra::build = NND+sort+opt): " << elapsed_step2 << "s\n";
  std::cout << "  Step 3 (Copy + Write npy):            " << elapsed_step3 << "s\n";
  std::cout << "  --------------------------------\n";
  std::cout << "  Total:                                " << elapsed_total << "s\n";
  std::cout << "  Output: " << graph_save_path << "\n";

  rmm::mr::set_current_device_resource(orig_mr);
  return 0;
}
