#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <filesystem>

// ====== 这些 include 需要按你的项目实际路径调整 ======
#include <raft/core/resources.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/neighbors/detail/nn_descent.cuh>
#include <raft/neighbors/detail/cagra/graph_core.cuh>
#include "utils.hpp"

// using namespace raft::neighbors::experimental::nn_descent;

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv)
{
  const std::string data_path =
    (argc > 1) ? argv[1]
               : "/home/lanlu/microBenchmark/dataset/sift1M/base.1M.u8bin";
  const uint32_t graph_degree              = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : 32;
  const uint32_t intermediate_graph_degree = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 64;
  const size_t   max_iterations            = (argc > 4) ? static_cast<size_t>(std::stoul(argv[4]))   : 20;
  const double   termination_threshold     = (argc > 5) ? std::stod(argv[5])                         : 0.0001;
  const bool     return_distances          = (argc > 6) ? static_cast<bool>(std::stoi(argv[6]))      : false;

  // 从 data_path 提取所在文件夹
  const std::string folder = std::filesystem::path(data_path).parent_path().string();
  const std::string graph_save_path = folder + "/neighbors_nnCAGRA_prune"
    + std::to_string(graph_degree) + "-" + std::to_string(intermediate_graph_degree) + ".npy";
  const std::string dist_save_path  = folder + "/dist_nnCAGRA_prune"
    + std::to_string(graph_degree) + "-" + std::to_string(intermediate_graph_degree) + ".npy";

  // -------- 读取 .u8bin 数据集 --------
  int64_t N, D;
  std::vector<float> h_data = read_u8bin(data_path, N, D);
  std::cout << "Loaded dataset: N=" << N << " D=" << D << "\n";

  // GNND 参数
  raft::neighbors::experimental::nn_descent::index_params params;
  params.graph_degree              = intermediate_graph_degree;
  params.intermediate_graph_degree = (size_t)(1.5 * intermediate_graph_degree);
  params.max_iterations            = max_iterations;
  params.termination_threshold     = termination_threshold;
  params.return_distances          = return_distances;

  // 用 host_matrix_view 包一层（row_major）
  auto dataset = raft::make_host_matrix_view<const float, int64_t>(h_data.data(), N, D);

  // -------- RAFT 资源 --------
  raft::resources res;

  // -------- build index --------
  auto idx = raft::neighbors::experimental::nn_descent::detail::build<float, uint32_t>(res, params, dataset);

  // （可选）按真实 L2 距离重排序邻居——CAGRA 也做了这步
  raft::neighbors::cagra::detail::graph::sort_knn_graph(res, dataset, idx.graph());
  // 分配最终图（[N, graph_degree] = [N, 32]）
  auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(N, graph_degree);
  // CAGRA prune：2-hop detour counting + reverse edge replacement
  raft::neighbors::cagra::detail::graph::optimize(res, idx.graph(), cagra_graph.view());

  // save_graph_as_npy<uint32_t>(res, cagra_graph, graph_save_path);
  write_npy_2d<uint32_t>(graph_save_path,
                       cagra_graph.data_handle(),
                       cagra_graph.extent(0),
                       cagra_graph.extent(1));
  // save_distances_as_npy<uint32_t, float>(res, idx, dist_save_path);

  // 取出图并打印前几个点的邻居
  // auto graph_view = idx.graph(); // shape: [N, graph_degree], already a host_matrix_view
  // auto graph_view = cagra_graph; // shape: [N, graph_degree], already a host_matrix_view
  // std::cout << "Graph built. First 5 nodes' neighbors:\n";
  // for (int i = 0; i < 5; i++) {
  //   std::cout << "i=" << i << ": ";
  //   for (int j = 0; j < (int)params.graph_degree; j++) {
  //     std::cout << graph_view(i, j) << (j + 1 == (int)params.graph_degree ? "" : ",");
  //   }
  //   std::cout << "\n";
  // }

  return 0;
}
