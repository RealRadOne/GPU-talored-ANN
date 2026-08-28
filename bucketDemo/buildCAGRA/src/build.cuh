template <typename Index_t>
GnndGraph<Index_t>::GnndGraph(const size_t nrow,
                              const size_t node_degree,
                              const size_t internal_node_degree,
                              const size_t num_samples)
  : nrow(nrow),
    node_degree(node_degree),
    num_samples(num_samples),
    bloom_filter(nrow, internal_node_degree / segment_size, 3),
    h_dists{raft::make_host_matrix<DistData_t, size_t, raft::row_major>(nrow, node_degree)},
    h_graph_new(nrow * num_samples),
    h_list_sizes_new(nrow),
    h_graph_old(nrow * num_samples),
    h_list_sizes_old{nrow}
{
  // node_degree must be a multiple of segment_size;
  assert(node_degree % segment_size == 0);
  assert(internal_node_degree % segment_size == 0);

  num_segments = node_degree / segment_size;
  // To save the CPU memory, graph should be allocated by external function
  h_graph = nullptr;
}

// This is the only operation on the CPU that cannot be overlapped.
// So it should be as fast as possible.
template <typename Index_t>
void GnndGraph<Index_t>::sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    auto list_new         = h_graph_new.data() + i * num_samples;
    h_list_sizes_new[i].x = 0;
    h_list_sizes_new[i].y = 0;

    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighb_id >= nrow) break;
      if (bloom_filter.check(i, new_neighb_id)) { continue; }
      bloom_filter.add(i, new_neighb_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new[i].x++] = new_neighb_id;
      if (h_list_sizes_new[i].x == num_samples) break;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::init_random_graph()
{
  for (size_t seg_idx = 0; seg_idx < static_cast<size_t>(num_segments); seg_idx++) {
    // random sequence (range: 0~nrow)
    // segment_x stores neighbors which id % num_segments == x
    std::vector<Index_t> rand_seq(nrow / num_segments);
    std::iota(rand_seq.begin(), rand_seq.end(), 0);
    auto gen = std::default_random_engine{seg_idx};
    std::shuffle(rand_seq.begin(), rand_seq.end(), gen);

#pragma omp parallel for
    for (size_t i = 0; i < nrow; i++) {
      size_t base_idx      = i * node_degree + seg_idx * segment_size;
      auto h_neighbor_list = h_graph + base_idx;
      auto h_dist_list     = h_dists.data_handle() + base_idx;
      for (size_t j = 0; j < static_cast<size_t>(segment_size); j++) {
        size_t idx = base_idx + j;
        Index_t id = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
        if ((size_t)id == i) {
          id = rand_seq[(idx + segment_size) % rand_seq.size()] * num_segments + seg_idx;
        }
        h_neighbor_list[j].id_with_flag() = id;
        h_dist_list[j]                    = std::numeric_limits<DistData_t>::max();
      }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sample_graph(bool sample_new)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    h_list_sizes_old[i].x = 0;
    h_list_sizes_old[i].y = 0;
    h_list_sizes_new[i].x = 0;
    h_list_sizes_new[i].y = 0;

    auto list     = h_graph + i * node_degree;
    auto list_old = h_graph_old.data() + i * num_samples;
    auto list_new = h_graph_new.data() + i * num_samples;
    for (int j = 0; j < segment_size; j++) {
      for (int k = 0; k < num_segments; k++) {
        auto neighbor = list[k * segment_size + j];
        if ((size_t)neighbor.id() >= nrow) continue;
        if (!neighbor.is_new()) {
          if (h_list_sizes_old[i].x < num_samples) {
            list_old[h_list_sizes_old[i].x++] = neighbor.id();
          }
        } else if (sample_new) {
          if (h_list_sizes_new[i].x < num_samples) {
            list[k * segment_size + j].mark_old();
            list_new[h_list_sizes_new[i].x++] = neighbor.id();
          }
        }
        if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
      }
      if (h_list_sizes_old[i].x == num_samples && h_list_sizes_new[i].x == num_samples) { break; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::update_graph(const InternalID_t<Index_t>* new_neighbors,
                                      const DistData_t* new_dists,
                                      const size_t width,
                                      std::atomic<int64_t>& update_counter)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j];
      auto new_dist      = new_dists[i * width + j];
      if (new_dist == std::numeric_limits<DistData_t>::max()) break;
      if ((size_t)new_neighb_id.id() == i) continue;
      int seg_idx    = new_neighb_id.id() % num_segments;
      auto list      = h_graph + i * node_degree + seg_idx * segment_size;
      auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * segment_size;
      int insert_pos =
        insert_to_ordered_list(list, dist_list, segment_size, new_neighb_id, new_dist);
      if (i % counter_interval == 0 && insert_pos != segment_size) { update_counter++; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sort_lists()
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    std::vector<std::pair<DistData_t, Index_t>> new_list;
    for (size_t j = 0; j < node_degree; j++) {
      new_list.emplace_back(h_dists.data_handle()[i * node_degree + j],
                            h_graph[i * node_degree + j].id());
    }
    std::sort(new_list.begin(), new_list.end());
    for (size_t j = 0; j < node_degree; j++) {
      h_graph[i * node_degree + j].id_with_flag() = new_list[j].second;
      h_dists.data_handle()[i * node_degree + j]  = new_list[j].first;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::clear()
{
  bloom_filter.clear();
}

template <typename Index_t>
GnndGraph<Index_t>::~GnndGraph()
{
  assert(h_graph == nullptr);
}

template <typename Data_t, typename Index_t, typename epilogue_op>
GNND<Data_t, Index_t, epilogue_op>::GNND(raft::resources const& res,
                                         const BuildConfig& build_config)
  : res(res),
    build_config_(build_config),
    graph_(build_config.max_dataset_size,
           align32::roundUp(build_config.node_degree),
           align32::roundUp(build_config.internal_node_degree ? build_config.internal_node_degree
                                                              : build_config.node_degree),
           NUM_SAMPLES),
    nrow_(build_config.max_dataset_size),
    ndim_(build_config.dataset_dim),
    d_data_{raft::make_device_matrix<__half, size_t, raft::row_major>(
      res, nrow_, build_config.dataset_dim)},
    l2_norms_{raft::make_device_vector<DistData_t, size_t>(res, nrow_)},
    graph_buffer_{
      raft::make_device_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    dists_buffer_{
      raft::make_device_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    graph_host_buffer_(nrow_ * DEGREE_ON_DEVICE),
    dists_host_buffer_(nrow_ * DEGREE_ON_DEVICE),
    d_locks_{raft::make_device_vector<int, size_t>(res, nrow_)},
    h_rev_graph_new_(nrow_ * NUM_SAMPLES),
    h_graph_old_(nrow_ * NUM_SAMPLES),
    h_rev_graph_old_(nrow_ * NUM_SAMPLES),
    d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, nrow_)},
    d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, nrow_)}
{
  static_assert(NUM_SAMPLES <= 32);
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);
};

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::reset(raft::resources const& res)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);
}

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::add_reverse_edges(Index_t* graph_ptr,
                                                           Index_t* h_rev_graph_ptr,
                                                           Index_t* d_rev_graph_ptr,
                                                           int2* list_sizes,
                                                           cudaStream_t stream)
{
  add_rev_edges_kernel<<<nrow_, raft::warp_size(), 0, stream>>>(
    graph_ptr, d_rev_graph_ptr, NUM_SAMPLES, list_sizes);
  raft::copy(
    h_rev_graph_ptr, d_rev_graph_ptr, nrow_ * NUM_SAMPLES, raft::resource::get_cuda_stream(res));
}

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::local_join(cudaStream_t stream,
                                                    epilogue_op distance_epilogue)
{
  thrust::fill(thrust::device.on(stream),
               dists_buffer_.data_handle(),
               dists_buffer_.data_handle() + dists_buffer_.size(),
               std::numeric_limits<float>::max());
  local_join_kernel<<<nrow_, BLOCK_SIZE, 0, stream>>>(
    thrust::raw_pointer_cast(graph_.h_graph_new.data()),
    thrust::raw_pointer_cast(h_rev_graph_new_.data()),
    d_list_sizes_new_.data_handle(),
    thrust::raw_pointer_cast(h_graph_old_.data()),
    thrust::raw_pointer_cast(h_rev_graph_old_.data()),
    d_list_sizes_old_.data_handle(),
    NUM_SAMPLES,
    d_data_.data_handle(),
    ndim_,
    graph_buffer_.data_handle(),
    dists_buffer_.data_handle(),
    DEGREE_ON_DEVICE,
    d_locks_.data_handle(),
    l2_norms_.data_handle(),
    distance_epilogue);
}

template <typename Data_t, typename Index_t, typename epilogue_op>
void GNND<Data_t, Index_t, epilogue_op>::build(Data_t* data,
                                               const Index_t nrow,
                                               Index_t* output_graph,
                                               bool return_distances,
                                               DistData_t* output_distances,
                                               epilogue_op distance_epilogue)
{
  using input_t = typename std::remove_const<Data_t>::type;

  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  nrow_               = nrow;
  graph_.nrow         = nrow;
  graph_.h_graph      = (InternalID_t<Index_t>*)output_graph;

  cudaPointerAttributes data_ptr_attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
  size_t batch_size = (data_ptr_attr.devicePointer == nullptr) ? 100000 : nrow_;

  raft::spatial::knn::detail::utils::batch_load_iterator vec_batches{
    data, static_cast<size_t>(nrow_), build_config_.dataset_dim, batch_size, stream};
  for (auto const& batch : vec_batches) {
    preprocess_data_kernel<<<
      batch.size(),
      raft::warp_size(),
      sizeof(Data_t) * ceildiv(build_config_.dataset_dim, static_cast<size_t>(raft::warp_size())) *
        raft::warp_size(),
      stream>>>(batch.data(),
                d_data_.data_handle(),
                build_config_.dataset_dim,
                l2_norms_.data_handle(),
                batch.offset());
  }

  thrust::fill(thrust::device.on(stream),
               (Index_t*)graph_buffer_.data_handle(),
               (Index_t*)graph_buffer_.data_handle() + graph_buffer_.size(),
               std::numeric_limits<Index_t>::max());

  graph_.clear();
  graph_.init_random_graph();
  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                          thrust::raw_pointer_cast(dists_host_buffer_.data()),
                          DEGREE_ON_DEVICE,
                          update_counter_);
      if (update_counter_ < build_config_.termination_threshold * nrow_ *
                              build_config_.dataset_dim / counter_interval) {
        update_counter_ = -1;
      }
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < build_config_.max_iterations; it++) {
    raft::copy(d_list_sizes_new_.data_handle(),
               thrust::raw_pointer_cast(graph_.h_list_sizes_new.data()),
               nrow_,
               raft::resource::get_cuda_stream(res));
    raft::copy(thrust::raw_pointer_cast(h_graph_old_.data()),
               thrust::raw_pointer_cast(graph_.h_graph_old.data()),
               nrow_ * NUM_SAMPLES,
               raft::resource::get_cuda_stream(res));
    raft::copy(d_list_sizes_old_.data_handle(),
               thrust::raw_pointer_cast(graph_.h_list_sizes_old.data()),
               nrow_,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    std::thread update_and_sample_thread(update_and_sample, it);

    RAFT_LOG_DEBUG("# GNND iteraton: %lu / %lu", it + 1, build_config_.max_iterations);

    // Reuse dists_buffer_ to save GPU memory. graph_buffer_ cannot be reused, because it
    // contains some information for local_join.
    static_assert(DEGREE_ON_DEVICE * sizeof(*(dists_buffer_.data_handle())) >=
                  NUM_SAMPLES * sizeof(*(graph_buffer_.data_handle())));
    add_reverse_edges(thrust::raw_pointer_cast(graph_.h_graph_new.data()),
                      thrust::raw_pointer_cast(h_rev_graph_new_.data()),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_new_.data_handle(),
                      stream);
    add_reverse_edges(thrust::raw_pointer_cast(h_graph_old_.data()),
                      thrust::raw_pointer_cast(h_rev_graph_old_.data()),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_old_.data_handle(),
                      stream);

    // Tensor operations from `mma.h` are guarded with archicteture
    // __CUDA_ARCH__ >= 700. Since RAFT supports compilation for ARCH 600,
    // we need to ensure that `local_join_kernel` (which uses tensor) operations
    // is not only not compiled, but also a runtime error is presented to the user
    auto kernel       = preprocess_data_kernel<input_t>;
    void* kernel_ptr  = reinterpret_cast<void*>(kernel);
    auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
    auto wmma_range =
      raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

    if (wmma_range.contains(runtime_arch)) {
      local_join(stream, distance_epilogue);
    } else {
      THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
    }

    update_and_sample_thread.join();

    if (update_counter_ == -1) { break; }
    raft::copy(thrust::raw_pointer_cast(graph_host_buffer_.data()),
               graph_buffer_.data_handle(),
               nrow_ * DEGREE_ON_DEVICE,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    raft::copy(thrust::raw_pointer_cast(dists_host_buffer_.data()),
               dists_buffer_.data_handle(),
               nrow_ * DEGREE_ON_DEVICE,
               raft::resource::get_cuda_stream(res));

    graph_.sample_graph_new(thrust::raw_pointer_cast(graph_host_buffer_.data()), DEGREE_ON_DEVICE);
  }

  graph_.update_graph(thrust::raw_pointer_cast(graph_host_buffer_.data()),
                      thrust::raw_pointer_cast(dists_host_buffer_.data()),
                      DEGREE_ON_DEVICE,
                      update_counter_);
  raft::resource::sync_stream(res);
  graph_.sort_lists();

  // Reuse graph_.h_dists as the buffer for shrink the lists in graph
  static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));

  if (return_distances) {
    auto graph_d_dists = raft::make_device_matrix<DistData_t, int64_t, raft::row_major>(
      res, nrow_, build_config_.node_degree);
    raft::copy(graph_d_dists.data_handle(),
               graph_.h_dists.data_handle(),
               nrow_ * build_config_.node_degree,
               raft::resource::get_cuda_stream(res));

    auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);

    raft::matrix::slice_coordinates coords{static_cast<int64_t>(0),
                                           static_cast<int64_t>(0),
                                           static_cast<int64_t>(nrow_),
                                           static_cast<int64_t>(build_config_.output_graph_degree)};
    raft::matrix::slice<DistData_t, int64_t, raft::row_major>(
      res, raft::make_const_mdspan(graph_d_dists.view()), output_dist_view, coords);
    raft::resource::sync_stream(res);
  }

  Index_t* graph_shrink_buffer = (Index_t*)graph_.h_dists.data_handle();

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      size_t idx = i * graph_.node_degree + j;
      int id     = graph_.h_graph[idx].id();
      if (id < static_cast<int>(nrow_)) {
        graph_shrink_buffer[i * build_config_.node_degree + j] = id;
      } else {
        graph_shrink_buffer[i * build_config_.node_degree + j] =
          raft::neighbors::cagra::detail::device::xorshift64(idx) % nrow_;
      }
    }
  }
  graph_.h_graph = nullptr;

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      output_graph[i * build_config_.node_degree + j] =
        graph_shrink_buffer[i * build_config_.node_degree + j];
    }
  }
}

template <typename T,
          typename IdxT        = uint32_t,
          typename epilogue_op = DistEpilogue<IdxT, T>,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
void build(raft::resources const& res,
           const index_params& params,
           mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
           index<IdxT>& idx,
           epilogue_op distance_epilogue = DistEpilogue<IdxT, T>())
{
  RAFT_EXPECTS(dataset.extent(0) < std::numeric_limits<int>::max() - 1,
               "The dataset size for GNND should be less than %d",
               std::numeric_limits<int>::max() - 1);
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = dataset.extent(0) - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // The elements in each knn-list are partitioned into different buckets, and we need more buckets
  // to mitigate bucket collisions. `intermediate_degree` is OK to larger than
  // extended_graph_degree.
  size_t extended_graph_degree =
    align32::roundUp(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
  size_t extended_intermediate_degree = align32::roundUp(
    static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

  auto int_graph = raft::make_host_matrix<int, int64_t, row_major>(
    dataset.extent(0), static_cast<int64_t>(extended_graph_degree));

  BuildConfig build_config{.max_dataset_size      = static_cast<size_t>(dataset.extent(0)),
                           .dataset_dim           = static_cast<size_t>(dataset.extent(1)),
                           .node_degree           = extended_graph_degree,
                           .internal_node_degree  = extended_intermediate_degree,
                           .max_iterations        = params.max_iterations,
                           .termination_threshold = params.termination_threshold,
                           .output_graph_degree   = params.graph_degree};

  GNND<const T, int, epilogue_op> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset.data_handle(),
              dataset.extent(0),
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle(),
              distance_epilogue);
  } else {
    RAFT_EXPECTS(!params.return_distances,
                 "Distance view not allocated. Using return_distances set to true requires "
                 "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      auto graph                  = idx.graph().data_handle();
      graph[i * graph_degree + j] = int_graph.data_handle()[i * extended_graph_degree + j];
    }
  }
}

template <typename T,
          typename IdxT        = uint32_t,
          typename epilogue_op = DistEpilogue<IdxT, T>,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
index<IdxT> build(raft::resources const& res,
                  const index_params& params,
                  mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
                  epilogue_op distance_epilogue = DistEpilogue<IdxT, T>())
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  index<IdxT> idx{
    res, dataset.extent(0), static_cast<int64_t>(graph_degree), params.return_distances};

  build(res, params, dataset, idx, distance_epilogue);

  return idx;
}
