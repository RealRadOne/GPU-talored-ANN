#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <limits>
#include <iomanip>
#include <atomic>
#include <unordered_set>
#include <cstring>
#include <type_traits>
#include <omp.h>
#include <chrono>
#include <future>
#include <memory>
#include <fmt/ostream.h>

// Boost
#include <boost/program_options.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// RAFT
#include <raft/core/resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/distance/distance.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/neighbors/detail/nn_descent.cuh>
#include <raft/neighbors/detail/cagra/graph_core.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/random/rng.cuh>
#include <raft/matrix/select_k.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>

// Local headers
#include "../utils.hpp"
#include "../load.hpp"
#include "../bucket_build.cuh"
#include "../bucket_order.hpp"
#include "../utils/timing.hpp"
#include "../utils/spot_metrics.hpp"
#include "../utils/step_timer.hpp"

namespace po = boost::program_options;
using namespace bucket;

// ============== Phase 1: LoadConfig & Memory Management ==============

/**
 * Configuration for flexible data loading and GPU KMeans++ centroid generation
 * Based on user requirements and DiskANN design patterns
 */
struct LoadConfig {
    // Memory limits
    size_t cpu_limit_bytes;        // CPU memory limit, default 16GB
    size_t gpu_limit_bytes;        // GPU memory limit, default 0 (auto-detect via cudaMemGetInfo)

    // Sampling and Centroid parameters
    float sample_rate;             // Sampling ratio [0, 1], default 0.1 (10%)
    float centroid_ratio;          // Centroid ratio relative to sampled data, default 0.01 (1%)

    // PQ parameters (DiskANN Per-Subspace Quantization style)
    uint32_t pq_bits_start;        // PQ starting bits, default 8
    uint32_t pq_bits_min;          // PQ minimum bits, default 4 (ensures compression)
    uint32_t pq_dim;               // PQ dimension, default 0 (auto = D/4)
    float pq_train_fraction;       // PQ training data fraction, default 0.05 (5%)
    uint32_t pq_train_max_rows;    // PQ training data row limit, default 65536

    // Control
    bool use_pq;                   // Force PQ quantization (skip judgment)
    uint32_t seed;                 // Random seed

    // Safety
    float gpu_safety_margin;       // Reserve fraction of GPU memory, default 0.1 (10%)

    // KNN graph parameters
    uint32_t knn_k;                // KNN graph degree for centroids, range [1, 1000], default 32

    static constexpr uint32_t MAX_KNN_K = 1000;

    // Constructor with safe defaults
    LoadConfig() :
        cpu_limit_bytes(16UL << 30),    // 16GB
        gpu_limit_bytes(0),             // Auto-detect
        sample_rate(0.1f),
        centroid_ratio(0.01f),
        pq_bits_start(8),
        pq_bits_min(4),
        pq_dim(0),
        pq_train_fraction(0.05f),
        pq_train_max_rows(65536),
        use_pq(false),
        seed(42),
        gpu_safety_margin(0.1f),
        knn_k(32)
    {}
};

/**
 * Auto-detect GPU memory if gpu_limit_bytes == 0
 * Uses cudaMemGetInfo to query actual available GPU memory
 */
void init_gpu_limit_if_needed(LoadConfig& config) {
    if (config.gpu_limit_bytes == 0) {  // 0 means auto-detect
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        // Use 95% of available memory, reserve 5% for system
        config.gpu_limit_bytes = static_cast<size_t>(free_bytes * 0.95);
        std::cout << "Auto-detected GPU memory: " << std::fixed << std::setprecision(2)
                  << total_bytes / 1e9 << " GB total, "
                  << config.gpu_limit_bytes / 1e9 << " GB available\n";
    }
}

/**
 * Memory requirement estimation structure
 * Key insight: Must include KMeans++ working space (distance matrix)
 */
struct MemoryEstimate {
    // Raw dataset
    size_t raw_data_bytes;
    int64_t raw_data_rows;

    // KMeans working space (CRITICAL!)
    // Distance matrix: n_trials × n_samples × sizeof(float)
    // where n_trials = 2 + ceil(log(n_centroids))
    size_t kmeans_working_bytes;
    int32_t kmeans_n_trials;

    // Sampled data (when sampling needed)
    size_t sampled_data_bytes;
    int64_t sampled_data_rows;

    // PQ related (when PQ needed)
    size_t pq_train_data_bytes;
    int64_t pq_train_rows;
    size_t pq_codebook_bytes;
    size_t pq_encoded_data_bytes;
    uint32_t final_pq_bits;

    // Centroid data
    size_t centroid_data_bytes;
    int64_t centroid_rows;

    // Decision flags
    bool fits_in_gpu;              // Full dataset fits in GPU (Step 1)
    bool sampled_fits_with_kmeans; // Sampled data + KMeans workspace fits (Step 2)
    bool need_pq;                  // PQ quantization required (Step 3)
};

/**
 * Estimate memory requirements for different scenarios
 *
 * Three-stage judgment flow (as per user requirement):
 * Step 1: Check if full dataset fits in GPU (with KMeans working space)
 * Step 2: Check if sampled data fits in GPU (with KMeans working space)
 * Step 3: Check if PQ quantized data fits (with dynamic pq_bits adjustment)
 */
