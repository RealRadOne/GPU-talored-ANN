#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <random>
#include <limits>
#include <iomanip>

// Boost
#include <boost/program_options.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// RAFT
#include <raft/core/resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/cluster/kmeans.cuh>
#include <rmm/device_uvector.hpp>

// Local headers
#include "utils.hpp"
#include "load.hpp"
#include "bucket_build.cuh"

namespace po = boost::program_options;
using namespace bucket;

// ============== Data Loading Configuration ==============

/**
 * Configuration struct for data loading and centroid generation
 */
struct LoadConfig {
    // Memory limits
    size_t cpu_limit_bytes;        // CPU memory limit, default 16GB
    size_t gpu_limit_bytes;        // GPU memory limit, default 0 (auto-detect)

    // Sampling/Centroid parameters
    float sample_rate;             // Sampling ratio [0, 1], default 0.1 (10%)
    float centroid_ratio;          // Centroid ratio relative to sampled data, default 0.01 (1%)

    // PQ parameters (DiskANN style)
    uint32_t pq_bits_start;        // PQ starting bits, default 8
    uint32_t pq_bits_min;          // PQ minimum bits, default 4
    uint32_t pq_dim;               // PQ dimension, default 0 (auto = D/4)
    float pq_train_fraction;       // PQ training data fraction, default 0.05 (5%)
    uint32_t pq_train_max_rows;    // PQ training data row limit, default 65536

    // Control
    bool use_pq;                   // Force PQ quantization
    uint32_t seed;                 // Random seed

    // Safety
    float gpu_safety_margin;       // Reserve fraction of GPU memory, default 0.1 (10%)

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
        gpu_safety_margin(0.1f)
    {}
};

/**
 * Auto-detect GPU memory if gpu_limit_bytes == 0
 */
inline void init_gpu_limit_if_needed(LoadConfig& config) {
    if (config.gpu_limit_bytes == 0) {  // 0 means auto-detect
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        // Use 95% of available memory, reserve 5% for system
        config.gpu_limit_bytes = static_cast<size_t>(free_bytes * 0.95);
        std::cout << "Auto-detected GPU memory: " << total_bytes / 1e9 << " GB total, "
                  << config.gpu_limit_bytes / 1e9 << " GB available\n";
    }
}

/**
 * Memory estimate structure (updated with KMeans working space)
 */
struct MemoryEstimate {
    // Raw dataset
    size_t raw_data_bytes;
    int64_t raw_data_rows;

    // KMeans working space (CRITICAL!)
    size_t kmeans_working_bytes;     // n_trials × n_samples sizeof(float)
    int32_t kmeans_n_trials;         // 2 + ceil(log(n_centroids))

    // Sampled data (when sampling needed)
    size_t sampled_data_bytes;
    int64_t sampled_data_rows;

    // PQ related (when PQ needed)
    size_t pq_codebook_bytes;
    size_t pq_encoded_data_bytes;    // Encoded data (int8)
    uint32_t final_pq_bits;          // Actual pq_bits used (may < 8)

    // Centroid data
    size_t centroid_data_bytes;
    int64_t centroid_rows;

    // Decision flags
    bool fits_in_gpu;                // Full dataset + KMeans space fits in GPU
    bool sampled_fits_with_kmeans;   // Sampled data + KMeans space fits in GPU
    bool need_pq;                    // PQ quantization required

    // Backward compatibility
    int64_t n_sampled;
    int64_t n_centroids;
    size_t pq_data_bytes;            // Total PQ data size
};

/**
 * Estimate memory requirements for data loading and processing
 */
MemoryEstimate estimate_memory_requirement(
    int64_t N,
    int64_t D,
    const LoadConfig& config,
    size_t element_size = sizeof(float)) {

    MemoryEstimate result{};
    result.raw_data_rows = N;

    // Step 1: Calculate KMeans working space requirement
    int64_t n_centroids = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
    result.kmeans_n_trials = 2 + static_cast<int32_t>(std::ceil(std::log2(n_centroids)));
    result.kmeans_working_bytes = static_cast<size_t>(result.kmeans_n_trials) * N * sizeof(float);

    // Step 2: Calculate raw data size
    result.raw_data_bytes = N * D * element_size;
    size_t safety_bytes = static_cast<size_t>(config.gpu_limit_bytes * config.gpu_safety_margin);

    // Step 2A: Check if full dataset + KMeans workspace fits in GPU
    size_t total_with_kmeans = result.raw_data_bytes + result.kmeans_working_bytes + safety_bytes;
    result.fits_in_gpu = total_with_kmeans <= config.gpu_limit_bytes;

    if (result.fits_in_gpu) {
        result.sampled_data_rows = N;
        result.sampled_data_bytes = result.raw_data_bytes;
        result.centroid_rows = n_centroids;
        result.centroid_data_bytes = n_centroids * D * sizeof(float);
        result.need_pq = false;
        result.n_sampled = N;
        result.n_centroids = n_centroids;
        result.pq_data_bytes = result.raw_data_bytes;
        result.sampled_fits_with_kmeans = true;
        return result;
    }

    // Step 2B: Try sampling
    result.sampled_data_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.sample_rate));
    result.sampled_data_bytes = result.sampled_data_rows * D * sizeof(float);

    size_t kmeans_working_bytes_sampled =
        static_cast<size_t>(result.kmeans_n_trials) * result.sampled_data_rows * sizeof(float);

    size_t total_sampled = result.sampled_data_bytes + kmeans_working_bytes_sampled + safety_bytes;
    result.sampled_fits_with_kmeans = total_sampled <= config.gpu_limit_bytes;

    if (result.sampled_fits_with_kmeans) {
        result.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(result.sampled_data_rows * config.centroid_ratio));
        result.centroid_data_bytes = result.centroid_rows * D * sizeof(float);
        result.need_pq = false;
        result.pq_data_bytes = result.sampled_data_bytes;
        result.n_sampled = result.sampled_data_rows;
        result.n_centroids = result.centroid_rows;
        return result;
    }

    // Step 3: Sampled data still too large, must use PQ
    result.need_pq = true;

    // Calculate PQ parameters
    uint32_t pq_dim = (config.pq_dim == 0) ?
        ((D / 4 + 7) / 8 * 8) : config.pq_dim;  // Auto or provided, rounded to 8
    uint32_t n_subspaces = D / pq_dim;

    // Try different pq_bits to fit in GPU memory
    uint32_t pq_bits = config.pq_bits_start;
    bool found_valid_bits = false;

    while (pq_bits >= config.pq_bits_min) {
        // Codebook size
        size_t codebook_size = static_cast<size_t>(n_subspaces) * (1 << pq_bits) * pq_dim;
        result.pq_codebook_bytes = codebook_size * sizeof(float);

        // Encoded data size
        uint32_t bytes_per_code = (pq_bits + 7) / 8;
        result.pq_encoded_data_bytes = static_cast<size_t>(result.sampled_data_rows) * n_subspaces * bytes_per_code;

        // Decoded (reconstructed) data will be float
        size_t pq_decoded_bytes = static_cast<size_t>(result.sampled_data_rows) * D * sizeof(float);

        // Total GPU requirement
        size_t total_pq = result.pq_encoded_data_bytes + result.pq_codebook_bytes +
                         pq_decoded_bytes + kmeans_working_bytes_sampled + safety_bytes;

        if (total_pq <= config.gpu_limit_bytes) {
            result.final_pq_bits = pq_bits;
            found_valid_bits = true;
            break;
        }

        pq_bits--;
    }

    if (!found_valid_bits) {
        std::cerr << "WARNING: Cannot fit data in GPU even with pq_bits=" << config.pq_bits_min << "\n";
        result.final_pq_bits = config.pq_bits_min;  // Use minimum anyway
    }

    result.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(result.sampled_data_rows * config.centroid_ratio));
    result.centroid_data_bytes = result.centroid_rows * D * sizeof(float);
    result.pq_data_bytes = result.pq_encoded_data_bytes + result.pq_codebook_bytes;
    result.n_sampled = result.sampled_data_rows;
    result.n_centroids = result.centroid_rows;

    return result;
}

/**
 * Validate load configuration
 */
void validate_load_config(const LoadConfig& config) {
    // Check: centroid_ratio < sample_rate
    if (config.centroid_ratio >= config.sample_rate) {
        throw std::runtime_error(
            "centroid_ratio (" + std::to_string(config.centroid_ratio) +
            ") must be < sample_rate (" + std::to_string(config.sample_rate) + ")"
        );
    }

    // Check: sample_rate range
    if (config.sample_rate <= 0.0f || config.sample_rate > 1.0f) {
        throw std::runtime_error("sample_rate must be in (0, 1]");
    }

    // Check: centroid_ratio range
    if (config.centroid_ratio <= 0.0f || config.centroid_ratio > 1.0f) {
        throw std::runtime_error("centroid_ratio must be in (0, 1]");
    }

    // Check: pq_bits range
    if (config.pq_bits_start < 4 || config.pq_bits_start > 8) {
        throw std::runtime_error("pq_bits_start must be in [4, 8]");
    }

    if (config.pq_bits_min >= config.pq_bits_start) {
        throw std::runtime_error("pq_bits_min must be < pq_bits_start");
    }

    // Check: GPU safety margin
    if (config.gpu_safety_margin < 0.0f || config.gpu_safety_margin >= 1.0f) {
        throw std::runtime_error("gpu_safety_margin must be in [0, 1)");
    }
}

/**
 * Print memory estimate in human-readable format
 */
void print_memory_estimate(const MemoryEstimate& est) {
    auto format_bytes = [](size_t bytes) {
        if (bytes < 1024) return std::to_string(bytes) + " B";
        if (bytes < 1024*1024) return std::to_string(bytes/1024) + " KB";
        if (bytes < 1024*1024*1024) return std::to_string(bytes/(1024*1024)) + " MB";
        return std::to_string(bytes/(1024.0*1024*1024)) + " GB";
    };

    std::cout << "  Memory Estimate:\n";
    std::cout << "    Raw data:       " << format_bytes(est.raw_data_bytes);
    std::cout << " (fits_in_gpu=" << (est.fits_in_gpu ? "true" : "false") << ")\n";
    std::cout << "    Sampled data:   " << format_bytes(est.sampled_data_bytes);
    std::cout << " (N=" << est.n_sampled << ")\n";
    if (est.need_pq) {
        std::cout << "    After PQ:       " << format_bytes(est.pq_data_bytes) << "\n";
    }
    std::cout << "    Centroids:      " << format_bytes(est.centroid_data_bytes);
    std::cout << " (K=" << est.n_centroids << ")\n";
    std::cout << "    Need PQ:        " << (est.need_pq ? "yes" : "no") << "\n\n";
}

// ============== Sampling Functions ==============

/**
 * Sample without replacement using Fisher-Yates algorithm
 */
std::vector<int64_t> sample_without_replacement(
    int64_t N,
    int64_t n_samples,
    uint32_t seed) {

    if (n_samples > N) n_samples = N;
    if (n_samples <= 0) return {};

    std::vector<int64_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0LL);

    std::mt19937_64 gen(seed);

    // Fisher-Yates shuffle on first n_samples elements
    for (int64_t i = 0; i < n_samples; ++i) {
        std::uniform_int_distribution<int64_t> dis(i, N - 1);
        int64_t j = dis(gen);
        std::swap(indices[i], indices[j]);
    }

    indices.resize(n_samples);
    return indices;
}

/**
 * Sample data points: returns (sampled_data, sampled_indices)
 */
std::pair<std::vector<float>, std::vector<int64_t>>
sample_data(
    const std::vector<float>& X,
    int64_t N, int D,
    float sample_rate,
    uint32_t seed) {

    int64_t n_samples = static_cast<int64_t>(N * sample_rate);
    n_samples = std::max(static_cast<int64_t>(1), std::min(n_samples, N));

    // Get sampled indices
    auto sampled_indices = sample_without_replacement(N, n_samples, seed);

    // Extract sampled data
    std::vector<float> sampled_data(n_samples * D);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_samples; ++i) {
        int64_t src_idx = sampled_indices[i];
        std::copy(X.begin() + src_idx * D,
                  X.begin() + (src_idx + 1) * D,
                  sampled_data.begin() + i * D);
    }

    return {sampled_data, sampled_indices};
}

// ============== Simplified PQ Quantization ==============

/**
 * Simple per-dimension quantizer
 */
struct SimpleQuantizer {
    std::vector<float> scale;        // [D]
    std::vector<float> offset;       // [D]
    uint32_t bits;
};

/**
 * Train a simple quantizer on sample data
 */
SimpleQuantizer train_simple_quantizer(
    const std::vector<float>& X_sample,
    int64_t n_samples,
    int D,
    uint32_t bits = 8) {

    SimpleQuantizer quantizer;
    quantizer.bits = bits;
    quantizer.scale.resize(D);
    quantizer.offset.resize(D);

    uint32_t max_val = (1U << bits) - 1;

    // Compute min/max for each dimension
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < D; ++d) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (int64_t i = 0; i < n_samples; ++i) {
            float val = X_sample[i * D + d];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        quantizer.offset[d] = min_val;
        quantizer.scale[d] = (max_val > min_val) ?
            (max_val - min_val) / static_cast<float>(max_val) : 1.0f;
    }

    return quantizer;
}

/**
 * Encode data using the quantizer
 */
std::vector<uint8_t> simple_encode(
    const std::vector<float>& X,
    int64_t n_vectors,
    int D,
    const SimpleQuantizer& q) {

    std::vector<uint8_t> codes(n_vectors * D);
    uint32_t max_val = (1U << q.bits) - 1;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int64_t i = 0; i < n_vectors; ++i) {
        for (int d = 0; d < D; ++d) {
            float val = X[i * D + d];
            float normalized = (val - q.offset[d]) / q.scale[d];
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            uint8_t code = static_cast<uint8_t>(normalized * max_val);
            codes[i * D + d] = code;
        }
    }

    return codes;
}

/**
 * Decode data using the quantizer (reconstruction)
 */
std::vector<float> simple_decode(
    const std::vector<uint8_t>& codes,
    int64_t n_vectors,
    int D,
    const SimpleQuantizer& q) {

    std::vector<float> reconstructed(n_vectors * D);
    uint32_t max_val = (1U << q.bits) - 1;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int64_t i = 0; i < n_vectors; ++i) {
        for (int d = 0; d < D; ++d) {
            uint8_t code = codes[i * D + d];
            float normalized = static_cast<float>(code) / max_val;
            reconstructed[i * D + d] = q.offset[d] + normalized * q.scale[d];
        }
    }

    return reconstructed;
}

// ============== CPU-based KNN helper ==============

/**
 * Simple CPU-based K nearest neighbors search
 * Returns: (indices [N_A * k], distances [N_A * k])
 */
std::pair<std::vector<int64_t>, std::vector<float>>
topk_from_A_to_B_cpu(const std::vector<float>& A, int N_A, int D,
                      const std::vector<float>& B, int N_B,
                      int k) {
    std::vector<int64_t> out_indices(static_cast<int64_t>(N_A) * k, -1);
    std::vector<float> out_dists(static_cast<int64_t>(N_A) * k, std::numeric_limits<float>::infinity());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_A; ++i) {
        const float* a = A.data() + i * D;
        std::vector<std::pair<float, int64_t>> candidates;
        candidates.reserve(N_B);

        for (int j = 0; j < N_B; ++j) {
            const float* b = B.data() + j * D;
            float dist = l2_distance_sq(a, b, D);
            candidates.push_back({dist, j});
        }

        // Partial sort to get top-k
        if (k >= N_B) {
            std::sort(candidates.begin(), candidates.end());
        } else {
            std::nth_element(candidates.begin(), candidates.begin() + k,
                           candidates.end());
            std::sort(candidates.begin(), candidates.begin() + k);
        }

        int limit = std::min(k, N_B);
        for (int j = 0; j < limit; ++j) {
            out_indices[i * k + j] = candidates[j].second;
            out_dists[i * k + j] = candidates[j].first;
        }
    }

    return {out_indices, out_dists};
}

/**
 * Assignment of N points to K centers - CPU version
 * Returns: assignment vector (N,) with values in [0, K)
 */
std::vector<int64_t>
pairwise_l2_min_assign_cpu(const std::vector<float>& X, int N, int D,
                            const std::vector<float>& centers, int K) {
    std::vector<int64_t> indices(N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const float* x = X.data() + i * D;
        float min_dist = std::numeric_limits<float>::infinity();
        int best_idx = 0;

        for (int k = 0; k < K; ++k) {
            const float* c = centers.data() + k * D;
            float dist = l2_distance_sq(x, c, D);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = k;
            }
        }

        indices[i] = best_idx;
    }

    return indices;
}

/**
 * GPU-accelerated K nearest neighbors search
 * Returns: (indices [N_A * k], distances [N_A * k])
 * Computes pairwise distances on GPU and finds top-k
 */
std::pair<std::vector<int64_t>, std::vector<float>>
topk_from_A_to_B_gpu(const std::vector<float>& A, int N_A, int D,
                     const std::vector<float>& B, int N_B,
                     int k) {
    std::vector<int64_t> out_indices(static_cast<int64_t>(N_A) * k, -1);
    std::vector<float> out_dists(static_cast<int64_t>(N_A) * k, std::numeric_limits<float>::infinity());

    try {
        // Allocate GPU buffers for A, B, and distances
        GPUBuffer<float> A_gpu(static_cast<int64_t>(N_A) * D);
        GPUBuffer<float> B_gpu(static_cast<int64_t>(N_B) * D);
        GPUBuffer<float> distances_gpu(static_cast<int64_t>(N_A) * N_B);
        GPUBuffer<int64_t> indices_gpu(static_cast<int64_t>(N_A) * k);
        GPUBuffer<float> dists_gpu(static_cast<int64_t>(N_A) * k);

        // Copy data to GPU
        A_gpu.copy_from_host(A.data(), N_A * D);
        B_gpu.copy_from_host(B.data(), N_B * D);

        // std::cout << "      [GPU] TopK computation: N_A=" << N_A
        //           << ", N_B=" << N_B << ", k=" << k
        //           << ", distance_matrix_size=" << (static_cast<int64_t>(N_A) * N_B * 4 / 1e9) << "GB\n";

        // Compute pairwise distances
        compute_distances_gpu(A_gpu, N_A, D, B_gpu, N_B, distances_gpu);

        // Find top-k indices and distances
        find_topk_gpu(distances_gpu, N_A, N_B, k, indices_gpu, dists_gpu);

        // Copy results back to CPU
        indices_gpu.copy_to_host(out_indices.data(), N_A * k);
        dists_gpu.copy_to_host(out_dists.data(), N_A * k);

        CUDA_CHECK(cudaDeviceSynchronize());

    } catch (const std::exception& e) {
        std::cerr << "GPU topk computation failed: " << e.what() << "\n";
        std::cerr << "Falling back to CPU...\n";
        return topk_from_A_to_B_cpu(A, N_A, D, B, N_B, k);
    }

    return {out_indices, out_dists};
}

/**
 * K-Means++ initialization with GPU acceleration
 * Uses CUDA for distance computations
 */
std::pair<std::vector<int32_t>, std::vector<float>>
kmeans_pp_centers_gpu(const std::vector<float>& X, int N, int D,
                       int B, uint32_t seed, int pool_ratio = 5) {
    RandomGenerator rng(seed);

    int pool_n = std::min(N, pool_ratio * B);
    auto pool_ids_uint = rng.randperm(N);
    pool_ids_uint.resize(pool_n);

    // Convert to int32_t for consistency
    std::vector<int32_t> pool_ids(pool_ids_uint.begin(), pool_ids_uint.end());

    std::vector<float> min_d2(pool_n, std::numeric_limits<float>::infinity());
    std::vector<int32_t> chosen;

    // Choose first center randomly
    int first_idx = pool_ids[rng.randint(pool_n)];
    chosen.push_back(first_idx);

    std::cout << "  [K-Means++] Pool: " << pool_n << ", Total iters: " << B << std::endl;

    // K-Means++ iterations
    for (int i = 1; i < B; ++i) {
        int last_idx = chosen.back();
        const float* last_c = X.data() + last_idx * D;

        // Compute distances for pool with OpenMP parallelization
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < pool_n; ++j) {
            int glob_idx = pool_ids[j];
            float d2 = l2_distance_sq(last_c, X.data() + glob_idx * D, D);
            min_d2[j] = std::min(min_d2[j], d2);
        }

        float total = std::accumulate(min_d2.begin(), min_d2.end(), 0.0f);
        if (total <= 0.0f) {
            chosen.push_back(pool_ids[rng.randint(pool_n)]);
        } else {
            // Weighted sampling
            float r = rng.uniform() * total;
            float acc = 0.0f;
            int nxt = 0;
            for (int j = 0; j < pool_n; ++j) {
                acc += min_d2[j];
                if (acc >= r) {
                    nxt = pool_ids[j];
                    break;
                }
            }
            chosen.push_back(nxt);
        }
    }

    // Create center matrix
    std::vector<float> centers(B * D);
    for (int i = 0; i < B; ++i) {
        int idx = chosen[i];
        std::copy(X.begin() + idx * D, X.begin() + (idx + 1) * D,
                  centers.begin() + i * D);
    }

    return {chosen, centers};
}

/**
 * Fast random center initialization (for speed over quality)
 * Simply pick B random points as centers
 */
std::pair<std::vector<int32_t>, std::vector<float>>
kmeans_random_init(const std::vector<float>& X, int N, int D,
                   int B, uint32_t seed) {
    RandomGenerator rng(seed);
    auto chosen_uint = rng.randperm(N);
    chosen_uint.resize(B);

    // Convert to int32_t
    std::vector<int32_t> chosen(chosen_uint.begin(), chosen_uint.end());

    std::vector<float> centers(B * D);
    for (int i = 0; i < B; ++i) {
        int idx = chosen[i];
        std::copy(X.begin() + idx * D, X.begin() + (idx + 1) * D,
                  centers.begin() + i * D);
    }

    return {chosen, centers};
}

/**
 * Pairwise L2 assignment using GPU with aggressive batching (memory-efficient)
 * Calculates batch_size * B * 4 bytes for distances, which should fit in GPU
 */
std::vector<int64_t>
pairwise_l2_min_assign_gpu(const std::vector<float>& X, int N, int D,
                            const std::vector<float>& centers, int B) {
    std::vector<int64_t> indices(N);

    // Calculate batch size to fit in GPU memory
    // GPU distance matrix: batch_size * B * sizeof(float) = batch_size * B * 4 bytes
    // Conservative: target 2GB for distances matrix
    long long max_distance_bytes = 2000000000LL;  // 2GB
    long long calculation = max_distance_bytes / (B * sizeof(float));
    int batch_size = std::max(1000, (int)calculation);

    // Also limit by dataset size
    batch_size = std::min(batch_size, N);

    // std::cout << "    [GPU] Distance computation: batch=" << batch_size
    //           << ", centers=" << B << ", per_batch_memory=" << (batch_size * B * 4 / 1e9) << "GB\n";

    // Process points in batches
    for (int batch_start = 0; batch_start < N; batch_start += batch_size) {
        int batch_end = std::min(batch_start + batch_size, N);
        int batch_n = batch_end - batch_start;

        try {
            // GPU buffers - explicitly scoped for auto-cleanup
            {
                GPUBuffer<float> X_batch_gpu(static_cast<int64_t>(batch_n) * D);
                GPUBuffer<float> C_gpu(static_cast<int64_t>(B) * D);
                GPUBuffer<float> distances_gpu(static_cast<int64_t>(batch_n) * B);
                GPUBuffer<int64_t> indices_batch_gpu(batch_n);

                // Copy batch data to GPU
                X_batch_gpu.copy_from_host(X.data() + batch_start * D, batch_n * D);
                C_gpu.copy_from_host(centers.data(), B * D);

                // Compute distances for this batch
                compute_distances_gpu(X_batch_gpu, batch_n, D, C_gpu, B, distances_gpu);

                // Find minimum indices for this batch
                find_assignments_gpu(distances_gpu, batch_n, B, indices_batch_gpu);

                // Copy results back
                indices_batch_gpu.copy_to_host(indices.data() + batch_start, batch_n);
            }  // GPU buffers freed here

            CUDA_CHECK(cudaDeviceSynchronize());

            if (batch_end % std::max(100000, N / 10) == 0) {
                std::cout << "      Processed " << batch_end << "/" << N << " points\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "GPU batch processing failed at batch [" << batch_start << ", " << batch_end << "): " << e.what() << "\n";
            std::cerr << "Falling back to CPU assignment...\n";
            return pairwise_l2_min_assign_cpu(X, N, D, centers, B);
        }
    }

    return indices;
}

/**
 * K-Means Lloyd refinement iterations - CPU version (memory efficient)
 * For very large datasets, CPU is faster and avoids GPU memory issues
 */
std::pair<std::vector<float>, std::vector<int64_t>>
refine_centers_kmeans_gpu(const std::vector<float>& X, int N, int D,
                          std::vector<float>& centers, int B,
                          int n_iters) {
    std::vector<int64_t> assign(N);
    std::vector<float> new_centers(B * D);
    std::vector<int32_t> counts(B);

    std::cout << "    [CPU] Using CPU for K-Means (more memory efficient)\n";

    for (int iter = 0; iter < n_iters; ++iter) {
        // Assign points to centers using CPU (fast with OpenMP)
        assign = pairwise_l2_min_assign_cpu(X, N, D, centers, B);

        // Update centers on CPU
        std::fill(new_centers.begin(), new_centers.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (int i = 0; i < N; ++i) {
            int b = assign[i];
            if (b >= 0 && b < B) {
                for (int d = 0; d < D; ++d) {
                    new_centers[b * D + d] += X[i * D + d];
                }
                counts[b]++;
            }
        }

        // Normalize
        for (int b = 0; b < B; ++b) {
            if (counts[b] > 0) {
                float inv_count = 1.0f / counts[b];
                for (int d = 0; d < D; ++d) {
                    new_centers[b * D + d] *= inv_count;
                }
            } else {
                // Keep old center
                std::copy(centers.begin() + b * D, centers.begin() + (b + 1) * D,
                         new_centers.begin() + b * D);
            }
        }

        centers = new_centers;
        std::cout << "      Iteration " << (iter + 1) << "/" << n_iters << " done\n";
    }

    return {centers, assign};
}

/**
 * K-Means++ initialization with D^2 weighted sampling on pool
 */
std::pair<std::vector<int32_t>, std::vector<float>>
kmeans_pp_centers(const std::vector<float>& X, int N, int D,
                   int B, uint32_t seed, int pool_ratio = 20) {
    RandomGenerator rng(seed);

    int pool_n = std::min(N, pool_ratio * B);
    auto pool_ids_uint = rng.randperm(N);
    pool_ids_uint.resize(pool_n);

    // Convert to int32_t for consistency
    std::vector<int32_t> pool_ids(pool_ids_uint.begin(), pool_ids_uint.end());

    std::vector<float> min_d2(pool_n, std::numeric_limits<float>::infinity());

    std::vector<int32_t> chosen;
    chosen.push_back(pool_ids[rng.randint(pool_n)]);

    // K-Means++ iterations
    for (int i = 1; i < B; ++i) {
        int last_idx = chosen.back();
        const float* last_c = X.data() + last_idx * D;

        // Update min distances
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < pool_n; ++j) {
            int glob_idx = pool_ids[j];
            const float* x = X.data() + glob_idx * D;
            float d2 = l2_distance_sq(last_c, x, D);
            min_d2[j] = std::min(min_d2[j], d2);
        }

        float total = std::accumulate(min_d2.begin(), min_d2.end(), 0.0f);
        if (total <= 0.0f) {
            chosen.push_back(pool_ids[rng.randint(pool_n)]);
        } else {
            // Weighted sampling
            float r = rng.uniform() * total;
            float acc = 0.0f;
            int nxt = 0;
            for (int j = 0; j < pool_n; ++j) {
                acc += min_d2[j];
                if (acc >= r) {
                    nxt = pool_ids[j];
                    break;
                }
            }
            chosen.push_back(nxt);
        }
    }

    // Create center matrix
    std::vector<float> centers(B * D);
    for (int i = 0; i < B; ++i) {
        int idx = chosen[i];
        std::copy(X.begin() + idx * D, X.begin() + (idx + 1) * D,
                  centers.begin() + i * D);
    }

    return {chosen, centers};
}

/**
 * K-Means Lloyd refinement iterations (optimized)
 * Uses thread-local accumulation to avoid lock contention
 */
std::pair<std::vector<float>, std::vector<int64_t>>
refine_centers_kmeans(const std::vector<float>& X, int N, int D,
                      std::vector<float>& centers, int B,
                      int n_iters) {
    std::vector<int64_t> assign(N);
    std::vector<float> new_centers(B * D);
    std::vector<int32_t> counts(B);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Assign points to centers
        assign = pairwise_l2_min_assign_cpu(X, N, D, centers, B);

        // Update centers with thread-local accumulators
        std::fill(new_centers.begin(), new_centers.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        int num_threads = omp_get_max_threads();
        std::vector<float> thread_centers(num_threads * B * D, 0.0f);
        std::vector<int32_t> thread_counts(num_threads * B, 0);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            int tid = omp_get_thread_num();
            int b = assign[i];
            if (b >= 0 && b < B) {
                for (int d = 0; d < D; ++d) {
                    thread_centers[tid * B * D + b * D + d] += X[i * D + d];
                }
                thread_counts[tid * B + b]++;
            }
        }

        // Merge thread-local results
        for (int tid = 0; tid < num_threads; ++tid) {
            for (int b = 0; b < B; ++b) {
                counts[b] += thread_counts[tid * B + b];
                for (int d = 0; d < D; ++d) {
                    new_centers[b * D + d] += thread_centers[tid * B * D + b * D + d];
                }
            }
        }

        // Normalize
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < B; ++b) {
            if (counts[b] > 0) {
                float inv_count = 1.0f / counts[b];
                for (int d = 0; d < D; ++d) {
                    new_centers[b * D + d] *= inv_count;
                }
            } else {
                // Keep old center
                std::copy(centers.begin() + b * D, centers.begin() + (b + 1) * D,
                         new_centers.begin() + b * D);
            }
        }

        centers = new_centers;
    }

    return {centers, assign};
}

/**
 * Main integration function: Prepare data and run GPU KMeans++ to generate centroids
 * Handles sampling, optional PQ quantization, and RAFT KMeans++ initialization
 *
 * Returns: (centroid_indices_in_original, centroid_data)
 *   centroid_indices_in_original: [n_centroids] - indices mapped back to original data
 *   centroid_data: [n_centroids, D] - centroid vectors
 */
std::pair<std::vector<int64_t>, std::vector<float>>
prepare_and_kmeans_pp_gpu(
    const std::vector<float>& X_full,
    int64_t N, int D,
    const LoadConfig& config,
    const MemoryEstimate& mem_est) {

    std::cout << "[GPU KMeans++ Centroid Generation]\n";

    // Step 1: Sample data if needed
    std::vector<float> X_work = X_full;
    std::vector<int64_t> sampled_indices;

    if (!mem_est.fits_in_gpu) {
        std::cout << "  Sampling " << (config.sample_rate * 100) << "% of data...\n";
        auto [sampled_data, indices] = sample_data(
            X_full, N, D, config.sample_rate, config.seed);
        X_work = sampled_data;
        sampled_indices = indices;
        N = mem_est.n_sampled;
    } else {
        // Full dataset, create identity mapping
        sampled_indices.resize(N);
        std::iota(sampled_indices.begin(), sampled_indices.end(), 0LL);
    }

    // Step 2: Apply PQ quantization if needed
    SimpleQuantizer quantizer{};
    bool quantized = false;

    if (mem_est.need_pq) {
        std::cout << "  Training PQ quantizer (" << mem_est.final_pq_bits << " bits)...\n";
        quantizer = train_simple_quantizer(X_work, N, D, mem_est.final_pq_bits);

        std::cout << "  Encoding sampled data...\n";
        auto codes = simple_encode(X_work, N, D, quantizer);

        std::cout << "  Decoding to recover approximate data...\n";
        X_work = simple_decode(codes, N, D, quantizer);
        quantized = true;
    }

    int64_t n_centroids = mem_est.n_centroids;
    std::cout << "  Preparing GPU data (N=" << N << ", D=" << D << ", K=" << n_centroids << ")...\n";

    try {
        // Step 3: Transfer data to GPU and run RAFT KMeans++
        raft::resources res;

        // Create host matrix view for input data
        auto dataset = raft::make_host_matrix_view<const float, int64_t>(
            X_work.data(), N, D);

        // Allocate GPU memory for dataset
        auto dataset_gpu = raft::make_device_matrix<float, int64_t>(res, N, D);
        raft::copy(dataset_gpu.data_handle(), dataset.data_handle(),
                   dataset.size(), raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);

        // Allocate GPU memory for centroids and indices
        auto centroids_gpu = raft::make_device_matrix<float, int64_t>(res, n_centroids, D);
        auto centroid_indices_gpu = raft::make_device_vector<int64_t, int64_t>(res, n_centroids);

        // Setup KMeans++ parameters
        raft::cluster::KMeansParams kmeans_params;
        kmeans_params.n_clusters = n_centroids;
        kmeans_params.metric = raft::distance::DistanceType::L2Expanded;
        kmeans_params.rng_state.seed = config.seed;

        std::cout << "  Running RAFT KMeans++ on GPU...\n";

        // Run KMeans++ initialization
        rmm::device_uvector<char> workspace(0, raft::resource::get_cuda_stream(res));
        raft::cluster::kmeans::init_plus_plus<float, int64_t>(
            res,
            kmeans_params,
            raft::make_device_matrix_view<const float, int64_t>(dataset_gpu.data_handle(), N, D),
            centroids_gpu.view(),
            workspace
        );

        raft::resource::sync_stream(res);

        // Copy results back to host
        std::vector<int64_t> centroid_indices_gpu_host(n_centroids);
        std::vector<float> centroids_data(n_centroids * D);

        raft::copy(centroid_indices_gpu_host.data(),
                   centroid_indices_gpu.data_handle(),
                   n_centroids,
                   raft::resource::get_cuda_stream(res));
        raft::copy(centroids_data.data(),
                   centroids_gpu.data_handle(),
                   centroids_gpu.size(),
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);

        // Step 4: Map centroid indices back to original dataset
        std::vector<int64_t> centroid_indices_original(n_centroids);
        for (int64_t i = 0; i < n_centroids; ++i) {
            int64_t idx_in_sampled = centroid_indices_gpu_host[i];
            centroid_indices_original[i] = sampled_indices[idx_in_sampled];
        }

        std::cout << "  Generated " << n_centroids << " centroids\n\n";

        return {centroid_indices_original, centroids_data};

    } catch (const std::exception& e) {
        std::cerr << "Error in GPU KMeans++: " << e.what() << "\n";
        std::cerr << "Falling back to CPU KMeans++...\n";

        // Fallback to CPU implementation
        auto [indices, centers] = kmeans_pp_centers(
            X_work, N, D, n_centroids, config.seed, 20);

        // Map back to original indices
        std::vector<int64_t> indices_original(n_centroids);
        for (int64_t i = 0; i < n_centroids; ++i) {
            indices_original[i] = sampled_indices[indices[i]];
        }

        std::cout << "  Generated " << n_centroids << " centroids (using CPU fallback)\n\n";

        return {indices_original, centers};
    }
}

// ============== Rebalance and Bucket Building ==============
std::vector<int64_t>
rebalance_assign(std::vector<int64_t> assign, int B,
                 const std::vector<int64_t>& top2_idx,
                 const std::vector<float>& top2_dist,
                 int target = 32, int slack = 4, int max_iters = 30) {
    int N = assign.size();

    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<int32_t> counts(B, 0);
        for (int i = 0; i < N; ++i) {
            if (assign[i] >= 0 && assign[i] < B) {
                counts[assign[i]]++;
            }
        }

        std::vector<int> over_buckets;
        for (int b = 0; b < B; ++b) {
            if (counts[b] > target + slack) {
                over_buckets.push_back(b);
            }
        }

        if (over_buckets.empty()) break;

        for (int b : over_buckets) {
            int excess = counts[b] - target;

            // Find movable points with smallest cost
            std::vector<std::pair<float, int>> candidates;
            for (int i = 0; i < N; ++i) {
                if (assign[i] != b) continue;

                int64_t idx1 = top2_idx[i * 2];
                int64_t idx2 = top2_idx[i * 2 + 1];
                float d1 = top2_dist[i * 2];
                float d2 = top2_dist[i * 2 + 1];

                if (idx2 >= 0 && counts[idx2] <= target + slack) {
                    float cost = d2 - d1;
                    candidates.push_back({cost, i});
                }
            }

            // Sort by cost and move cheapest points
            std::sort(candidates.begin(), candidates.end());
            int moved = 0;
            for (auto [cost, i] : candidates) {
                if (moved >= excess) break;
                int64_t new_center = top2_idx[i * 2 + 1];
                assign[i] = new_center;
                counts[new_center]++;
                counts[b]--;
                moved++;
            }
        }
    }

    return assign;
}

/**
 * Build buckets - lightweight version without K-Means
 * Only: random center init + single assignment pass
 */
std::pair<std::vector<std::vector<int32_t>>, std::vector<float>>
build_buckets(const std::vector<float>& X, int N, int D,
              uint32_t seed, int kmeans_iters = 5,
              bool balance = true, int balance_slack = 4,
              const std::string& init_method = "auto") {
    int B = std::max(1, N / 32);
    int target = 32;

    std::cout << "  [bucket] Random center initialization ...\n";
    auto [center_ids, centers] = kmeans_random_init(X, N, D, B, seed);

    std::cout << "  [bucket] Single assignment pass using GPU (no refinement) ...\n";
    std::vector<int64_t> assign = pairwise_l2_min_assign_gpu(X, N, D, centers, B);

    // Print statistics
    std::vector<int> assign_int(assign.begin(), assign.end());
    print_bucket_stats(assign_int, B, target);

    // Convert assignments to bucket lists
    std::vector<std::vector<int32_t>> buckets(B);
    for (int i = 0; i < N; ++i) {
        if (assign[i] >= 0 && assign[i] < B) {
            buckets[assign[i]].push_back(i);
        }
    }

    return {buckets, centers};
}

/**
 * Find k nearest buckets by center distance
 */
std::vector<std::vector<int64_t>>
k_nearest_buckets(const std::vector<float>& centers, int B, int D,
                  int k) {
    std::cout << "  [KNB] Finding k nearest buckets using CPU (OpenMP) ...\n";
    // Use CPU version - more efficient for this size due to better sorting
    auto [indices, dists] = topk_from_A_to_B_cpu(centers, B, D, centers, B, k + 1);

    // Remove self-loops
    std::vector<std::vector<int64_t>> result(B);
    for (int b = 0; b < B; ++b) {
        std::vector<int64_t> neighbors;
        for (int j = 0; j < k + 1; ++j) {
            int64_t neighbor = indices[b * (k + 1) + j];
            if (neighbor != b && neighbor >= 0) {
                neighbors.push_back(neighbor);
                if (neighbors.size() >= k) break;
            }
        }
        result[b] = neighbors;
    }

    return result;
}

// Forward declaration
std::pair<std::vector<int64_t>, std::vector<float>>
neighbors_within_knn_buckets(const std::vector<float>& X, int N, int D,
                             const std::vector<std::vector<int32_t>>& buckets,
                             const std::vector<std::vector<int64_t>>& knb,
                             int m, int B);

/**
 * Find m nearest neighbors within bucket's k nearest buckets - GPU Tensor Core version
 */
std::pair<std::vector<int64_t>, std::vector<float>>
neighbors_within_knn_buckets_gpu(const std::vector<float>& X, int N, int D,
                                 const std::vector<std::vector<int32_t>>& buckets,
                                 const std::vector<std::vector<int64_t>>& knb,
                                 int m, int B) {
    std::vector<int64_t> all_neighbors(static_cast<int64_t>(N) * m, -1);
    std::vector<float> all_dists(static_cast<int64_t>(N) * m, std::numeric_limits<float>::infinity());

    try {
        int processed_buckets = 0;
        for (int b = 0; b < B; ++b) {
            const auto& q_indices = buckets[b];
            if (q_indices.empty()) continue;

            if (processed_buckets % 100 == 0) {
                std::cout << "      [Progress] Processed " << processed_buckets << "/" << B << " buckets\n";
            }
            processed_buckets++;

            // Gather candidate indices from b and its k nearest buckets
            std::vector<int32_t> cand_indices;
            for (int nb : knb[b]) {
                cand_indices.insert(cand_indices.end(), buckets[nb].begin(), buckets[nb].end());
            }
            cand_indices.insert(cand_indices.end(), q_indices.begin(), q_indices.end());

            // Remove duplicates
            std::sort(cand_indices.begin(), cand_indices.end());
            cand_indices.erase(std::unique(cand_indices.begin(), cand_indices.end()), cand_indices.end());

            if (cand_indices.empty()) continue;

            // Extract query and candidate data
            int nq = q_indices.size();
            int nc = cand_indices.size();

            // For small buckets, fall back to CPU
            if (nq * nc < 100000) {
                std::vector<float> Q(nq * D), Cand(nc * D);

                #pragma omp parallel for collapse(2) schedule(static)
                for (int i = 0; i < nq; ++i) {
                    for (int d = 0; d < D; ++d) {
                        Q[i * D + d] = X[q_indices[i] * D + d];
                    }
                }

                #pragma omp parallel for collapse(2) schedule(static)
                for (int i = 0; i < nc; ++i) {
                    for (int d = 0; d < D; ++d) {
                        Cand[i * D + d] = X[cand_indices[i] * D + d];
                    }
                }

                auto [indices, dists] = topk_from_A_to_B_gpu(Q, nq, D, Cand, nc, m + 1);

                // Map local indices to global and filter self
                for (int i = 0; i < nq; ++i) {
                    int qi = q_indices[i];
                    std::vector<std::pair<float, int64_t>> neighbors;

                    for (int j = 0; j < m + 1; ++j) {
                        int64_t local_idx = indices[i * (m + 1) + j];
                        if (local_idx < 0 || local_idx >= nc) continue;
                        int64_t global_idx = cand_indices[local_idx];
                        float dist = dists[i * (m + 1) + j];

                        if (global_idx != qi) {  // Skip self
                            neighbors.push_back({dist, global_idx});
                        }
                        if (neighbors.size() >= m) break;
                    }

                    // Fill result
                    for (size_t j = 0; j < neighbors.size(); ++j) {
                        all_neighbors[qi * m + j] = neighbors[j].second;
                        all_dists[qi * m + j] = neighbors[j].first;
                    }
                }
                continue;
            }

            // GPU path for larger buckets
            // std::cout << "    [GPU TC] Bucket " << b << ": nq=" << nq << " nc=" << nc
            //           << " size=" << (nq * nc * 4 / 1e6) << "MB\n";

            // Allocate GPU buffers
            GPUBuffer<float> Q_gpu(static_cast<int64_t>(nq) * D);
            GPUBuffer<float> Cand_gpu(static_cast<int64_t>(nc) * D);
            GPUBuffer<float> distances_gpu(static_cast<int64_t>(nq) * nc);
            GPUBuffer<int64_t> indices_gpu(static_cast<int64_t>(nq) * (m + 1));
            GPUBuffer<float> dists_gpu(static_cast<int64_t>(nq) * (m + 1));

            // Copy query and candidate data to GPU
            std::vector<float> Q(nq * D), Cand(nc * D);

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < nq; ++i) {
                for (int d = 0; d < D; ++d) {
                    Q[i * D + d] = X[q_indices[i] * D + d];
                }
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < nc; ++i) {
                for (int d = 0; d < D; ++d) {
                    Cand[i * D + d] = X[cand_indices[i] * D + d];
                }
            }

            Q_gpu.copy_from_host(Q.data(), nq * D);
            Cand_gpu.copy_from_host(Cand.data(), nc * D);

            // Compute distances using tensor cores
            compute_distances_gpu(Q_gpu, nq, D, Cand_gpu, nc, distances_gpu);

            // Find top-k indices and distances
            find_topk_gpu(distances_gpu, nq, nc, m + 1, indices_gpu, dists_gpu);

            // Copy results back to CPU
            std::vector<int64_t> h_indices(static_cast<int64_t>(nq) * (m + 1));
            std::vector<float> h_dists(static_cast<int64_t>(nq) * (m + 1));

            indices_gpu.copy_to_host(h_indices.data(), nq * (m + 1));
            dists_gpu.copy_to_host(h_dists.data(), nq * (m + 1));

            CUDA_CHECK(cudaDeviceSynchronize());

            // Map local indices to global and filter self
            for (int i = 0; i < nq; ++i) {
                int qi = q_indices[i];
                std::vector<std::pair<float, int64_t>> neighbors;

                for (int j = 0; j < m + 1; ++j) {
                    int64_t local_idx = h_indices[i * (m + 1) + j];
                    if (local_idx < 0 || local_idx >= nc) continue;
                    int64_t global_idx = cand_indices[local_idx];
                    float dist = h_dists[i * (m + 1) + j];

                    if (global_idx != qi) {  // Skip self
                        neighbors.push_back({dist, global_idx});
                    }
                    if (neighbors.size() >= m) break;
                }

                // Fill result
                for (size_t j = 0; j < neighbors.size(); ++j) {
                    all_neighbors[qi * m + j] = neighbors[j].second;
                    all_dists[qi * m + j] = neighbors[j].first;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU tensor core computation failed: " << e.what() << "\n";
        std::cerr << "Falling back to CPU version...\n";
        return neighbors_within_knn_buckets(X, N, D, buckets, knb, m, B);
    }

    return {all_neighbors, all_dists};
}

/**
 * Find m nearest neighbors within bucket's k nearest buckets
 */
std::pair<std::vector<int64_t>, std::vector<float>>
neighbors_within_knn_buckets(const std::vector<float>& X, int N, int D,
                             const std::vector<std::vector<int32_t>>& buckets,
                             const std::vector<std::vector<int64_t>>& knb,
                             int m, int B) {
    std::vector<int64_t> all_neighbors(static_cast<int64_t>(N) * m, -1);
    std::vector<float> all_dists(static_cast<int64_t>(N) * m, std::numeric_limits<float>::infinity());

    for (int b = 0; b < B; ++b) {
        const auto& q_indices = buckets[b];
        if (q_indices.empty()) continue;

        // Gather candidate indices from b and its k nearest buckets
        std::vector<int32_t> cand_indices;
        for (int nb : knb[b]) {
            cand_indices.insert(cand_indices.end(), buckets[nb].begin(), buckets[nb].end());
        }
        cand_indices.insert(cand_indices.end(), q_indices.begin(), q_indices.end());

        // Remove duplicates
        std::sort(cand_indices.begin(), cand_indices.end());
        cand_indices.erase(std::unique(cand_indices.begin(), cand_indices.end()), cand_indices.end());

        if (cand_indices.empty()) continue;

        // Extract query and candidate data
        int nq = q_indices.size();
        int nc = cand_indices.size();
        std::vector<float> Q(nq * D), Cand(nc * D);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nq; ++i) {
            for (int d = 0; d < D; ++d) {
                Q[i * D + d] = X[q_indices[i] * D + d];
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nc; ++i) {
            for (int d = 0; d < D; ++d) {
                Cand[i * D + d] = X[cand_indices[i] * D + d];
            }
        }

        // Find m+1 nearest neighbors (to filter self-loops)
        auto [indices, dists] = topk_from_A_to_B_cpu(Q, nq, D, Cand, nc, m + 1);

        // Map local indices to global and filter self
        for (int i = 0; i < nq; ++i) {
            int qi = q_indices[i];
            std::vector<std::pair<float, int64_t>> neighbors;

            for (int j = 0; j < m + 1; ++j) {
                int64_t local_idx = indices[i * (m + 1) + j];
                if (local_idx < 0 || local_idx >= nc) continue;
                int64_t global_idx = cand_indices[local_idx];
                float dist = dists[i * (m + 1) + j];

                if (global_idx != qi) {  // Skip self
                    neighbors.push_back({dist, global_idx});
                }
                if (neighbors.size() >= m) break;
            }

            // Fill result
            for (size_t j = 0; j < neighbors.size(); ++j) {
                all_neighbors[qi * m + j] = neighbors[j].second;
                all_dists[qi * m + j] = neighbors[j].first;
            }
        }
    }

    return {all_neighbors, all_dists};
}

// ============== Main ==============

int main(int argc, char** argv) {
    po::options_description desc("Bucket Index Builder");
    desc.add_options()
        ("help", "show help message")
        ("data", po::value<std::string>()->required(), "Path to input data (.fbin/.ibin/.i8bin)")
        ("k", po::value<int>()->required(), "Number of nearest neighbor buckets")
        ("m", po::value<int>()->required(), "Number of nearest neighbors per point")
        ("out_dir", po::value<std::string>()->required(), "Output directory")
        ("seed", po::value<uint32_t>()->default_value(0), "Random seed")
        ("kmeans-iters", po::value<int>()->default_value(5), "K-Means iterations")
        ("no-balance", "Disable bucket rebalancing")
        ("balance-slack", po::value<int>()->default_value(4), "Balance slack")
        ("init-method", po::value<std::string>()->default_value("kmeans"), "Initialization: kmeans, kmeans-fast, or random")
        ("t", po::value<int>()->default_value(1), "Number of rounds")
        // New parameters for data loading and centroid generation
        ("cpu-limit", po::value<size_t>()->default_value(16UL * 1024 * 1024 * 1024),
         "CPU memory limit in bytes (default 16GB)")
        ("gpu-limit", po::value<size_t>()->default_value(4UL * 1024 * 1024 * 1024),
         "GPU memory limit in bytes (default 4GB)")
        ("sample-rate", po::value<float>()->default_value(0.1f),
         "Sampling ratio [0, 1] (default 0.1 = 10%)")
        ("centroid-ratio", po::value<float>()->default_value(0.01f),
         "Centroid ratio relative to full dataset [0, 1] (default 0.01 = 1%)")
        ("use-pq", po::bool_switch()->default_value(false),
         "Force PQ quantization")
        ("pq-bits-start", po::value<uint32_t>()->default_value(8),
         "PQ starting bits [4-8] (default 8)")
        ("pq-bits-min", po::value<uint32_t>()->default_value(4),
         "PQ minimum bits [4-8] (default 4)")
        ("pq-train-fraction", po::value<float>()->default_value(0.05f),
         "PQ training data fraction [0, 1] (default 0.05 = 5%)")
        ("pq-train-max-rows", po::value<uint32_t>()->default_value(65536),
         "PQ training data row limit (default 65536)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::string data_path = vm["data"].as<std::string>();
    int k = vm["k"].as<int>();
    int m = vm["m"].as<int>();
    std::string out_dir = vm["out_dir"].as<std::string>();
    uint32_t seed = vm["seed"].as<uint32_t>();
    int kmeans_iters = vm["kmeans-iters"].as<int>();
    bool balance = !vm.count("no-balance");
    int balance_slack = vm["balance-slack"].as<int>();
    std::string init_method = vm["init-method"].as<std::string>();
    int t = vm["t"].as<int>();

    // Parse data loading parameters
    LoadConfig load_config;
    load_config.cpu_limit_bytes = vm["cpu-limit"].as<size_t>();
    load_config.gpu_limit_bytes = vm["gpu-limit"].as<size_t>();
    load_config.sample_rate = vm["sample-rate"].as<float>();
    load_config.centroid_ratio = vm["centroid-ratio"].as<float>();
    load_config.use_pq = vm["use-pq"].as<bool>();
    load_config.pq_bits_start = vm["pq-bits-start"].as<uint32_t>();
    load_config.pq_bits_min = vm["pq-bits-min"].as<uint32_t>();
    load_config.pq_dim = 0;  // Auto
    load_config.pq_train_fraction = vm["pq-train-fraction"].as<float>();
    load_config.pq_train_max_rows = vm["pq-train-max-rows"].as<uint32_t>();
    load_config.seed = seed;
    load_config.gpu_safety_margin = 0.1f;  // Reserve 10% for system

    // Auto-detect GPU memory if needed
    init_gpu_limit_if_needed(load_config);

    // Validate configuration
    validate_load_config(load_config);

    // Create output directory
    std::filesystem::create_directories(out_dir);

    // Load data
    std::cout << "[1] Loading data from " << data_path << " ...\n";
    int32_t N, D;
    std::vector<float> X;

    // Detect file format by extension
    std::string ext;
    size_t dot_pos = data_path.rfind('.');
    if (dot_pos != std::string::npos) {
        ext = data_path.substr(dot_pos);
    }

    if (ext == ".u8bin" || ext == ".i8bin") {
        load::read_u8bin_to_f32(data_path, X, N, D);
    } else if (ext == ".ibin") {
        std::vector<int32_t> X_int;
        load::read_ibin_i32(data_path, X_int, N, D);
        X.resize(X_int.size());
        std::copy(X_int.begin(), X_int.end(), X.begin());
    } else {
        // Default to .fbin/.bin
        load::read_fbin_f32(data_path, X, N, D);
    }
    std::cout << "Loaded X: shape=(" << N << ", " << D << ")\n\n";

    // Estimate memory requirements
    std::cout << "[1.5] Estimating memory requirements...\n";
    validate_load_config(load_config);
    auto mem_est = estimate_memory_requirement(N, D, load_config);
    print_memory_estimate(mem_est);

    // Generate centroids using GPU KMeans++ with sampling and optional PQ
    std::cout << "[1.7] Generating centroids with GPU KMeans++...\n";
    auto [centroid_indices, centroids] = prepare_and_kmeans_pp_gpu(
        X, N, D, load_config, mem_est);

    // Use generated centroids for bucket building instead of random initialization
    int B = centroids.size() / D;
    std::cout << "Using " << B << " centroids from GPU KMeans++ initialization\n\n";

    // Create bucket structure from centroids
    std::vector<std::vector<int32_t>> buckets(B);

    // Assign all points to nearest centroid
    auto assignments = pairwise_l2_min_assign_cpu(X, N, D, centroids, B);
    for (int64_t i = 0; i < N; ++i) {
        buckets[assignments[i]].push_back(static_cast<int32_t>(i));
    }

    std::cout << "[2] Buckets created with centroid-based assignment\n";

    // Find k nearest buckets
    std::cout << "[3] Finding k=" << k << " nearest buckets ...\n";
    auto knb = k_nearest_buckets(centroids, B, D, k);

    // Find neighbors within buckets
    std::cout << "[4] Finding m=" << m << " nearest neighbors ...\n";
    auto [neighbors, dists] = neighbors_within_knn_buckets_gpu(X, N, D, buckets, knb, m, B);

    // Save results
    std::cout << "[5] Saving results ...\n";
    std::string suffix = "_k" + std::to_string(k) + "m" + std::to_string(m) + "t" + std::to_string(t);
    std::string neighbors_path = out_dir + "/neighbors" + suffix + ".npy";
    std::string dists_path = out_dir + "/neighbors_dist" + suffix + ".npy";

    load::write_npy_int64_2d(neighbors_path, neighbors.data(), N, m);
    load::write_npy_f32_2d(dists_path, dists.data(), N, m);

    std::cout << "Done. Files written to: " << out_dir << "\n";

    return 0;
}
