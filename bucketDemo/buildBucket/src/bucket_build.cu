#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <random>
#include <limits>
#include <iomanip>
#include <unordered_set>

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
#include "bucket_order.hpp"

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

// ============== Format-aware chunked/sampled loading ==============
// 统一处理 .fbin/.u8bin/.i8bin/.ibin 的分块顺序读取与按下标采样读取，
// 使得质心生成和分桶都不需要把整份数据集常驻内存。

inline std::string detect_ext(const std::string& path) {
    size_t dot_pos = path.rfind('.');
    return (dot_pos != std::string::npos) ? path.substr(dot_pos) : std::string();
}

inline void load_chunk_as_f32(const std::string& path, const std::string& ext,
                              int64_t start, int64_t count,
                              std::vector<float>& out, int32_t& D) {
    if (ext == ".u8bin" || ext == ".i8bin") {
        load::read_u8bin_chunk_to_f32(path, start, count, out, D);
    } else if (ext == ".ibin") {
        load::read_ibin_chunk_to_f32(path, start, count, out, D);
    } else {
        load::read_fbin_chunk(path, start, count, out, D);
    }
}

inline void load_sampled_as_f32(const std::string& path, const std::string& ext,
                                const std::vector<int64_t>& sorted_indices,
                                std::vector<float>& out) {
    int32_t N_tmp, D_tmp;
    if (ext == ".u8bin" || ext == ".i8bin") {
        std::vector<uint8_t> tmp;
        load::read_bigann_raw_sampled<uint8_t>(path, sorted_indices, tmp, N_tmp, D_tmp);
        out.resize(tmp.size());
        std::copy(tmp.begin(), tmp.end(), out.begin());
    } else if (ext == ".ibin") {
        std::vector<int32_t> tmp;
        load::read_bigann_raw_sampled<int32_t>(path, sorted_indices, tmp, N_tmp, D_tmp);
        out.resize(tmp.size());
        std::copy(tmp.begin(), tmp.end(), out.begin());
    } else {
        load::read_fbin_sampled(path, sorted_indices, out, N_tmp, D_tmp);
    }
}

/**
 * A bucket's members loaded on demand: global ids plus their raw vectors
 * (contiguous, ids[i] <-> vecs[i*D .. (i+1)*D)). This is a transient,
 * short-lived object - only bucket_ids[b] (just the int32 ids, see
 * assign_to_buckets_chunked) is kept resident for the life of the program;
 * a Bucket's vecs are materialized on demand (see load_bucket_vectors) by
 * reading straight from the original dataset file, and discarded once used.
 */
struct Bucket {
    std::vector<int32_t> ids;
    std::vector<float> vecs;
};

/**
 * Materialize one bucket's vectors on demand by reading them straight out of
 * the original dataset file (ids must already be known, e.g. from
 * bucket_ids[b] built by assign_to_buckets_chunked). Nothing here is cached
 * or kept beyond the returned Bucket's lifetime.
 */
inline Bucket load_bucket_vectors(const std::vector<int32_t>& ids,
                                  const std::string& data_path, const std::string& ext) {
    Bucket bk;
    bk.ids = ids;  // assign_to_buckets_chunked builds these in ascending global-id order already
    std::vector<int64_t> ids64(ids.begin(), ids.end());
    load_sampled_as_f32(data_path, ext, ids64, bk.vecs);
    return bk;
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
    const std::string& data_path, const std::string& ext,
    int64_t N_full, int D,
    const LoadConfig& config,
    const MemoryEstimate& mem_est) {

    std::cout << "[GPU KMeans++ Centroid Generation]\n";

    // Step 1: Load only what's needed directly from disk (full dataset only
    // when it actually fits the GPU budget; otherwise just the sampled rows)
    std::vector<float> X_work;
    std::vector<int64_t> sampled_indices;
    int64_t N = N_full;

    if (!mem_est.fits_in_gpu) {
        std::cout << "  Sampling " << (config.sample_rate * 100) << "% of data from disk...\n";
        sampled_indices = sample_without_replacement(N_full, mem_est.n_sampled, config.seed);
        std::sort(sampled_indices.begin(), sampled_indices.end());  // sequential reads are faster
        load_sampled_as_f32(data_path, ext, sampled_indices, X_work);
        N = static_cast<int64_t>(sampled_indices.size());
    } else {
        std::cout << "  Loading full dataset (fits in GPU budget)...\n";
        int32_t D_loaded;
        load_chunk_as_f32(data_path, ext, 0, N_full, X_work, D_loaded);
        sampled_indices.resize(N_full);
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
 * Assign the FULL dataset to the nearest centroid, reading it in sequential
 * chunks straight from disk instead of requiring it all resident in CPU RAM
 * up front. Only each point's global id is kept (appended to its bucket's
 * id list) - the vector itself is dropped once the chunk is processed.
 * Step 4 re-reads whichever vectors it needs later, on demand, straight
 * from the original dataset file (see load_bucket_vectors).
 *
 * Peak resident memory here is the transient chunk buffer (bounded by
 * cpu_limit_bytes) plus the id lists themselves, which total N*4 bytes
 * across all buckets - independent of D. For typical embedding dimensions
 * (D=100-1000+) that's a 100-1000x smaller footprint than holding the full
 * N*D*4-byte dataset, which is what actually made Step 1 (loading) and
 * Step 4 (neighbor search) memory-heavy before. It is not literally O(1)
 * in N: a dataset large enough that even N*4 bytes of ids doesn't fit would
 * need a further disk-spilled (two-level radix bucketing) design, which is
 * a fair amount more moving parts for a regime this project isn't in yet.
 */
std::vector<std::vector<int32_t>>
assign_to_buckets_chunked(const std::string& data_path, const std::string& ext,
                          int64_t N, int D,
                          const std::vector<float>& centroids, int B,
                          size_t cpu_limit_bytes) {
    std::vector<std::vector<int32_t>> bucket_ids(B);

    size_t bytes_per_row = static_cast<size_t>(D) * sizeof(float);
    int64_t chunk_rows = static_cast<int64_t>(
        std::max<size_t>(1, (cpu_limit_bytes / 4) / bytes_per_row));
    chunk_rows = std::max<int64_t>(1, std::min(chunk_rows, N));

    std::cout << "  [assign] Chunked assignment: chunk_rows=" << chunk_rows
              << " (~" << (chunk_rows * bytes_per_row / 1e6) << " MB/chunk)\n";

    for (int64_t start = 0; start < N; start += chunk_rows) {
        int64_t cur = std::min(chunk_rows, N - start);

        std::vector<float> X_chunk;
        int32_t D_chunk;
        load_chunk_as_f32(data_path, ext, start, cur, X_chunk, D_chunk);

        auto assign = pairwise_l2_min_assign_cpu(X_chunk, static_cast<int>(cur), D, centroids, B);
        // X_chunk is discarded at the end of this iteration - only the ids survive.

        for (int64_t i = 0; i < cur; ++i) {
            int64_t b = assign[i];
            if (b < 0 || b >= B) continue;
            bucket_ids[b].push_back(static_cast<int32_t>(start + i));
        }

        std::cout << "      Processed " << (start + cur) << "/" << N << " points\n";
    }

    std::vector<int> counts(B);
    for (int b = 0; b < B; ++b) counts[b] = static_cast<int>(bucket_ids[b].size());
    int minc = *std::min_element(counts.begin(), counts.end());
    int maxc = *std::max_element(counts.begin(), counts.end());
    double avg = static_cast<double>(N) / B;
    std::cout << "  [bucket] min=" << minc << "  max=" << maxc << "  avg=" << avg << "\n";

    return bucket_ids;
}

/**
 * CPU brute-force fallback for k_nearest_buckets: one exact B x B distance
 * pass via topk_from_A_to_B_cpu, then self-loop removal.
 */
std::vector<std::vector<int64_t>>
k_nearest_buckets_cpu(const std::vector<float>& centers, int B, int D, int k) {
    auto [indices, dists] = topk_from_A_to_B_cpu(centers, B, D, centers, B, k + 1);

    std::vector<std::vector<int64_t>> result(B);
    for (int b = 0; b < B; ++b) {
        std::vector<int64_t> neighbors;
        for (int j = 0; j < k + 1; ++j) {
            int64_t neighbor = indices[b * (k + 1) + j];
            if (neighbor != b && neighbor >= 0) {
                neighbors.push_back(neighbor);
                if (static_cast<int>(neighbors.size()) >= k) break;
            }
        }
        result[b] = neighbors;
    }
    return result;
}

/**
 * Find k nearest buckets by center distance - GPU, batched over rows.
 *
 * Centroids (B x D) are small and stay fully resident on the GPU for the
 * whole call, but the B x B distance matrix is not computed in one shot:
 * it's built in row-batches sized to keep each batch's distance matrix
 * under ~2GB (same batching pattern as pairwise_l2_min_assign_gpu). That
 * avoids the single dense-B*B-float allocation that made the naive GPU path
 * (topk_from_A_to_B_gpu) infeasible for large B, while doing the O(B^2)
 * distance work at GPU throughput instead of the CPU brute force (which,
 * for large B, was both O(B^2) compute AND heavy on allocation churn from
 * a per-row candidate vector).
 *
 * find_topk_gpu's kernel keeps its running top-k in a fixed 256-slot stack
 * array, so this only takes the GPU path when k+1 <= 256; otherwise it uses
 * the CPU version, which has no such limit.
 */
std::vector<std::vector<int64_t>>
k_nearest_buckets(const std::vector<float>& centers, int B, int D, int k) {
    if (k + 1 > 256) {
        std::cout << "  [KNB] k+1=" << (k + 1)
                  << " exceeds find_topk_gpu's fixed 256-slot buffer; using CPU.\n";
        return k_nearest_buckets_cpu(centers, B, D, k);
    }

    std::vector<std::vector<int64_t>> result(B);

    try {
        std::cout << "  [KNB] Finding k nearest buckets using GPU (batched) ...\n";

        // Keep each batch's [batch_size x B] distance matrix under ~2GB.
        long long max_distance_bytes = 2000000000LL;
        long long calc = max_distance_bytes / (static_cast<long long>(B) * sizeof(float));
        int batch_size = std::max(1, static_cast<int>(std::min<long long>(calc, B)));

        GPUBuffer<float> C_gpu(static_cast<int64_t>(B) * D);
        C_gpu.copy_from_host(centers.data(), static_cast<size_t>(B) * D);

        for (int start = 0; start < B; start += batch_size) {
            int cur = std::min(batch_size, B - start);

            GPUBuffer<float> Q_gpu(static_cast<int64_t>(cur) * D);
            GPUBuffer<float> distances_gpu(static_cast<int64_t>(cur) * B);
            GPUBuffer<int64_t> indices_gpu(static_cast<int64_t>(cur) * (k + 1));
            GPUBuffer<float> dists_gpu(static_cast<int64_t>(cur) * (k + 1));

            Q_gpu.copy_from_host(centers.data() + static_cast<size_t>(start) * D,
                                 static_cast<size_t>(cur) * D);

            compute_distances_gpu(Q_gpu, cur, D, C_gpu, B, distances_gpu);
            find_topk_gpu(distances_gpu, cur, B, k + 1, indices_gpu, dists_gpu);

            std::vector<int64_t> h_indices(static_cast<size_t>(cur) * (k + 1));
            indices_gpu.copy_to_host(h_indices.data(), static_cast<size_t>(cur) * (k + 1));
            CUDA_CHECK(cudaDeviceSynchronize());

            for (int i = 0; i < cur; ++i) {
                int b = start + i;
                std::vector<int64_t> neighbors;
                for (int j = 0; j < k + 1; ++j) {
                    int64_t neighbor = h_indices[i * (k + 1) + j];
                    if (neighbor != b && neighbor >= 0) {
                        neighbors.push_back(neighbor);
                        if (static_cast<int>(neighbors.size()) >= k) break;
                    }
                }
                result[b] = neighbors;
            }

            std::cout << "      Processed " << (start + cur) << "/" << B << " buckets\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU k-nearest-buckets computation failed: " << e.what() << "\n";
        std::cerr << "Falling back to CPU...\n";
        return k_nearest_buckets_cpu(centers, B, D, k);
    }

    return result;
}

/**
 * Bundles the two open output files (neighbors.npy, neighbors_dist.npy) so
 * results can be streamed out row-by-row as they're computed, instead of
 * accumulating N*m-sized arrays in memory. Files are pre-created (header +
 * sentinel-filled) by the caller in main(); this just seeks and writes one
 * row at a time.
 */
struct ResultWriter {
    std::fstream* neighbors_out;
    size_t neighbors_header;
    std::fstream* dists_out;
    size_t dists_header;
    int m;

    void write_row(int64_t qi, const std::vector<std::pair<float, int64_t>>& neighbors) const {
        std::vector<int64_t> row_n(m, -1);
        std::vector<float> row_d(m, std::numeric_limits<float>::infinity());
        for (size_t j = 0; j < neighbors.size() && static_cast<int>(j) < m; ++j) {
            row_n[j] = neighbors[j].second;
            row_d[j] = neighbors[j].first;
        }
        load::write_npy_row_int64(*neighbors_out, neighbors_header, qi, m, row_n.data());
        load::write_npy_row_f32(*dists_out, dists_header, qi, m, row_d.data());
    }
};

// Forward declaration
void neighbors_within_knn_buckets(int64_t N, int D,
                                  const std::vector<std::vector<int32_t>>& bucket_ids,
                                  const std::string& data_path, const std::string& ext,
                                  const std::vector<std::vector<int64_t>>& knb,
                                  int m, int B, const ResultWriter& writer,
                                  const std::vector<int32_t>& order, size_t cache_bytes);

/**
 * Byte footprint of a materialized Bucket, for sizing the Belady cache below.
 */
inline size_t bucket_bytes(const Bucket& bk) {
    return bk.vecs.size() * sizeof(float) + bk.ids.size() * sizeof(int32_t);
}

/**
 * Gather (deduped) candidate ids/vectors for bucket b and its k nearest
 * buckets, routing every per-bucket load through `cache` (a
 * bucket_order::BeladyBucketCache<Bucket>) instead of reading straight from
 * disk each time. Consecutive positions in the processing order tend to
 * share most of their neighbor buckets (that's what
 * compute_bucket_processing_order optimizes for), so most of these end up as
 * cache hits rather than fresh disk reads.
 *
 * Buckets partition all N points, so members never truly overlap across
 * buckets; the `seen` dedup is defensive, not load-bearing (mirrors the
 * pre-cache version's sort+unique pass).
 */
template <typename Cache>
inline Bucket gather_candidates_cached(const std::vector<std::vector<int32_t>>& bucket_ids,
                                       const std::string& data_path, const std::string& ext,
                                       int32_t pos, int b,
                                       const std::vector<int64_t>& neighbor_buckets,
                                       Cache& cache) {
    Bucket merged;
    std::unordered_set<int32_t> seen;

    auto loader = [&](int32_t id) {
        return load_bucket_vectors(bucket_ids[id], data_path, ext);
    };
    auto append = [&](int32_t id) {
        const Bucket& src = cache.get(pos, id, loader);
        if (src.ids.empty()) return;
        int Dloc = static_cast<int>(src.vecs.size() / src.ids.size());
        for (size_t i = 0; i < src.ids.size(); ++i) {
            if (!seen.insert(src.ids[i]).second) continue;
            merged.ids.push_back(src.ids[i]);
            merged.vecs.insert(merged.vecs.end(),
                               src.vecs.begin() + i * Dloc, src.vecs.begin() + (i + 1) * Dloc);
        }
    };

    append(static_cast<int32_t>(b));
    for (int64_t nb : neighbor_buckets) append(static_cast<int32_t>(nb));

    return merged;
}

/**
 * Find m nearest neighbors within bucket's k nearest buckets - GPU Tensor Core version.
 * Out-of-core on both ends: only bucket_ids[b] (int32 ids, N*4 bytes total)
 * is resident on the input side; bucket vectors are loaded on demand through
 * a Belady-optimal cache (see bucket_order.hpp) sized to `cache_bytes`,
 * keyed on the bucket processing `order` (a DiskJoin-style task ordering
 * over the k-nearest-bucket graph `knb` - see compute_bucket_processing_order
 * in main()) so that consecutive buckets share most of their k-nearest
 * neighbor buckets and mostly hit cache instead of re-reading from disk. On
 * the output side, each query's row is written straight to disk via
 * `writer` as soon as it's computed, so there's no N*m-sized result array
 * either.
 */
void neighbors_within_knn_buckets_gpu(int64_t N, int D,
                                      const std::vector<std::vector<int32_t>>& bucket_ids,
                                      const std::string& data_path, const std::string& ext,
                                      const std::vector<std::vector<int64_t>>& knb,
                                      int m, int B, const ResultWriter& writer,
                                      const std::vector<int32_t>& order, size_t cache_bytes) {
    auto adj = bucket_order::adjacency_from_nested(knb);
    auto future_access = bucket_order::compute_future_access_lists(order, adj);
    bucket_order::BeladyBucketCache<Bucket> cache(std::move(future_access), cache_bytes, bucket_bytes);

    try {
        int processed_buckets = 0;
        for (int32_t pos = 0; pos < B; ++pos) {
            int b = order[pos];
            if (bucket_ids[b].empty()) continue;

            if (processed_buckets % 100 == 0) {
                std::cout << "      [Progress] Processed " << processed_buckets << "/" << B
                          << " buckets (cache: " << cache.resident_count() << " buckets, "
                          << cache.used_bytes() / 1e6 << " MB)\n";
            }
            processed_buckets++;

            // Query = this bucket's own vectors, via cache (copied out - the
            // gather_candidates_cached call below issues further cache.get()
            // calls that may evict it otherwise, see BeladyBucketCache's
            // usage contract in bucket_order.hpp).
            Bucket query = cache.get(pos, static_cast<int32_t>(b), [&](int32_t id) {
                return load_bucket_vectors(bucket_ids[id], data_path, ext);
            });
            // Candidates = this bucket + its k nearest buckets, deduped, via cache
            Bucket cand = gather_candidates_cached(bucket_ids, data_path, ext, pos, b, knb[b], cache);
            if (cand.ids.empty()) continue;

            const auto& q_ids = query.ids;
            const auto& cand_ids = cand.ids;
            int nq = static_cast<int>(q_ids.size());
            int nc = static_cast<int>(cand_ids.size());
            const float* Q = query.vecs.data();

            // For small buckets, fall back to CPU
            if (static_cast<int64_t>(nq) * nc < 100000) {
                auto [indices, dists] = topk_from_A_to_B_gpu(query.vecs, nq, D, cand.vecs, nc, m + 1);

                // Map local indices to global and filter self
                for (int i = 0; i < nq; ++i) {
                    int qi = q_ids[i];
                    std::vector<std::pair<float, int64_t>> neighbors;

                    for (int j = 0; j < m + 1; ++j) {
                        int64_t local_idx = indices[i * (m + 1) + j];
                        if (local_idx < 0 || local_idx >= nc) continue;
                        int64_t global_idx = cand_ids[local_idx];
                        float dist = dists[i * (m + 1) + j];

                        if (global_idx != qi) {  // Skip self
                            neighbors.push_back({dist, global_idx});
                        }
                        if (neighbors.size() >= m) break;
                    }

                    writer.write_row(qi, neighbors);
                }
                continue;
            }

            // GPU path for larger buckets
            // Allocate GPU buffers
            GPUBuffer<float> Q_gpu(static_cast<int64_t>(nq) * D);
            GPUBuffer<float> Cand_gpu(static_cast<int64_t>(nc) * D);
            GPUBuffer<float> distances_gpu(static_cast<int64_t>(nq) * nc);
            GPUBuffer<int64_t> indices_gpu(static_cast<int64_t>(nq) * (m + 1));
            GPUBuffer<float> dists_gpu(static_cast<int64_t>(nq) * (m + 1));

            Q_gpu.copy_from_host(Q, static_cast<size_t>(nq) * D);
            Cand_gpu.copy_from_host(cand.vecs.data(), cand.vecs.size());

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
                int qi = q_ids[i];
                std::vector<std::pair<float, int64_t>> neighbors;

                for (int j = 0; j < m + 1; ++j) {
                    int64_t local_idx = h_indices[i * (m + 1) + j];
                    if (local_idx < 0 || local_idx >= nc) continue;
                    int64_t global_idx = cand_ids[local_idx];
                    float dist = h_dists[i * (m + 1) + j];

                    if (global_idx != qi) {  // Skip self
                        neighbors.push_back({dist, global_idx});
                    }
                    if (neighbors.size() >= m) break;
                }

                writer.write_row(qi, neighbors);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "GPU tensor core computation failed: " << e.what() << "\n";
        std::cerr << "Falling back to CPU version...\n";
        neighbors_within_knn_buckets(N, D, bucket_ids, data_path, ext, knb, m, B, writer,
                                     order, cache_bytes);
    }
}

/**
 * Find m nearest neighbors within bucket's k nearest buckets - CPU version.
 * Same out-of-core loading/writing strategy as the GPU version above,
 * including the processing order + Belady cache.
 */
void neighbors_within_knn_buckets(int64_t N, int D,
                                  const std::vector<std::vector<int32_t>>& bucket_ids,
                                  const std::string& data_path, const std::string& ext,
                                  const std::vector<std::vector<int64_t>>& knb,
                                  int m, int B, const ResultWriter& writer,
                                  const std::vector<int32_t>& order, size_t cache_bytes) {
    auto adj = bucket_order::adjacency_from_nested(knb);
    auto future_access = bucket_order::compute_future_access_lists(order, adj);
    bucket_order::BeladyBucketCache<Bucket> cache(std::move(future_access), cache_bytes, bucket_bytes);

    for (int32_t pos = 0; pos < B; ++pos) {
        int b = order[pos];
        if (bucket_ids[b].empty()) continue;

        Bucket query = cache.get(pos, static_cast<int32_t>(b), [&](int32_t id) {
            return load_bucket_vectors(bucket_ids[id], data_path, ext);
        });
        Bucket cand = gather_candidates_cached(bucket_ids, data_path, ext, pos, b, knb[b], cache);
        if (cand.ids.empty()) continue;

        const auto& q_ids = query.ids;
        const auto& cand_ids = cand.ids;
        int nq = static_cast<int>(q_ids.size());
        int nc = static_cast<int>(cand_ids.size());

        // Find m+1 nearest neighbors (to filter self-loops)
        auto [indices, dists] = topk_from_A_to_B_cpu(query.vecs, nq, D, cand.vecs, nc, m + 1);

        // Map local indices to global and filter self
        for (int i = 0; i < nq; ++i) {
            int qi = q_ids[i];
            std::vector<std::pair<float, int64_t>> neighbors;

            for (int j = 0; j < m + 1; ++j) {
                int64_t local_idx = indices[i * (m + 1) + j];
                if (local_idx < 0 || local_idx >= nc) continue;
                int64_t global_idx = cand_ids[local_idx];
                float dist = dists[i * (m + 1) + j];

                if (global_idx != qi) {  // Skip self
                    neighbors.push_back({dist, global_idx});
                }
                if (neighbors.size() >= m) break;
            }

            writer.write_row(qi, neighbors);
        }
    }
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
         "PQ training data row limit (default 65536)")
        // Bucket processing order (DiskJoin-style task ordering, see
        // bucket_order.hpp) for Step 4's load order + cache.
        ("order-window", po::value<int32_t>()->default_value(0),
         "Sliding-window size for bucket processing order (0 = auto: 4*k)")
        ("cache-mb", po::value<size_t>()->default_value(0),
         "Bucket vector cache budget in MB for Step 4 (0 = auto: cpu-limit/2)");

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
    int32_t order_window_arg = vm["order-window"].as<int32_t>();
    size_t cache_mb_arg = vm["cache-mb"].as<size_t>();

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

    // Read just the header - the full dataset is never loaded into one
    // resident array; each stage below pulls only the rows it needs,
    // straight from disk.
    std::cout << "[1] Reading header from " << data_path << " ...\n";
    auto [N, D] = load::read_fbin_header(data_path);
    std::string ext = detect_ext(data_path);
    std::cout << "Dataset shape=(" << N << ", " << D << ")\n\n";

    // Estimate memory requirements
    std::cout << "[1.5] Estimating memory requirements...\n";
    validate_load_config(load_config);
    auto mem_est = estimate_memory_requirement(N, D, load_config);
    print_memory_estimate(mem_est);

    // Generate centroids using GPU KMeans++ with sampling and optional PQ
    // (this reads only the sampled rows it needs from disk, or the full
    // file only when mem_est says it actually fits the GPU budget)
    std::cout << "[1.7] Generating centroids with GPU KMeans++...\n";
    auto [centroid_indices, centroids] = prepare_and_kmeans_pp_gpu(
        data_path, ext, N, D, load_config, mem_est);

    // Use generated centroids for bucket building instead of random initialization
    int B = centroids.size() / D;
    std::cout << "Using " << B << " centroids from GPU KMeans++ initialization\n\n";

    // Assign the full dataset to buckets, streaming it in chunks from disk.
    // Only each point's global id is kept resident (bucket_ids); vectors are
    // re-read on demand in Step 4.
    std::cout << "[2] Assigning full dataset to buckets (chunked) ...\n";
    auto bucket_ids = assign_to_buckets_chunked(
        data_path, ext, N, D, centroids, B, load_config.cpu_limit_bytes);

    // Find k nearest buckets
    std::cout << "[3] Finding k=" << k << " nearest buckets ...\n";
    auto knb = k_nearest_buckets(centroids, B, D, k);

    // Order buckets so Step 4 processes ones with overlapping k-nearest-bucket
    // sets close together (DiskJoin's task ordering, Algorithm 2 - see
    // bucket_order.hpp), then size a Belady-optimal cache for that order.
    std::cout << "[3.5] Computing bucket processing order ...\n";
    int32_t order_window = (order_window_arg > 0) ? order_window_arg : std::max(4 * k, 16);
    auto bucket_process_order = bucket_order::compute_bucket_processing_order(
        bucket_order::adjacency_from_nested(knb), order_window);
    size_t cache_bytes = (cache_mb_arg > 0)
        ? cache_mb_arg * 1024ULL * 1024ULL
        : load_config.cpu_limit_bytes / 2;
    std::cout << "  order_window=" << order_window
              << " cache_budget=" << (cache_bytes / 1e6) << " MB\n";

    // Prepare output files up front: create the .npy files with the right
    // header/shape, then sentinel-fill them in chunks so any row Step 4
    // never touches (shouldn't happen - every point lands in some bucket)
    // still reads back as -1/inf instead of a zeroed hole.
    std::cout << "[5] Preparing output files ...\n";
    std::string suffix = "_k" + std::to_string(k) + "m" + std::to_string(m) + "t" + std::to_string(t);
    std::string neighbors_path = out_dir + "/neighbors" + suffix + ".npy";
    std::string dists_path = out_dir + "/neighbors_dist" + suffix + ".npy";

    size_t neighbors_header = load::create_npy_int64_2d(neighbors_path, N, m);
    size_t dists_header = load::create_npy_f32_2d(dists_path, N, m);

    std::fstream neighbors_out(neighbors_path, std::ios::binary | std::ios::in | std::ios::out);
    std::fstream dists_out(dists_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!neighbors_out.is_open() || !dists_out.is_open()) {
        throw std::runtime_error("Failed to reopen output files for row writes");
    }
    load::prefill_npy_int64_2d(neighbors_out, neighbors_header, N, m, load_config.cpu_limit_bytes / 4);
    load::prefill_npy_f32_2d(dists_out, dists_header, N, m, load_config.cpu_limit_bytes / 4);

    ResultWriter writer{&neighbors_out, neighbors_header, &dists_out, dists_header, m};

    // Find neighbors within buckets - out-of-core end to end: reads each
    // bucket's vectors from data_path on demand (instead of a resident
    // array) and streams each query's result row straight into the files
    // above as soon as it's computed (instead of accumulating an N*m result
    // in memory before writing it all at once). Steps 4 and 5 are
    // interleaved this way by design - there's no point separating "compute
    // everything" from "write everything" once neither side is resident.
    std::cout << "[4] Finding m=" << m << " nearest neighbors (streamed to disk) ...\n";
    neighbors_within_knn_buckets_gpu(N, D, bucket_ids, data_path, ext, knb, m, B, writer,
                                     bucket_process_order, cache_bytes);

    neighbors_out.close();
    dists_out.close();
    std::cout << "Done. Files written to: " << out_dir << "\n";

    return 0;
}
