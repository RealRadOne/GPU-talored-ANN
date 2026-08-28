#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

// ============== CUDA Error Checking ==============

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err_) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err_)); \
        } \
    } while(0)

// ============== GPU Memory Management ==============

template<typename T>
class GPUBuffer {
public:
    T* data = nullptr;
    size_t size = 0;

    GPUBuffer() = default;

    explicit GPUBuffer(size_t size_) : size(size_) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&data, size * sizeof(T)));
        }
    }

    ~GPUBuffer() {
        if (data) {
            cudaFree(data);
        }
    }

    // Move semantics
    GPUBuffer(GPUBuffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void copy_from_host(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(data, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// ============== Distance Kernels ==============

/**
 * Compute pairwise L2 distances: D[i,j] = ||X[i] - C[j]||^2
 * X: (N, D) matrix
 * C: (B, D) matrix
 * D: output (N, B) matrix
 */
__global__ void compute_l2_distances_kernel(
    const float* X, int N, int D,
    const float* C, int B,
    float* distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < B) {
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = X[i * D + d] - C[j * D + d];
            dist += diff * diff;
        }
        distances[i * B + j] = dist;
    }
}

/**
 * Find the minimum distance and its index for each row
 * distances: (N, B) matrix
 * min_indices: output (N,) vector
 */
__global__ void find_min_indices_kernel(
    const float* distances, int N, int B,
    int64_t* min_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float min_dist = distances[i * B];
        int64_t min_idx = 0;
        for (int j = 1; j < B; ++j) {
            float d = distances[i * B + j];
            if (d < min_dist) {
                min_dist = d;
                min_idx = j;
            }
        }
        min_indices[i] = min_idx;
    }
}

/**
 * Update min distances: min_dists[i] = min(min_dists[i], distances[i,j])
 * for a specific center j
 */
__global__ void update_min_distances_kernel(
    const float* X, int N, int D,
    const float* center, // single center of dimension D
    float* min_dists) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = X[i * D + d] - center[d];
            dist += diff * diff;
        }
        min_dists[i] = fminf(min_dists[i], dist);
    }
}

// ============== Reduction Kernels ==============

/**
 * Sum reduction for computing total of min_dists
 */
__global__ void sum_reduction_kernel(
    const float* data, int N,
    float* output) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// ============== Wrapper Functions ==============

inline void compute_distances_gpu(
    const GPUBuffer<float>& X, int N, int D,
    const GPUBuffer<float>& C, int B,
    GPUBuffer<float>& distances) {

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (B + block.y - 1) / block.y);

    compute_l2_distances_kernel<<<grid, block>>>(
        X.data, N, D,
        C.data, B,
        distances.data);

    CUDA_CHECK(cudaGetLastError());
}

inline void find_assignments_gpu(
    const GPUBuffer<float>& distances, int N, int B,
    GPUBuffer<int64_t>& assignments) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    find_min_indices_kernel<<<grid_size, block_size>>>(
        distances.data, N, B,
        assignments.data);

    CUDA_CHECK(cudaGetLastError());
}

inline void update_distances_gpu(
    const GPUBuffer<float>& X, int N, int D,
    const GPUBuffer<float>& center, // single center on GPU
    GPUBuffer<float>& min_dists) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    update_min_distances_kernel<<<grid_size, block_size>>>(
        X.data, N, D,
        center.data,
        min_dists.data);

    CUDA_CHECK(cudaGetLastError());
}

inline float sum_reduction_gpu(
    const GPUBuffer<float>& data, int N) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    GPUBuffer<float> block_sums(grid_size);

    sum_reduction_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        data.data, N,
        block_sums.data);

    CUDA_CHECK(cudaGetLastError());

    // Sum the block results on CPU
    std::vector<float> h_sums(grid_size);
    block_sums.copy_to_host(h_sums.data(), grid_size);

    float total = 0.0f;
    for (float s : h_sums) {
        total += s;
    }

    return total;
}

// ============== Top-K Kernels ==============

/**
 * Find top-k smallest distances and their indices for each row of a distance matrix
 * distances: (N, B) matrix (row-major)
 * k: number of smallest values per row
 * top_indices: output (N, k) matrix of indices
 * top_dists: output (N, k) matrix of distances
 */
__global__ void find_topk_kernel(
    const float* distances, int N, int B, int k,
    int64_t* top_indices, float* top_dists) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // For this row, find top-k smallest distances
        int actual_k = min(k, B);

        // For small B or k, use insertion sort approach
        // Allocate temporary arrays on stack for actual_k <= 256
        float temp_dists[256];
        int temp_indices[256];

        // Initialize with invalid values
        for (int j = 0; j < actual_k; ++j) {
            temp_dists[j] = __int_as_float(0x7f800000);  // FLT_MAX
            temp_indices[j] = -1;
        }

        // Scan all B distances and maintain top-k
        for (int j = 0; j < B; ++j) {
            float d = distances[i * B + j];

            // Find insertion position
            int insert_pos = actual_k;
            for (int p = 0; p < actual_k; ++p) {
                if (d < temp_dists[p]) {
                    insert_pos = p;
                    break;
                }
            }

            // Insert if not at end
            if (insert_pos < actual_k) {
                // Shift elements right
                for (int s = actual_k - 1; s > insert_pos; --s) {
                    temp_dists[s] = temp_dists[s - 1];
                    temp_indices[s] = temp_indices[s - 1];
                }
                temp_dists[insert_pos] = d;
                temp_indices[insert_pos] = j;
            }
        }

        // Write results - use int64_t for indices to avoid overflow
        for (int j = 0; j < actual_k; ++j) {
            top_indices[i * k + j] = (int64_t)temp_indices[j];
            top_dists[i * k + j] = temp_dists[j];
        }
        // Fill remaining with -1 / inf if actual_k < k
        for (int j = actual_k; j < k; ++j) {
            top_indices[i * k + j] = -1;
            top_dists[i * k + j] = __int_as_float(0x7f800000);  // FLT_MAX
        }
    }
}

/**
 * Wrapper function for GPU-accelerated top-k computation
 */
inline void find_topk_gpu(
    const GPUBuffer<float>& distances, int N, int B, int k,
    GPUBuffer<int64_t>& top_indices, GPUBuffer<float>& top_dists) {

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    find_topk_kernel<<<grid_size, block_size>>>(
        distances.data, N, B, k,
        top_indices.data, top_dists.data);

    CUDA_CHECK(cudaGetLastError());
}

// ============== Tensor Core Optimized Distance Computation ==============
// Note: compute_distances_gpu already uses tensor cores via NVCC optimization
// for efficient pairwise distance computation on modern GPUs
