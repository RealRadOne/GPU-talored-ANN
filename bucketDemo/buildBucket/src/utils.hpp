#pragma once

#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace bucket {

// ============== Random utilities ==============

/**
 * Seed-controlled random number generation for reproducibility
 */
class RandomGenerator {
    std::mt19937 gen;

public:
    explicit RandomGenerator(uint32_t seed = 0) : gen(seed) {}

    // Generate random integer in [0, max)
    uint32_t randint(uint32_t max) {
        std::uniform_int_distribution<uint32_t> dis(0, max - 1);
        return dis(gen);
    }

    // Generate random uniform float in [0, 1)
    float uniform() {
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        return dis(gen);
    }

    // Generate random permutation indices [0, n)
    std::vector<uint32_t> randperm(uint32_t n) {
        std::vector<uint32_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), gen);
        return perm;
    }
};

// ============== Math utilities ==============

/**
 * Squared L2 distance between two vectors
 */
inline float l2_distance_sq(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * Compute squared norms of all vectors
 * norms: output array of size N
 * vectors: N x D matrix in row-major layout
 */
inline void compute_norms_sq(std::vector<float>& norms,
                              const std::vector<float>& vectors,
                              int N, int D) {
    norms.resize(N);
    #pragma omp parallel for collapse(1) schedule(static)
    for (int i = 0; i < N; ++i) {
        float norm = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = vectors[i * D + j];
            norm += v * v;
        }
        norms[i] = norm;
    }
}

/**
 * Simple argmin for finding closest center
 */
inline int argmin_vec(const std::vector<float>& values) {
    if (values.empty()) return -1;
    int idx = 0;
    float minval = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] < minval) {
            minval = values[i];
            idx = static_cast<int>(i);
        }
    }
    return idx;
}

// ============== Vector utilities ==============

/**
 * Find top-k indices with smallest values
 * values: input values
 * k: number of smallest values to find
 * Returns: pair of (indices, values) both of size k
 */
inline std::pair<std::vector<int>, std::vector<float>>
topk_smallest(const std::vector<float>& values, int k) {
    if (k > static_cast<int>(values.size())) {
        k = values.size();
    }

    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort: find top-k smallest
    std::nth_element(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&values](int a, int b) { return values[a] < values[b]; }
    );

    std::vector<int> topk_indices(indices.begin(), indices.begin() + k);
    std::vector<float> topk_values;
    topk_values.reserve(k);
    for (int idx : topk_indices) {
        topk_values.push_back(values[idx]);
    }

    // Sort results by value
    std::sort(topk_indices.begin(), topk_indices.end(),
              [&values](int a, int b) { return values[a] < values[b]; });
    topk_values.clear();
    for (int idx : topk_indices) {
        topk_values.push_back(values[idx]);
    }

    return {topk_indices, topk_values};
}

// ============== Printing utilities ==============

inline void print_bucket_stats(const std::vector<int>& assignment, int B, int target) {
    std::vector<int> counts(B, 0);
    int N = assignment.size();

    for (int i = 0; i < N; ++i) {
        if (assignment[i] >= 0 && assignment[i] < B) {
            counts[assignment[i]]++;
        }
    }

    int minc = *std::min_element(counts.begin(), counts.end());
    int maxc = *std::max_element(counts.begin(), counts.end());
    float avg = static_cast<float>(N) / B;

    double var = 0.0;
    for (int c : counts) {
        double diff = c - avg;
        var += diff * diff;
    }
    double std = std::sqrt(var / B);

    int empty = std::count(counts.begin(), counts.end(), 0);
    int within4 = std::count_if(counts.begin(), counts.end(),
                                 [target](int c) { return std::abs(c - target) <= 4; });

    std::cout << "  [bucket] min=" << minc << "  max=" << maxc
              << "  avg=" << avg << "  std=" << std
              << "  empty=" << empty
              << "  ±4=" << (within4 * 100.0f / B) << "%\n";
}

} // namespace bucket
