#pragma once

// ============== Phase 2: Sampling Functions ==============

/**
 * Sample using fast random sampling (with replacement)
 */
std::vector<int64_t> sample_without_replacement(
    int64_t N,
    int64_t n_samples,
    uint32_t seed) {

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, N - 1);
    std::vector<int64_t> indices(n_samples);

    // Fast random sampling: directly generate random indices
    for (int64_t i = 0; i < n_samples; ++i) {
        indices[i] = dist(rng);
    }

    return indices;
}

