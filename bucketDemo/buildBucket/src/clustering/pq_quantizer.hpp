#pragma once

// ============== Phase 3: PQ Quantization (DiskANN style) ==============

/**
 * Simple quantizer using per-dimension quantization
 * (Simplified version for quick deployment)
 */
struct SimpleQuantizer {
    std::vector<float> scale;        // [D]
    std::vector<float> offset;       // [D]
    uint32_t bits;
};

/**
 * Train simple quantizer from sampled data
 */
SimpleQuantizer train_simple_quantizer(
    const std::vector<float>& X,
    int64_t N, int D,
    uint32_t bits) {

    SimpleQuantizer q;
    q.bits = bits;
    q.scale.resize(D);
    q.offset.resize(D);

    uint32_t n_levels = 1 << bits;

    #pragma omp parallel for
    for (int d = 0; d < D; ++d) {
        float min_val = X[d];
        float max_val = X[d];

        for (int64_t i = 0; i < N; ++i) {
            float val = X[i * D + d];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        q.offset[d] = min_val;
        q.scale[d] = (max_val - min_val) / (n_levels - 1);
        if (q.scale[d] < 1e-6f) q.scale[d] = 1.0f;
    }

    return q;
}

/**
 * Encode float data to uint8 codes using simple quantizer
 */
std::vector<uint8_t> simple_encode(
    const std::vector<float>& X,
    int64_t N, int D,
    const SimpleQuantizer& q) {

    std::vector<uint8_t> codes(N * D);
    uint32_t n_levels = 1 << q.bits;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            float val = X[i * D + d];
            float normalized = (val - q.offset[d]) / q.scale[d];
            normalized = std::max(0.0f, std::min((float)(n_levels - 1), normalized));
            codes[i * D + d] = static_cast<uint8_t>(normalized);
        }
    }

    return codes;
}

/**
 * Decode uint8 codes back to float data
 */
std::vector<float> simple_decode(
    const std::vector<uint8_t>& codes,
    int64_t N, int D,
    const SimpleQuantizer& q) {

    std::vector<float> X_decoded(N * D);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            uint8_t code = codes[i * D + d];
            X_decoded[i * D + d] = q.offset[d] + code * q.scale[d];
        }
    }

    return X_decoded;
}

