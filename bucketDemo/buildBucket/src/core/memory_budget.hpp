#pragma once

MemoryEstimate estimate_memory_requirement(
    int64_t N,
    int64_t D,
    int64_t n_centroids,
    const LoadConfig& config,
    const std::string& dtype = "float32") {

    MemoryEstimate est{};

    est.raw_data_rows = N;
    size_t dtype_size = (dtype == "float32") ? 4 : ((dtype == "uint8" || dtype == "int8") ? 1 : 4);
    est.raw_data_bytes = N * D * dtype_size;

    // Calculate KMeans+ working space
    est.kmeans_n_trials = 2 + static_cast<int32_t>(std::ceil(std::log2(std::max(static_cast<int64_t>(2), n_centroids))));
    est.kmeans_working_bytes = static_cast<size_t>(est.kmeans_n_trials) * N * sizeof(float);

    size_t safety_bytes = static_cast<size_t>(config.gpu_limit_bytes * config.gpu_safety_margin);

    // Step 1: Check if full dataset fits in GPU (with KMeans workspace)
    size_t total_with_kmeans = est.raw_data_bytes + est.kmeans_working_bytes + safety_bytes;
    est.fits_in_gpu = total_with_kmeans <= config.gpu_limit_bytes;

    if (est.fits_in_gpu) {
        est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
        est.centroid_data_bytes = est.centroid_rows * D * sizeof(float);
        est.sampled_fits_with_kmeans = true;
        est.need_pq = false;
        return est;
    }

    // Step 2: Try sampling
    est.sampled_data_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.sample_rate));
    est.sampled_data_bytes = est.sampled_data_rows * D * sizeof(float);

    size_t kmeans_working_bytes_sampled =
        static_cast<size_t>(est.kmeans_n_trials) * est.sampled_data_rows * sizeof(float);

    size_t total_sampled = est.sampled_data_bytes + kmeans_working_bytes_sampled + safety_bytes;
    est.sampled_fits_with_kmeans = total_sampled <= config.gpu_limit_bytes;

    if (est.sampled_fits_with_kmeans) {
        est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(est.sampled_data_rows * config.centroid_ratio));
        est.centroid_data_bytes = est.centroid_rows * D * sizeof(float);
        est.need_pq = false;
        return est;
    }

    // Step 3: Sampled data still too large, must use PQ with adaptive compression
    // Strategy: first reduce pq_bits, then reduce pq_dim if needed
    est.need_pq = true;

    // Initial PQ parameters (DiskANN style)
    uint32_t initial_pq_dim = (config.pq_dim == 0) ?
        ((D / 4 + 7) / 8 * 8) : config.pq_dim;

    uint32_t pq_bits = config.pq_bits_start;
    uint32_t pq_dim = initial_pq_dim;
    bool found_valid_config = false;

    // PQ training data: 5% of sampled data, capped at 65536 rows
    est.pq_train_rows = std::min(
        static_cast<int64_t>(est.sampled_data_rows * config.pq_train_fraction),
        static_cast<int64_t>(config.pq_train_max_rows)
    );
    est.pq_train_data_bytes = est.pq_train_rows * D * sizeof(float);

    // Outer loop: try different pq_dim values (smaller pq_dim = more subspaces = better compression)
    // Minimum viable pq_dim is 4 (at least 1 subspace with 4 dims)
    while (pq_dim >= 4 && !found_valid_config) {
        uint32_t n_subspaces = D / pq_dim;

        // Inner loop: try different pq_bits for current pq_dim
        // No longer restricted to config.pq_bits_min - allow flexible range
        pq_bits = config.pq_bits_start;
        while (!found_valid_config) {
            // Codebook size: n_subspaces × 2^pq_bits × pq_dim × sizeof(float)
            size_t codebook_size = static_cast<size_t>(n_subspaces) * (1 << pq_bits) * pq_dim;
            est.pq_codebook_bytes = codebook_size * sizeof(float);

            // Encoded data size: n_sampled × n_subspaces × ceil(pq_bits / 8.0)
            uint32_t bytes_per_code = (pq_bits + 7) / 8;
            est.pq_encoded_data_bytes = static_cast<size_t>(est.sampled_data_rows) * n_subspaces * bytes_per_code;

            // Decoded (reconstructed) data will be float
            size_t pq_decoded_bytes = static_cast<size_t>(est.sampled_data_rows) * D * sizeof(float);

            // Total GPU requirement for PQ pipeline
            size_t total_pq = est.pq_encoded_data_bytes + est.pq_codebook_bytes +
                             pq_decoded_bytes + kmeans_working_bytes_sampled + safety_bytes;

            if (total_pq <= config.gpu_limit_bytes) {
                est.final_pq_bits = pq_bits;
                found_valid_config = true;
                break;
            }

            if (pq_bits <= config.pq_bits_min) {  // Minimum is pq_bits_min bit
                break;
            }
            pq_bits--;
        }

        // If no valid config found with current pq_dim, try larger pq_dim to reduce subspaces
        if (!found_valid_config) {
            // Increase pq_dim to reduce number of subspaces and achieve better compression
            uint32_t new_pq_dim = pq_dim * 2;
            if (new_pq_dim > D || new_pq_dim == pq_dim) {
                break;  // Can't increase further (pq_dim cannot exceed D)
            }
            // Round to power of 2 for efficiency
            uint32_t power = 1;
            while ((power * 2) <= new_pq_dim) power *= 2;
            pq_dim = power;
            // Continue outer loop to retry with new pq_dim and increased pq_bits range
        }
    }

    if (!found_valid_config) {
        std::cerr << "ERROR: Cannot fit data in GPU even with aggressive PQ compression\n";
        std::cerr << "  Tried pq_dim from " << initial_pq_dim << " down to 4\n";
        std::cerr << "  Tried pq_bits from " << config.pq_bits_start << " down to 1\n";
        std::cerr << "  Minimum required: " << (est.pq_encoded_data_bytes + est.pq_codebook_bytes) / 1e9 << " GB\n";
        std::cerr << "  Available: " << config.gpu_limit_bytes / 1e9 << " GB\n";
        throw std::runtime_error("Data too large for GPU memory even with maximum compression");
    }

    est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(est.sampled_data_rows * config.centroid_ratio));
    est.centroid_data_bytes = est.centroid_rows * D * sizeof(float);

    return est;
}

/**
 * Validate LoadConfig parameters
 */
void validate_load_config(const LoadConfig& config, int64_t N) {
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
    // if (config.pq_bits_start < 4 || config.pq_bits_start > 8) {
    //     throw std::runtime_error("pq_bits_start must be in [4, 8]");
    // }
    if (config.pq_bits_start < config.pq_bits_min) {
        throw std::runtime_error("pq_bits_start must be greater than or equal to pq_bits_min");
    }
    if (config.pq_bits_min >= config.pq_bits_start) {
        throw std::runtime_error("pq_bits_min must be < pq_bits_start");
    }

    // Check: knn_k range
    if (config.knn_k == 0 || config.knn_k > LoadConfig::MAX_KNN_K) {
        throw std::runtime_error(
            "knn_k must be in [1, " + std::to_string(LoadConfig::MAX_KNN_K) +
            "], got " + std::to_string(config.knn_k));
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

    std::cout << "\n=== Memory Estimate ===\n";
    std::cout << "Raw data:           " << format_bytes(est.raw_data_bytes) << "\n";
    std::cout << "Fits in GPU:        " << (est.fits_in_gpu ? "YES" : "NO") << "\n";

    if (!est.fits_in_gpu) {
        std::cout << "Sampled data:       " << format_bytes(est.sampled_data_bytes)
                  << " (" << (100.0 * est.sampled_data_rows / est.raw_data_rows) << "%)\n";
        std::cout << "Fits with KMeans:   " << (est.sampled_fits_with_kmeans ? "YES" : "NO") << "\n";

        if (est.need_pq) {
            std::cout << "PQ Quantization:    REQUIRED\n";
            std::cout << "  Final pq_bits:    " << est.final_pq_bits << "\n";
            std::cout << "  Encoded data:     " << format_bytes(est.pq_encoded_data_bytes) << "\n";
            std::cout << "  Codebook:         " << format_bytes(est.pq_codebook_bytes) << "\n";
        }
    }

    std::cout << "KMeans workspace:   " << format_bytes(est.kmeans_working_bytes) << "\n";
    std::cout << "  n_trials:         " << est.kmeans_n_trials << "\n";
    std::cout << "Centroids:          " << est.centroid_rows << " points\n";
    std::cout << "========================\n\n";
}

