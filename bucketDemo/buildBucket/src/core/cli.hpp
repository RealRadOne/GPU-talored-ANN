#pragma once

// ============== Phase 5: CLI and Main Integration ==============

/**
 * Parse LoadConfig from command-line arguments
 */
LoadConfig parse_load_config(const po::variables_map& vm) {
    LoadConfig config;

    if (vm.count("cpu-limit")) {
        config.cpu_limit_bytes = vm["cpu-limit"].as<size_t>();
    }
    if (vm.count("gpu-limit")) {
        config.gpu_limit_bytes = vm["gpu-limit"].as<size_t>();
    }
    if (vm.count("sample-rate")) {
        config.sample_rate = vm["sample-rate"].as<float>();
    }
    if (vm.count("centroid-ratio")) {
        config.centroid_ratio = vm["centroid-ratio"].as<float>();
    }
    if (vm.count("use-pq")) {
        config.use_pq = vm["use-pq"].as<bool>();
    }
    if (vm.count("pq-bits-start")) {
        config.pq_bits_start = vm["pq-bits-start"].as<uint32_t>();
    }
    if (vm.count("pq-bits-min")) {
        config.pq_bits_min = vm["pq-bits-min"].as<uint32_t>();
    }
    if (vm.count("seed")) {
        config.seed = vm["seed"].as<uint32_t>();
    }
    if (vm.count("knn-k")) {
        config.knn_k = vm["knn-k"].as<uint32_t>();
        if (config.knn_k > LoadConfig::MAX_KNN_K) {
            throw std::runtime_error(
                "knn-k must be <= " + std::to_string(LoadConfig::MAX_KNN_K) +
                ", got " + std::to_string(config.knn_k));
        }
    }

    return config;
}

