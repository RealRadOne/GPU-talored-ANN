#pragma once

// ============== Phase 11: Bucket Disk Writer ==============

/**
 * 将每个 centroid 的 bucket（包含的数据点 global index）写入磁盘。
 *
 * 磁盘格式（紧凑二进制，方便随机读取单个 bucket）：
 *
 * 文件: <output_dir>/bucket_index.bin
 *   header:
 *     int64_t  n_centroids
 *     int64_t  N   (总数据点数)
 *   body (n_centroids 条记录):
 *     int64_t  offset     // 在 bucket_data.bin 中的字节偏移
 *     int32_t  count      // 该 bucket 包含的点数
 *
 * 文件: <output_dir>/bucket_data.bin
 *   连续存储每个 bucket 的 int32_t point_ids[]
 *   (用 int32 而非 int64，因为 N < 2^31 时节省一半空间)
 *
 * 读取 bucket c 的方法:
 *   1. 从 bucket_index.bin 读取 offset[c] 和 count[c]
 *   2. seek 到 bucket_data.bin 的 offset[c]
 *   3. 读取 count[c] 个 int32_t
 */
void write_buckets_to_disk(
    const std::string& output_dir,
    const std::vector<int64_t>& assignments,
    int64_t N,
    int64_t n_centroids)
{
    std::cout << "[WriteBuckets] Writing to " << output_dir << "\n";

    std::filesystem::create_directories(output_dir);

    // Step 1: Gather points per centroid
    std::vector<std::vector<int32_t>> buckets(n_centroids);
    for (int64_t i = 0; i < N; ++i) {
        int64_t c = assignments[i];
        if (c >= 0 && c < n_centroids) {
            buckets[c].push_back(static_cast<int32_t>(i));
        }
    }

    // Step 2: Write bucket_data.bin (contiguous int32 arrays)
    std::string data_path = output_dir + "/bucket_data.bin";
    std::ofstream data_out(data_path, std::ios::binary);
    if (!data_out.is_open())
        throw std::runtime_error("Cannot open: " + data_path);

    std::vector<int64_t> offsets(n_centroids);
    std::vector<int32_t> counts(n_centroids);
    int64_t current_offset = 0;

    for (int64_t c = 0; c < n_centroids; ++c) {
        offsets[c] = current_offset;
        counts[c] = static_cast<int32_t>(buckets[c].size());

        if (!buckets[c].empty()) {
            data_out.write(reinterpret_cast<const char*>(buckets[c].data()),
                           buckets[c].size() * sizeof(int32_t));
        }
        current_offset += static_cast<int64_t>(buckets[c].size()) * sizeof(int32_t);
    }
    data_out.close();

    // Step 3: Write bucket_index.bin (header + offset/count table)
    std::string index_path = output_dir + "/bucket_index.bin";
    std::ofstream index_out(index_path, std::ios::binary);
    if (!index_out.is_open())
        throw std::runtime_error("Cannot open: " + index_path);

    // Header
    index_out.write(reinterpret_cast<const char*>(&n_centroids), sizeof(int64_t));
    index_out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));

    // Offset + count per centroid
    for (int64_t c = 0; c < n_centroids; ++c) {
        index_out.write(reinterpret_cast<const char*>(&offsets[c]), sizeof(int64_t));
        index_out.write(reinterpret_cast<const char*>(&counts[c]), sizeof(int32_t));
    }
    index_out.close();

    // Stats
    size_t data_bytes = static_cast<size_t>(current_offset);
    size_t index_bytes = sizeof(int64_t) * 2 + n_centroids * (sizeof(int64_t) + sizeof(int32_t));
    std::cout << "[WriteBuckets] Done. data=" << data_bytes / 1e6
              << "MB, index=" << index_bytes / 1e3 << "KB\n";
}

