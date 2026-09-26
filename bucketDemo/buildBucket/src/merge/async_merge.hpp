#pragma once

// ============== Phase 12: Optional bucket-aligned ID reorder ==============

/**
 * Compute the bucket-aligned permutation: new_id 区间 [offsets[pos], offsets[pos+1])
 * 占据 bucket_order[pos] 这个桶的所有点 —— 也就是说 offsets 是按 `bucket_order`
 * (DiskJoin 风格的 task ordering, 见 bucket_order.hpp) 的顺序排列的，不是按原始
 * 桶编号。optimize_chunked 的 method C/D 只按 offsets 数组顺序把连续几个桶划进
 * 一个 chunk，从不关心某个 chunk 对应哪个原始桶编号，所以这个改动对它完全透明；
 * 而这正是重排真正要解决的问题：只有当"新 ID 空间里相邻"的桶在特征空间里也相邻，
 * chunk 边界才会真的贴合数据分布，method C/D 文档里"相邻 chunk 在特征空间也相邻"
 * 的假设才成立。
 *
 * 输入 assignments (老 ID 空间)，输出:
 *   perm[old_id]         = new_id    (未分配的点为 0xFFFFFFFFu)
 *   inverse_perm[new_id] = old_id
 *   bucket_offsets       = bucket 边界（新 ID 空间，按 bucket_order 排列），长度 n_buckets+1
 */
struct ReorderInfo {
    std::vector<uint32_t> perm;
    std::vector<uint32_t> inverse_perm;
    std::vector<uint32_t> bucket_offsets;
    int64_t total_in_buckets;
};

static ReorderInfo compute_bucket_reorder(
    const std::vector<int64_t>& assignments, int64_t N, int64_t n_buckets,
    const std::vector<int32_t>& bucket_order)
{
    ReorderInfo r;
    r.bucket_offsets.assign(n_buckets + 1, 0);

    // Count per bucket (indexed by raw bucket id, same as `assignments`).
    std::vector<int64_t> counts(n_buckets, 0);
    for (int64_t i = 0; i < N; ++i) {
        int64_t b = assignments[i];
        if (b >= 0 && b < n_buckets) counts[b]++;
    }

    // Lay ranges out in bucket_order sequence (position pos -> raw bucket
    // bucket_order[pos]), not raw bucket id order.
    std::vector<uint32_t> new_id_range_start(n_buckets, 0);
    uint32_t running = 0;
    for (int64_t pos = 0; pos < n_buckets; ++pos) {
        int32_t b = bucket_order[pos];
        new_id_range_start[b] = running;
        running += static_cast<uint32_t>(counts[b]);
        r.bucket_offsets[pos + 1] = running;
    }

    r.total_in_buckets = static_cast<int64_t>(running);
    r.perm.assign(N, 0xFFFFFFFFu);
    r.inverse_perm.assign(r.total_in_buckets, 0);

    std::vector<uint32_t> cursor = new_id_range_start;
    for (int64_t old_id = 0; old_id < N; ++old_id) {
        int64_t b = assignments[old_id];
        if (b >= 0 && b < n_buckets) {
            uint32_t new_id = cursor[b]++;
            r.perm[old_id] = new_id;
            r.inverse_perm[new_id] = static_cast<uint32_t>(old_id);
        }
    }
    return r;
}

/**
 * 写出所有 reorder 产物（与 reorder.cpp 对齐，可被 optimize_chunked --method C/D 直接消费）：
 *   data_reordered.<ext>         重排后的 dataset，保留原始 DataT element type
 *   vector_knn_reordered.bin     重排 + 邻居 ID 重映射后的 KNN 图（仅当 neighbors_m > 0）
 *   bucket_offsets.bin           bucket 边界 (新 ID 空间)
 *   perm.bin / inverse_perm.bin  ID 翻译表 (uint32[N])
 *
 * X_full 现在是 DataT，原本格式直写一遍 row 重排即可（无类型转换）。
 */
template <typename DataT>
static void write_reordered_outputs(
    const std::string& output_dir,
    const std::string& input_ext,
    const std::vector<DataT>& X_full, int64_t /*N*/, int D,
    const std::vector<int32_t>& vector_knn, int M_neighbors,
    const ReorderInfo& r, int64_t n_buckets)
{
    const int64_t N = static_cast<int64_t>(X_full.size() / D);
    const int64_t total = r.total_in_buckets;

    // 1) data_reordered.<ext>：直接 row-level memcpy，element type = DataT
    {
        std::string path = output_dir + "/data_reordered" + input_ext;
        std::ofstream out(path, std::ios::binary);
        int32_t Nh = static_cast<int32_t>(total), Dh = static_cast<int32_t>(D);
        out.write(reinterpret_cast<const char*>(&Nh), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&Dh), sizeof(int32_t));

        std::vector<DataT> reord(static_cast<size_t>(total) * D);
        const size_t row_bytes = static_cast<size_t>(D) * sizeof(DataT);
        #pragma omp parallel for schedule(static)
        for (int64_t new_id = 0; new_id < total; ++new_id) {
            uint32_t old_id = r.inverse_perm[new_id];
            std::memcpy(reord.data() + static_cast<size_t>(new_id) * D,
                        X_full.data() + static_cast<size_t>(old_id) * D,
                        row_bytes);
        }
        out.write(reinterpret_cast<const char*>(reord.data()),
                  static_cast<std::streamsize>(reord.size() * sizeof(DataT)));
        std::cout << "  Wrote " << path
                  << " (" << reord.size() * sizeof(DataT) / 1e9 << " GB)\n";
    }

    // 2) vector_knn_reordered.bin（如果有 vector_knn）
    if (M_neighbors > 0 && !vector_knn.empty()) {
        std::vector<int32_t> reord_knn(static_cast<size_t>(total) * M_neighbors);
        int64_t bad = 0;
        #pragma omp parallel for schedule(static) reduction(+:bad)
        for (int64_t new_id = 0; new_id < total; ++new_id) {
            uint32_t old_id = r.inverse_perm[new_id];
            const int32_t* src = vector_knn.data() + static_cast<size_t>(old_id) * M_neighbors;
            int32_t* dst = reord_knn.data() + static_cast<size_t>(new_id) * M_neighbors;
            for (int k = 0; k < M_neighbors; ++k) {
                int32_t old_nb = src[k];
                if (old_nb < 0) {
                    dst[k] = -1;
                } else if (static_cast<int64_t>(old_nb) >= N
                           || r.perm[old_nb] == 0xFFFFFFFFu) {
                    dst[k] = -1; bad++;
                } else {
                    dst[k] = static_cast<int32_t>(r.perm[old_nb]);
                }
            }
        }
        if (bad > 0)
            std::cout << "  [Warn] " << bad
                      << " neighbor entries → -1 (unassigned/out-of-range)\n";
        write_vector_knn_to_disk(
            output_dir + "/vector_knn_reordered.bin", reord_knn, total, M_neighbors);
    }

    // 3) bucket_offsets.bin / perm.bin / inverse_perm.bin
    {
        std::string p = output_dir + "/bucket_offsets.bin";
        std::ofstream out(p, std::ios::binary);
        int32_t nb32 = static_cast<int32_t>(n_buckets);
        out.write(reinterpret_cast<const char*>(&nb32), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(r.bucket_offsets.data()),
                  static_cast<std::streamsize>(r.bucket_offsets.size() * sizeof(uint32_t)));
        std::cout << "  Wrote " << p << "\n";
    }
    {
        std::ofstream out(output_dir + "/perm.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(r.perm.data()),
                  static_cast<std::streamsize>(r.perm.size() * sizeof(uint32_t)));
    }
    {
        std::ofstream out(output_dir + "/inverse_perm.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(r.inverse_perm.data()),
                  static_cast<std::streamsize>(r.inverse_perm.size() * sizeof(uint32_t)));
    }
    std::cout << "  Wrote perm.bin / inverse_perm.bin\n";
}

// ============== Main ==============

// run_pipeline_impl<DataT> 是真正的工作函数：根据输入文件 element type 实例化一份。
// main() 解析 CLI 后据 ext 分派到对应实例。
//
// iterations > 1: 重复跑 Step 2-6 (centroid 选取/分桶/邻居), 每次 seed 不同;
//                 把 T 次 per-vector KNN 结果按距离 dedupe-merge 输出。
//                 Bucket 文件 (Step 5) 仅保留最后一次 iteration 的结果。
template <typename DataT>
