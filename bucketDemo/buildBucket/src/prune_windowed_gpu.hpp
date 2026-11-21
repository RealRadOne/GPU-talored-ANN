// prune_windowed_gpu.hpp — GPU kernel for prune_windowed's batched path (see prune_windowed_gpu.cu).
#pragma once
#include <cstdint>
#include <vector>
#include <functional>

namespace prune_windowed_gpu {
// 多 GPU 辅助 (prune_windowed.cpp 不直接 include CUDA 头)
int device_count();
void set_device(int dev);
// true if a CUDA device is present, M <= 64 and the working set fits in free GPU memory
// (elem_bytes = sizeof the dataset element: 1 for uint8 / int8, 4 for float)
bool available(int64_t N, int M, int D, int n_buckets, int W, int out_deg, int elem_bytes = 1);
// T = uint8_t / int8_t / float (instantiated in prune_windowed_gpu.cu).
// cand (N,M) int32 (-1 padding), vec (N,D) T, offsets (n_buckets+1) by processing position,
// win (n_buckets, W) sorted window positions per position (padded), win_len (n_buckets).
// Writes out (N, out_deg). Throws on CUDA errors / too few valid candidates.
template <typename T>
void run(const int32_t* cand, const T* vec, int64_t N, int M, int D,
         const uint32_t* offsets, int n_buckets,
         const std::vector<int32_t>& win, const std::vector<int32_t>& win_len, int W,
         int out_deg,
         // 结果按行块交回 (rows 指向 pinned 暂存区，row0 起 n 行，行宽 out_deg)，
         // 依次递增、覆盖 [0, N)；回调返回后暂存区会被复用
         const std::function<void(const uint32_t* rows, int64_t row0, int64_t n)>& sink);
struct ChunkStats { int64_t chunks = 0, loaded_rows = 0; };
// spot (SPOT_DESIGN §4.3): claim(p0, p1) 在块 [p0, p1) 开始前 (false → 不再开新块，已开的块写完后返回)；
// done(p0, p1) 在块的行全部交给 sink 之后 (按块顺序)。
struct ChunkHooks {
    std::function<bool(int p0, int p1)> claim;
    std::function<void(int p0, int p1)> done;
};
// B13: optional reader for the centers' own candidate distances (vector_dists.bin rows, M floats per
// row, same order as the candidate rows). When set, run_chunked ranks each center's candidates by
// these distances (out-of-window candidates get their Step 6 distance instead of +inf).
void set_center_dists(std::function<void(int64_t lo, int64_t n, float* out)> f);
// Same prune, out-of-core: positions are processed in chunks whose window union (rows of the
// buckets in the windows of those positions) fits budget_bytes (cand + vec rows); load_rows(lo, n,
// cand, vec) must fill n consecutive rows (new ids lo..lo+n) of candidates (int32, M per row) and
// T vectors (D per row). Results go to sink in increasing row order like run().
template <typename T>
void run_chunked(int64_t N, int M, int D, const uint32_t* offsets, int n_buckets,
                 const std::vector<int32_t>& win, const std::vector<int32_t>& win_len, int W,
                 int out_deg, size_t budget_bytes,
                 const std::function<void(int64_t lo, int64_t n, int32_t* cand, T* vec)>& load_rows,
                 const std::function<void(const uint32_t* rows, int64_t row0, int64_t n)>& sink,
                 ChunkStats* stats = nullptr,
                 // 多 GPU: 只处理位置 [pos_lo, pos_hi) (行 [offsets[pos_lo], offsets[pos_hi]))；调用方已 cudaSetDevice
                 int pos_lo = 0, int pos_hi = -1,
                 const ChunkHooks* hooks = nullptr);
}
