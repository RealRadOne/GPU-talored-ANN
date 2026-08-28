#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <random>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <numeric>
#include <iomanip>

#include "load.hpp"

// ======================== Metrics Helpers ========================

float l2sqr(const float* a, const float* b, int D) {
    float diff_sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        float diff = a[i] - b[i];
        diff_sum += diff * diff;
    }
    return diff_sum;
}

// ======================== Graph Search (efSearch style) ========================

struct SearchResult {
    std::vector<int64_t> indices;
    std::vector<float> distances;
};

SearchResult graph_search_single(
    const float* q,                                  // query vector [D]
    const std::vector<float>& base,                  // base vectors [N, D]
    const std::vector<int64_t>& neighbors,           // neighbor graph [N, M]
    int N,                                           // number of base points
    int M,                                           // M neighbors per point
    int D,                                           // dimension
    int topk,
    int ef,
    std::mt19937& rng,
    int num_entry) {

    std::vector<bool> visited(N, false);

    // Priority queue: (distance, node_id)
    std::priority_queue<std::pair<float, int64_t>> cand_heap;
    std::vector<std::pair<float, int64_t>> best_list;

    // Random entry points
    std::uniform_int_distribution<int> dist_entry(0, N - 1);
    std::vector<int> entries_set;
    for (int i = 0; i < num_entry; ++i) {
        int entry = dist_entry(rng);
        if (std::find(entries_set.begin(), entries_set.end(), entry) == entries_set.end()) {
            entries_set.push_back(entry);
        }
    }

    // Initialize with entry points
    for (int entry : entries_set) {
        if (!visited[entry]) {
            float dist = l2sqr(q, base.data() + entry * D, D);
            visited[entry] = true;
            cand_heap.push({-dist, entry});  // negative dist for max-heap as min-heap
            best_list.push_back({dist, entry});
        }
    }

    if (best_list.empty()) {
        return {{}, {}};
    }

    // Heap search
    while (!cand_heap.empty()) {
        auto [neg_dist_u, u] = cand_heap.top();
        float dist_u = -neg_dist_u;
        cand_heap.pop();

        // Check stopping condition
        if (best_list.size() >= ef) {
            float worst_dist = 0.0f;
            for (auto [d, _] : best_list) {
                worst_dist = std::max(worst_dist, d);
            }
            if (dist_u > worst_dist) {
                break;
            }
        }

        // Process neighbors
        for (int j = 0; j < M; ++j) {
            int64_t v = neighbors[u * M + j];
            if (v < 0 || v >= N) continue;
            if (visited[v]) continue;

            visited[v] = true;
            float dv = l2sqr(q, base.data() + v * D, D);
            cand_heap.push({-dv, v});

            // Update best_list
            if (best_list.size() < ef) {
                best_list.push_back({dv, v});
                std::push_heap(best_list.begin(), best_list.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
            } else {
                float worst_dist = best_list[0].first;
                if (dv < worst_dist) {
                    std::pop_heap(best_list.begin(), best_list.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
                    best_list.pop_back();
                    best_list.push_back({dv, v});
                    std::push_heap(best_list.begin(), best_list.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                }
            }
        }
    }

    // Sort best_list by distance
    std::sort(best_list.begin(), best_list.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });

    SearchResult result;
    result.indices.resize(topk, -1);
    result.distances.resize(topk, std::numeric_limits<float>::infinity());

    for (int i = 0; i < std::min(topk, (int)best_list.size()); ++i) {
        result.indices[i] = best_list[i].second;
        result.distances[i] = best_list[i].first;
    }

    return result;
}

// ======================== Recall vs groundtruth ========================

struct RecallResult {
    float recall_first;   // Recall@topk (hit GT[:,0])
    float recall_any;     // Recall@topk (hit any of GT[:,:])
    float recall_overlap; // Recall@topk (mean overlap)
};

RecallResult recall_vs_gt(
    const std::vector<int64_t>& pred,  // [nq, topk]
    const std::vector<int32_t>& gt,    // [nq, k_gt]
    int nq, int topk, int k_gt) {

    int hit_first = 0;
    int hit_any = 0;
    float total_overlap = 0.0f;

    for (int i = 0; i < nq; ++i) {
        // Check if GT[i, 0] is in prediction
        if (gt[i * k_gt] >= 0) {
            for (int j = 0; j < topk; ++j) {
                if (pred[i * topk + j] == gt[i * k_gt]) {
                    hit_first++;
                    break;
                }
            }
        }

        // Check if any GT is in prediction
        std::unordered_set<int32_t> gt_set;
        for (int j = 0; j < k_gt; ++j) {
            if (gt[i * k_gt + j] >= 0) {
                gt_set.insert(gt[i * k_gt + j]);
            }
        }

        if (!gt_set.empty()) {
            int hit_count = 0;
            for (int j = 0; j < topk; ++j) {
                if (gt_set.count(pred[i * topk + j])) {
                    hit_count++;
                }
            }
            if (hit_count > 0) {
                hit_any++;
            }
        }

        // Calculate overlap for this query
        int k = std::min({topk, (int)gt_set.size(), k_gt});
        if (k > 0) {
            std::unordered_set<int64_t> pred_set;
            for (int j = 0; j < k; ++j) {
                if (pred[i * topk + j] >= 0) {
                    pred_set.insert(pred[i * topk + j]);
                }
            }

            int overlap = 0;
            for (const auto& val : pred_set) {
                if (gt_set.count(val)) {
                    overlap++;
                }
            }
            total_overlap += overlap / static_cast<float>(k);
        }
    }

    RecallResult result;
    result.recall_first = hit_first / static_cast<float>(nq > 0 ? nq : 1);
    result.recall_any = hit_any / static_cast<float>(nq > 0 ? nq : 1);
    result.recall_overlap = total_overlap / static_cast<float>(nq > 0 ? nq : 1);

    return result;
}

// ======================== Search with Recall ========================

void search_graph_recall(
    const std::string& base_path,
    const std::string& query_path,
    const std::string& index_path,
    const std::string& gt_path,
    int topk,
    int ef,
    int num_entry,
    bool normalize) {

    std::cout << "[eval] Loading data...\n";

    // Helper lambda to detect and load data
    auto load_data = [](const std::string& path, std::vector<float>& data, int32_t& N, int32_t& D) {
        std::string ext;
        size_t dot_pos = path.rfind('.');
        if (dot_pos != std::string::npos) {
            ext = path.substr(dot_pos);
        }

        if (ext == ".u8bin" || ext == ".i8bin") {
            load::read_u8bin_to_f32(path, data, N, D);
        } else if (ext == ".ibin") {
            std::vector<int32_t> data_int;
            load::read_ibin_i32(path, data_int, N, D);
            data.resize(data_int.size());
            std::copy(data_int.begin(), data_int.end(), data.begin());
        } else {
            // Default to .fbin/.bin
            load::read_fbin_f32(path, data, N, D);
        }
    };

    // Load base
    int32_t N, D;
    std::vector<float> base;
    load_data(base_path, base, N, D);
    std::cout << "  base shape: (" << N << ", " << D << ")\n";

    // Load queries
    int32_t nq, dq;
    std::vector<float> Q;
    load_data(query_path, Q, nq, dq);
    std::cout << "  query shape: (" << nq << ", " << dq << ")\n";

    if (dq != D) {
        throw std::runtime_error("Dimension mismatch between base and query");
    }

    // Load neighbors (index)
    int64_t n_idx, m_idx;
    std::vector<int64_t> neighbors;
    load::read_npy_int64_2d(index_path, neighbors, n_idx, m_idx);
    std::cout << "  index shape: (" << n_idx << ", " << m_idx << ")\n";

    if (n_idx != N) {
        throw std::runtime_error("Index N mismatch with base");
    }

    // Load ground truth
    int32_t gt_nq_tmp, k_gt_tmp;
    std::vector<int32_t> GT;
    load::read_ibin_i32(gt_path, GT, gt_nq_tmp, k_gt_tmp);
    int64_t gt_nq = gt_nq_tmp;
    int64_t k_gt = k_gt_tmp;
    std::cout << "  GT shape: (" << gt_nq << ", " << k_gt << ")\n";

    if (gt_nq != nq) {
        throw std::runtime_error("GT nq mismatch with query");
    }

    // Normalize if needed
    if (normalize) {
        std::cout << "[eval] L2-normalizing base & queries\n";
        auto normalize_rows = [D](std::vector<float>& X, int n) {
            for (int i = 0; i < n; ++i) {
                float norm = 0.0f;
                for (int j = 0; j < D; ++j) {
                    norm += X[i * D + j] * X[i * D + j];
                }
                norm = std::sqrt(norm + 1e-12f);
                for (int j = 0; j < D; ++j) {
                    X[i * D + j] /= norm;
                }
            }
        };
        normalize_rows(base, N);
        normalize_rows(Q, nq);
    }

    // Prepare for search
    std::mt19937 rng(42);
    ef = std::max(ef, topk);
    std::vector<int64_t> preds(nq * topk, -1);

    std::cout << "[eval] start searching ...\n";
    for (int i = 0; i < nq; ++i) {
        if (i % 1000 == 0) {
            std::cout << "  query " << i << "/" << nq << "\n";
        }

        auto result = graph_search_single(
            Q.data() + i * D, base, neighbors, N, m_idx, D, topk, ef, rng, num_entry);

        for (int j = 0; j < topk; ++j) {
            if (j < result.indices.size()) {
                preds[i * topk + j] = result.indices[j];
            }
        }
    }

    // Calculate recall
    RecallResult recall = recall_vs_gt(preds, GT, nq, topk, k_gt);

    std::cout << "\n======== Recall vs GT ========\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Recall@" << topk << " (hit GT[:,0])         : " << recall.recall_first << "\n";
    std::cout << "Recall@" << topk << " (hit any of GT[:,:]) : " << recall.recall_any << "\n";
    std::cout << "Recall@" << topk << " (recall mean overlap): " << recall.recall_overlap << "\n";
}

// ======================== Index Comparison ========================

void compare_indices_setwise(
    const std::string& index_a_path,
    const std::string& index_b_path,
    int k_cmp,
    const std::string& save_csv) {

    // Load indices
    int64_t N_a, Ma;
    std::vector<int64_t> A;
    load::read_npy_int64_2d(index_a_path, A, N_a, Ma);

    int64_t N_b, Mb;
    std::vector<int64_t> B;
    load::read_npy_int64_2d(index_b_path, B, N_b, Mb);

    if (N_a != N_b) {
        throw std::runtime_error("N mismatch between indices");
    }

    int64_t N = N_a;
    int64_t k = k_cmp > 0 ? static_cast<int64_t>(k_cmp) : std::min(Ma, Mb);
    k = std::min(k, std::min(Ma, Mb));

    if (k <= 0) {
        throw std::runtime_error("k_cmp must be > 0");
    }

    int hits_any = 0;
    int set_equal = 0;
    float total_overlap = 0.0f;

    for (int64_t i = 0; i < N; ++i) {
        std::unordered_set<int64_t> SA, SB;

        for (int64_t j = 0; j < k && j < Ma; ++j) {
            if (A[i * Ma + j] >= 0) {
                SA.insert(A[i * Ma + j]);
            }
        }

        for (int64_t j = 0; j < k && j < Mb; ++j) {
            if (B[i * Mb + j] >= 0) {
                SB.insert(B[i * Mb + j]);
            }
        }

        // Calculate intersection
        int inter_sz = 0;
        for (const auto& val : SA) {
            if (SB.count(val)) {
                inter_sz++;
            }
        }

        float overlap = inter_sz / static_cast<float>(k);
        total_overlap += overlap;

        if (inter_sz >= 1) {
            hits_any++;
        }

        if (SA.size() == k && SB.size() == k && inter_sz == k) {
            set_equal++;
        }
    }

    float mean_overlap = total_overlap / static_cast<float>(N > 0 ? N : 1);
    float hits_any_rate = hits_any / static_cast<float>(N > 0 ? N : 1);
    float set_equal_rate = set_equal / static_cast<float>(N > 0 ? N : 1);

    std::cout << "\n======= Index Comparison (set-wise) =======\n";
    std::cout << "N=" << N << ", k=" << k << " (A:" << Ma << ", B:" << Mb << ")\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "MeanOverlap@" << k << " (|A∩B|/k) : " << mean_overlap << "\n";
    std::cout << "HitsAny@" << k << "               : " << hits_any_rate << "\n";
    std::cout << "SetEqual@" << k << "              : " << set_equal_rate << "\n";
}

// ======================== CLI ========================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  search    Run graph search and evaluate recall\n";
    std::cout << "  compare   Compare two indices (set-wise)\n\n";
    std::cout << "Search options:\n";
    std::cout << "  --base <path>       Path to base vectors (.fbin)\n";
    std::cout << "  --query <path>      Path to query vectors (.fbin)\n";
    std::cout << "  --index <path>      Path to index (.npy)\n";
    std::cout << "  --gt <path>         Path to ground truth (.ibin)\n";
    std::cout << "  --topk <int>        Return top-k results (default: 10)\n";
    std::cout << "  --ef <int>          efSearch breadth (default: 200)\n";
    std::cout << "  --num-entry <int>   Number of entry points (default: 4)\n";
    std::cout << "  --normalize         L2-normalize for cosine search\n\n";
    std::cout << "Compare options:\n";
    std::cout << "  --index1 <path>     First index (.npy)\n";
    std::cout << "  --index2 <path>     Second index (.npy)\n";
    std::cout << "  --k-cmp <int>       Number of neighbors to compare\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string cmd = argv[1];

        if (cmd == "search") {
            std::string base_path, query_path, index_path, gt_path;
            int topk = 10;
            int ef = 200;
            int num_entry = 4;
            bool normalize = false;

            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--base" && i + 1 < argc) base_path = argv[++i];
                else if (arg == "--query" && i + 1 < argc) query_path = argv[++i];
                else if (arg == "--index" && i + 1 < argc) index_path = argv[++i];
                else if (arg == "--gt" && i + 1 < argc) gt_path = argv[++i];
                else if (arg == "--topk" && i + 1 < argc) topk = std::stoi(argv[++i]);
                else if (arg == "--ef" && i + 1 < argc) ef = std::stoi(argv[++i]);
                else if (arg == "--num-entry" && i + 1 < argc) num_entry = std::stoi(argv[++i]);
                else if (arg == "--normalize") normalize = true;
            }

            if (base_path.empty() || query_path.empty() || index_path.empty() || gt_path.empty()) {
                std::cerr << "Error: missing required arguments for search\n";
                return 1;
            }

            search_graph_recall(base_path, query_path, index_path, gt_path, topk, ef, num_entry, normalize);

        } else if (cmd == "compare") {
            std::string index1_path, index2_path;
            int k_cmp = -1;
            std::string save_csv;

            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--index1" && i + 1 < argc) index1_path = argv[++i];
                else if (arg == "--index2" && i + 1 < argc) index2_path = argv[++i];
                else if ((arg == "--k-cmp" || arg == "--k_cmp") && i + 1 < argc) k_cmp = std::stoi(argv[++i]);
                else if ((arg == "--save-csv" || arg == "--save_csv") && i + 1 < argc) save_csv = argv[++i];
                else {
                    std::cerr << "Warning: unknown argument: " << arg << "\n";
                }
            }

            if (index1_path.empty() || index2_path.empty()) {
                std::cerr << "Error: missing required arguments for compare\n";
                return 1;
            }

            compare_indices_setwise(index1_path, index2_path, k_cmp, save_csv);

        } else {
            std::cerr << "Unknown command: " << cmd << "\n";
            print_usage(argv[0]);
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
