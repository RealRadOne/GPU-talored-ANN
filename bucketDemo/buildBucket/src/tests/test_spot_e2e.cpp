// 端到端: spot_sim.py run (真实 mmap 状态文件 + SIGUSR1) + 这个进程里的段调度器 (模拟 worker，不用 GPU)。
// 用法: BUCKET2_SPOT_DIR=<dir> ./test_spot_e2e <G> <n_units> <us_per_unit>
#include "../spot_sched.hpp"

#include <deque>

using namespace spot;

int main(int argc, char** argv) {
    Client::get().init();   // 必须最先: 屏蔽 SIGUSR1、映射状态文件、登记 pid
    if (!Client::get().enabled()) { std::cerr << "set BUCKET2_SPOT_DIR\n"; return 2; }
    const int G = argc > 1 ? std::atoi(argv[1]) : 4;
    const int64_t n = argc > 2 ? std::atoll(argv[2]) : 200000;
    const int us = argc > 3 ? std::atoi(argv[3]) : 50;
    bool shared = false;
    auto ws = make_workers(G, &shared);
    auto opt = options_from_env(shared);
    std::vector<int64_t> cuts{0};
    std::vector<int> owners;
    std::vector<int> sw;
    for (size_t i = 0; i < ws.size(); ++i) if (!ws[i].ondemand) sw.push_back(static_cast<int>(i));
    for (size_t k = 0; k < sw.size(); ++k) { cuts.push_back(n * static_cast<int64_t>(k + 1) / static_cast<int64_t>(sw.size())); owners.push_back(sw[k]); }
    SegmentScheduler S("e2e", n, ws, cuts, owners, opt);
    std::vector<std::atomic<int>> cnt(static_cast<size_t>(n));
    for (auto& c : cnt) c = 0;
    std::vector<std::thread> th;
    for (size_t w = 0; w < ws.size(); ++w)
        th.emplace_back([&, w] {
            S.run_worker(static_cast<int>(w), [] { std::this_thread::sleep_for(std::chrono::milliseconds(200)); },   // 冷启动
                [&](const Lease& L) {
                    std::deque<std::pair<int64_t, int64_t>> fl;
                    auto pop = [&] {
                        auto [a, b] = fl.front(); fl.pop_front();
                        std::this_thread::sleep_for(std::chrono::microseconds(us * (b - a)));
                        S.commit(L, b, [&] { for (int64_t i = a; i < b; ++i) cnt[static_cast<size_t>(i)]++; });
                    };
                    for (int64_t u = L.begin;;) {
                        const int64_t b = std::min<int64_t>(S.end_of(L), u + 64);
                        if (u >= b || !S.may_start(L, u, b)) break;
                        fl.emplace_back(u, b); u = b;
                        if (fl.size() >= 4) pop();
                    }
                    while (!fl.empty()) pop();
                },
                [] {});
        });
    for (auto& t : th) t.join();
    int64_t bad = 0;
    for (auto& c : cnt) if (c != 1) ++bad;
    std::string why;
    const bool ok = S.verify_complete(&why) && bad == 0;
    S.print_stats(std::cout);
    std::cout << (ok ? "E2E PASS" : "E2E FAIL") << " bad_units=" << bad << " " << why << "\n";
    return ok ? 0 : 1;
}
