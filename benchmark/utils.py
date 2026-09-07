import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def generate_fbin_if_needed(path: str, N: int, D: int):
    if os.path.exists(path) and os.path.getsize(path) == (8 + N * D * 4):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Generating synthetic test set: {path} ({N:,} vectors, {D}D)...")
    np.random.seed(42)
    data = np.random.randn(N, D).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        f.write(data.tobytes())

def plot_benchmark_results(dataset_names, base_totals, opt_totals, base_gemms, opt_gemms, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    x = np.arange(len(dataset_names))
    width = 0.35

    # Panel 1: Indexing Time Comparison
    ax1.bar(x - width/2, base_totals, width, color="#d9534f", alpha=0.85, label="Baseline (bucket2)")
    ax1.bar(x + width/2, opt_totals, width, color="#5cb85c", alpha=0.85, label="Cache-Opt (bucket_reordered)")
    ax1.set_title("Total Indexing Time (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Speedup Multipliers
    gemm_speedups = [b / max(1e-6, o) for b, o in zip(base_gemms, opt_gemms)]
    total_speedups = [b / max(1e-6, o) for b, o in zip(base_totals, opt_totals)]

    ax2.bar(x - width/2, gemm_speedups, width, color="#0275d8", alpha=0.85, label="GEMM Speedup")
    ax2.bar(x + width/2, total_speedups, width, color="#f0ad4e", alpha=0.85, label="Overall Speedup")
    ax2.axhline(1.0, color="gray", linestyle=":")
    ax2.set_title("Speedup Factor (Higher is Better)", fontweight="bold")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Benchmark comparison graph saved to: {out_path}")


def plot_cache_misses(hw_metrics: dict, algo_metrics: dict, out_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    if hw_metrics:
        labels = list(hw_metrics.keys())
        x = np.arange(len(labels))
        width = 0.35

        l2_base = [hw_metrics[k].get("base_l2_hit", 0.0) for k in labels]
        l2_opt = [hw_metrics[k].get("opt_l2_hit", 0.0) for k in labels]

        ax1.bar(x - width/2, l2_base, width, label="Baseline L2 Hit %", color="#d9534f", alpha=0.85)
        ax1.bar(x + width/2, l2_opt, width, label="Cache-Opt L2 Hit %", color="#5cb85c", alpha=0.85)
        ax1.set_ylabel("Cache Hit Rate (%)")
        ax1.set_title("Hardware GPU L2 Cache Hit Rate", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

    if algo_metrics:
        capacities = algo_metrics.get("capacities", [])
        raw_misses = algo_metrics.get("raw_order_miss_rate", [])
        reorder_misses = algo_metrics.get("reordered_miss_rate", [])
        belady_misses = algo_metrics.get("belady_miss_rate", [])

        if raw_misses:
            ax2.plot(capacities, raw_misses, "x--", color="#d9534f", label="Raw Order (LRU)")
        if reorder_misses:
            ax2.plot(capacities, reorder_misses, "s-", color="#f0ad4e", label="Reordered (LRU)")
        if belady_misses:
            ax2.plot(capacities, belady_misses, "o-", color="#5cb85c", label="Reordered (Belady OPT)")

        ax2.set_xlabel("Cache Capacity (Buckets)")
        ax2.set_ylabel("Miss Rate (%)")
        ax2.set_title("Algorithmic Miss Rate vs Cache Budget", fontweight="bold")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Cache miss visualization saved to: {out_path}")
