#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Benchmarking Pipeline for GPANN
Compares Baseline (bucket2) vs Cache-Optimized (bucket_reordered)
"""

import os
import re
import subprocess
import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt

def generate_fbin_if_needed(path: str, N: int, D: int):
    """Creates synthetic vector dataset if not already present."""
    if os.path.exists(path) and os.path.getsize(path) == (8 + N * D * 4):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"  [Data] Generating synthetic test set: {path} ({N:,} vectors, {D}D)...")
    np.random.seed(42)
    data = np.random.randn(N, D).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, D))
        f.write(data.tobytes())

def run_binary(bin_path: str, data_path: str, out_dir: str, k: int, m: int) -> dict:
    """Executes a GPANN C++ binary and parses timing metrics from stdout."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        bin_path,
        "--data", data_path,
        "--output_dir", out_dir,
        "--knn-k", str(k),
        "--neighbors-m", str(m),
        "--iterations", "1"
    ]
    print(f"  [Run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = proc.stdout

    if proc.returncode != 0:
        print(f"❌ Error running {bin_path}:\n{output}")
        return {"total_s": 0.0, "gemm_ms": 0.0, "success": False}

    # Parse timing metrics from output
    gemm_match = re.search(r"pure GEMM:\s*([\d\.]+)\s*ms", output)
    total_match = re.search(r"Total pipeline elapsed:\s*([\d\.]+)\s*s", output)

    gemm_ms = float(gemm_match.group(1)) if gemm_match else 0.0
    total_s = float(total_match.group(1)) if total_match else 0.0

    return {"total_s": total_s, "gemm_ms": gemm_ms, "success": True}

def plot_benchmark_results(sizes, base_totals, opt_totals, base_gemms, opt_gemms, out_path):
    """Generates a clean 2-panel comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    sizes_k = [s // 1000 for s in sizes]

    # Panel 1: Execution Time
    ax1.plot(sizes_k, base_totals, "o-", color="#d9534f", lw=2, label="Baseline (bucket2)")
    ax1.plot(sizes_k, opt_totals, "s-", color="#5cb85c", lw=2, label="Cache-Opt (bucket_reordered)")
    ax1.set_title("Total Indexing Time (Lower is Better)", fontweight="bold")
    ax1.set_xlabel("Dataset Size (K vectors)")
    ax1.set_ylabel("Time (seconds)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Speedup Multipliers
    gemm_speedups = [b / max(1e-6, o) for b, o in zip(base_gemms, opt_gemms)]
    total_speedups = [b / max(1e-6, o) for b, o in zip(base_totals, opt_totals)]

    ax2.bar([f"{s}K" for s in sizes_k], gemm_speedups, width=0.35, color="#0275d8", alpha=0.85, label="GEMM Speedup")
    ax2.plot([f"{s}K" for s in sizes_k], total_speedups, "D-", color="#f0ad4e", lw=2, label="Overall Speedup")
    ax2.axhline(1.0, color="gray", linestyle=":")
    ax2.set_title("Speedup Factor (Higher is Better)", fontweight="bold")
    ax2.set_xlabel("Dataset Size")
    ax2.set_ylabel("Speedup (x)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\n📊 Benchmark comparison graph saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="GPANN Minimal Benchmark Suite")
    parser.add_argument("--sizes", nargs="+", type=int, default=[50000, 100000, 200000])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--build_dir", type=str, default="./buildBucket/build")
    parser.add_argument("--out_dir", type=str, default="./output/benchmark")
    args = parser.parse_args()

    bin_base = os.path.join(args.build_dir, "bucket2")
    bin_opt  = os.path.join(args.build_dir, "bucket_reordered")

    base_totals, opt_totals = [], []
    base_gemms, opt_gemms = [], []

    print("=" * 60)
    print("  GPANN Cache-Locality Benchmark Suite")
    print(f"  Sizes: {args.sizes} | Dim: {args.dim} | K={args.k}, M={args.m}")
    print("=" * 60)

    for N in args.sizes:
        data_path = f"test_data/vectors_{N//1000}k_{args.dim}d.fbin"
        generate_fbin_if_needed(data_path, N, args.dim)

        print(f"\n🚀 Testing N = {N:,} vectors:")
        # 1. Run Baseline
        res_base = run_binary(bin_base, data_path, f"{args.out_dir}/base_{N}", args.k, args.m)
        # 2. Run Cache-Optimized
        res_opt  = run_binary(bin_opt, data_path, f"{args.out_dir}/opt_{N}", args.k, args.m)

        if res_base["success"] and res_opt["success"]:
            base_totals.append(res_base["total_s"])
            opt_totals.append(res_opt["total_s"])
            base_gemms.append(res_base["gemm_ms"])
            opt_gemms.append(res_opt["gemm_ms"])
            sp = res_base["total_s"] / max(1e-6, res_opt["total_s"])
            print(f"  ⏱️ Baseline: {res_base['total_s']:.2f}s | Reordered: {res_opt['total_s']:.2f}s | Speedup: {sp:.2f}x")

    if base_totals:
        plot_benchmark_results(args.sizes, base_totals, opt_totals, base_gemms, opt_gemms, f"{args.out_dir}/benchmark_results.png")

if __name__ == "__main__":
    main()
