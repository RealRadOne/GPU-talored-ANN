#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Runner Script for GPANN Benchmarking
Builds bucket2 and bucket_reordered, then executes the benchmark suite.
"""

import os
import sys
import subprocess
import shutil

def run_cmd(cmd: str, cwd: str = None):
    print(f"🚀 [Executing] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    return res

def main():
    print("=" * 60)
    print("  GPANN: Standalone Benchmark Runner")
    print("=" * 60)

    # 1. Ensure Dependencies & RAFT/RMM headers
    if not os.path.exists("/content/raft"):
        run_cmd("git clone --depth 1 -b branch-24.04 https://github.com/rapidsai/raft.git /content/raft")
    if not os.path.exists("/content/rmm"):
        run_cmd("git clone --depth 1 -b branch-24.04 https://github.com/rapidsai/rmm.git /content/rmm")
    if not os.path.exists("/content/cuco"):
        run_cmd("git clone --depth 1 https://github.com/NVIDIA/cuCollections.git /content/cuco")

    # Install apt dependencies
    run_cmd("apt-get update -qq && apt-get install -y -qq libspdlog-dev libboost-program-options-dev libfmt-dev")

    # Set CUDA environment
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

    # 2. Clean & Build
    build_dir = "/content/GPU-talored-ANN/bucketDemo/buildBucket/build"
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = (
        "cmake .. "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++ "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
        "-DCMAKE_CUDA_ARCHITECTURES=75"
    )
    run_cmd(cmake_cmd, cwd=build_dir)
    run_cmd(f"make -j{os.cpu_count() or 4} bucket2 bucket_reordered", cwd=build_dir)

    # 3. Run Benchmark
    benchmark_dir = "/content/GPU-talored-ANN/bucketDemo"
    run_cmd("python3 benchmark.py --sizes 50000 100000 200000 --dim 128 --k 32 --m 32", cwd=benchmark_dir)
    print("\n✅ Benchmark completed successfully! Results saved to output/benchmark/benchmark_results.png")

if __name__ == "__main__":
    main()
