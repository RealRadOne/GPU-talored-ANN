#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Runner Script for GPANN Benchmarking
Automates dependency setup (RAFT, RMM, CUTLASS, cuCollections), builds both targets, and benchmarks.
"""

import os
import sys
import subprocess
import shutil

def run_cmd(cmd: str, cwd: str = None):
    print(f"🚀 [Executing] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    return res

def setup_dependencies():
    print("=" * 60)
    print("  1. Setting up NVIDIA / RAPIDS Dependencies")
    print("=" * 60)
    
    # 1. System packages
    run_cmd("apt-get update -qq && apt-get install -y -qq libboost-program-options-dev libfmt-dev nlohmann-json3-dev libspdlog-dev")

    # 2. NVIDIA / RAPIDS header repositories
    repos = {
        "/content/raft": "https://github.com/rapidsai/raft.git -b branch-24.04",
        "/content/rmm": "https://github.com/rapidsai/rmm.git -b branch-24.04",
        "/content/cutlass": "https://github.com/NVIDIA/cutlass.git -b v3.5.0",
        "/content/cuco": "https://github.com/NVIDIA/cuCollections.git"
    }

    for path, repo in repos.items():
        if not os.path.exists(path):
            print(f"Cloning {path}...")
            run_cmd(f"git clone --depth 1 {repo} {path}")
        else:
            print(f"✅ {path} already exists.")

    # 3. Environment variables for CUDA
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

def build_project():
    print("\n" + "=" * 60)
    print("  2. Building GPANN (bucket2 & bucket_reordered)")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(base_dir, "bucketDemo/buildBucket/build")
    
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
    print("✅ Binaries built successfully!")

def run_benchmark():
    print("\n" + "=" * 60)
    print("  3. Running Cache-Optimization Benchmark")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(base_dir, "bucketDemo")
    run_cmd("python3 benchmark.py --sizes 50000 100000 200000 --dim 128 --k 32 --m 32", cwd=benchmark_dir)
    print("\n🎉 Benchmark completed! Results plot saved to bucketDemo/output/benchmark/benchmark_results.png")

if __name__ == "__main__":
    setup_dependencies()
    build_project()
    run_benchmark()
