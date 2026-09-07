#!/usr/bin/env python3
import os
import subprocess

def sh(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def setup_dependencies():
    print("1. Installing system dependencies and cloning RAPIDS/CUDA headers...")
    sh("apt-get update -qq && apt-get install -y -qq libboost-program-options-dev nlohmann-json3-dev")

    repos = [
        ("https://github.com/fmtlib/fmt.git", "10.2.1", "/content/fmt"),
        ("https://github.com/gabime/spdlog.git", "v1.13.0", "/content/spdlog"),
        ("https://github.com/rapidsai/raft.git", "branch-24.04", "/content/raft"),
        ("https://github.com/rapidsai/rmm.git", "branch-24.04", "/content/rmm"),
        ("https://github.com/NVIDIA/cutlass.git", "v2.10.0", "/content/cutlass"),
        ("https://github.com/NVIDIA/cuCollections.git", None, "/content/cuco"),
    ]
    for url, branch, path in repos:
        if not os.path.exists(path):
            b_flag = f"-b {branch}" if branch else ""
            sh(f"git clone --depth 1 {b_flag} {url} {path}")

    # Patch RAFT header for CUTLASS 2.10 compatibility
    raft_hdr = "/content/raft/cpp/include/raft/distance/detail/pairwise_distance_epilogue_elementwise.h"
    if os.path.exists(raft_hdr):
        sh(f"grep -q kIsSingleSource {raft_hdr} || sed -i '/kCount = kElementsPerAccess;/a \\  static bool const kIsSingleSource = true;' {raft_hdr}")

    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

def build_and_benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(base_dir, "bucketDemo/buildBucket/build")
    os.makedirs(build_dir, exist_ok=True)

    print("\n2. Building GPANN binaries...")
    cmake_flags = "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75"
    sh(f"cmake .. {cmake_flags} && make -j1 bucket2 && make -j1 bucket_reordered", cwd=build_dir)

    print("\n3. Running benchmark orchestrator...")
    benchmark_dir = os.path.join(base_dir, "benchmark")
    sh("python3 benchmark.py", cwd=benchmark_dir)

if __name__ == "__main__":
    setup_dependencies()
    build_and_benchmark()

