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
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    captured_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        print("\n" + "=" * 60)
        print("❌ COMMAND FAILED!")
        print("=" * 60)
        error_lines = [
            l.strip() for l in captured_lines
            if any(k in l.lower() for k in ["error:", "fatal:", "undefined reference", "killed", "signal 9", "cannot find"])
        ]
        if error_lines:
            print("\n🚨 Extracted Compiler / Linker Errors:")
            for err in error_lines[-30:]:
                print(f"  ▶ {err}")
            print("=" * 60)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc

def setup_dependencies():
    print("=" * 60)
    print("  1. Setting up NVIDIA / RAPIDS Dependencies")
    print("=" * 60)
    
    # 1. System packages
    run_cmd("apt-get update -qq && apt-get install -y -qq libboost-program-options-dev libfmt-dev nlohmann-json3-dev libspdlog-dev")

    # 2. NVIDIA / RAPIDS header repositories
    # Note: RAFT 24.04 officially targets CUTLASS v2.10.0. CUTLASS v3+ added
    # OutputOp_::kIsSingleSource to EpilogueWithBroadcast, which is missing in RAFT 24.04's
    # PairwiseDistanceEpilogueElementwise and causes build failure.
    repos = {
        "/content/fmt": "https://github.com/fmtlib/fmt.git -b 10.2.1",
        "/content/spdlog": "https://github.com/gabime/spdlog.git -b v1.13.0",
        "/content/raft": "https://github.com/rapidsai/raft.git -b branch-24.04",
        "/content/rmm": "https://github.com/rapidsai/rmm.git -b branch-24.04",
        "/content/cutlass": "https://github.com/NVIDIA/cutlass.git -b v2.10.0",
        "/content/cuco": "https://github.com/NVIDIA/cuCollections.git"
    }

    # Clean up existing CUTLASS if it was previously cloned as v3.x
    cutlass_dir = "/content/cutlass"
    epilogue_hdr = os.path.join(cutlass_dir, "include/cutlass/epilogue/threadblock/epilogue_with_broadcast.h")
    if os.path.exists(epilogue_hdr):
        try:
            with open(epilogue_hdr, "r", errors="ignore") as f:
                if "kIsSingleSource" in f.read():
                    print("⚠️ Incompatible CUTLASS v3+ detected in /content/cutlass. Removing to re-clone v2.10.0...")
                    shutil.rmtree(cutlass_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning checking cutlass version: {e}")

    # Clean up existing spdlog if it was previously cloned as v1.12.0 (lacks nvcc/fmt constexpr fix #2901)
    spdlog_dir = "/content/spdlog"
    spdlog_hdr = os.path.join(spdlog_dir, "include/spdlog/common.h")
    if os.path.exists(spdlog_hdr):
        try:
            with open(spdlog_hdr, "r", errors="ignore") as f:
                if "SPDLOG_CONSTEXPR_FUNC FMT_CONSTEXPR" not in f.read():
                    print("⚠️ Incompatible spdlog v1.12 detected in /content/spdlog. Removing to re-clone v1.13.0...")
                    shutil.rmtree(spdlog_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning checking spdlog version: {e}")

    for path, repo in repos.items():
        if not os.path.exists(path):
            print(f"Cloning {path}...")
            run_cmd(f"git clone --depth 1 {repo} {path}")
        else:
            print(f"✅ {path} already exists.")

    # Additional safeguard: ensure RAFT's PairwiseDistanceEpilogueElementwise defines kIsSingleSource
    raft_epilogue_hdr = "/content/raft/cpp/include/raft/distance/detail/pairwise_distance_epilogue_elementwise.h"
    if os.path.exists(raft_epilogue_hdr):
        try:
            with open(raft_epilogue_hdr, "r") as f:
                raft_hdr_content = f.read()
            if "kIsSingleSource" not in raft_hdr_content:
                target_str = "static int const kCount             = kElementsPerAccess;"
                if target_str in raft_hdr_content:
                    raft_hdr_content = raft_hdr_content.replace(
                        target_str,
                        f"{target_str}\n  static bool const kIsSingleSource   = true;"
                    )
                    with open(raft_epilogue_hdr, "w") as f:
                        f.write(raft_hdr_content)
                    print("✅ Applied compatibility patch: Defined kIsSingleSource in RAFT PairwiseDistanceEpilogueElementwise.")
        except Exception as e:
            print(f"Warning patching RAFT header: {e}")

    # Additional safeguard: ensure spdlog's common.h doesn't define SPDLOG_CONSTEXPR_FUNC as constexpr under nvcc
    if os.path.exists(spdlog_hdr):
        try:
            with open(spdlog_hdr, "r") as f:
                spd_content = f.read()
            if "SPDLOG_CONSTEXPR_FUNC FMT_CONSTEXPR" not in spd_content and "#        define SPDLOG_CONSTEXPR_FUNC constexpr" in spd_content:
                spd_content = spd_content.replace(
                    "#        define SPDLOG_CONSTEXPR_FUNC constexpr",
                    "#        define SPDLOG_CONSTEXPR_FUNC inline"
                )
                with open(spdlog_hdr, "w") as f:
                    f.write(spd_content)
                print("✅ Applied compatibility patch: Inlined SPDLOG_CONSTEXPR_FUNC in spdlog/common.h.")
        except Exception as e:
            print(f"Warning patching spdlog header: {e}")

    # Build & install fmt 10.2.1 to /usr/local so both header-only and static linking are guaranteed compatible
    fmt_dir = "/content/fmt"
    if os.path.exists(fmt_dir):
        fmt_lib = os.path.join(fmt_dir, "build/libfmt.a")
        if not os.path.exists(fmt_lib):
            try:
                print("📦 Building & installing fmt 10.2.1 to /usr/local...")
                run_cmd("cmake -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DFMT_DOC=OFF -DFMT_TEST=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && cmake --build build -j2 && cmake --install build", cwd=fmt_dir)
                print("✅ fmt 10.2.1 installed to /usr/local.")
            except Exception as e:
                print(f"Warning building fmt: {e}")

    # 3. Environment variables for CUDA
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

def build_project():
    print("\n" + "=" * 60)
    print("  2. Building GPANN (bucket2 & bucket_reordered)")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(base_dir, "bucketDemo/buildBucket/build")
    
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = (
        "cmake .. "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++ "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
        "-DCMAKE_CUDA_ARCHITECTURES=75"
    )
    run_cmd(cmake_cmd, cwd=build_dir)
    # Build targets sequentially to prevent memory exhaustion (OOM) on 2-vCPU / 12GB instances
    print("🔨 Building target 'bucket2'...")
    run_cmd("make -j1 bucket2", cwd=build_dir)
    print("🔨 Building target 'bucket_reordered'...")
    run_cmd("make -j1 bucket_reordered", cwd=build_dir)
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
