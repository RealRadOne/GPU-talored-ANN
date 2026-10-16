#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess

def sh(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def fetch_dependency(name, archive_url, git_url, git_ref, target_path, verify_subpath):
    verify_file = os.path.join(target_path, verify_subpath)
    if os.path.exists(verify_file):
        print(f"Dependency {name} already present at {target_path}")
        return

    print(f"Fetching dependency: {name} -> {target_path}")
    shutil.rmtree(target_path, ignore_errors=True)
    os.makedirs(target_path, exist_ok=True)

    # 1. Fast HTTP archive extraction via curl with retries
    tar_cmd = f"curl -sSL --connect-timeout 20 --retry 3 --retry-delay 2 '{archive_url}' | tar -xz --strip-components=1 -C '{target_path}'"
    try:
        res = subprocess.run(tar_cmd, shell=True, timeout=120)
        if res.returncode == 0 and os.path.exists(verify_file):
            print(f"Successfully extracted {name} archive.")
            return
    except Exception as e:
        print(f"Archive download for {name} failed: {e}")

    # 2. Fallback to git clone if archive download failed
    print(f"Fallback: Cloning {name} via git...")
    shutil.rmtree(target_path, ignore_errors=True)
    b_flag = f"-b {git_ref}" if git_ref else ""
    sh(f"git clone --depth 1 {b_flag} {git_url} {target_path}")

def setup_dependencies():
    print("1. Installing system dependencies and RAPIDS/CUDA headers...")
    sh("apt-get update -qq && apt-get install -y -qq libboost-program-options-dev nlohmann-json3-dev")
    sh("pip install -q pylance h5py pyarrow")

    deps = [
        ("fmt", "https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz", "https://github.com/fmtlib/fmt.git", "10.2.1", "/content/fmt", "include/fmt/core.h"),
        ("spdlog", "https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.tar.gz", "https://github.com/gabime/spdlog.git", "v1.13.0", "/content/spdlog", "include/spdlog/spdlog.h"),
        ("raft", "https://github.com/rapidsai/raft/archive/refs/heads/branch-24.04.tar.gz", "https://github.com/rapidsai/raft.git", "branch-24.04", "/content/raft", "cpp/include/raft"),
        ("rmm", "https://github.com/rapidsai/rmm/archive/refs/heads/branch-24.04.tar.gz", "https://github.com/rapidsai/rmm.git", "branch-24.04", "/content/rmm", "include/rmm"),
        ("cutlass", "https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.10.0.tar.gz", "https://github.com/NVIDIA/cutlass.git", "v2.10.0", "/content/cutlass", "include/cutlass"),
        ("cuco", "https://github.com/NVIDIA/cuCollections/archive/refs/heads/main.tar.gz", "https://github.com/NVIDIA/cuCollections.git", None, "/content/cuco", "include"),
    ]

    for name, archive_url, git_url, git_ref, target_path, verify_subpath in deps:
        fetch_dependency(name, archive_url, git_url, git_ref, target_path, verify_subpath)

    # Patch RAFT header for CUTLASS 2.10 compatibility
    raft_hdr = "/content/raft/cpp/include/raft/distance/detail/pairwise_distance_epilogue_elementwise.h"
    if os.path.exists(raft_hdr):
        sh(f"grep -q kIsSingleSource {raft_hdr} || sed -i '/kCount = kElementsPerAccess;/a \\  static bool const kIsSingleSource = true;' {raft_hdr}")

    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Shim headers so RAFT/RMM branch-24.04 compile against CUDA 13's CCCL.
CUDA13_SHIMS = {
    # CCCL dropped thrust::optional; alias it to std::optional.
    "thrust/optional.h": (
        "#pragma once\n"
        "#include <optional>\n"
        "namespace thrust { using std::optional; using std::nullopt; }\n"
    ),
    # CCCL removed cub::TransformInputIterator; map it onto Thrust.
    # (CUB order: value, op, iterator.  Thrust order: op, iterator, reference, value.)
    "cub_compat.h": (
        "#pragma once\n"
        "#include <thrust/iterator/transform_iterator.h>\n"
        "#include <thrust/iterator/iterator_traits.h>\n"
        "namespace cub {\n"
        "template <typename ValueT, typename OpT, typename IterT>\n"
        "using TransformInputIterator =\n"
        "    thrust::transform_iterator<OpT, IterT, thrust::use_default, ValueT>;\n"
        "}\n"
    ),
}

def write_if_changed(filepath, content):
    """Write content unless the file already holds exactly that, so make stays incremental."""
    if os.path.exists(filepath):
        with open(filepath) as existing_file:
            if existing_file.read() == content:
                return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as shim_file:
        shim_file.write(content)

def write_cuda13_shims(include_dir="/content/rmm/include"):
    for relative_path, content in CUDA13_SHIMS.items():
        write_if_changed(os.path.join(include_dir, relative_path), content)

def build_gpann(project_root, build_dir, targets=("gpann_modular",), cuda_arch="75", log_dir="/content"):
    """Configure and build with the CUDA 13 shims; full logs go to log_dir/{cmake,build}.log."""
    cuda_flags = " ".join([
        f"-I{project_root}/src",
        "-I/content/rmm/include",
        "-I/content/raft/cpp/include",
        "--pre-include /content/rmm/include/cub_compat.h",
    ])
    cmake_options = " ".join([
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=17",
        "-DCMAKE_CUDA_STANDARD=17",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        f'-DCMAKE_CUDA_FLAGS="{cuda_flags}"',
    ])
    steps = [
        ("cmake", f"cmake -B {build_dir} -S {project_root} {cmake_options}"),
        ("build", f"make -C {build_dir} -j$(nproc) {' '.join(targets)}"),
    ]
    for name, cmd in steps:
        log_path = os.path.join(log_dir, f"{name}.log")
        with open(log_path, "w") as log_file:
            result = subprocess.run(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            with open(log_path, errors="replace") as log_file:
                lines = log_file.readlines()
            errors = [line for line in lines if "error" in line.lower()][:40]
            print("".join(errors or lines[-30:]))
            raise RuntimeError(f"{name} failed (exit {result.returncode}); full log: {log_path}")
    for target in targets:
        print(f"Built {os.path.join(build_dir, target)}")

def build_and_benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(base_dir, "bucketDemo/buildBucket/build")
    os.makedirs(build_dir, exist_ok=True)

    print("\n2. Building GPANN binaries...")
    cmake_flags = "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75"
    sh(f"cmake .. {cmake_flags} && make -j1 bucket2 && make -j4 gpann_modular", cwd=build_dir)

    print("\n3. Running benchmark orchestrator...")
    benchmark_dir = os.path.join(base_dir, "benchmark")
    extra_args = " ".join(sys.argv[1:])
    sh(f"python3 benchmark.py {extra_args}".strip(), cwd=benchmark_dir)

if __name__ == "__main__":
    setup_dependencies()
    build_and_benchmark()


