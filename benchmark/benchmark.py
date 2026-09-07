import os
import sys
import re
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import plot_benchmark_results, generate_fbin_if_needed
from stream_data import prepare_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_binary(bin_path: str, data_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        bin_path,
        "-i", data_path,
        "-o", out_dir,
        "--knn-k", "32",
        "--neighbors-m", "32",
        "--iterations", "1"
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = proc.stdout

    if proc.returncode != 0:
        print(f"Error running {bin_path}:\n{output}")
        return {"total_s": 0.0, "gemm_ms": 0.0, "success": False}

    gemm_match = re.search(r"pure GEMM:\s*([\d\.]+)\s*ms", output)
    total_match = re.search(r"(?:Total pipeline elapsed|Total):\s*([\d\.]+)\s*s", output)

    gemm_ms = float(gemm_match.group(1)) if gemm_match else 0.0
    total_s = float(total_match.group(1)) if total_match else 0.0

    return {"total_s": total_s, "gemm_ms": gemm_ms, "success": True}

def main():
    parser = argparse.ArgumentParser(description="GPANN Benchmark Suite")
    parser.add_argument("--trial-only", action="store_true", help="Run only the 1 synthetic test trial")
    parser.add_argument("--dataset", type=str, choices=["sift1m", "deep1m", "laion1m"], default=None, help="Run a specific dataset only")
    args = parser.parse_args()

    bin_base = os.path.join(REPO_ROOT, "bucketDemo", "buildBucket", "build", "bucket2")
    bin_opt = os.path.join(REPO_ROOT, "bucketDemo", "buildBucket", "build", "bucket_reordered")
    data_dir = os.path.join(REPO_ROOT, "test_data")
    out_dir = os.path.join(os.path.dirname(__file__), "output")

    print("=" * 60)
    print("GPANN Benchmark Suite: Baseline (bucket2) vs Cache-Opt (bucket_reordered)")
    print("=" * 60)

    # 1 Test trial on generated data, followed by mandated real benchmark datasets
    if args.dataset:
        tasks = [(args.dataset.upper(), args.dataset, 1000000, 128)]
    elif args.trial_only:
        tasks = [("Trial-50K", "synthetic", 50000, 128)]
    else:
        tasks = [
            ("Trial-50K", "synthetic", 50000, 128),
            ("SIFT-1M", "sift1m", 1000000, 128),
            ("DEEP-1M", "deep1m", 1000000, 96),
            ("LAION-1M", "laion1m", 1000000, 512),
        ]

    names, base_totals, opt_totals = [], [], []
    base_gemms, opt_gemms = [], []

    for name, dtype, n, d in tasks:
        if dtype == "synthetic":
            data_path = os.path.join(data_dir, f"trial_{n//1000}k_{d}d.fbin")
            generate_fbin_if_needed(data_path, n, d)
        else:
            data_path = prepare_dataset(dtype, data_dir)

        if not data_path or not os.path.exists(data_path):
            print(f"Skipping {name}: file not ready")
            continue

        print(f"\nEvaluating {name} ({data_path}):")
        res_base = run_binary(bin_base, data_path, os.path.join(out_dir, f"base_{name}"))
        res_opt = run_binary(bin_opt, data_path, os.path.join(out_dir, f"opt_{name}"))

        if res_base["success"] and res_opt["success"]:
            names.append(name)
            base_totals.append(res_base["total_s"])
            opt_totals.append(res_opt["total_s"])
            base_gemms.append(res_base["gemm_ms"])
            opt_gemms.append(res_opt["gemm_ms"])
            sp = res_base["total_s"] / max(1e-6, res_opt["total_s"])
            print(f"{name} -> Baseline: {res_base['total_s']:.2f}s | Reordered: {res_opt['total_s']:.2f}s | Speedup: {sp:.2f}x")

    if names:
        plot_benchmark_results(
            names,
            base_totals,
            opt_totals,
            base_gemms,
            opt_gemms,
            os.path.join(out_dir, "benchmark_results.png")
        )

if __name__ == "__main__":
    main()


