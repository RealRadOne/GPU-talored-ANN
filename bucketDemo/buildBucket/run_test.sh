#!/bin/bash
# Test script for bucket building with GPU KMeans++ centroid generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SRC_DIR="$SCRIPT_DIR/src"
BIN="$BUILD_DIR/bucket"

echo "=========================================="
echo "Bucket Building Test Script"
echo "=========================================="

# Step 1: Generate test data
echo ""
echo "[Step 1] Generating test data..."
if command -v python3 &> /dev/null; then
    cd "$SCRIPT_DIR"
    python3 generate_test_data.py
else
    echo "Python3 not found. Skipping test data generation."
    exit 1
fi

# Step 2: Compile the project
echo ""
echo "[Step 2] Compiling bucket building project..."
cd "$BUILD_DIR"
make -j$(nproc)
echo "Build completed successfully!"

# Step 3: Run tests with different configurations
echo ""
echo "[Step 3] Running bucket building tests..."
echo ""

# Test 1: Small dataset without sampling (should fit in GPU)
echo "=== Test 1: Small dataset (10K vectors) ==="
mkdir -p "$SCRIPT_DIR/output/test1"
time $BIN \
    --data "$SCRIPT_DIR/test_data/vectors_10k_128d.fbin" \
    --k 32 \
    --m 64 \
    --out_dir "$SCRIPT_DIR/output/test1" \
    --seed 42 \
    --gpu-limit $((4 * 1024 * 1024 * 1024)) \
    --sample-rate 1.0 \
    --centroid-ratio 0.1 \
    --pq-bits-start 8

echo ""
echo "Results saved to $SCRIPT_DIR/output/test1"
ls -lh "$SCRIPT_DIR/output/test1"

# Test 2: Medium dataset with sampling
echo ""
echo "=== Test 2: Medium dataset (100K vectors) with sampling ==="
mkdir -p "$SCRIPT_DIR/output/test2"
time $BIN \
    --data "$SCRIPT_DIR/test_data/vectors_100k_128d.fbin" \
    --k 128 \
    --m 64 \
    --out_dir "$SCRIPT_DIR/output/test2" \
    --seed 42 \
    --gpu-limit $((4 * 1024 * 1024 * 1024)) \
    --sample-rate 0.2 \
    --centroid-ratio 0.01 \
    --pq-bits-start 8

echo ""
echo "Results saved to $SCRIPT_DIR/output/test2"
ls -lh "$SCRIPT_DIR/output/test2"

# Test 3: Large dataset with PQ compression and dynamic bit adjustment
echo ""
echo "=== Test 3: Large dataset (1M vectors) with PQ compression ==="
mkdir -p "$SCRIPT_DIR/output/test3"
time $BIN \
    --data "$SCRIPT_DIR/test_data/vectors_1m_128d.fbin" \
    --k 256 \
    --m 64 \
    --out_dir "$SCRIPT_DIR/output/test3" \
    --seed 42 \
    --gpu-limit $((4 * 1024 * 1024 * 1024)) \
    --sample-rate 0.1 \
    --centroid-ratio 0.01 \
    --pq-bits-start 8 \
    --pq-bits-min 2 \
    --use-pq

echo ""
echo "Results saved to $SCRIPT_DIR/output/test3"
ls -lh "$SCRIPT_DIR/output/test3"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
