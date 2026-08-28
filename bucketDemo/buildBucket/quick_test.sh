#!/bin/bash
# Quick test - single command

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Generating test data..."
python3 generate_test_data.py

echo ""
echo "Building project..."
cd build && make -j$(nproc) && cd ..

echo ""
echo "Running single test: 100K vectors, 2GB GPU memory, with PQ compression"
mkdir -p output/quick_test

./build/bucket \
    --data test_data/vectors_100k_128d.fbin \
    --k 128 \
    --m 32 \
    --out_dir output/quick_test \
    --seed 42 \
    --gpu-limit $((2 * 1024 * 1024 * 1024)) \
    --sample-rate 0.1 \
    --centroid-ratio 0.01 \
    --pq-bits-start 8 \
    --pq-bits-min 1

echo ""
echo "✓ Test completed! Results in output/quick_test/"
ls -lh output/quick_test/
