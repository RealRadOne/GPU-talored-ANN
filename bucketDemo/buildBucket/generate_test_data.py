#!/usr/bin/env python3
"""
Generate test data in .fbin format for bucket building
"""
import numpy as np
import struct
import os

def write_fbin(filename, data):
    """Write float32 data to .fbin format"""
    N, D = data.shape
    with open(filename, 'wb') as f:
        # Write N and D as int32
        f.write(struct.pack('I', N))
        f.write(struct.pack('I', D))
        # Write data as float32
        data.astype(np.float32).tofile(f)
    print(f"Created {filename}: shape=({N}, {D}), size={os.path.getsize(filename) / 1e6:.2f}MB")

def main():
    # Create test data directory
    os.makedirs('test_data', exist_ok=True)

    # Test case 1: Small dataset (10K vectors, 128 dims)
    print("Generating small test dataset (10K vectors, 128D)...")
    N, D = 10000, 128
    X_small = np.random.randn(N, D).astype(np.float32)
    write_fbin('test_data/vectors_10k_128d.fbin', X_small)

    # Test case 2: Medium dataset (100K vectors, 128 dims)
    print("Generating medium test dataset (100K vectors, 128D)...")
    N, D = 100000, 128
    X_medium = np.random.randn(N, D).astype(np.float32)
    write_fbin('test_data/vectors_100k_128d.fbin', X_medium)

    # Test case 3: Large dataset (1M vectors, 128 dims) - for compression testing
    print("Generating large test dataset (1M vectors, 128D)...")
    N, D = 1000000, 128
    X_large = np.random.randn(N, D).astype(np.float32)
    write_fbin('test_data/vectors_1m_128d.fbin', X_large)

if __name__ == '__main__':
    main()
