# GPANN: Future Optimization Roadmap

## 1. Memory & Compression Optimizations
- [ ] **2D Adaptive Grid Search in `estimate_memory_requirement`**:
  - *Context:* Current version uses a 1D loop adjusting `pq_bits` ($8 \to 4$).
  - *Future Optimization:* Implement a 2D adaptive loop exploring both subspace dimensions `pq_dim` ($D/4 \to 4$) and `pq_bits` ($8 \to 1$) to dynamically fit extreme datasets on $\le 4\text{GB}$ VRAM GPUs.

- [ ] **GPUDirect Storage (GDS / `cuFile`)**:
  - Direct NVMe SSD to GPU VRAM DMA streaming, bypassing Host CPU RAM.

## 2. Kernel & Hardware Optimizations
- [ ] **Fused GEMM + Top-K Kernel**:
  - Use CUTLASS / Triton to maintain warp-level bitonic top-$M$ sorting in Shared Memory / Registers, eliminating global VRAM writes of raw distance matrices.

- [ ] **FP8 / FP4 Tensor Core Support**:
  - Add native FP8 (`E4M3`/`E5M2`) Tensor Core execution on Ada/Hopper/Blackwell GPUs.

- [ ] **Scalable Parallel KMeans (`KMeans||`)**:
  - Replace sequential centroid sampling with 5 parallel rounds of $O(B)$ sampling.

## 3. Distributed & Multi-GPU Scaling
- [ ] **Multi-GPU Sharding via NCCL**:
  - Shard Voronoi bucket candidate calculations across multiple GPUs on a node.

## 4. Architecture & Codebase Refactoring
- [ ] **Modularize 3500+ LOC Monolithic Files (`bucket.cu` / `bucket_reordered.cu`)**:
  - *Context:* Current `bucket.cu` is 3,500+ lines in a single file, making debugging and experimental changes difficult.
  - *Goal:* Decompose into clean, testable C++/CUDA sub-modules:
    - `src/core/memory_budget.hpp`: RAM/VRAM estimation and fallback logic.
    - `src/clustering/kmeans_gpu.cuh`: GPU KMeans++ and centroid selection.
    - `src/routing/batch_assign.cuh`: Voronoi routing and boundary rebalancing.
    - `src/gemm/tensorcore_knn.cuh`: cuBLAS Tensor Core GEMM and double buffering.
    - `src/merge/async_merge.hpp`: Asynchronous multi-iteration CPU deduplication.
