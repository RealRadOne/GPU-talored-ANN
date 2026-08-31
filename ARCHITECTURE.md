# GPANN: System Architecture & Flowcharts

This document visualizes the complete end-to-end pipeline of the **GPU-tailored-ANN (GPANN)** framework.

---

## 1. Master End-to-End Architecture Flowchart

```mermaid
flowchart TD
    classDef io fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef gpu fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef cpu fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#311b92;
    classDef out fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;

    START(["Raw Vector Dataset<br/>(.fbin / .u8bin / .ibin)"]):::io --> S1

    subgraph PHASE1 ["Phase 1: Memory-Adaptive Ingestion"]
        S1{"Does Full Dataset +<br/>KMeans Workspace<br/>Fit in GPU VRAM?"}:::decision
        S1 -- "YES" --> L1["Full Dataset In-Memory Load"]:::gpu
        S1 -- "NO" --> S2{"Does Sampled Subset<br/>(e.g., 10%) Fit in VRAM?"}:::decision
        S2 -- "YES" --> L2["Sequential Disk Sampling<br/>(Sorted NVMe I/O)"]:::io
        S2 -- "NO" --> L3["Adaptive Subspace PQ<br/>(Dynamically compress to 8-1 bits)"]:::gpu
    end

    L1 & L2 & L3 --> S3

    subgraph PHASE2 ["Phase 2: Centroid Backbone Construction"]
        S3["Parallel GPU KMeans++<br/>(Sample B ≈ N/32 centroids)"]:::gpu --> S4["Centroid KNN Graph Build<br/>(RAFT CAGRA graph on centroids)"]:::gpu
    end

    S4 --> S5

    subgraph PHASE3 ["Phase 3: Vector Assignment to Voronoi Buckets"]
        S5["Stream Dataset in GPU Batches"]:::gpu --> S6["GPU Greedy Graph ANNS<br/>(Route each vector to nearest centroid)"]:::gpu
        S6 --> S7["Top-2 Boundary Rebalance<br/>(Balance bucket size to 32 ± slack)"]:::cpu
    end

    S7 --> S8

    subgraph PHASE4 ["Phase 4: Tensor Core Per-Vector KNN"]
        S8["Gather Candidate Vectors<br/>(Self-bucket + K centroid neighbor buckets)"]:::gpu --> S9["Compute Pairwise L2 Distance<br/>via cuBLAS Tensor Core GEMM<br/>(INT8 IMMA / FP32 TF32)"]:::gpu
        S9 --> S10["GPU select_k<br/>(Extract top-M nearest neighbors)"]:::gpu
    end

    S10 --> S11

    subgraph PHASE5 ["Phase 5: Multi-Iteration Overlapped Merge"]
        S11{"Iteration t < T?"}:::decision
        S11 -- "YES (t < T)" --> S12["Async CPU Dedupe-Merge<br/>(Overlapped via std::async)"]:::cpu
        S12 -. "Runs in background while GPU starts next round" .-> S3
        S11 -- "NO (Done T rounds)" --> S13["Consolidated Forward KNN Graph"]:::io
    end

    S13 --> S14

    subgraph PHASE6 ["Phase 6: Out-of-Core Graph Optimization"]
        S14["Bucket ID Reordering<br/>(Consecutive ID mapping [Ob, Ob+1))"]:::cpu --> S15["3-Slot Sliding-Window GPU Inversion<br/>(Reverse edges without OOM)"]:::gpu
        S15 --> S16["CAGRA Detour Occlusion Pruning<br/>(Target degree K_out)"]:::gpu
    end

    S16 --> FINAL(["Final High-Recall Graph Index<br/>(cagra_graph.bin / neighbors.npy)"]):::out
```

---

## 2. Adaptive Memory Decision Tree

```mermaid
flowchart TD
    classDef check fill:#e0f7fa,stroke:#00838f,stroke-width:2px;
    classDef pass fill:#e8f8f5,stroke:#27ae60,stroke-width:2px;
    classDef adapt fill:#fef9e7,stroke:#f39c12,stroke-width:2px;
    classDef fail fill:#fdedec,stroke:#e74c3c,stroke-width:2px;

    IN["Input: N vectors, D dimensions, GPU limit M_bytes"]:::check --> C1

    C1{"Raw Data + (2+log B)*N*4B<br/>≤ M_bytes * 0.9?"}:::check
    C1 -- "True" --> T1["Tier 1: Direct In-Memory<br/>(Load all N points to GPU)"]:::pass
    
    C1 -- "False" --> C2{"Sampled Data (N * 10%) +<br/>KMeans Workspace ≤ VRAM?"}:::check
    C2 -- "True" --> T2["Tier 2: Sequential Disk Sampling<br/>(Order sample IDs → Sequential Read)"]:::pass
    
    C2 -- "False" --> C3["Initialize PQ: bits=8, pq_dim=D/4"]:::adapt
    C3 --> C4{"Encoded Data + Codebook +<br/>Workspace ≤ VRAM?"}:::check
    C4 -- "True" --> T3["Tier 3: Compressed PQ Pipeline<br/>(Train on GPU, uint8 representation)"]:::pass
    C4 -- "False" --> C5{"Can reduce bits?<br/>(bits > 1)"}:::check
    C5 -- "Yes" --> C6["Decrement bits (bits = bits - 1)"]:::adapt --> C4
    C5 -- "No" --> C7["Double pq_dim (halve subspaces)"]:::adapt --> C4
```

---

## 3. Tensor Core GEMM & Asynchronous Merge Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant GPU as NVIDIA GPU (Tensor Cores)
    participant HostRAM as Host RAM
    participant CPUWorker as Background CPU Thread (std::async)

    Note over GPU, HostRAM: Iteration 1 (Seed = 42)
    GPU->>GPU: Compute KMeans++ & Centroid KNN Graph
    GPU->>GPU: Dense GEMM (INT8/TF32): X · Yᵀ for all buckets
    GPU->>HostRAM: Download Iteration 1 KNN Graph & Distances
    HostRAM->>HostRAM: Set Iteration 1 as Running Graph

    Note over GPU, CPUWorker: Iteration 2 (Seed = 43) - Pipelined Execution
    par GPU Computes Round 2
        GPU->>GPU: Compute KMeans++ with Seed 43
        GPU->>GPU: Dense GEMM & select_k
        GPU->>HostRAM: Download Iteration 2 KNN Graph
    and Background CPU Merges Round 1 + 2
        CPUWorker->>CPUWorker: Read Iteration 1 (Running) & Iteration 2
        CPUWorker->>CPUWorker: Sort candidates by L2 distance
        CPUWorker->>CPUWorker: Deduplicate IDs & Retain Top-M
        CPUWorker->>HostRAM: Overwrite Running Graph with Merged Result
    end

    Note over HostRAM: Final High-Recall Merged KNN Graph Ready
```

---

## 4. Online Query Search Pipeline

```mermaid
flowchart TD
    classDef search fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px;
    classDef state fill:#e0f2f1,stroke:#00796b,stroke-width:2px;

    Q["Query Vector q (D-dim)"]:::search --> E1["Pick 'num_entry' random entry points (e.g. 4 nodes)"]:::search
    E1 --> H1["Compute distance to entry points<br/>Initialize Min-Heap: cand_heap & Best-List (size ef)"]:::state

    H1 --> W1{"cand_heap is not empty<br/>AND best_dist < worst in Best-List?"}:::search
    W1 -- "NO (Converged)" --> OUT["Sort Best-List<br/>Return Top-K Nearest Neighbors"]:::state

    W1 -- "YES" --> POP["Pop closest unvisited node 'u' from cand_heap"]:::search
    POP --> EXP["Inspect all M neighbors of 'u' from Graph Index"]:::search
    EXP --> DIST["Compute L2² distance from q to each neighbor v"]:::search
    DIST --> UPD["If v is closer than worst in Best-List:<br/>1. Insert v into Best-List<br/>2. Push (dist, v) into cand_heap"]:::state
    UPD --> W1
```
