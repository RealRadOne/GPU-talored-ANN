import argparse
import os
import struct
import numpy as np
import torch
from typing import Tuple, List

# --------------------------
# Loaders
# --------------------------

def get_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def load_bigann_to_f32(path: str) -> Tuple[np.ndarray, int, int]:
    """
    Read BIGANN-style binary with header (int32 N, int32 D):
      - .fbin/.bin: float32
      - .u8bin/.i8bin: uint8 (cast to float32)
    return X (float32, shape [N, D]), N, D
    """
    ext = get_ext(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"File too small: {path}")
        N, D = struct.unpack("ii", head)
        cnt = N * D
        if ext in (".fbin", ".bin"):
            raw = f.read(cnt * 4)
            if len(raw) != cnt * 4:
                raise ValueError("Size mismatch for fbin")
            X = np.frombuffer(raw, dtype=np.float32).reshape(N, D)
        elif ext in (".u8bin", ".i8bin"):
            raw = f.read(cnt)
            if len(raw) != cnt:
                raise ValueError("Size mismatch for u8bin/i8bin")
            X = np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(N, D)
        else:
            raise ValueError(f"Unsupported data ext: {ext}")
    return X, N, D

def load_ibin(path: str) -> Tuple[np.ndarray, int, int]:
    """
    groundtruth .ibin: int32, header (N, D) then N*D values.
    Returns GT numpy int32 [N, D]
    """
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"File too small: {path}")
        N, D = struct.unpack("ii", head)
        cnt = N * D
        raw = f.read(cnt * 4)
        if len(raw) != cnt * 4:
            raise ValueError("Size mismatch for ibin")
        G = np.frombuffer(raw, dtype=np.int32).reshape(N, D)
    return G, N, D

def load_neighbors(path: str) -> np.ndarray:
    """
    读取 prune 后的邻接矩阵：
        shape (N, M)
        dtype int64
        padding 用 -1
    返回:
        np.ndarray[int64] shape (N, M)
    """
    idx = np.load(path)

    # 必须是二维矩阵
    if idx.ndim != 2:
        raise ValueError(
            f"Expected 2D neighbors matrix, got shape={idx.shape}"
        )
    # 必须是整数类型
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(
            f"neighbors must be integer dtype, got {idx.dtype}"
        )
    # 强制转成 int64（因为 -1 padding 以及图搜索需要 int 边）
    if idx.dtype != np.int64:
        if np.any(idx < -1) or np.any(idx > np.iinfo(np.uint64).max): 
            raise ValueError("neighbors matrix contains invalid negative ids (< -1), or out of uint64 range")
        idx = idx.astype(np.int64, copy=False)

    return idx

# ----------------------------
# I/O: BIGANN bin loaders
# ----------------------------

def load_bigann_bin(path: str) -> torch.Tensor:
    """
    Load .fbin/.ibin/.i8bin with BIGANN header format:
      int32 N, int32 D, then N*D elements in the corresponding dtype.
    Returns float32 torch tensor of shape (N, D) on CPU.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("File too small to contain BIGANN header (N,D).")
        N, D = struct.unpack("ii", header)
        # Compute item count and dtype
        if ext == ".fbin":
            dtype_np = np.float32
            item_bytes = 4
        elif ext == ".ibin":
            # Some corpora store ints; we cast to float32 for distance
            dtype_np = np.int32
            item_bytes = 4
        elif ext == ".u8bin":
            # BIGANN i8 typically unsigned (0..255); some sets are signed int8.
            # We first try uint8; you can switch to int8 if needed.
            dtype_np = np.uint8
            item_bytes = 1
        else:
            raise ValueError(f"Unsupported extension: {ext}")

        expected = N * D * item_bytes
        raw = f.read()
        if len(raw) != expected:
            raise ValueError(
                f"Data size mismatch: expect {expected} bytes for N={N},D={D}, got {len(raw)}."
            )

        arr = np.frombuffer(raw, dtype=dtype_np, count=N*D).reshape(N, D)

        # Cast to float32 for downstream
        if dtype_np == np.uint8:
            # Common practice: cast to float32; optional mean-centering or normalization left to user
            X = arr.astype(np.float32)
        elif dtype_np == np.int32:
            X = arr.astype(np.float32)
        else:
            X = arr  # float32 already

        # return torch tensor on CPU
        return torch.from_numpy(X.astype(np.float32))


def save_npy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)