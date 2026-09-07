import os
import struct
import numpy as np

def stream_huggingface_to_fbin(dataset_id: str, split: str, vector_col: str, out_path: str, max_vectors: int = 1000000, chunk_size: int = 50000):
    if os.path.exists(out_path):
        print(f"Dataset already exists at: {out_path}")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Streaming dataset {dataset_id} ({max_vectors:,} vectors target)...")
    ds = load_dataset(dataset_id, split=split, streaming=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    buffer = []
    total_written = 0
    dim = None

    with open(out_path, "wb") as f:
        f.write(struct.pack("ii", 0, 0))

        for row in ds:
            vec = row[vector_col]
            if dim is None:
                dim = len(vec)
            buffer.append(vec)

            if len(buffer) >= chunk_size:
                arr = np.asarray(buffer, dtype=np.float32)
                f.write(arr.tobytes())
                total_written += len(buffer)
                buffer.clear()
                print(f"  Streamed {total_written:,} / {max_vectors:,} vectors...")

            if total_written + len(buffer) >= max_vectors:
                break

        if buffer:
            arr = np.asarray(buffer, dtype=np.float32)
            f.write(arr.tobytes())
            total_written += len(buffer)
            buffer.clear()

        f.seek(0)
        f.write(struct.pack("ii", total_written, dim))

    print(f"Stream complete: {out_path} ({total_written:,} vectors, {dim}D)")
    return out_path

def prepare_dataset(name: str, out_dir: str) -> str:
    name_lower = name.lower()
    os.makedirs(out_dir, exist_ok=True)

    if name_lower == "sift1m":
        out_path = os.path.join(out_dir, "sift1m_128d.fbin")
        return stream_huggingface_to_fbin("maknee/sift1m", "train", "vector", out_path, max_vectors=1000000)
    elif name_lower == "laion1m":
        out_path = os.path.join(out_dir, "laion1m_512d.fbin")
        return stream_huggingface_to_fbin("lance-format/laion-1m", "train", "vector", out_path, max_vectors=1000000)
    elif name_lower == "deep1m":
        out_path = os.path.join(out_dir, "deep1m_96d.fbin")
        if os.path.exists(out_path):
            return out_path
        print(f"Deep1M should be placed at: {out_path} (Download from Kaggle: cypherxray/deepnet-1m-dataset)")
        return out_path
    else:
        raise ValueError(f"Unknown dataset name: {name}")
