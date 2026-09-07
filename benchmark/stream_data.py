import os
import struct
import urllib.request
import numpy as np

DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")

def download_file_stream(url: str, out_path: str, token: str = None) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Dataset already exists at: {out_path}")
        return out_path

    headers = {"User-Agent": "Mozilla/5.0"}
    auth_token = token or DEFAULT_HF_TOKEN
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    print(f"Streaming authenticated download from {url} to {out_path}...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    part_path = out_path + ".tmp"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(part_path, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"  Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)...")
            else:
                print(f"  Downloaded {downloaded / (1024*1024):.1f} MB...")

    os.rename(part_path, out_path)
    print(f"Download complete: {out_path} ({os.path.getsize(out_path):,} bytes)")
    return out_path

def stream_lance_laion_to_fbin(out_path: str, max_vectors: int = 1000000) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Dataset already exists at: {out_path}")
        return out_path

    try:
        import lance
    except ImportError:
        print("Note: lance is not installed. To stream LAION-1M, run: pip install pylance")
        return None

    print(f"Streaming LAION-1M embeddings via lance...")
    token = DEFAULT_HF_TOKEN
    storage_options = {"hf_token": token} if token else {}
    ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance", storage_options=storage_options)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("ii", 0, 0))
        total = 0
        dim = None
        for batch in ds.to_batches(columns=["img_emb"], batch_size=50000):
            arr = np.stack(batch["img_emb"].to_numpy()).astype(np.float32)
            if dim is None:
                dim = arr.shape[1]
            f.write(arr.tobytes())
            total += len(arr)
            print(f"  Streamed {total:,} / {max_vectors:,} LAION vectors...")
            if total >= max_vectors:
                break
        f.seek(0)
        f.write(struct.pack("ii", total, dim))
    return out_path

def prepare_dataset(name: str, out_dir: str) -> str:
    name_lower = name.lower()
    os.makedirs(out_dir, exist_ok=True)

    if name_lower == "sift1m":
        out_path = os.path.join(out_dir, "sift1m_128d.fbin")
        url = "https://huggingface.co/datasets/maknee/sift1m/resolve/main/fbin/base.fbin"
        return download_file_stream(url, out_path)

    elif name_lower == "laion1m":
        out_path = os.path.join(out_dir, "laion1m_768d.fbin")
        res = stream_lance_laion_to_fbin(out_path)
        return res

    elif name_lower == "deep1m":
        out_path = os.path.join(out_dir, "deep1m_96d.fbin")
        if os.path.exists(out_path):
            return out_path
        print(f"Deep1M not found at {out_path}. Please place cypherxray/deepnet-1m-dataset there.")
        return None

    else:
        raise ValueError(f"Unknown dataset name: {name}")

