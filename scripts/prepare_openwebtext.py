"""Data prep for OpenWebText, adapted from nanoGPT's data/openwebtext/prepare.py.

Pure data plumbing (download parquet shards, BPE-encode with tiktoken, write
memmap-able uint16 .bin shards) -- unrelated to the no-autograd constraint,
so there's no reason to reinvent nanoGPT's approach here (the assignment
explicitly allows reusing nanoGPT for "dataset shuffling and scaffolding").

Downloads whole parquet shards from the public Skylion007/openwebtext mirror
(80 shards, ~110M tokens each) via huggingface_hub rather than the `datasets`
streaming iterator: the streaming path's background prefetch threads proved
flaky under unauthenticated rate limits on this box (dropped connections
during retries crashed the interpreter's GIL state on cleanup, after the
data had already been written -- see REASONING.md roadblock log). Whole-shard
download is simpler, resumable, and each shard alone comfortably covers the
scaled-down token budget from REASONING.md sec 7.

Writes data/openwebtext/{train,val}.bin as flat uint16 token arrays.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "openwebtext"
REPO_ID = "Skylion007/openwebtext"
NUM_SHARDS_TOTAL = 80

parser = argparse.ArgumentParser()
parser.add_argument("--num-shards", type=int, default=2, help="how many of the 80 train parquet shards to pull")
parser.add_argument("--val-fraction", type=float, default=0.0005, help="fraction of docs held out for val.bin")
parser.add_argument("--seed", type=int, default=2357)
args = parser.parse_args()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(args.seed)

    train_chunks, val_chunks = [], []
    train_tok, val_tok = 0, 0

    for shard_idx in range(args.num_shards):
        fname = f"plain_text/train-{shard_idx:05d}-of-{NUM_SHARDS_TOTAL:05d}.parquet"
        print(f"downloading shard {shard_idx + 1}/{args.num_shards}: {fname}")
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=fname)

        table = pq.read_table(local_path, columns=["text"])
        texts = table.column("text").to_pylist()
        print(f"  {len(texts):,} documents, tokenizing...")

        for text in tqdm(texts, desc=f"shard {shard_idx}"):
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            ids = np.array(ids, dtype=np.uint16)
            if rng.random() < args.val_fraction:
                val_chunks.append(ids)
                val_tok += len(ids)
            else:
                train_chunks.append(ids)
                train_tok += len(ids)

        print(f"  running total: train={train_tok:,} val={val_tok:,} tokens")

    for name, chunks, path in [
        ("train", train_chunks, OUT_DIR / "train.bin"),
        ("val", val_chunks, OUT_DIR / "val.bin"),
    ]:
        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
        mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        print(f"{name}: {arr.size:,} tokens -> {path}")


if __name__ == "__main__":
    main()
