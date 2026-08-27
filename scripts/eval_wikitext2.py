import argparse
import sys
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanograd_gpt.backend import get_xp  # noqa: E402
from nanograd_gpt.model import GPT, GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-embd", type=int, default=384)
parser.add_argument("--block-size", type=int, default=256)
parser.add_argument("--vocab-size", type=int, default=50304)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()


def load_wikitext2_test_text():
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n".join(t for t in ds["text"] if t.strip())


def main():
    xp = get_xp(args.device)
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg, xp)

    ckpt = np.load(args.ckpt)
    params = list(model.parameters())
    assert len(params) == len(ckpt.files), f"checkpoint has {len(ckpt.files)} arrays, model has {len(params)} params"
    for p, key in zip(params, ckpt.files):
        arr = ckpt[key]
        p.data[...] = xp.asarray(arr) if args.device == "cuda" else arr
    print(f"loaded checkpoint {args.ckpt} ({sum(p.data.size for p in params):,} params)")

    enc = tiktoken.get_encoding("gpt2")
    print("loading WikiText-2 (raw, test split)...")
    text = load_wikitext2_test_text()
    ids = np.array(enc.encode_ordinary(text), dtype=np.int64)
    print(f"WikiText-2 test: {len(ids):,} tokens")

    T = args.block_size
    n_windows = (len(ids) - 1) // T
    ids = ids[: n_windows * T + 1]

    total_loss, total_tokens = 0.0, 0
    B = args.batch_size
    for start in range(0, n_windows, B):
        chunk_idx = range(start, min(start + B, n_windows))
        x = np.stack([ids[i * T : i * T + T] for i in chunk_idx])
        y = np.stack([ids[i * T + 1 : i * T + T + 1] for i in chunk_idx])
        if args.device == "cuda":
            x, y = xp.asarray(x), xp.asarray(y)
        _, loss, _ = model.forward(x, targets=y, training=False)
        loss_val = float(loss) if args.device == "cpu" else float(loss.get())
        n_tok = len(chunk_idx) * T
        total_loss += loss_val * n_tok
        total_tokens += n_tok

    mean_loss = total_loss / total_tokens
    ppl = float(np.exp(mean_loss))
    print(f"\nWikiText-2 (raw, test) | tokens evaluated: {total_tokens:,}")
    print(f"cross-entropy: {mean_loss:.4f}  |  perplexity: {ppl:.2f}")

    out_path = Path(args.ckpt).parent / "wikitext2_eval.json"
    import json

    with open(out_path, "w") as f:
        json.dump({"tokens": total_tokens, "cross_entropy": mean_loss, "perplexity": ppl}, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
