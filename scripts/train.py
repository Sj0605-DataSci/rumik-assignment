"""Training loop: data batching, cosine LR schedule w/ warmup, AdamW step,
checkpointing, and loss/perplexity logging. Mirrors nanoGPT's train.py
scaffolding (data loading + schedule), but every forward/backward call goes
through the hand-derived gradients in nanograd_gpt/layers.py -- no autograd.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanograd_gpt.backend import get_xp  # noqa: E402
from nanograd_gpt.model import GPT, GPTConfig  # noqa: E402
from nanograd_gpt.optim import AdamW, cosine_lr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=str(ROOT / "data" / "openwebtext"))
parser.add_argument("--out-dir", default=str(ROOT / "out"))
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

parser.add_argument("--n-layer", type=int, default=12)
parser.add_argument("--n-head", type=int, default=12)
parser.add_argument("--n-embd", type=int, default=768)
parser.add_argument("--block-size", type=int, default=1024)
parser.add_argument("--vocab-size", type=int, default=50304)
parser.add_argument("--dropout", type=float, default=0.0, help="nanoGPT's own pretraining default is 0.0")

parser.add_argument("--batch-size", type=int, default=12)
parser.add_argument("--max-steps", type=int, default=5000)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--max-lr", type=float, default=6e-4)
parser.add_argument("--min-lr", type=float, default=6e-5)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=1.0)

parser.add_argument("--eval-interval", type=int, default=250)
parser.add_argument("--eval-iters", type=int, default=50)
parser.add_argument("--log-interval", type=int, default=20)
parser.add_argument("--ckpt-interval", type=int, default=500)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()


def get_batch(data_np, block_size, batch_size, xp, rng):
    ix = rng.integers(0, len(data_np) - block_size - 1, size=batch_size)
    x = np.stack([data_np[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data_np[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    if xp.__name__ != "numpy":
        x, y = xp.asarray(x), xp.asarray(y)
    return x, y


@np.errstate(over="ignore")
def clip_grad_norm(params, max_norm, xp):
    total_sq = 0.0
    for p in params:
        total_sq += float((p.grad * p.grad).sum())
    total_norm = total_sq**0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale
    return total_norm


def estimate_loss(model, data, block_size, batch_size, xp, rng, eval_iters):
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, xp, rng)
        _, loss, _ = model.forward(x, targets=y, training=False)  # dropout off for eval
        losses.append(float(loss) if xp.__name__ == "numpy" else float(loss.get()))
    return float(np.mean(losses))


def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xp = get_xp(args.device)
    xp.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg, xp)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"model params: {n_params:,}  device={args.device}")

    opt = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    history = {"step": [], "train_loss": [], "val_loss": [], "val_ppl": [], "lr": [], "tok_per_sec": []}
    t0 = time.time()
    tokens_since_log = 0

    for step in range(args.max_steps + 1):
        lr = cosine_lr(
            step, warmup_steps=args.warmup_steps, max_steps=args.max_steps, max_lr=args.max_lr, min_lr=args.min_lr
        )

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, xp, rng, args.eval_iters)
            train_loss_est = estimate_loss(
                model, train_data, args.block_size, args.batch_size, xp, rng, args.eval_iters
            )
            val_ppl = float(np.exp(val_loss))
            elapsed = time.time() - t0
            tps = tokens_since_log / elapsed if elapsed > 0 and step > 0 else 0.0
            print(
                f"step {step:5d} | train_loss {train_loss_est:.4f} | val_loss {val_loss:.4f} "
                f"| val_ppl {val_ppl:.2f} | lr {lr:.2e} | tok/s {tps:.0f}"
            )
            history["step"].append(step)
            history["train_loss"].append(train_loss_est)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["lr"].append(lr)
            history["tok_per_sec"].append(tps)
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            t0 = time.time()
            tokens_since_log = 0

        if step == args.max_steps:
            break

        x, y = get_batch(train_data, args.block_size, args.batch_size, xp, rng)
        model.zero_grad()
        _, loss, dlogits = model.forward(x, targets=y)
        model.backward(dlogits)
        clip_grad_norm(list(model.parameters()), args.grad_clip, xp)
        opt.step(lr=lr)
        tokens_since_log += args.batch_size * args.block_size

        if step % args.log_interval == 0:
            lv = float(loss) if xp.__name__ == "numpy" else float(loss.get())
            print(f"  step {step:5d} | loss {lv:.4f} | lr {lr:.2e}")

        if step % args.ckpt_interval == 0 and step > 0:
            save_checkpoint(model, out_dir / f"ckpt_{step}.npz", xp)

    save_checkpoint(model, out_dir / "ckpt_final.npz", xp)
    print("training complete")


def save_checkpoint(model, path, xp):
    arrays = {}
    for i, p in enumerate(model.parameters()):
        data = p.data if xp.__name__ == "numpy" else p.data.get()
        arrays[f"param_{i}"] = data
    np.savez(path, **arrays)
    print(f"checkpoint saved: {path}")


if __name__ == "__main__":
    main()
