"""Sanity check #4 from REASONING.md: a tiny model should drive loss to ~0
on a single repeated batch. First real end-to-end signal that backward
wiring + AdamW actually work together, before spending compute on real data.

Usage: python scripts/overfit_one_batch.py [--device cpu|cuda]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanograd_gpt.backend import get_xp  # noqa: E402
from nanograd_gpt.model import GPT, GPTConfig  # noqa: E402
from nanograd_gpt.optim import AdamW  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()

xp = get_xp(args.device)
xp.random.seed(0)

cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)
model = GPT(cfg, xp)
opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

B, T = 4, 32
idx = xp.asarray((xp.random.rand(B, T) * cfg.vocab_size).astype("int64"))
targets = xp.asarray((xp.random.rand(B, T) * cfg.vocab_size).astype("int64"))

print(f"device={args.device}  params={sum(p.data.size for p in model.parameters()):,}")
t0 = time.time()
losses = []
for step in range(args.steps):
    model.zero_grad()
    _, loss, dlogits = model.forward(idx, targets=targets)
    model.backward(dlogits)
    opt.step()
    loss_val = float(loss) if args.device == "cpu" else float(loss.get())
    losses.append(loss_val)
    if step % 20 == 0 or step == args.steps - 1:
        print(f"step {step:4d}  loss {loss_val:.4f}")

dt = time.time() - t0
print(f"\n{args.steps} steps in {dt:.1f}s ({args.steps/dt:.1f} steps/s)")

start, end = losses[0], losses[-1]
print(f"loss: {start:.4f} -> {end:.4f}")
assert end < start * 0.05, f"expected loss to collapse toward ~0, got {end:.4f} from {start:.4f}"
print("PASS: model overfits a single batch as expected")
