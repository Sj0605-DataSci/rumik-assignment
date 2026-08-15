"""Measure real forward+backward+AdamW tokens/sec on GB10 for a candidate
config, on random data, so the token budget in REASONING.md sec 7 is picked
from a measurement rather than a guess.
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
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--n-layer", type=int, default=12)
parser.add_argument("--n-head", type=int, default=12)
parser.add_argument("--n-embd", type=int, default=768)
parser.add_argument("--block-size", type=int, default=1024)
parser.add_argument("--vocab-size", type=int, default=50304)
parser.add_argument("--batch-size", type=int, default=12)
parser.add_argument("--warmup-iters", type=int, default=5)
parser.add_argument("--bench-iters", type=int, default=20)
args = parser.parse_args()

xp = get_xp(args.device)
xp.random.seed(0)

cfg = GPTConfig(
    vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd
)
model = GPT(cfg, xp)
opt = AdamW(model.parameters(), lr=3e-4)
n_params = sum(p.data.size for p in model.parameters())
print(f"config: n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} block_size={args.block_size}")
print(f"params: {n_params:,}  device={args.device}  batch_size={args.batch_size}")

B, T = args.batch_size, args.block_size
idx = xp.asarray((xp.random.rand(B, T) * cfg.vocab_size).astype("int64"))
targets = xp.asarray((xp.random.rand(B, T) * cfg.vocab_size).astype("int64"))


def sync():
    if args.device == "cuda":
        xp.cuda.Stream.null.synchronize()


for _ in range(args.warmup_iters):
    model.zero_grad()
    _, loss, dlogits = model.forward(idx, targets=targets)
    model.backward(dlogits)
    opt.step()
sync()

t0 = time.time()
for _ in range(args.bench_iters):
    model.zero_grad()
    _, loss, dlogits = model.forward(idx, targets=targets)
    model.backward(dlogits)
    opt.step()
sync()
dt = time.time() - t0

tokens = args.bench_iters * B * T
tok_per_s = tokens / dt
step_time = dt / args.bench_iters
print(f"\n{args.bench_iters} steps in {dt:.2f}s -> {step_time*1000:.0f} ms/step, {tok_per_s:,.0f} tok/s")

for hours in [1, 4, 8, 24]:
    print(f"  in {hours:>2}h wall clock: ~{tok_per_s*3600*hours:,.0f} tokens, ~{3600*hours/step_time:,.0f} steps")
