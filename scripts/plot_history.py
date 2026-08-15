"""Plot train/val loss and val perplexity curves from out/history.json."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--history", default=str(ROOT / "out" / "history.json"))
parser.add_argument("--out", default=str(ROOT / "out" / "loss_curves.png"))
args = parser.parse_args()

with open(args.history) as f:
    h = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(h["step"], h["train_loss"], label="train")
axes[0].plot(h["step"], h["val_loss"], label="val")
axes[0].set_xlabel("step")
axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("train/val loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(h["step"], h["val_ppl"], color="tab:red")
axes[1].set_xlabel("step")
axes[1].set_ylabel("perplexity")
axes[1].set_title("val perplexity")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"saved {args.out}")
