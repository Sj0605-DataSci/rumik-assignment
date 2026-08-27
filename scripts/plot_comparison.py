import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    ("Run 1: nano (30M)", ROOT / "out" / "history.json", 32 * 256, "tab:blue"),
    ("Run 2: GPT-2-small, 53M tok (undertrained)", ROOT / "out" / "run2_gpt2small" / "history.json", 12 * 1024, "tab:orange"),
    ("Run 2b: GPT-2-small, 135M tok (corrected)", ROOT / "out" / "run2b_gpt2small_135M" / "history.json", 12 * 1024, "tab:green"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for label, path, tokens_per_step, color in RUNS:
    if not path.exists():
        continue
    h = json.load(open(path))
    tokens_m = [s * tokens_per_step / 1e6 for s in h["step"]]
    axes[0].plot(tokens_m, h["val_loss"], label=label, color=color)
    axes[1].plot(tokens_m, h["val_ppl"], label=label, color=color)

axes[0].set_xlabel("tokens processed (millions)")
axes[0].set_ylabel("val cross-entropy loss")
axes[0].set_title("Validation loss vs. tokens seen")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("tokens processed (millions)")
axes[1].set_ylabel("val perplexity")
axes[1].set_title("Validation perplexity vs. tokens seen")
axes[1].set_yscale("log")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
out_path = ROOT / "out" / "comparison_curves.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
