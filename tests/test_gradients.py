"""Numerical gradient checks for every hand-written backward pass.

This is the primary evidence that the calculus in layers.py/model.py is
correct: central-difference numerical gradients on float64 tensors, compared
against the analytic gradients produced by each layer's backward(). Runs on
plain numpy (xp=np) -- these tensors are tiny, no reason to touch the GPU.

Run directly with `python tests/test_gradients.py`, or via `pytest tests/`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanograd_gpt.layers import (  # noqa: E402
    MLP,
    Block,
    CausalSelfAttention,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    softmax_cross_entropy,
)
from nanograd_gpt.model import GPT, GPTConfig  # noqa: E402

RNG = np.random.default_rng(0)
EPS = 1e-6
RTOL = 1e-4
ATOL = 1e-6


def numerical_grad(loss_fn, x, eps=EPS, max_elems=200):
    """Central-difference dL/dx for every element of x (or a random subset
    if x is larger than max_elems, to keep runtime sane)."""
    grad = np.zeros_like(x)
    flat_idx = list(np.ndindex(x.shape))
    if len(flat_idx) > max_elems:
        sel = RNG.choice(len(flat_idx), size=max_elems, replace=False)
        flat_idx = [flat_idx[i] for i in sel]
    for idx in flat_idx:
        orig = x[idx]
        x[idx] = orig + eps
        fp = loss_fn()
        x[idx] = orig - eps
        fm = loss_fn()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
    return grad, flat_idx


def assert_close_at(analytic, numeric, idx_list, name):
    a = np.array([analytic[i] for i in idx_list])
    n = np.array([numeric[i] for i in idx_list])
    if not np.allclose(a, n, rtol=RTOL, atol=ATOL):
        diff = np.abs(a - n)
        worst = np.argmax(diff)
        raise AssertionError(
            f"{name}: mismatch at {idx_list[worst]}: analytic={a[worst]:.8f} "
            f"numeric={n[worst]:.8f} diff={diff[worst]:.2e}"
        )
    print(f"  ok  {name:<28} max|diff|={np.max(np.abs(a - n)):.2e}  ({len(idx_list)} elems checked)")


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float64) * 0.1


# ---------------------------------------------------------------------------


def test_linear():
    print("Linear")
    lin = Linear(5, 7, np, dtype=np.float64)
    x = rand(3, 4, 5)
    dy = rand(3, 4, 7)

    def loss():
        return float(np.sum(dy * lin.forward(x)))

    lin.zero_grad()
    lin.forward(x)
    dx = lin.backward(dy)

    g, idx = numerical_grad(loss, x)
    assert_close_at(dx, g, idx, "Linear dx")
    g, idx = numerical_grad(loss, lin.W.data)
    assert_close_at(lin.W.grad, g, idx, "Linear dW")
    g, idx = numerical_grad(loss, lin.b.data)
    assert_close_at(lin.b.grad, g, idx, "Linear db")


def test_layernorm():
    print("LayerNorm")
    ln = LayerNorm(6, np, dtype=np.float64)
    x = rand(3, 4, 6) + 1.0
    dy = rand(3, 4, 6)

    def loss():
        return float(np.sum(dy * ln.forward(x)))

    ln.zero_grad()
    ln.forward(x)
    dx = ln.backward(dy)

    g, idx = numerical_grad(loss, x)
    assert_close_at(dx, g, idx, "LayerNorm dx")
    g, idx = numerical_grad(loss, ln.gamma.data)
    assert_close_at(ln.gamma.grad, g, idx, "LayerNorm dgamma")
    g, idx = numerical_grad(loss, ln.beta.data)
    assert_close_at(ln.beta.grad, g, idx, "LayerNorm dbeta")


def test_gelu():
    print("GELU")
    gelu = GELU(np)
    x = rand(3, 4, 6)
    dy = rand(3, 4, 6)

    def loss():
        return float(np.sum(dy * gelu.forward(x)))

    gelu.forward(x)
    dx = gelu.backward(dy)

    g, idx = numerical_grad(loss, x)
    assert_close_at(dx, g, idx, "GELU dx")


def test_embedding():
    print("Embedding")
    emb = Embedding(9, 5, np, dtype=np.float64)
    idx = RNG.integers(0, 9, size=(2, 6))
    idx[0, 0] = idx[0, 1] = 3  # force a repeated index to exercise scatter-add accumulation
    dy = rand(2, 6, 5)

    def loss():
        return float(np.sum(dy * emb.forward(idx)))

    emb.zero_grad()
    emb.forward(idx)
    emb.backward(dy)

    g, gidx = numerical_grad(loss, emb.weight.data)
    assert_close_at(emb.weight.grad, g, gidx, "Embedding dweight")


def test_causal_self_attention():
    print("CausalSelfAttention")
    attn = CausalSelfAttention(8, 2, block_size=6, xp=np, dtype=np.float64)
    x = rand(2, 5, 8)
    dy = rand(2, 5, 8)

    def loss():
        return float(np.sum(dy * attn.forward(x)))

    attn.zero_grad()
    attn.forward(x)
    dx = attn.backward(dy)

    g, idx = numerical_grad(loss, x)
    assert_close_at(dx, g, idx, "Attention dx")
    g, idx = numerical_grad(loss, attn.c_attn.W.data)
    assert_close_at(attn.c_attn.W.grad, g, idx, "Attention dW_qkv")
    g, idx = numerical_grad(loss, attn.c_proj.W.data)
    assert_close_at(attn.c_proj.W.grad, g, idx, "Attention dW_proj")


def test_mlp():
    print("MLP")
    mlp = MLP(6, np, dtype=np.float64)
    x = rand(2, 3, 6)
    dy = rand(2, 3, 6)

    def loss():
        return float(np.sum(dy * mlp.forward(x)))

    mlp.zero_grad()
    mlp.forward(x)
    dx = mlp.backward(dy)

    g, idx = numerical_grad(loss, x)
    assert_close_at(dx, g, idx, "MLP dx")
    g, idx = numerical_grad(loss, mlp.fc.W.data)
    assert_close_at(mlp.fc.W.grad, g, idx, "MLP dW_fc")


def test_block():
    print("Block (attn+mlp+2xLN, full residual wiring)")
    block = Block(8, 2, block_size=6, xp=np, dtype=np.float64)
    x = rand(2, 5, 8)
    dy = rand(2, 5, 8)

    def loss():
        return float(np.sum(dy * block.forward(x)))

    block.zero_grad()
    block.forward(x)
    dx = block.backward(dy)

    g, idx = numerical_grad(loss, x)
    assert_close_at(dx, g, idx, "Block dx")
    g, idx = numerical_grad(loss, block.ln1.gamma.data)
    assert_close_at(block.ln1.gamma.grad, g, idx, "Block dln1.gamma")


def test_softmax_cross_entropy():
    print("softmax_cross_entropy")
    N, V = 5, 9
    logits = rand(N, V) * 10
    targets = RNG.integers(0, V, size=N)

    def loss():
        l, _ = softmax_cross_entropy(np, logits.copy(), targets)
        return float(l)

    l, dlogits = softmax_cross_entropy(np, logits.copy(), targets)
    g, idx = numerical_grad(loss, logits)
    assert_close_at(dlogits, g, idx, "CE dlogits")


def test_full_model_tiny():
    print("Full GPT (tiny) — catches wiring bugs per-layer checks can't see")
    cfg = GPTConfig(vocab_size=13, block_size=6, n_layer=2, n_head=2, n_embd=8)
    model = GPT(cfg, np, dtype=np.float64)
    B, T = 2, 5
    idx = RNG.integers(0, cfg.vocab_size, size=(B, T))
    targets = RNG.integers(0, cfg.vocab_size, size=(B, T))

    def loss():
        _, l, _ = model.forward(idx, targets=targets)
        return float(l)

    model.zero_grad()
    _, l, dlogits = model.forward(idx, targets=targets)
    model.backward(dlogits)

    # spot-check a representative parameter from each stage of the pipeline:
    # tied embedding, an interior block's attention + MLP + LayerNorm, final LN.
    checks = [
        (model.wte.weight, "wte.weight (tied emb + lm_head)"),
        (model.blocks[0].attn.c_attn.W, "blocks[0].attn.c_attn.W"),
        (model.blocks[0].mlp.fc.W, "blocks[0].mlp.fc.W"),
        (model.blocks[1].ln1.gamma, "blocks[1].ln1.gamma"),
        (model.ln_f.gamma, "ln_f.gamma"),
        (model.wpe.weight, "wpe.weight"),
    ]
    for param, name in checks:
        g, gidx = numerical_grad(loss, param.data, max_elems=40)
        assert_close_at(param.grad, g, gidx, name)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nall {len(tests)} gradient checks passed")
