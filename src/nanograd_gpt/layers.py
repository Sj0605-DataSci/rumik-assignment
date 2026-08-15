"""Hand-written forward/backward for every layer used by the GPT model.

No autograd tape, no generic Tensor class with overloaded ops. Each layer
caches whatever it needs during forward() and consumes that cache in
backward() to produce local gradients via a closed-form derivative I derived
on paper first (see REASONING.md for the per-layer derivation notes). The
"graph" is just the fixed Python composition order in model.py.

Shape convention throughout: batch B, sequence length T, model dim C
(a.k.a. n_embd), number of heads n_head, per-head dim hd = C // n_head.
"""

import math

from .backend import scatter_add
from .param import Param


class Module:
    def __init__(self):
        self._params = []
        self._modules = []

    def add_param(self, data, xp):
        p = Param(data, xp)
        self._params.append(p)
        return p

    def add_module(self, m):
        self._modules.append(m)
        return m

    def parameters(self):
        for p in self._params:
            yield p
        for m in self._modules:
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """y = x @ W + b, x: (..., Cin) -> y: (..., Cout)."""

    def __init__(self, cin, cout, xp, bias=True, std=0.02, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        self.cin, self.cout = cin, cout
        W = (xp.random.randn(cin, cout) * std).astype(dtype)
        self.W = self.add_param(W, xp)
        self.b = self.add_param(xp.zeros(cout, dtype=dtype), xp) if bias else None
        self._x = None

    def forward(self, x):
        self._x = x
        y = x @ self.W.data
        if self.b is not None:
            y = y + self.b.data
        return y

    def backward(self, dy):
        x = self._x
        flat_x = x.reshape(-1, self.cin)
        flat_dy = dy.reshape(-1, self.cout)
        self.W.grad += flat_x.T @ flat_dy
        if self.b is not None:
            self.b.grad += flat_dy.sum(axis=0)
        dx = dy @ self.W.data.T
        self._x = None
        return dx


class LayerNorm(Module):
    """Normalize over the last axis, then scale/shift by learned gamma/beta."""

    def __init__(self, dim, xp, eps=1e-5, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        self.eps = eps
        self.gamma = self.add_param(xp.ones(dim, dtype=dtype), xp)
        self.beta = self.add_param(xp.zeros(dim, dtype=dtype), xp)
        self._cache = None

    def forward(self, x):
        xp = self.xp
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        std = xp.sqrt(var + self.eps)
        xhat = xc / std
        y = self.gamma.data * xhat + self.beta.data
        self._cache = (xhat, std)
        return y

    def backward(self, dy):
        xp = self.xp
        xhat, std = self._cache
        n = xhat.shape[-1]

        reduce_axes = tuple(range(dy.ndim - 1))
        self.gamma.grad += (dy * xhat).sum(axis=reduce_axes)
        self.beta.grad += dy.sum(axis=reduce_axes)

        dxhat = dy * self.gamma.data
        # standard LayerNorm backward: dxhat depends on all n inputs through
        # both the mean and the variance, hence the two correction terms.
        dx = (
            n * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        ) / (n * std)
        self._cache = None
        return dx


class GELU(Module):
    """GPT-2's tanh approximation of GELU (matches nanoGPT/HF exactly)."""

    _C = math.sqrt(2.0 / math.pi)

    def __init__(self, xp):
        super().__init__()
        self.xp = xp
        self._x = None

    def forward(self, x):
        xp = self.xp
        c = self._C
        inner = c * (x + 0.044715 * x**3)
        t = xp.tanh(inner)
        self._x = x
        self._t = t
        return 0.5 * x * (1.0 + t)

    def backward(self, dy):
        x, t = self._x, self._t
        c = self._C
        dinner_dx = c * (1.0 + 3 * 0.044715 * x**2)
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner_dx
        self._x = None
        self._t = None
        return dy * dy_dx


class Embedding(Module):
    """Row lookup into a (num_embeddings, dim) table. Backward is scatter-add."""

    def __init__(self, num_embeddings, dim, xp, std=0.02, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        table = (xp.random.randn(num_embeddings, dim) * std).astype(dtype)
        self.weight = self.add_param(table, xp)
        self._idx = None

    def forward(self, idx):
        self._idx = idx
        return self.weight.data[idx]

    def backward(self, dy):
        # gather's backward is scatter-add, not scatter-assign: the same
        # token id can appear multiple times in a batch and every occurrence
        # must contribute its own gradient into that one row.
        scatter_add(self.xp, self.weight.grad, self._idx, dy)
        self._idx = None
        return None  # idx is not differentiable


class Dropout(Module):
    """Inverted dropout: zero each element independently w.p. p, and rescale
    survivors by 1/(1-p) so the expected activation magnitude is unchanged --
    that's what lets eval mode simply skip dropout with no separate rescaling.
    Backward reapplies the exact same mask (dropout has no learnable params).
    """

    def __init__(self, p, xp):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
        self.xp = xp
        self._mask = None

    def forward(self, x, training):
        if not training or self.p == 0.0:
            return x
        xp = self.xp
        keep = 1.0 - self.p
        mask = (xp.random.rand(*x.shape) < keep).astype(x.dtype) / keep
        self._mask = mask
        return x * mask

    def backward(self, dy):
        if self._mask is None:
            return dy  # forward ran in eval mode (or p==0): identity, so backward is too
        dx = dy * self._mask
        self._mask = None
        return dx


def softmax(xp, x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = xp.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(xp, p, dp, axis=-1):
    # d(softmax)/dx contracted with upstream dp, without materializing the
    # full Jacobian: ds = p * (dp - sum(dp * p, axis))
    return p * (dp - (dp * p).sum(axis=axis, keepdims=True))


class CausalSelfAttention(Module):
    def __init__(self, n_embd, n_head, block_size, xp, std=0.02, proj_std=None, dropout=0.0, dtype=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.xp = xp
        dtype = dtype or xp.float32
        proj_std = std if proj_std is None else proj_std
        self.n_embd, self.n_head = n_embd, n_head
        self.hd = n_embd // n_head
        self.c_attn = self.add_module(Linear(n_embd, 3 * n_embd, xp, std=std, dtype=dtype))
        # c_proj feeds straight into the residual stream, so it gets the
        # GPT-2-paper 1/sqrt(2*n_layer) scaled init, not the general `std`
        # (see GPT.__init__ for where proj_std is computed) -- keeps
        # residual-stream variance from growing unboundedly with depth.
        self.c_proj = self.add_module(Linear(n_embd, n_embd, xp, std=proj_std, dtype=dtype))
        # additive causal mask, -inf strictly above the diagonal
        mask = xp.triu(xp.full((block_size, block_size), -xp.inf, dtype=dtype), k=1)
        self.mask = mask
        self.attn_dropout = self.add_module(Dropout(dropout, xp))
        self.resid_dropout = self.add_module(Dropout(dropout, xp))
        self._cache = None

    def _split_heads(self, x, B, T):
        # (B,T,C) -> (B,nh,T,hd)
        return x.reshape(B, T, self.n_head, self.hd).transpose(0, 2, 1, 3)

    def _merge_heads(self, x, B, T):
        # (B,nh,T,hd) -> (B,T,C)
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.n_embd)

    def forward(self, x, training=True):
        xp = self.xp
        B, T, C = x.shape
        qkv = self.c_attn.forward(x)
        q, k, v = xp.split(qkv, 3, axis=-1)
        q = self._split_heads(q, B, T)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        scale = 1.0 / math.sqrt(self.hd)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale
        att = att + self.mask[:T, :T]
        p = softmax(xp, att, axis=-1)             # true attention probabilities, each row sums to 1
        p_drop = self.attn_dropout.forward(p, training)  # what actually multiplies v below
        y = p_drop @ v  # (B,nh,T,hd)
        y = self._merge_heads(y, B, T)
        out = self.c_proj.forward(y)
        out = self.resid_dropout.forward(out, training)

        # cache BOTH p (needed by softmax_backward's Jacobian formula, which
        # assumes rows summing to 1 -- no longer true after dropout) and
        # p_drop (needed for dv, since the forward matmul actually used p_drop)
        self._cache = (q, k, v, p, p_drop, B, T, scale)
        return out

    def backward(self, dout):
        xp = self.xp
        q, k, v, p, p_drop, B, T, scale = self._cache

        dout = self.resid_dropout.backward(dout)
        dy = self.c_proj.backward(dout)  # (B,T,C)
        dy = self._split_heads(dy, B, T)  # (B,nh,T,hd)

        dp_drop = dy @ v.transpose(0, 1, 3, 2)  # (B,nh,T,T), gradient w.r.t. p_drop
        dv = p_drop.transpose(0, 1, 3, 2) @ dy  # (B,nh,T,hd)

        dp = self.attn_dropout.backward(dp_drop)  # un-does the dropout mask -> gradient w.r.t. p
        datt = softmax_backward(xp, p, dp, axis=-1)  # mask positions: p=0 -> datt=0 there too
        datt = datt * scale

        dq = datt @ k  # (B,nh,T,hd)
        dk = datt.transpose(0, 1, 3, 2) @ q  # (B,nh,T,hd)

        dq = self._merge_heads(dq, B, T)
        dk = self._merge_heads(dk, B, T)
        dv = self._merge_heads(dv, B, T)
        dqkv = xp.concatenate([dq, dk, dv], axis=-1)

        dx = self.c_attn.backward(dqkv)
        self._cache = None
        return dx


class MLP(Module):
    def __init__(self, n_embd, xp, std=0.02, proj_std=None, dropout=0.0, dtype=None):
        super().__init__()
        dtype = dtype or xp.float32
        proj_std = std if proj_std is None else proj_std
        self.fc = self.add_module(Linear(n_embd, 4 * n_embd, xp, std=std, dtype=dtype))
        self.gelu = self.add_module(GELU(xp))
        # proj feeds the residual stream, same 1/sqrt(2*n_layer) scaled init as attn's c_proj
        self.proj = self.add_module(Linear(4 * n_embd, n_embd, xp, std=proj_std, dtype=dtype))
        self.dropout = self.add_module(Dropout(dropout, xp))

    def forward(self, x, training=True):
        x = self.fc.forward(x)
        x = self.gelu.forward(x)
        x = self.proj.forward(x)
        x = self.dropout.forward(x, training)
        return x

    def backward(self, dy):
        dy = self.dropout.backward(dy)
        dy = self.proj.backward(dy)
        dy = self.gelu.backward(dy)
        dy = self.fc.backward(dy)
        return dy


class Block(Module):
    """Pre-norm GPT-2 transformer block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, n_embd, n_head, block_size, xp, std=0.02, proj_std=None, dropout=0.0, dtype=None):
        super().__init__()
        dtype = dtype or xp.float32
        self.ln1 = self.add_module(LayerNorm(n_embd, xp, dtype=dtype))
        self.attn = self.add_module(
            CausalSelfAttention(
                n_embd, n_head, block_size, xp, std=std, proj_std=proj_std, dropout=dropout, dtype=dtype
            )
        )
        self.ln2 = self.add_module(LayerNorm(n_embd, xp, dtype=dtype))
        self.mlp = self.add_module(MLP(n_embd, xp, std=std, proj_std=proj_std, dropout=dropout, dtype=dtype))

    def forward(self, x, training=True):
        a = self.attn.forward(self.ln1.forward(x), training=training)
        x = x + a
        self._res1_in = x  # input to the second residual branch
        m = self.mlp.forward(self.ln2.forward(x), training=training)
        x = x + m
        return x

    def backward(self, dout):
        # residual add: gradient splits identically into both branches
        dm = dout
        dx_res2 = dout
        dln2_out = self.mlp.backward(dm)
        dx_res2 = dx_res2 + self.ln2.backward(dln2_out)

        da = dx_res2
        dx_res1 = dx_res2
        dln1_out = self.attn.backward(da)
        dx_res1 = dx_res1 + self.ln1.backward(dln1_out)
        return dx_res1


def softmax_cross_entropy(xp, logits, targets):
    """logits: (N, V) float, targets: (N,) int. Returns (loss, dlogits).

    Uses the closed-form combined gradient dlogits = softmax(logits) - one_hot(target),
    derived once analytically rather than differentiating softmax and NLL
    separately (that combined form is both cheaper and numerically nicer -
    no division by small softmax probabilities anywhere).
    """
    N, V = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
    logprobs = shifted - logsumexp
    nll = -logprobs[xp.arange(N), targets]
    loss = nll.mean()

    p = xp.exp(logprobs)
    dlogits = p
    dlogits[xp.arange(N), targets] -= 1.0
    dlogits /= N
    return loss, dlogits
