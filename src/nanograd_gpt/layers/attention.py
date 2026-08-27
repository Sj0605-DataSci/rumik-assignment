import math

from .dropout import Dropout
from .linear import Linear
from .module import Module
from .softmax import softmax, softmax_backward


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
