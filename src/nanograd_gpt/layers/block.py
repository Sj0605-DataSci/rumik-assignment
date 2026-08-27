from .attention import CausalSelfAttention
from .layernorm import LayerNorm
from .mlp import MLP
from .module import Module


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
