from .attention import CausalSelfAttention
from .dropout import Dropout
from .layernorm import LayerNorm
from .mlp import MLP
from .module import Module


class Block(Module):
    def __init__(self, embed_dim, num_heads, block_size, xp, dim_mult=4, dropout_p=0.0, std=0.02, proj_std=None, dtype=None):
        super().__init__()
        dtype = dtype or xp.float32
        self.embed_dim, self.num_heads, self.dropout_p, self.dim_mult = embed_dim, num_heads, dropout_p, dim_mult

        self.attention = self.add_module(
            CausalSelfAttention(embed_dim, num_heads, block_size, xp, std=std, proj_std=proj_std, dtype=dtype)
        )
        self.ff = self.add_module(MLP(embed_dim, xp, dim_mult=dim_mult, std=std, proj_std=proj_std, dtype=dtype))
        self.norm1 = self.add_module(LayerNorm(embed_dim, xp, dtype=dtype))
        self.norm2 = self.add_module(LayerNorm(embed_dim, xp, dtype=dtype))
        self.dropout1 = self.add_module(Dropout(dropout_p, xp))
        self.dropout2 = self.add_module(Dropout(dropout_p, xp))

    def forward(self, x, training=True):
        # pre-norm: Attention + Residual (norm applied to the branch's
        # input, BEFORE it joins the residual stream -- this is the GPT-2
        # ordering; see the module docstring for why it differs from the
        # source repo's post-norm ordering)
        attn = self.attention.forward(self.norm1.forward(x))
        attn = self.dropout1.forward(attn, training)
        x = x + attn

        # pre-norm: FeedForward + Residual
        ff_out = self.ff.forward(self.norm2.forward(x))
        ff_out = self.dropout2.forward(ff_out, training)
        x = x + ff_out

        return x

    def backward(self, output_grad):
        # residual add #2 backward: gradient copies into both branches
        grad_drop = self.dropout2.backward(output_grad)
        grad_norm2_out = self.ff.backward(grad_drop)
        grad_x1_from_ff = self.norm2.backward(grad_norm2_out)
        grad_x1 = output_grad + grad_x1_from_ff

        # residual add #1 backward: same idea
        grad_drop = self.dropout1.backward(grad_x1)
        grad_norm1_out = self.attention.backward(grad_drop)
        grad_x_from_attn = self.norm1.backward(grad_norm1_out)
        grad_x = grad_x1 + grad_x_from_attn

        return grad_x

    def __repr__(self):
        return (
            f"Block(embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout_p={self.dropout_p}, dim_mult={self.dim_mult})"
        )
