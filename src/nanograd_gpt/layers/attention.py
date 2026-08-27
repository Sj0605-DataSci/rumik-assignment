import math

from .linear import Linear
from .module import Module
from .softmax import softmax, softmax_backward


class CausalSelfAttention(Module):
    def __init__(self, embed_dim, num_heads, block_size, xp, std=0.02, proj_std=None, dtype=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.xp = xp
        dtype = dtype or xp.float32
        proj_std = std if proj_std is None else proj_std
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = self.add_module(Linear(embed_dim, embed_dim, xp, std=std, dtype=dtype))
        self.k_linear = self.add_module(Linear(embed_dim, embed_dim, xp, std=std, dtype=dtype))
        self.v_linear = self.add_module(Linear(embed_dim, embed_dim, xp, std=std, dtype=dtype))
        self.out_linear = self.add_module(Linear(embed_dim, embed_dim, xp, std=proj_std, dtype=dtype))

        self.mask = xp.triu(xp.full((block_size, block_size), -xp.inf, dtype=dtype), k=1)
        self._cache = None

    def forward(self, x):
        """x: (batch_size, seq_len, embed_dim)."""
        xp = self.xp
        batch_size, seq_len, _ = x.shape

        x_flat = x.reshape(batch_size * seq_len, self.embed_dim)
        q = self.q_linear.forward(x_flat)
        k = self.k_linear.forward(x_flat)
        v = self.v_linear.forward(x_flat)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        scores = scores + self.mask[:seq_len, :seq_len]

        probs = softmax(xp, scores, axis=-1)
        attn = probs @ v  # (batch, heads, seq, head_dim)

        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        attn_flat = attn.reshape(batch_size * seq_len, self.embed_dim)
        out = self.out_linear.forward(attn_flat)
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        self._cache = (q, k, v, probs, batch_size, seq_len, scale)
        return out

    def backward(self, output_grad):
        xp = self.xp
        q, k, v, probs, batch_size, seq_len, scale = self._cache

        output_grad_flat = output_grad.reshape(batch_size * seq_len, self.embed_dim)
        grad_attn_flat = self.out_linear.backward(output_grad_flat)
        grad_attn = grad_attn_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Backward through attn = probs @ v: dL/dprobs = dL/dattn . v^T, dL/dv = probs^T . dL/dattn
        grad_probs = grad_attn @ v.transpose(0, 1, 3, 2)
        grad_v = probs.transpose(0, 1, 3, 2) @ grad_attn

        grad_scores = softmax_backward(xp, probs, grad_probs, axis=-1)
        # masked positions already have probs=0, so softmax_backward already
        # produced exactly 0 there -- no extra masking step needed here
        grad_scores = grad_scores * scale

        # scores = q @ k^T: dL/dQ = dL/dS . K, dL/dK = dL/dS^T . Q
        grad_q = grad_scores @ k
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ q

        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        grad_q_flat = grad_q.reshape(batch_size * seq_len, self.embed_dim)
        grad_k_flat = grad_k.reshape(batch_size * seq_len, self.embed_dim)
        grad_v_flat = grad_v.reshape(batch_size * seq_len, self.embed_dim)

        grad_query = self.q_linear.backward(grad_q_flat)
        grad_key = self.k_linear.backward(grad_k_flat)
        grad_value = self.v_linear.backward(grad_v_flat)

        grad_query = grad_query.reshape(batch_size, seq_len, self.embed_dim)
        grad_key = grad_key.reshape(batch_size, seq_len, self.embed_dim)
        grad_value = grad_value.reshape(batch_size, seq_len, self.embed_dim)

        self._cache = None
        # self-attention: query=key=value=x, so the three gradient paths
        # back into x all add together
        return grad_query + grad_key + grad_value
