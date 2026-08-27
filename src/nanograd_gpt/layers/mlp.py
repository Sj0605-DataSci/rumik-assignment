from .gelu import GELU
from .linear import Linear
from .module import Module


class MLP(Module):
    def __init__(self, embed_dim, xp, dim_mult=4, std=0.02, proj_std=None, dtype=None):
        super().__init__()
        dtype = dtype or xp.float32
        proj_std = std if proj_std is None else proj_std
        self.linear1 = self.add_module(Linear(embed_dim, embed_dim * dim_mult, xp, std=std, dtype=dtype))
        self.gelu = self.add_module(GELU(xp))
        self.linear2 = self.add_module(Linear(embed_dim * dim_mult, embed_dim, xp, std=proj_std, dtype=dtype))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # flatten (B,S,E) -> (B*S,E) since Linear only takes 2D input
        x_flat = x.reshape(batch_size * seq_len, embed_dim)
        x_flat = self.linear1.forward(x_flat)
        x_flat = self.gelu.forward(x_flat)
        x_flat = self.linear2.forward(x_flat)
        return x_flat.reshape(batch_size, seq_len, embed_dim)

    def backward(self, output_grad):
        batch_size, seq_len, embed_dim = output_grad.shape
        grad_flat = output_grad.reshape(batch_size * seq_len, embed_dim)
        grad_flat = self.linear2.backward(grad_flat)
        grad_flat = self.gelu.backward(grad_flat)
        grad_flat = self.linear1.backward(grad_flat)
        return grad_flat.reshape(batch_size, seq_len, embed_dim)
