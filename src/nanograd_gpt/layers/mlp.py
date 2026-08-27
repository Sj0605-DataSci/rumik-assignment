from .dropout import Dropout
from .gelu import GELU
from .linear import Linear
from .module import Module


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
