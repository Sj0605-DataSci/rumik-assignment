from ..backend import scatter_add
from .module import Module


class Embedding(Module):
    def __init__(self, vocab_size, embed_dim, xp, std=0.02, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.weight = self.add_param((xp.random.randn(vocab_size, embed_dim) * std).astype(dtype), xp)
        self._x = None

    def forward(self, x):
        """This just indexes our embedding matrix."""
        self._x = x
        return self.weight.data[x]

    def backward(self, output_grad):
        """add.at (scatter_add on cupy) accumulates repeated indices at
        their positions, e.g. if token id 0 appears twice in a batch, both
        occurrences' gradients land in row 0 of weight.grad, summed."""
        scatter_add(self.xp, self.weight.grad, self._x, output_grad)
        self._x = None
        return None
