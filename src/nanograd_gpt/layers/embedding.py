from ..backend import scatter_add
from .module import Module


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
