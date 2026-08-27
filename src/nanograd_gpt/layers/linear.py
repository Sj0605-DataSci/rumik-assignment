from .module import Module


class Linear(Module):
    """y = x @ W + b. Input must already be 2D (flatten batch/seq dims
    before calling, same convention the source repo uses)."""

    def __init__(self, in_features, out_features, xp, bias=True, std=0.02, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        self.in_features, self.out_features = in_features, out_features
        self.W = self.add_param((xp.random.randn(in_features, out_features) * std).astype(dtype), xp)
        self.b = self.add_param(xp.zeros(out_features, dtype=dtype), xp) if bias else None
        self._x = None

    def forward(self, x):
        self._x = x
        out = x @ self.W.data
        if self.b is not None:
            out = out + self.b.data
        return out

    def backward(self, output_grad):
        # Grad w.r.t weight
        self.W.grad += self._x.T @ output_grad
        # Grad w.r.t bias
        if self.b is not None:
            self.b.grad += output_grad.sum(axis=0)
        # Grad w.r.t input
        dx = output_grad @ self.W.data.T
        self._x = None
        return dx
