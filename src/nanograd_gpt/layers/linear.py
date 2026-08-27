from .module import Module


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
