from .module import Module


class LayerNorm(Module):
    """Normalize over the last axis, then scale/shift by learned gamma/beta."""

    def __init__(self, dim, xp, eps=1e-5, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        self.eps = eps
        self.gamma = self.add_param(xp.ones(dim, dtype=dtype), xp)
        self.beta = self.add_param(xp.zeros(dim, dtype=dtype), xp)
        self._cache = None

    def forward(self, x):
        xp = self.xp
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        std = xp.sqrt(var + self.eps)
        xhat = xc / std
        y = self.gamma.data * xhat + self.beta.data
        self._cache = (xhat, std)
        return y

    def backward(self, dy):
        xp = self.xp
        xhat, std = self._cache
        n = xhat.shape[-1]

        reduce_axes = tuple(range(dy.ndim - 1))
        self.gamma.grad += (dy * xhat).sum(axis=reduce_axes)
        self.beta.grad += dy.sum(axis=reduce_axes)

        dxhat = dy * self.gamma.data
        # standard LayerNorm backward: dxhat depends on all n inputs through
        # both the mean and the variance, hence the two correction terms.
        dx = (
            n * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        ) / (n * std)
        self._cache = None
        return dx
