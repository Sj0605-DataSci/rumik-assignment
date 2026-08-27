from .module import Module


class LayerNorm(Module):
    def __init__(self, num_features, xp, eps=1e-5, dtype=None):
        super().__init__()
        self.xp = xp
        dtype = dtype or xp.float32
        self.num_features = num_features
        self.eps = eps
        self.gamma = self.add_param(xp.ones(num_features, dtype=dtype), xp)
        self.beta = self.add_param(xp.zeros(num_features, dtype=dtype), xp)
        self._cache = None

    def forward(self, x):
        """x: (B, S, E) -> normalized and affine-transformed output."""
        xp = self.xp
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / xp.sqrt(var + self.eps)
        out = self.gamma.data * x_hat + self.beta.data
        self._cache = (x_hat, var)
        return out

    def backward(self, output_grad):
        xp = self.xp
        x_hat, var = self._cache
        n = x_hat.shape[-1]
        reduce_axes = tuple(range(output_grad.ndim - 1))

        # Gradient w.r.t gamma and beta (affine transform)
        self.gamma.grad += (output_grad * x_hat).sum(axis=reduce_axes)
        self.beta.grad += output_grad.sum(axis=reduce_axes)

        # Gradient w.r.t normalized input
        dx_hat = output_grad * self.gamma.data
        var_eps = var + self.eps

        # LayerNorm backward formula (vectorized): dx_hat depends on all n
        # inputs through both the mean and the variance, hence the two
        # correction terms mean1 (from the mean) and mean2 (from the variance).
        mean1 = dx_hat.mean(axis=-1, keepdims=True)
        mean2 = (dx_hat * x_hat).mean(axis=-1, keepdims=True)
        dx = (dx_hat - mean1 - x_hat * mean2) / xp.sqrt(var_eps)

        self._cache = None
        return dx
