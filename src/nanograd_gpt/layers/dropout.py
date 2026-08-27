from .module import Module


class Dropout(Module):
    def __init__(self, p, xp):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
        self.xp = xp
        self._mask = None

    def forward(self, x, training):
        if not training or self.p == 0.0:
            return x
        xp = self.xp
        # mask of zeros/ones, scaled by 1/(1-p) so the expected activation
        # magnitude is unchanged -- that's what lets eval mode just skip
        # dropout entirely with no separate rescaling needed there.
        mask = (xp.random.rand(*x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        self._mask = mask
        return x * mask

    def backward(self, output_grad):
        if self._mask is None:
            return output_grad  # forward ran in eval mode (or p==0): identity, so backward is too
        dx = output_grad * self._mask
        self._mask = None
        return dx
