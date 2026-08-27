from .module import Module


class Dropout(Module):
    """Inverted dropout: zero each element independently w.p. p, and rescale
    survivors by 1/(1-p) so the expected activation magnitude is unchanged --
    that's what lets eval mode simply skip dropout with no separate rescaling.
    Backward reapplies the exact same mask (dropout has no learnable params).
    """

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
        keep = 1.0 - self.p
        mask = (xp.random.rand(*x.shape) < keep).astype(x.dtype) / keep
        self._mask = mask
        return x * mask

    def backward(self, dy):
        if self._mask is None:
            return dy  # forward ran in eval mode (or p==0): identity, so backward is too
        dx = dy * self._mask
        self._mask = None
        return dx
