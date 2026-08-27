import math

from .module import Module


class GELU(Module):
    """GPT-2's tanh approximation of GELU (matches nanoGPT/HF exactly)."""

    _C = math.sqrt(2.0 / math.pi)

    def __init__(self, xp):
        super().__init__()
        self.xp = xp
        self._x = None

    def forward(self, x):
        xp = self.xp
        c = self._C
        inner = c * (x + 0.044715 * x**3)
        t = xp.tanh(inner)
        self._x = x
        self._t = t
        return 0.5 * x * (1.0 + t)

    def backward(self, dy):
        x, t = self._x, self._t
        c = self._C
        dinner_dx = c * (1.0 + 3 * 0.044715 * x**2)
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner_dx
        self._x = None
        self._t = None
        return dy * dy_dx
