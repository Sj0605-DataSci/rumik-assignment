from ..param import Param


class Module:
    """Base class every layer inherits: tracks its own Params and sub-Modules
    so parameters()/zero_grad() can walk the whole model without any layer
    needing to know about its parents or siblings.
    """

    def __init__(self):
        self._params = []
        self._modules = []

    def add_param(self, data, xp):
        p = Param(data, xp)
        self._params.append(p)
        return p

    def add_module(self, m):
        self._modules.append(m)
        return m

    def parameters(self):
        for p in self._params:
            yield p
        for m in self._modules:
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
