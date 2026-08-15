class Param:
    """A trainable array plus its gradient buffer.

    .grad is accumulated (+=) during backward rather than overwritten, so
    that weight-tied parameters (wte shared between the embedding lookup and
    the output projection) collect gradient contributions from every place
    they're used instead of the second use clobbering the first.
    """

    def __init__(self, data, xp):
        self.data = data
        self.xp = xp
        self.grad = xp.zeros_like(data)

    def zero_grad(self):
        self.grad[...] = 0

    @property
    def shape(self):
        return self.data.shape
