"""AdamW: pure parameter-update math, no backward pass involved here at all."""


class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [p.xp.zeros_like(p.data) for p in self.params]
        self.v = [p.xp.zeros_like(p.data) for p in self.params]

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_c1 = 1 - b1**self.t
        bias_c2 = 1 - b2**self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
            mhat = self.m[i] / bias_c1
            vhat = self.v[i] / bias_c2
            # decoupled weight decay: skip it on 1-D params (biases, LayerNorm
            # gain/shift) per GPT-2/nanoGPT convention -- only matrices get decayed
            if self.weight_decay > 0 and p.data.ndim >= 2:
                p.data -= lr * self.weight_decay * p.data
            p.data -= lr * mhat / (p.xp.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


def cosine_lr(step, *, warmup_steps, max_steps, max_lr, min_lr):
    import math

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
