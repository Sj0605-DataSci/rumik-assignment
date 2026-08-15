import math
from dataclasses import dataclass

from .layers import Block, Dropout, Embedding, LayerNorm, Module, softmax, softmax_cross_entropy


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # nanoGPT convention: padded up from GPT-2's 50257 for tensor-core-friendly shapes
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # nanoGPT's own pretraining default; nonzero mainly matters for finetuning


class GPT(Module):
    def __init__(self, config: GPTConfig, xp, std=0.02, dtype=None):
        super().__init__()
        self.config = config
        self.xp = xp
        dtype = dtype or xp.float32

        # GPT-2-paper init: every residual-stream-writing projection (attn's
        # c_proj, MLP's final proj) gets its init std scaled down by
        # 1/sqrt(2*n_layer), so that as depth grows, each block's
        # contribution to the residual stream's variance doesn't grow
        # unboundedly with it. Only c_proj/proj get this; c_attn/fc (which
        # don't write directly into the residual stream) keep the plain std.
        proj_std = std / math.sqrt(2 * config.n_layer)

        self.wte = self.add_module(Embedding(config.vocab_size, config.n_embd, xp, std=std, dtype=dtype))
        self.wpe = self.add_module(Embedding(config.block_size, config.n_embd, xp, std=std, dtype=dtype))
        self.drop = self.add_module(Dropout(config.dropout, xp))
        self.blocks = [
            self.add_module(
                Block(
                    config.n_embd, config.n_head, config.block_size, xp,
                    std=std, proj_std=proj_std, dropout=config.dropout, dtype=dtype,
                )
            )
            for _ in range(config.n_layer)
        ]
        self.ln_f = self.add_module(LayerNorm(config.n_embd, xp, dtype=dtype))
        # lm_head is weight-tied to wte: no separate Linear/param, logits = h @ wte.T.
        # See GPT.backward for how gradient from *this* usage accumulates
        # into the same wte.weight.grad buffer as the embedding lookup's.

        self._cache = None

    def forward(self, idx, targets=None, training=True):
        xp = self.xp
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = xp.arange(T, dtype=xp.int64)
        pos = xp.broadcast_to(pos, (B, T))
        tok_emb = self.wte.forward(idx)
        pos_emb = self.wpe.forward(pos)
        x = tok_emb + pos_emb
        x = self.drop.forward(x, training)

        for block in self.blocks:
            x = block.forward(x, training=training)
        h = self.ln_f.forward(x)  # (B,T,C)

        logits = h @ self.wte.weight.data.T  # weight tying, no bias (GPT-2 convention)

        loss = None
        dlogits = None
        if targets is not None:
            N, V = B * T, self.config.vocab_size
            loss, dlogits = softmax_cross_entropy(xp, logits.reshape(N, V), targets.reshape(N))

        self._cache = (h, B, T)
        return logits, loss, dlogits

    def backward(self, dlogits):
        """Call after forward(..., targets=...); pass the dlogits it returned."""
        xp = self.xp
        h, B, T = self._cache
        C = self.config.n_embd
        N, V = B * T, self.config.vocab_size

        h_flat = h.reshape(N, C)
        dlogits = dlogits.reshape(N, V)

        # logits = h @ W.T  (W == wte.weight.data, shape (V,C))
        dh = dlogits @ self.wte.weight.data  # (N,C)
        self.wte.weight.grad += dlogits.T @ h_flat  # accumulates alongside the embedding-lookup usage below

        dh = dh.reshape(B, T, C)
        dx = self.ln_f.backward(dh)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        dx = self.drop.backward(dx)

        dtok_emb = dx
        dpos_emb = dx
        self.wte.backward(dtok_emb)  # scatter-adds into the SAME wte.weight.grad as above
        self.wpe.backward(dpos_emb)
        self._cache = None

    @staticmethod
    def loss_and_backward(model: "GPT", idx, targets):
        logits, loss, dlogits = model.forward(idx, targets=targets)
        model.backward(dlogits)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """Greedy/temperature sampling for eval/inspection. Forward-only, no backward."""
        xp = self.xp
        rng = rng or xp.random
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self.forward(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                kth = xp.sort(logits, axis=-1)[:, -k][:, None]
                logits = xp.where(logits < kth, -xp.inf, logits)
            probs = softmax(xp, logits, axis=-1)
            probs_host = probs if xp.__name__ == "numpy" else probs.get()
            import numpy as np

            next_ids = np.array(
                [np.random.choice(probs_host.shape[-1], p=probs_host[b]) for b in range(probs_host.shape[0])]
            )
            next_ids = xp.asarray(next_ids).reshape(-1, 1)
            idx = xp.concatenate([idx, next_ids], axis=1)
        return idx
