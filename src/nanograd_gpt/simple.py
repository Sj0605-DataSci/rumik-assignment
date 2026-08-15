"""GPT-2, rewritten with zero classes and zero hidden state.

This file is a teaching companion to layers.py/model.py (the version actually
training on the GPU right now) -- same math, same architecture, but every
function is a pure function: it takes plain numpy/cupy arrays in, returns
plain arrays out, and NEVER stashes anything on `self` because there is no
`self`. Parameters live in a plain nested dict. Every function takes `xp`
(numpy or cupy) as an explicit argument, so swapping backend is just passing
a different module in -- nothing about the functions themselves changes.

Read top to bottom: linear -> layernorm -> gelu -> embedding -> softmax ->
attention -> mlp -> block -> full model -> loss -> optimizer.
"""

import math


# ============================================================
# LINEAR: y = x @ W + b
# ============================================================

def linear_forward(x, W, b, xp):
    # x: (..., cin), W: (cin, cout), b: (cout,) -> y: (..., cout)
    y = x @ W + b
    cache = (x, W)  # need x and W again in the backward pass
    return y, cache


def linear_backward(dy, cache, xp):
    x, W = cache
    cin, cout = W.shape
    x_flat = x.reshape(-1, cin)     # flatten every leading axis (B,T,...) into one
    dy_flat = dy.reshape(-1, cout)
    dW = x_flat.T @ dy_flat         # (cin, cout), same shape as W
    db = dy_flat.sum(axis=0)        # (cout,), same shape as b
    dx = dy @ W.T                   # (..., cin), same shape as x
    return dx, dW, db


# ============================================================
# LAYERNORM: normalize over the last axis, then scale/shift
# ============================================================

def layernorm_forward(x, gamma, beta, xp, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    std = xp.sqrt(var + eps)
    xhat = xc / std
    y = gamma * xhat + beta
    cache = (xhat, std)
    return y, cache


def layernorm_backward(dy, cache, gamma, xp):
    xhat, std = cache
    n = xhat.shape[-1]
    reduce_axes = tuple(range(dy.ndim - 1))  # every axis except the last (C) one
    dgamma = (dy * xhat).sum(axis=reduce_axes)
    dbeta = dy.sum(axis=reduce_axes)
    dxhat = dy * gamma
    dx = (
        n * dxhat
        - dxhat.sum(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
    ) / (n * std)
    return dx, dgamma, dbeta


# ============================================================
# GELU (GPT-2's tanh approximation)
# ============================================================

_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_forward(x, xp):
    inner = _GELU_C * (x + 0.044715 * x**3)
    t = xp.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    cache = (x, t)
    return y, cache


def gelu_backward(dy, cache, xp):
    x, t = cache
    dinner_dx = _GELU_C * (1.0 + 3 * 0.044715 * x**2)
    dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner_dx
    dx = dy * dy_dx
    return dx


# ============================================================
# EMBEDDING: row lookup by integer index
# ============================================================

def embedding_forward(table, idx, xp):
    y = table[idx]     # fancy indexing: gathers one row per index
    cache = idx
    return y, cache


def embedding_backward(dy, cache, table_shape, xp):
    idx = cache
    dtable = xp.zeros(table_shape, dtype=dy.dtype)
    if xp.__name__ == "numpy":
        xp.add.at(dtable, idx, dy)          # accumulate, don't overwrite, at repeated idx
    else:
        import cupyx
        cupyx.scatter_add(dtable, idx, dy)  # cupy's equivalent of np.add.at
    return dtable


# ============================================================
# SOFTMAX
# ============================================================

def softmax_forward(x, xp):
    x = x - x.max(axis=-1, keepdims=True)   # subtract max so exp() never overflows
    e = xp.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_backward(p, dp, xp):
    # contracted Jacobian-vector product: ds = p * (dp - sum(dp*p))
    return p * (dp - (dp * p).sum(axis=-1, keepdims=True))


# ============================================================
# CAUSAL SELF-ATTENTION
# ============================================================

def make_causal_mask(block_size, xp, dtype):
    return xp.triu(xp.full((block_size, block_size), -xp.inf, dtype=dtype), k=1)


def split_heads(t, n_head, xp):
    B, T, C = t.shape
    hd = C // n_head
    return t.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)  # (B,T,C) -> (B,nh,T,hd)


def merge_heads(t, xp):
    B, nh, T, hd = t.shape
    return t.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)  # (B,nh,T,hd) -> (B,T,C)


def attention_forward(x, W_qkv, b_qkv, W_proj, b_proj, n_head, mask, xp):
    B, T, C = x.shape
    hd = C // n_head

    qkv, lin1_cache = linear_forward(x, W_qkv, b_qkv, xp)   # (B,T,3C)
    q, k, v = xp.split(qkv, 3, axis=-1)                     # each (B,T,C)
    q, k, v = split_heads(q, n_head, xp), split_heads(k, n_head, xp), split_heads(v, n_head, xp)

    scale = 1.0 / math.sqrt(hd)
    att = (q @ k.transpose(0, 1, 3, 2)) * scale             # (B,nh,T,T)
    att = att + mask[:T, :T]                                 # -inf above the diagonal
    p = softmax_forward(att, xp)                             # attention weights

    y = p @ v                                                 # (B,nh,T,hd)
    y = merge_heads(y, xp)                                    # (B,T,C)
    out, lin2_cache = linear_forward(y, W_proj, b_proj, xp)   # (B,T,C)

    cache = (q, k, v, p, lin1_cache, lin2_cache, n_head, scale)
    return out, cache


def attention_backward(dout, cache, xp):
    q, k, v, p, lin1_cache, lin2_cache, n_head, scale = cache

    dy, dW_proj, db_proj = linear_backward(dout, lin2_cache, xp)
    dy = split_heads(dy, n_head, xp)                          # (B,nh,T,hd)

    dp = dy @ v.transpose(0, 1, 3, 2)                         # (B,nh,T,T)
    dv = p.transpose(0, 1, 3, 2) @ dy                         # (B,nh,T,hd)

    datt = softmax_backward(p, dp, xp) * scale
    dq = datt @ k
    dk = datt.transpose(0, 1, 3, 2) @ q

    dq, dk, dv = merge_heads(dq, xp), merge_heads(dk, xp), merge_heads(dv, xp)
    dqkv = xp.concatenate([dq, dk, dv], axis=-1)

    dx, dW_qkv, db_qkv = linear_backward(dqkv, lin1_cache, xp)
    return dx, dW_qkv, db_qkv, dW_proj, db_proj


# ============================================================
# MLP: Linear -> GELU -> Linear
# ============================================================

def mlp_forward(x, W_fc, b_fc, W_proj, b_proj, xp):
    h, c1 = linear_forward(x, W_fc, b_fc, xp)
    a, c2 = gelu_forward(h, xp)
    y, c3 = linear_forward(a, W_proj, b_proj, xp)
    return y, (c1, c2, c3)


def mlp_backward(dy, cache, xp):
    c1, c2, c3 = cache
    da, dW_proj, db_proj = linear_backward(dy, c3, xp)
    dh = gelu_backward(da, c2, xp)
    dx, dW_fc, db_fc = linear_backward(dh, c1, xp)
    return dx, dW_fc, db_fc, dW_proj, db_proj


# ============================================================
# ONE TRANSFORMER BLOCK: pre-norm, attn + residual, mlp + residual
# ============================================================

def block_forward(x, p, n_head, mask, xp):
    # p is this block's own little dict of parameters
    ln1_out, ln1_cache = layernorm_forward(x, p["ln1_gamma"], p["ln1_beta"], xp)
    attn_out, attn_cache = attention_forward(
        ln1_out, p["attn_W_qkv"], p["attn_b_qkv"], p["attn_W_proj"], p["attn_b_proj"], n_head, mask, xp
    )
    x1 = x + attn_out  # residual add #1

    ln2_out, ln2_cache = layernorm_forward(x1, p["ln2_gamma"], p["ln2_beta"], xp)
    mlp_out, mlp_cache = mlp_forward(ln2_out, p["mlp_W_fc"], p["mlp_b_fc"], p["mlp_W_proj"], p["mlp_b_proj"], xp)
    x2 = x1 + mlp_out  # residual add #2

    cache = (ln1_cache, attn_cache, ln2_cache, mlp_cache, p["ln1_gamma"], p["ln2_gamma"])
    return x2, cache


def block_backward(dout, cache, xp):
    ln1_cache, attn_cache, ln2_cache, mlp_cache, ln1_gamma, ln2_gamma = cache

    # residual add #2 backward: gradient copies into both branches unchanged
    dx1_from_residual = dout
    dln2_out, dW_fc, db_fc, dW_proj_mlp, db_proj_mlp = mlp_backward(dout, mlp_cache, xp)
    dx1_from_mlp, dgamma2, dbeta2 = layernorm_backward(dln2_out, ln2_cache, ln2_gamma, xp)
    dx1 = dx1_from_residual + dx1_from_mlp

    # residual add #1 backward: same idea
    dx_from_residual = dx1
    dln1_out, dW_qkv, db_qkv, dW_proj_attn, db_proj_attn = attention_backward(dx1, attn_cache, xp)
    dx_from_attn, dgamma1, dbeta1 = layernorm_backward(dln1_out, ln1_cache, ln1_gamma, xp)
    dx = dx_from_residual + dx_from_attn

    grads = {
        "ln1_gamma": dgamma1, "ln1_beta": dbeta1,
        "attn_W_qkv": dW_qkv, "attn_b_qkv": db_qkv,
        "attn_W_proj": dW_proj_attn, "attn_b_proj": db_proj_attn,
        "ln2_gamma": dgamma2, "ln2_beta": dbeta2,
        "mlp_W_fc": dW_fc, "mlp_b_fc": db_fc,
        "mlp_W_proj": dW_proj_mlp, "mlp_b_proj": db_proj_mlp,
    }
    return dx, grads


# ============================================================
# FULL MODEL: params dict in, logits/loss out
# ============================================================

def init_params(vocab_size, block_size, n_layer, n_head, n_embd, xp, std=0.02, dtype=None):
    dtype = dtype or xp.float32

    def randn(*shape):
        return (xp.random.randn(*shape) * std).astype(dtype)

    params = {
        "wte": randn(vocab_size, n_embd),
        "wpe": randn(block_size, n_embd),
        "lnf_gamma": xp.ones(n_embd, dtype=dtype),
        "lnf_beta": xp.zeros(n_embd, dtype=dtype),
        "blocks": [],
    }
    for _ in range(n_layer):
        params["blocks"].append({
            "ln1_gamma": xp.ones(n_embd, dtype=dtype), "ln1_beta": xp.zeros(n_embd, dtype=dtype),
            "attn_W_qkv": randn(n_embd, 3 * n_embd), "attn_b_qkv": xp.zeros(3 * n_embd, dtype=dtype),
            "attn_W_proj": randn(n_embd, n_embd), "attn_b_proj": xp.zeros(n_embd, dtype=dtype),
            "ln2_gamma": xp.ones(n_embd, dtype=dtype), "ln2_beta": xp.zeros(n_embd, dtype=dtype),
            "mlp_W_fc": randn(n_embd, 4 * n_embd), "mlp_b_fc": xp.zeros(4 * n_embd, dtype=dtype),
            "mlp_W_proj": randn(4 * n_embd, n_embd), "mlp_b_proj": xp.zeros(n_embd, dtype=dtype),
        })
    return params


def softmax_cross_entropy(logits, targets, xp):
    N, V = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
    logprobs = shifted - logsumexp
    nll = -logprobs[xp.arange(N), targets]
    loss = nll.mean()

    p = xp.exp(logprobs)
    dlogits = p.copy()
    dlogits[xp.arange(N), targets] -= 1.0
    dlogits /= N
    return loss, dlogits


def gpt_forward(params, idx, n_head, mask, xp, targets=None):
    B, T = idx.shape
    pos = xp.broadcast_to(xp.arange(T, dtype=xp.int64), (B, T))

    tok_emb, tok_cache = embedding_forward(params["wte"], idx, xp)
    pos_emb, pos_cache = embedding_forward(params["wpe"], pos, xp)
    x = tok_emb + pos_emb

    block_caches = []
    for bp in params["blocks"]:
        x, c = block_forward(x, bp, n_head, mask, xp)
        block_caches.append(c)

    h, lnf_cache = layernorm_forward(x, params["lnf_gamma"], params["lnf_beta"], xp)
    logits = h @ params["wte"].T  # weight tying: reuse wte, no separate output matrix

    loss, dlogits = None, None
    if targets is not None:
        C = h.shape[-1]
        N, V = B * T, params["wte"].shape[0]
        loss, dlogits = softmax_cross_entropy(logits.reshape(N, V), targets.reshape(N), xp)

    cache = (tok_cache, pos_cache, block_caches, lnf_cache, h, B, T)
    return logits, loss, dlogits, cache


def gpt_backward(dlogits, params, cache, xp):
    tok_cache, pos_cache, block_caches, lnf_cache, h, B, T = cache
    C = h.shape[-1]
    N, V = B * T, params["wte"].shape[0]

    h_flat = h.reshape(N, C)
    dlogits = dlogits.reshape(N, V)

    # logits = h @ wte.T -- two things fall out of this one line's backward:
    dh = dlogits @ params["wte"]                 # gradient flowing back into h
    dwte_from_head = dlogits.T @ h_flat           # gradient into wte, source #1 of 2

    dh = dh.reshape(B, T, C)
    dx, dlnf_gamma, dlnf_beta = layernorm_backward(dh, lnf_cache, params["lnf_gamma"], xp)

    grads = {"blocks": [None] * len(block_caches)}
    for i in reversed(range(len(block_caches))):
        dx, block_grads = block_backward(dx, block_caches[i], xp)
        grads["blocks"][i] = block_grads

    dwte_from_emb = embedding_backward(dx, tok_cache, params["wte"].shape, xp)  # source #2 of 2
    dwpe = embedding_backward(dx, pos_cache, params["wpe"].shape, xp)

    grads["wte"] = dwte_from_head + dwte_from_emb  # the two sources ADD -- this is weight tying
    grads["wpe"] = dwpe
    grads["lnf_gamma"] = dlnf_gamma
    grads["lnf_beta"] = dlnf_beta
    return grads


# ============================================================
# ADAMW -- plain parameter-update math, walks the same nested
# dict shape as params/grads, no forward/backward involved
# ============================================================

def adamw_init_state(params, xp):
    if isinstance(params, dict):
        return {k: adamw_init_state(v, xp) for k, v in params.items()}
    if isinstance(params, list):
        return [adamw_init_state(v, xp) for v in params]
    return {"m": xp.zeros_like(params), "v": xp.zeros_like(params)}


def adamw_step(params, grads, state, xp, t, lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1):
    if isinstance(params, dict):
        for key in params:
            adamw_step(params[key], grads[key], state[key], xp, t, lr, beta1, beta2, eps, weight_decay)
        return
    if isinstance(params, list):
        for i in range(len(params)):
            adamw_step(params[i], grads[i], state[i], xp, t, lr, beta1, beta2, eps, weight_decay)
        return

    # base case: params is a single array (a leaf of the params tree)
    m, v = state["m"], state["v"]
    m[...] = beta1 * m + (1 - beta1) * grads
    v[...] = beta2 * v + (1 - beta2) * (grads * grads)
    mhat = m / (1 - beta1**t)
    vhat = v / (1 - beta2**t)
    if weight_decay > 0 and params.ndim >= 2:
        params -= lr * weight_decay * params
    params -= lr * mhat / (xp.sqrt(vhat) + eps)
