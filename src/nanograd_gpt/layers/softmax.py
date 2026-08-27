def softmax(xp, x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = xp.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(xp, p, dp, axis=-1):
    # d(softmax)/dx contracted with upstream dp, without materializing the
    # full Jacobian: ds = p * (dp - sum(dp * p, axis))
    return p * (dp - (dp * p).sum(axis=axis, keepdims=True))


def softmax_cross_entropy(xp, logits, targets):
    """logits: (N, V) float, targets: (N,) int. Returns (loss, dlogits).

    Uses the closed-form combined gradient dlogits = softmax(logits) - one_hot(target),
    derived once analytically rather than differentiating softmax and NLL
    separately (that combined form is both cheaper and numerically nicer -
    no division by small softmax probabilities anywhere).
    """
    N, V = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
    logprobs = shifted - logsumexp
    nll = -logprobs[xp.arange(N), targets]
    loss = nll.mean()

    p = xp.exp(logprobs)
    dlogits = p
    dlogits[xp.arange(N), targets] -= 1.0
    dlogits /= N
    return loss, dlogits
