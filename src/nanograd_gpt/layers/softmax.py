def softmax_slow(xp, x, axis=-1):
    """Naive softmax backward: builds each sample's full n x n Jacobian
    explicitly. O(n^2) memory per row; only usable for small n. Included
    for the derivation, not used in the actual attention path below."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = xp.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def softmax_slow_backward(xp, probs, output_grad):
    assert probs.ndim == 2, "softmax_slow_backward is the explicit-Jacobian path, 2D only"
    grad = xp.zeros_like(probs)
    for i in range(probs.shape[0]):
        p = probs[i]
        jacobian = -p.reshape(-1, 1) * p.reshape(1, -1)
        diag = xp.arange(p.shape[0])
        jacobian[diag, diag] = p * (1 - p)
        grad[i] = output_grad[i] @ jacobian
    return grad


def softmax(xp, x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = xp.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(xp, probs, output_grad, axis=-1):
    # Vectorized equivalent of softmax_slow_backward's explicit Jacobian:
    # ds = p * (dy - sum(dy * p)). Same result, no O(n^2) Jacobian built.
    dot_product = (output_grad * probs).sum(axis=axis, keepdims=True)
    return probs * (output_grad - dot_product)


def softmax_cross_entropy(xp, logits, targets):
    """logits: (N, V) raw scores (not softmaxed). targets: (N,) integer
    class indices. Returns (loss, dlogits) using the combined closed-form
    gradient softmax(logits) - one_hot(target) -- no separate softmax
    backward needed, and no dividing by a possibly-tiny predicted
    probability anywhere (see ManualTransformer's SLOW_CrossEntropyLoss for
    the version that does divide by p, which is both slower and the reason
    the fast combined form is preferred)."""
    N, V = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = xp.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    correct_class_probs = probs[xp.arange(N), targets]
    loss = -xp.mean(xp.log(correct_class_probs + 1e-12))

    dlogits = probs.copy()
    dlogits[xp.arange(N), targets] -= 1
    dlogits /= N  # bug fix vs. the source: forward is a MEAN over N, so backward must divide by N too
    return loss, dlogits
