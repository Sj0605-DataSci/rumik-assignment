"""Array backend switch: numpy for CPU (gradient checks, tests), cupy for GPU training.

No autograd lives here or anywhere else in this package - cupy is used purely
as a drop-in ndarray/BLAS replacement, exactly like numpy. Every gradient in
this codebase is computed by hand in layers.py.
"""

import numpy as np

_CUPY = None


def get_xp(device: str = "cpu"):
    if device == "cpu":
        return np
    if device == "cuda":
        global _CUPY
        if _CUPY is None:
            import cupy as cp

            _CUPY = cp
        return _CUPY
    raise ValueError(f"unknown device {device!r}")


def scatter_add(xp, dst, idx, src):
    """dst[idx] += src, with accumulation on repeated indices (unlike dst[idx] = ...)."""
    if xp is np:
        np.add.at(dst, idx, src)
    else:
        import cupyx

        cupyx.scatter_add(dst, idx, src)
    return dst


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.get()  # cupy -> host
