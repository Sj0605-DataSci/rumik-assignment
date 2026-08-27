"""Hand-written forward/backward for every layer used by the GPT model, one
class per file, grouped by what it does:

  module.py     Module      base class: parameter/sub-module bookkeeping
  linear.py     Linear      y = x @ W + b
  layernorm.py  LayerNorm   normalize + learned scale/shift
  gelu.py       GELU        GPT-2's tanh-approximation activation
  embedding.py  Embedding   row lookup, scatter-add backward
  dropout.py    Dropout     inverted dropout
  softmax.py    softmax, softmax_backward, softmax_cross_entropy
  attention.py  CausalSelfAttention
  mlp.py        MLP         Linear -> GELU -> Linear
  block.py      Block       one pre-norm transformer block

No autograd tape, no generic Tensor class with overloaded ops, anywhere in
this package. Each layer caches whatever it needs during forward() and
consumes that cache in backward() to produce local gradients via a
closed-form derivative I derived on paper first (see REASONING.md for the
per-layer derivation notes). The "graph" is just the fixed Python
composition order in model.py.

Shape convention throughout: batch B, sequence length T, model dim C
(a.k.a. n_embd), number of heads n_head, per-head dim hd = C // n_head.
"""

from .attention import CausalSelfAttention
from .block import Block
from .dropout import Dropout
from .embedding import Embedding
from .gelu import GELU
from .layernorm import LayerNorm
from .linear import Linear
from .mlp import MLP
from .module import Module
from .softmax import softmax, softmax_backward, softmax_cross_entropy

__all__ = [
    "CausalSelfAttention",
    "Block",
    "Dropout",
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "MLP",
    "Module",
    "softmax",
    "softmax_backward",
    "softmax_cross_entropy",
]
