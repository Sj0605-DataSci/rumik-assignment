# GPT-2 from scratch, no autograd — approach & reasoning

This is my planning doc before writing any code. Goal: pin down the stack, the
architecture, and exactly how backprop gets implemented by hand, so the actual
build is just execution against this plan.

## 1. Restating the constraint

Train a GPT-2-style decoder-only transformer on OpenWebText **without using
`.backward()` / autodiff from torch or jax**. Every gradient that flows through
every op has to be either (a) hand-derived and hand-coded per layer, or (b) the
product of a small autograd engine I write myself. The point of the exercise
isn't "can it train" — nanoGPT already proves that — it's whether I actually
understand the chain rule through LayerNorm, softmax attention, and GELU well
enough to write the backward pass myself, and can show my work.

## 2. Stack decision: NumPy, hand-coded gradients (not a custom autograd engine)

Two options were offered:
1. Hand-derive and hand-code the backward for every layer/module.
2. Write a small tensor autograd engine (tape-based, like a mini-tinygrad) and
   let it compose backward passes automatically.

**Going with (1).** Reasoning: the assignment says explicitly it's scoring
"math and research rigor... ability to go in depth" and "hand-coding gradients
forces you to write and reason about tensor shapes flowing through forward and
backward passes." A generic autograd engine is a software-engineering exercise
in graph construction and topological sort — it actually *hides* the
per-layer calculus behind a generic `Tensor.backward()`, once written. Hand-coding
each layer's local Jacobian-vector product is the part that's actually being
graded. It's also more debuggable: if block 4's gradient is wrong, I know
exactly which closed-form derivative to re-check, instead of tracing a tape.

Concretely: every layer is a small class with an explicit `forward(x)` that
caches what it needs, and an explicit `backward(dout)` that returns `dx` (and
accumulates `dW`, `db` etc. on itself). This is "PyTorch nn.Module, but the
backward method is math I wrote," not a computation-graph tracer. No tape, no
topological sort, no generic `Tensor` class with overloaded operators — just
plain NumPy arrays passed between layer objects I control in a fixed order
(the transformer block order), so the "graph" is just a Python list.

**Why NumPy over tinygrad (bonus) or raw torch/jax tensors with detach:**
- tinygrad still means fighting its own autograd/UOp graph to keep it out of
  the loop, and there's no real benefit for a from-scratch build — the whole
  value in the assignment is deriving and coding the math, not learning
  tinygrad's internals. Not worth the risk of subtly leaning on its lazy
  autodiff without noticing.
- torch/jax tensors + `.detach()` everywhere is more error-prone than just...
  not importing torch. One missed `.detach()` and I've silently used autograd.
  NumPy makes the constraint physically impossible to violate.
- NumPy has full BLAS-backed matmul (OpenBLAS/MKL), which is what actually
  matters for wall-clock time on CPU. `cupy` is a drop-in NumPy-API GPU array
  library — worth switching to it later purely as a speed upgrade (same code,
  same math, just GPU-resident arrays) if pure-CPU NumPy is too slow for even
  the scaled-down run. This does **not** reintroduce autograd — cupy has no
  autodiff of its own unless I use `cupy` alongside a tracking layer, which I
  won't.
- I will use `tiktoken` for **tokenization only** (GPT-2 BPE encode/decode).
  That's pure preprocessing, not part of the differentiable forward/backward
  path, so it doesn't touch the no-autograd rule.
- I may use `torch` in a **separate, isolated unit-test file** to
  finite-difference / autograd-check my hand-derived gradients against
  `torch.autograd` on tiny random tensors (batch=2, seq=4, dim=8). This is a
  correctness harness, never imported by the training code, and worth calling
  out explicitly so it's clear it's not doing any of the real work.

## 3. Architecture (mirrors nanoGPT's `model.py`, same shapes/conventions)

- Token embedding `wte` (V, C) + learned positional embedding `wpe` (T, C),
  summed.
- N x transformer block, each:
  - LayerNorm → causal multi-head self-attention → residual add
  - LayerNorm → MLP (Linear → GELU → Linear) → residual add
  - (pre-norm, GPT-2 style, not post-norm)
- Final LayerNorm
- LM head: `Linear(C, V)`, weight-tied to `wte`
- Loss: softmax cross-entropy over next-token targets, causal mask applied as
  `-inf` on upper-triangular attention scores before softmax

Every one of these is a layer object with forward/backward, listed below with
the derivative I need to have written out on paper before coding it:

| Layer | Forward | Backward I need to derive |
|---|---|---|
| Embedding lookup | gather rows of `wte`/`wpe` by index | scatter-add `dout` back into `dwte` at the same indices (no matmul — it's a gather, so backward is index_add, this trips people up) |
| Linear (`y = xW + b`) | matmul + bias | `dx = dy @ W.T`, `dW = x.T @ dy`, `db = sum(dy, axis=0)` — but batched over (B,T), so care with reshape/sum axes |
| LayerNorm | normalize over last dim, scale+shift by `gamma`,`beta` | the standard LN backward with the `1/N`, `mean`, `var` cross-terms — this is the one everyone gets wrong by forgetting the two correction terms from `mean` and `var` depending on all `N` inputs |
| GELU (tanh approx, matches GPT-2) | `0.5x(1+tanh(...))` | derivative of the tanh approximation, chain-ruled through |
| Causal self-attention | `softmax(QK^T/√d_k + mask) V` | softmax Jacobian (`diag(s) - s sᵀ` contracted efficiently, not materialized), plus backprop through the three matmuls (`Q,K,V` projections) and the mask (mask positions get zero gradient) |
| Residual add | `x + f(x)` | gradient just splits and adds into both branches — trivial but easy to forget when wiring the block |
| Softmax cross-entropy | softmax + NLL | the clean combined form `dlogits = softmax(logits) - one_hot(target)`, derived once and used directly (never differentiate softmax and NLL separately — numerically worse and pointless extra work) |
| Weight tying (`wte` == lm_head.weight) | shared matrix | gradients from both the embedding lookup and the output projection must accumulate into the *same* `dwte`, not overwrite |

Every layer caches whatever it needs for backward on `self` during `forward`
(the classic "save for backward" pattern), then `backward` consumes and clears
it. That's the entire "autograd" here — explicit state on each object, no
generic graph.

## 4. Correctness validation plan (this is the part I actually care about getting right)

Hand-derived gradients are worthless if they're subtly wrong and I don't
notice until loss curves look "fine but not great." Plan, in order, before any
real training run:

1. **Per-layer numerical gradient check.** For every layer class, tiny random
   input, compute `backward()` analytically, compare against central-difference
   numerical gradient (`(f(x+h)-f(x-h))/2h`) on every parameter and input,
   relative error < 1e-5ish for float64, looser for float32. This is the
   primary correctness gate — every layer must pass this in isolation before
   it's allowed into the full model.
2. **Full-model numerical check** on a tiny GPT (2 layers, dim 16, seq 4) —
   catches wiring bugs (wrong residual order, tied-weight gradient not
   accumulating, mask backward wrong) that per-layer checks can't see.
3. **Cross-check against `torch.autograd`** on the same tiny model/inputs as a
   second independent oracle (isolated test file, not part of the training
   code — see §2).
4. **Overfit-one-batch sanity check** — tiny model should drive loss to ~0 on
   a single repeated batch. Standard first real training signal that the
   optimizer + backward wiring works end to end, before spending compute on
   real data.

Only after all four pass does real training start.

## 5. Optimizer

AdamW, implemented directly as NumPy array update rules over `(param, grad,
m, v)` — this part involves no backward pass at all, it's just the update
equations from the paper applied per-parameter, plus manual weight decay
(decoupled, skip decay on biases/LayerNorm gains per GPT-2 convention) and a
cosine LR schedule with linear warmup, matching nanoGPT's schedule so results
are comparable.

## 6. Data pipeline

Reusing nanoGPT's approach for OpenWebText since the assignment explicitly
allows building on it for "dataset shuffling and scaffolding": `tiktoken`
GPT-2 BPE encode → binary `.bin` token shards → `np.memmap` for random-access
batch sampling. This is pure data plumbing, unrelated to the autograd
constraint, no reason to reinvent it.

## 7. Scale — DGX Spark (GB10) is available, so use it properly

Pure NumPy on CPU can't train a real 124M GPT-2 on the full ~9B-token
OpenWebText in any reasonable time. But there's a DGX Spark (GB10) box
available, so the plan is to target `cupy` as the **primary** array backend
from the start, not a fallback reached for only if CPU turns out too slow.

`cupy` is a drop-in NumPy-API array library — every hand-written
`forward`/`backward` method stays exactly the same Python/math, arrays just
live on the GPU and matmuls run through cuBLAS instead of the CPU BLAS. This
doesn't reintroduce autograd anywhere: cupy has no autodiff of its own unless
I explicitly opt into `cupy`'s (unused) tracking, which I won't — it's purely
a faster `ndarray`. I'll keep a thin `xp = numpy or cupy` switch so the exact
same layer code can still run on CPU for the gradient-check suite in §4
(cheap, small tensors, no reason to burn GPU time on unit tests) and switch to
GPU only for real training.

With GB10 actually driving the matmuls, the target model size moves closer to
real GPT-2 "small" territory (aiming for something in the 12-layer,
d_model 768, 12-head, block_size 1024 neighborhood — the actual 124M GPT-2
config) rather than a toy. Still won't process the full ~9B-token OpenWebText
end to end in one shot — token budget will be set from measured tokens/sec on
GB10 against however much wall-clock time is reasonable for this assignment,
not assumed up front. Plan:
- Start with a nanoGPT-"shakespeare_char"-scale sanity model to validate the
  full pipeline cheaply (CPU numpy is fine here, it's tiny).
- Measure real cupy throughput on GB10 with the actual model config before
  committing to a token budget, then pick n_layer/d_model/block_size/steps
  to fit the available time, biasing toward the real 124M config if the
  numbers support it.
- Two operational notes from having used this box before: (a) cap GPU clocks
  (`nvidia-smi -lgc 0,1500`) before a long run — uncapped clocks under
  sustained heavy load have crashed the OS on this machine before, and the
  cap doesn't survive a reboot so it needs re-applying per session; (b) don't
  run anything else that loads onto the GPU (inference, notebooks, etc.)
  while a training job is active — shared GPU, and it'll contend with or
  crash the training run.
- Document the exact scale (config, token budget, wall-clock, GB10
  tokens/sec measured) honestly in the writeup — the assignment is
  clear it's assessing gradient/math rigor, not SOTA loss numbers, so being
  explicit about "here's why this config, here's the token budget" is
  the right move rather than quietly training something tiny and calling it
  GPT-2.

## 8. Evaluation plan

- Train/val loss curves and perplexity (`exp(loss)`) logged over training,
  standard nanoGPT-style eval-interval loop, held-out OpenWebText val split.
- One downstream LM benchmark for an eval table — leaning toward **WikiText-2
  perplexity** (cheap, standard, directly reflects LM quality, no extra
  labeled-task scaffolding needed) as the primary number, and noting
  LAMBADA-style last-word accuracy as a stretch goal if time allows, since
  it's a better test of the model actually using context vs. just fitting
  local statistics.

## 9. Deliverables checklist (mapping back to what's asked)

- [ ] Training code (custom layers + hand-derived backward, AdamW, data prep,
      training loop)
- [ ] Gradient-check test suite (numerical + torch cross-check) — this is the
      evidence the math is right, should be front and center, not buried
- [ ] Loss/perplexity plots (train + val)
- [ ] Eval table on a standard LM benchmark
- [ ] Written roadblock log — kept as I go, not reconstructed after
- [ ] This reasoning doc, kept up to date if the plan changes

## 10. Anticipated roadblocks (best guesses before writing code — will update as I hit real ones)

- Softmax/LayerNorm numerical stability (overflow in `exp`, `eps` in variance)
  — standard max-subtraction trick, standard eps placement, but worth being
  careful with in both forward *and* backward (backward has its own stability
  traps, e.g. dividing by `std` again).
- Embedding backward is a scatter-add over a (B,T) batch of indices into a
  (V,C) table — needs to accumulate correctly when the same token index
  appears more than once in a batch (`np.add.at`, not plain fancy-index
  assignment, which would overwrite instead of accumulate).
- Weight-tying gradient accumulation (embedding table gets gradient
  contributions from two places — must add, not overwrite, whichever backward
  runs second).
- Activation memory: since there's no autograd tape doing this for me, every
  layer instance is responsible for caching its own forward activations for
  backward — need to be deliberate about clearing/reusing this per training
  step so memory doesn't balloon over the sequence of blocks.
- Attention backward is the most error-prone piece (softmax Jacobian through
  a batched, masked, multi-head matmul chain) — this is where the numerical
  gradient check in §4 matters most and will get the most scrutiny before I
  trust it.

## 11. Next steps

1. Write layer classes + numerical gradient check suite first, nothing else.
   No training loop until every layer passes its own check.
2. Wire full model, run full-model numerical check + overfit-one-batch.
3. Data prep script (adapted from nanoGPT) for a subset of OpenWebText.
4. Real training run at the scaled-down size, with logging.
5. Eval (WikiText-2 perplexity) + plots + writeup of roadblocks hit along the
   way.

## 12. Mid-project: swapping in priyammaz/ManualTransformer as the layer source

After the above was built, gradient-checked, and used to run three training
runs (see README's Results table), the layer implementations in
`src/nanograd_gpt/layers/` were replaced with code adapted from
[priyammaz/ManualTransformer](https://github.com/priyammaz/ManualTransformer)
(MIT License; see `LICENSE-ManualTransformer.txt`), at explicit user request,
because they had studied that repo's code line-by-line and wanted the shipped
implementation to match what they'd been practicing from.

This is worth being honest about against the assignment's own rule:
**"own each and every line that you ship."** I raised that tension directly
before doing it — replacing hand-derived code with an adapted third-party
implementation is a real deviation from "derived it myself," even reorganized
into multiple files and even with full attribution. The user made an informed
call to proceed anyway, understanding the risk. What actually happened,
concretely:

- Ported as adapted-but-faithful: `Linear`, `LayerNorm`, `SoftMax` (both the
  fast vectorized backward and the slow explicit-Jacobian version, kept for
  the derivation), `Embedding` (scatter-add), `Dropout`, `MultiHeadAttention`'s
  core mechanics (separate q/k/v Linear layers rather than this project's
  earlier fused `c_attn`, a genuine structural change, not just a rename).
- Fixed a real bug found while porting: the source's `CrossEntropyLoss`
  computes a mean loss forward but doesn't divide by batch size backward —
  internally inconsistent with its own reduction, and fails numerical
  gradient checking unfixed. Confirmed via `tests/test_gradients.py` before
  and after the fix.
- Deliberately NOT ported, because doing so would have reversed earlier
  decisions the user explicitly asked for: weight tying and AdamW (their
  `model.py`/`optim.py` have neither — separate untied output head, plain
  `Adam` with no weight decay/clipping/schedule); GELU over their ReLU
  (real GPT-2 uses GELU); pre-norm block ordering over their post-norm
  (real GPT-2 is pre-norm, which is what makes it trainable at depth).
  This project's `model.py`/`optim.py`/`train.py` were kept as-is throughout.

All three prior training runs' results, checkpoints, and eval numbers in the
README predate this swap and are unaffected — they were produced by the
original hand-derived code, not this adapted version. The full
numerical-gradient-check suite (`tests/test_gradients.py`) was re-run after
every file swap and passes identically (~1e-10 to ~1e-14 max diff) against
the adapted code, so the correctness evidence stands regardless of origin —
but "gradient-checked" and "independently derived" are different claims, and
this section exists so that distinction isn't buried.
