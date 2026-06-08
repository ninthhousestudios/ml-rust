# ML Learning Program

A self-directed path into machine learning, **for understanding, not employment**. Built in Rust,
hand-rolling the core so the machine is never a black box. Crates (`candle`, `burn`, `ndarray`) come
in only once the fundamentals are owned.

## Principles

- **Build to understand.** Writing `loss.backward()` myself *is* the curriculum, not a detour to skip.
- **Milestone-gated, not calendar-based.** Progress when I feel like it; gates tell me when a milestone
  is genuinely done.
- **Derive the load-bearing math by hand; recognize the rest.** Four derivations carry the whole thing
  (see below). Everything else is "I know what that word means" level.
- **Hand-roll core, crates for scale.** Toy autograd first, then see how the real frameworks structure
  the same idea.
- **Learn by interpreted walkthrough.** I attempt; we read the code/math together until it clicks.

## The arc

```
Phase 0  Rust numerical substrate      (small, doubles as Rust practice)
Phase 1  Classical ML, hand-rolled     - north star #2
Phase 2  Neural nets + backprop        - the hinge, makes #1 possible
Phase 3  Transformer from scratch      - north star #1 (Claude stops being magic)
Phase 4  Crates for scale              - use a real framework on a real project (TBD at the time)
Phase 5  Embeddings & the ML in manas  - north star #3 (last; the idea is already grasped)
```

## The four load-bearing derivations

Owned with pencil and paper. Everything else is recognize-only.

1. **Gradient descent + ∂MSE/∂w** (Milestone 1)
2. **Sigmoid + cross-entropy gradient** (Milestone 2)
3. **Backprop / chain rule** (Milestone 4) — the big one
4. **Scaled dot-product attention** (Milestone 7)

## Milestones

Each = a thing built in Rust + the math derived by hand + a gate ("got it when…").

Effort tiers: **light** (a session or two), **medium** (a few sessions), **heavy** (weeks of
on/off work). The early milestones are light; the weight is in Phases 2–3.

| # | Phase | Effort | Build target | Derive by hand | Gate |
|---|-------|--------|--------------|----------------|------|
| 0 | 0 | light | Minimal `Matrix` type (Vec-backed): matmul, transpose, elementwise. Adopt `ndarray` once boring. | — | Can multiply matrices and know the memory layout |
| 1 | 1 | light | Linear regression via batch gradient descent | **Gradient descent; ∂MSE/∂w** | Watch loss fall and say *why* the gradient points where it does |
| 2 | 1 | medium | Logistic regression, binary classifier | **Sigmoid + cross-entropy gradient** | Can draw and explain the decision boundary |
| 3 | 1 | medium | Decision tree (entropy/info-gain); recognize k-NN, k-means, PCA, SVM | — | Know when *not* to reach for gradient descent |
| — | 1 | — | *Woven through 1–3:* train/test split, precision/recall, overfitting, cross-validation | — | Tell a real result from a memorized one |
| 4 | 2 | heavy | **micrograd-in-Rust**: scalar autograd engine with `.backward()` | **Backprop / chain rule** | `.backward()` computes correct grads; can trace the graph by hand |
| 5 | 2 | medium | Small MLP on the autograd engine (ReLU, layers, SGD), train on MNIST-ish | optimizer mechanics | It learns and every line is explicable |
| 6 | 3 | medium | Tokenizer + learned embeddings (also seeds Phase 5) | — | Tokens → vectors, pipeline under control |
| 7 | 3 | heavy | Self-attention, then multi-head, from scratch | **Scaled dot-product attention; why softmax, why √d** | Can explain Q/K/V without hand-waving |
| 8 | 3 | heavy | Assemble a tiny GPT (positional enc, residual, layernorm, FFN), train char-level | — | It generates plausible text; attention is no longer magic |
| 9 | 4 | medium | Use a real Rust ML framework (candle or burn) on a project that's interesting at the time | recognize the framework's abstractions vs what you hand-rolled | Can map framework API calls to the primitives you built |
| 10 | 5 | medium | Reimplement smriti/chitta similarity search; read their real code together | contrastive/word2vec intuition, cosine sim, HNSW vs flat | Can explain — and rebuild — the ML inside my own stack |

## Source

Sparked by a YouTube roadmap (`learning-ml-readable.md`), then deliberately inverted: that video
optimizes for getting hired fast in Python by *avoiding* the math and *using* libraries. This program
optimizes for the opposite — understanding the machine, in Rust, by building it.

## Status

This doc is the **stable map** (the arc and the gates). Live status — which milestone is active,
what's verified, what's next — lives in the **yojana project `ml-rust`** (its `handoff` field and
task graph). One source of truth for status; this doc doesn't track progress so it can't rot.
