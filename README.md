# ml-rust

Learning machine learning by building it from scratch in Rust — for understanding, not for a job.

## Why this exists

Most ML learning paths optimize for getting hired fast: learn Python, skip the math, call
`scikit-learn` and `PyTorch`, and treat the internals as a black box you don't have time to open.
That's a reasonable path to a paycheck. It is the opposite of what I want here.

I have a math background and I'm learning Rust, and I'm curious about how machine learning *actually
works* — not how to wire libraries together. So this repo inverts the usual advice:

- **Hand-roll the core.** Matrices, gradient descent, an autograd engine, a transformer — written by
  hand, in Rust, before reaching for any framework. Real libraries (`candle`, `burn`, `ndarray`)
  come in only once the fundamentals are genuinely understood, and then mostly to scale up.
- **Derive the load-bearing math.** Not every proof — just the handful that carry everything:
  gradient descent, the cross-entropy gradient, backpropagation, and attention.
- **Build to understand.** Writing `loss.backward()` myself *is* the curriculum. The point isn't a
  working model; it's knowing why it works.

## The arc

| Phase | Focus |
|-------|-------|
| 0 | A minimal hand-rolled matrix type — the numerical substrate |
| 1 | Classical ML: linear & logistic regression, trees, k-means |
| 2 | Neural nets: a scalar autograd engine, then a small MLP + backprop |
| 3 | A transformer from scratch — attention, then a tiny GPT |
| 4 | Scaling up with [`candle`](https://github.com/huggingface/candle): real model inference |
| 5 | Embeddings and vector search — the ML hiding in everyday tools |

The full curriculum, with per-milestone gates, lives in
[`docs/ml-learning-program.md`](docs/ml-learning-program.md).

## Layout

```
docs/ml-learning-program.md   the curriculum and milestone gates
crates/linalg/                 Phase 0: the hand-rolled matrix type
```

More crates appear as the phases progress.

## Running it

```sh
cargo test
```

Each milestone is structured as an exercise: a skeleton with failing tests that define "done." When
the tests go green, the gate is cleared.

## A note to anyone passing through

This is a personal learning journal, not a tutorial or a library — I don't expect much of an
audience. But I believe in open learning, so it's public in case the approach (build it yourself, in
a language you like, to actually understand it) is useful to someone else chasing the same thing.
