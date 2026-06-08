# Machine Learning Resources

## Knowledge

### Math Foundations

- [Book: _Mathematics for Machine Learning_ — Deisenroth, Faisal, Ong (Cambridge, 2020)](https://mml-book.github.io/book/mml-book.pdf)
  Free PDF. Covers linear algebra, calculus, probability, and optimization with ML applications.
  Use for: the mathematical backbone — when a derivation needs more rigour than a walkthrough provides.

- [Video: _Essence of Linear Algebra_ — 3Blue1Brown](https://www.3blue1brown.com/topics/linear-algebra)
  Visual intuition for vectors, transformations, eigenvalues. 16 short videos.
  Use for: geometric intuition behind matrix operations — what matmul, transpose, and dot products _mean_.

- [Video: _Neural Networks_ — 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
  Visual walkthrough of neural nets, gradient descent, and backpropagation.
  Use for: building intuition before the formal derivation. Watch before Milestone 4.

### ML from Scratch

- [Video: _Neural Networks: Zero to Hero_ — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html)
  Builds micrograd (scalar autograd), makemore, and a GPT from scratch in Python. The closest
  thing to this project's philosophy, but in Python.
  Use for: conceptual reference for Milestones 4–8. Watch the micrograd lecture before building
  the Rust autograd engine.

- [Video: _Stanford CS229: Machine Learning_ — Andrew Ng (Autumn 2018)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
  Full university course. Covers linear regression, logistic regression, SVMs, decision trees,
  neural nets, and more with mathematical rigour.
  Use for: the classical ML milestones (1–3). Ng's derivation of gradient descent is excellent.

- [Course site: _CS229: Machine Learning_ — Stanford](https://cs229.stanford.edu/)
  Lecture notes, problem sets, and section materials. The written notes are sometimes clearer
  than the lectures for derivation work.
  Use for: supplementary reading alongside the video lectures.

### Rust ML Ecosystem

- [GitHub: _candle_ — Hugging Face](https://github.com/huggingface/candle)
  Minimalist ML framework for Rust. This is what the Phase 4 Flux inference binary targets.
  Use for: Milestone 9 onwards. Don't touch until autograd and transformer are hand-rolled.

- [Site: _Burn_](https://burn.dev/)
  Modular deep learning framework in Rust, more comprehensive than candle.
  Use for: reference and comparison during Phase 4. Not the primary target.

- [Book: _The Rust Programming Language_](https://doc.rust-lang.org/book/)
  The official Rust book. Already familiar territory, but useful for looking up ownership,
  traits, and iterators when the ML code gets complex.
  Use for: Rust-specific questions as they arise.

### Curated Collections

- [GitHub: _dair-ai/Mathematics-for-ML_](https://github.com/dair-ai/Mathematics-for-ML)
  Community-vetted collection of math-for-ML resources organised by topic (linear algebra,
  calculus, probability) and format (videos, books, courses).
  Use for: finding targeted resources when a specific math concept is unclear.

## Gaps

- No good Rust-specific "build ML from scratch" tutorial series — this project is essentially
  writing one. Karpathy's Zero to Hero is the closest analogue but in Python.
- No resource yet for flow-matching / DiT theory (needed for Milestone 9, Flux inference).
  Find before Phase 4.
- Embedding/similarity-search theory resources (needed for Milestone 10) not yet identified.
  Find before Phase 5.
