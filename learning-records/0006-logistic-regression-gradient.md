# Logistic regression gradient — load-bearing derivation #2

Josh implemented the logistic regression training loop (gradient descent with cross-entropy
loss), demonstrating the second load-bearing derivation: dL/dw = (1/n) Xt (yhat - y). His
`fit()` correctly computes predictions via sigmoid, derives error terms, and updates weights
and bias using the gradient. The code shows he understood the key insight: the gradient has
the same shape as linear regression's (Xt * errors), with no factor of 2, because the sigmoid
and log-loss derivatives cancel.

Evidence: `fit()` and supporting functions in `crates/logreg/src/lib.rs`;
`fit_learns_separable_1d` and `fit_loss_decreases` tests pass.

Implications: Two of four load-bearing derivations now complete. Next (Milestone 4): backprop /
chain rule — the hardest one. Before that, Milestone 3 (decision trees) provides breadth
beyond gradient descent.
