What you're building

  Three functions:
  1. sigmoid() — the squashing function. Quick to implement, but derive σ'(z) = σ(z)·(1 − σ(z))
     on paper first so you know why the derivative is free once you have the forward value.
  2. cross_entropy() — the loss function. Understand why not MSE (see Lesson 2).
  3. fit() — gradient descent again, same loop shape as linreg. The gradient simplifies to
     (1/n) · Xᵀ · (ŷ − y) — almost identical to linear regression, without the factor of 2.

  Before you code: the derivation

  This is load-bearing derivation #2. Grab pen and paper and work through:

  - σ'(z) = σ(z) · (1 − σ(z)). Start from σ(z) = (1 + e⁻ᶻ)⁻¹ and apply the chain rule.
  - The cross-entropy gradient: chain ∂L/∂σ · ∂σ/∂z · ∂z/∂w and watch the sigmoid terms cancel
    to give (ŷ − y) · x. This cancellation is the whole point — it's why cross-entropy is the
    "right" loss for sigmoid.

  The module doc and Lesson 2 (lessons/0002-logistic-regression-classification.html) walk through
  the steps, but do it yourself first.

  Implementation hints

  - sigmoid() operates element-wise on a Matrix. You'll need to iterate over every element.
  - The fit() loop is nearly identical to linreg's fit(). Swap in sigmoid(X·w + b) for the
    prediction and drop the factor of 2 from the gradient.
  - predict() is provided as a worked example, same as in linreg.

  The gate

  All 7 tests green + you can explain the decision boundary and why the gradient has the form
  it does.
