What you're building

  Two functions:
  1. mse() — mean squared error. Straightforward, start here to warm up.
  2. fit() — the real exercise. A batch gradient descent loop that learns weights w and bias b.

  Before you code: the derivation

  This is your first load-bearing derivation. Grab pen and paper and work through:

  - Why ∂MSE/∂w = (2/n) · Xᵀ · errors — not memorize the formula, but do the chain rule on MSE(w) = (1/n) Σ(w·xᵢ + b − yᵢ)² and see where each
  term comes from.
  - Why ∂MSE/∂b = (2/n) · sum(errors) — the bias partial is simpler but follows the same logic.
  - What the update rule w ← w − lr · ∂MSE/∂w is actually doing geometrically — you're stepping downhill on the loss surface.

  The module doc in the file walks through the steps, but do it yourself first — reading someone else's derivation isn't the same as doing it.

  Implementation hint
  
  You'll need elementwise subtraction and scalar multiplication on your Matrix. You can either add those methods to the linalg crate (cleaner,
  and you'll need them later) or do it inline with get/set loops. Your call.

  The gate
  
  All 5 tests green + you can explain the gradient, not just recite it.
