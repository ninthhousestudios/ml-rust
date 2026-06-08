# MSE implementation

Josh implemented `mse()` (mean squared error) in `crates/linreg/src/lib.rs`. The function
computes (1/n) Σ (pred_i − actual_i)², iterating over paired elements from two (n × 1)
matrices. Both MSE tests pass.

This is a warm-up for the gradient descent exercise — he understands the loss function before
deriving its gradient. The `fit()` function (the actual gradient descent loop) is still
`unimplemented!()` and is the active exercise.
