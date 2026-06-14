# Sigmoid function and cross-entropy loss

Josh implemented sigmoid and binary cross-entropy loss for the logistic regression milestone.
The sigmoid handles the full range correctly (no overflow for large negative z), and the
cross-entropy uses clamping (1e-15) to prevent log(0). He demonstrated understanding of why
MSE fails for classification (vanishing gradients in the sigmoid's flat regions) and why
cross-entropy penalises confident wrong predictions exponentially.

Evidence: working `sigmoid`, `cross_entropy` functions in `crates/logreg/src/lib.rs`; all 7
tests pass including `cross_entropy_confident_correct_is_low` and
`cross_entropy_confident_wrong_is_high`.
