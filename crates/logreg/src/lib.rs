//! Milestone 2: Logistic regression — binary classification via gradient descent.
//!
//! ## What we're building
//!
//! A model that predicts P(class = 1 | x) by passing a linear combination through the sigmoid:
//!
//! ```text
//!   z = X · w + b
//!   ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
//! ```
//!
//! We measure wrongness with **binary cross-entropy** (log loss):
//!
//! ```text
//!   L = −(1/n) Σ [ y_i · log(ŷ_i) + (1 − y_i) · log(1 − ŷ_i) ]
//! ```
//!
//! ## The gradient (derive this yourself — load-bearing derivation #2)
//!
//! The beautiful result: after the sigmoid derivative and log derivatives cancel, the gradient is:
//!
//! ```text
//!   ∂L/∂w = (1/n) · Xᵀ · (ŷ − y)
//!   ∂L/∂b = (1/n) · sum(ŷ − y)
//! ```
//!
//! Same shape as linear regression, no factor of 2. The training loop barely changes.
//!
//! See `lessons/0002-logistic-regression-classification.html` for the full walkthrough, or
//! `exercise.md` for the condensed version.

use linalg::Matrix;

/// A trained logistic regression model: ŷ = σ(X·w + b).
pub struct LogisticRegression {
    /// Weight vector (n_features × 1).
    pub weights: Matrix,
    /// Bias (scalar).
    pub bias: f64,
}

impl LogisticRegression {
    /// Predict probabilities: ŷ = σ(X · w + b).
    ///
    /// Worked example — this one's free. Same pattern as linreg's predict, plus sigmoid.
    pub fn predict(&self, x: &Matrix) -> Matrix {
        let z = add_bias(&x.matmul(&self.weights), self.bias);
        sigmoid(&z)
    }

    /// Classify: 1.0 if P(class=1) > threshold, else 0.0.
    pub fn classify(&self, x: &Matrix, threshold: f64) -> Matrix {
        let probs = self.predict(x);
        let mut out = Matrix::zeros(probs.rows, 1);
        for r in 0..probs.rows {
            out.set(r, 0, if probs.get(r, 0) > threshold { 1.0 } else { 0.0 });
        }
        out
    }
}

/// Add a scalar bias to every element of an (n × 1) matrix.
fn add_bias(m: &Matrix, bias: f64) -> Matrix {
    let mut out = Matrix::zeros(m.rows, m.cols);
    for r in 0..m.rows {
        for c in 0..m.cols {
            out.set(r, c, m.get(r, c) + bias);
        }
    }
    out
}

/// Apply the sigmoid function element-wise: σ(z) = 1 / (1 + e⁻ᶻ).
///
/// EXERCISE: Implement this. Input and output are both (n × 1) matrices.
///
/// Hint: Rust's `f64::exp()` gives you eˣ. You want 1.0 / (1.0 + (-z).exp()).
pub fn sigmoid(z: &Matrix) -> Matrix {
    unimplemented!("implement sigmoid — derive σ'(z) = σ(z)·(1 − σ(z)) on paper first")
}

/// Compute binary cross-entropy loss.
///
/// ```text
///   L = −(1/n) Σ [ y_i · log(ŷ_i) + (1 − y_i) · log(1 − ŷ_i) ]
/// ```
///
/// EXERCISE: Implement this. `predictions` are σ(z) values in (0, 1), `targets` are 0.0 or 1.0.
///
/// Hint: use `f64::ln()` for natural log. Clamp predictions away from exactly 0 or 1 to avoid
/// log(0) = −∞. A small epsilon like 1e-15 works: `pred.clamp(1e-15, 1.0 - 1e-15)`.
pub fn cross_entropy(predictions: &Matrix, targets: &Matrix) -> f64 {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, 1);
    assert_eq!(targets.cols, 1);

    unimplemented!("implement cross-entropy loss")
}

/// Train a logistic regression model using batch gradient descent.
///
/// Arguments:
/// - `x`: input features, shape (n_samples × n_features)
/// - `y`: target labels, shape (n_samples × 1), values are 0.0 or 1.0
/// - `lr`: learning rate
/// - `epochs`: number of gradient descent iterations
///
/// Returns a trained `LogisticRegression`.
///
/// EXERCISE: Implement the training loop. Each epoch:
///   1. Compute z = X · w + b
///   2. Predict: ŷ = σ(z)                         (n × 1)
///   3. Compute errors: e = ŷ − y                  (n × 1)
///   4. Compute gradients:
///        dw = (1/n) · Xᵀ · e                      (d × 1)
///        db = (1/n) · sum(e)                       (scalar)
///   5. Update:
///        w ← w − lr · dw
///        b ← b − lr · db
///
/// Initialize w to zeros (n_features × 1) and b to 0.0.
///
/// Compare this to linreg's fit(). The only differences: sigmoid in step 2, no factor of 2 in
/// step 4. The loop shape is the same.
pub fn fit(x: &Matrix, y: &Matrix, lr: f64, epochs: usize) -> LogisticRegression {
    assert_eq!(x.rows, y.rows);
    assert_eq!(y.cols, 1);

    unimplemented!("implement logistic regression training — same loop shape as linreg")
}

/// Compute classification accuracy: fraction of predictions matching targets.
///
/// Provided for free — use it to check your model after training.
pub fn accuracy(predicted_classes: &Matrix, targets: &Matrix) -> f64 {
    assert_eq!(predicted_classes.rows, targets.rows);
    let mut correct = 0;
    for r in 0..predicted_classes.rows {
        if (predicted_classes.get(r, 0) - targets.get(r, 0)).abs() < 1e-10 {
            correct += 1;
        }
    }
    correct as f64 / predicted_classes.rows as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sigmoid tests ---

    #[test]
    fn sigmoid_at_zero_is_half() {
        let z = Matrix::from_rows(&[&[0.0]]);
        let s = sigmoid(&z);
        assert!(
            (s.get(0, 0) - 0.5).abs() < 1e-10,
            "σ(0) should be 0.5, got {}",
            s.get(0, 0)
        );
    }

    #[test]
    fn sigmoid_large_positive_near_one() {
        let z = Matrix::from_rows(&[&[10.0]]);
        let s = sigmoid(&z);
        assert!(
            s.get(0, 0) > 0.999,
            "σ(10) should be near 1.0, got {}",
            s.get(0, 0)
        );
    }

    #[test]
    fn sigmoid_symmetry() {
        // σ(-z) = 1 - σ(z)
        let pos = Matrix::from_rows(&[&[2.0]]);
        let neg = Matrix::from_rows(&[&[-2.0]]);
        let s_pos = sigmoid(&pos).get(0, 0);
        let s_neg = sigmoid(&neg).get(0, 0);
        assert!(
            (s_pos + s_neg - 1.0).abs() < 1e-10,
            "σ(2) + σ(-2) should be 1.0, got {} + {} = {}",
            s_pos, s_neg, s_pos + s_neg
        );
    }

    // --- Cross-entropy tests ---

    #[test]
    fn cross_entropy_confident_correct_is_low() {
        // Model predicts 0.99 for true 1, and 0.01 for true 0 → low loss
        let preds = Matrix::from_rows(&[&[0.99], &[0.01]]);
        let targets = Matrix::from_rows(&[&[1.0], &[0.0]]);
        let loss = cross_entropy(&preds, &targets);
        assert!(
            loss < 0.02,
            "confident correct predictions should have low loss, got {loss}"
        );
    }

    #[test]
    fn cross_entropy_confident_wrong_is_high() {
        // Model predicts 0.01 for true 1 → high loss
        let preds = Matrix::from_rows(&[&[0.01], &[0.99]]);
        let targets = Matrix::from_rows(&[&[1.0], &[0.0]]);
        let loss = cross_entropy(&preds, &targets);
        assert!(
            loss > 2.0,
            "confident wrong predictions should have high loss, got {loss}"
        );
    }

    // --- Training tests ---

    #[test]
    fn fit_learns_separable_1d() {
        // Class 1 when x > 3, class 0 when x < 3.
        let x = Matrix::from_rows(&[
            &[1.0], &[1.5], &[2.0], &[2.5],
            &[4.0], &[4.5], &[5.0], &[5.5],
        ]);
        let y = Matrix::from_rows(&[
            &[0.0], &[0.0], &[0.0], &[0.0],
            &[1.0], &[1.0], &[1.0], &[1.0],
        ]);

        let model = fit(&x, &y, 0.5, 1000);
        let classes = model.classify(&x, 0.5);
        let acc = accuracy(&classes, &y);

        assert!(
            acc > 0.99,
            "should classify separable 1D data perfectly, got accuracy {acc}"
        );
    }

    #[test]
    fn fit_loss_decreases() {
        let x = Matrix::from_rows(&[
            &[1.0, 0.5], &[1.5, 1.0], &[2.0, 0.8],
            &[4.0, 3.5], &[4.5, 4.0], &[5.0, 3.8],
        ]);
        let y = Matrix::from_rows(&[
            &[0.0], &[0.0], &[0.0],
            &[1.0], &[1.0], &[1.0],
        ]);

        // Before training: w=0, b=0 → all predictions are σ(0) = 0.5
        let initial_preds = Matrix::from_rows(&[&[0.5], &[0.5], &[0.5], &[0.5], &[0.5], &[0.5]]);
        let initial_loss = cross_entropy(&initial_preds, &y);

        // After training
        let model = fit(&x, &y, 0.1, 500);
        let final_preds = model.predict(&x);
        let final_loss = cross_entropy(&final_preds, &y);

        assert!(
            final_loss < initial_loss * 0.1,
            "loss should drop significantly: {initial_loss} → {final_loss}"
        );
    }
}
