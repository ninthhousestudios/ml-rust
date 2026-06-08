//! Milestone 1: Linear regression via batch gradient descent.
//!
//! ## What we're building
//!
//! A model that predicts a continuous value from input features by learning weights `w` and a bias
//! `b`. Given inputs matrix X (n_samples × n_features) and targets y (n_samples × 1):
//!
//! ```text
//!   predictions = X · w + b
//! ```
//!
//! We measure how wrong we are with **Mean Squared Error (MSE)**:
//!
//! ```text
//!   MSE = (1 / n) * Σ (predicted_i − actual_i)²
//! ```
//!
//! Then we adjust w and b to reduce MSE. That adjustment process is gradient descent.
//!
//! ## The math you need to derive (pen and paper)
//!
//! This is the first load-bearing derivation. You need to work through *why* the gradient has the
//! form it does — not memorize it, derive it.
//!
//! ### Step 1: MSE as a function of w
//!
//! For a single weight (1D case), predictions are ŷ_i = w · x_i + b. Then:
//!
//! ```text
//!   MSE(w) = (1/n) Σ (ŷ_i − y_i)²
//!          = (1/n) Σ (w·x_i + b − y_i)²
//! ```
//!
//! ### Step 2: Differentiate with respect to w
//!
//! Apply the chain rule. Let e_i = ŷ_i − y_i (the error). Then:
//!
//! ```text
//!   ∂MSE/∂w = (1/n) Σ  2·e_i · ∂e_i/∂w
//!           = (1/n) Σ  2·e_i · x_i
//!           = (2/n) Σ  (ŷ_i − y_i) · x_i
//! ```
//!
//! The 2 is conventional — many implementations drop it (absorbed into the learning rate). We'll
//! keep it for correctness, then you'll see it doesn't matter.
//!
//! Similarly:
//!
//! ```text
//!   ∂MSE/∂b = (2/n) Σ (ŷ_i − y_i)
//! ```
//!
//! ### Step 3: The gradient descent update rule
//!
//! ```text
//!   w ← w − lr · ∂MSE/∂w
//!   b ← b − lr · ∂MSE/∂b
//! ```
//!
//! `lr` is the learning rate — a small number (0.01, 0.001) that controls step size.
//!
//! ### In matrix form (multiple features)
//!
//! With X as (n × d) and w as (d × 1):
//!
//! ```text
//!   predictions = X · w  (+ b broadcast to each row)
//!   errors      = predictions − y           (n × 1)
//!   ∂MSE/∂w     = (2/n) · Xᵀ · errors      (d × 1)
//!   ∂MSE/∂b     = (2/n) · sum(errors)       (scalar)
//! ```
//!
//! That `Xᵀ · errors` is doing all the per-feature gradient accumulation in one matmul. This is
//! why you built Matrix first.
//!
//! ## Worked example: predict_single
//!
//! Below is a complete single-feature prediction function to show the pattern. Your exercise is to
//! implement the training loop that learns `w` and `b` via gradient descent.

use linalg::Matrix;

/// A trained linear regression model: ŷ = X·w + b.
pub struct LinearRegression {
    /// Weight vector (n_features × 1).
    pub weights: Matrix,
    /// Bias (scalar, stored as f64).
    pub bias: f64,
}

impl LinearRegression {
    /// Predict: ŷ = X · w + b (broadcast b to every row).
    ///
    /// Worked example — this one's free. Study how it uses your Matrix type, then implement `fit`.
    pub fn predict(&self, x: &Matrix) -> Matrix {
        let mut preds = x.matmul(&self.weights); // (n × 1)
        for r in 0..preds.rows {
            preds.set(r, 0, preds.get(r, 0) + self.bias);
        }
        preds
    }
}

/// Compute Mean Squared Error between predictions and targets.
///
/// ```text
///   MSE = (1/n) Σ (pred_i − actual_i)²
/// ```
///
/// EXERCISE: Implement this. Both `predictions` and `targets` are (n × 1) matrices.
/// Return the scalar MSE value.
pub fn mse(predictions: &Matrix, targets: &Matrix) -> f64 {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, 1);
    assert_eq!(targets.cols, 1);
    let n = predictions.rows;
    let mut sum = 0.0;
    for i in 0..n {
        sum += (predictions.get(i,0)-targets.get(i,0)).powf(2.0);
    }
    sum / (n as f64)
}

/// Train a linear regression model using batch gradient descent.
///
/// Arguments:
/// - `x`: input features, shape (n_samples × n_features)
/// - `y`: target values, shape (n_samples × 1)
/// - `lr`: learning rate (e.g. 0.01)
/// - `epochs`: number of gradient descent iterations
///
/// Returns a trained `LinearRegression`.
///
/// EXERCISE: Implement the training loop. Each epoch:
///   1. Predict: ŷ = X · w + b
///   2. Compute errors: e = ŷ − y              (n × 1)
///   3. Compute gradients:
///        dw = (2/n) · Xᵀ · e                  (d × 1)
///        db = (2/n) · sum(e)                   (scalar)
///   4. Update:
///        w ← w − lr · dw
///        b ← b − lr · db
///
/// Initialize w to zeros (n_features × 1) and b to 0.0.
///
/// Hint: you'll need to add elementwise subtraction and scalar multiplication to your Matrix
/// type — or do the arithmetic element-by-element with get/set. Either works; adding the ops
/// to Matrix is cleaner but not required for the gate.
pub fn fit(x: &Matrix, y: &Matrix, lr: f64, epochs: usize) -> LinearRegression {
    assert_eq!(x.rows, y.rows);
    assert_eq!(y.cols, 1);
    let n = x.rows as f64;
    let d = x.cols;

    unimplemented!("implement batch gradient descent — derive the gradients first")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MSE tests ---

    #[test]
    fn mse_perfect_predictions() {
        let preds = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let targets = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0]]);
        let loss = mse(&preds, &targets);
        assert!((loss - 0.0).abs() < 1e-10, "perfect predictions → MSE = 0");
    }

    #[test]
    fn mse_known_value() {
        // errors: [1, -1, 2], squared: [1, 1, 4], mean: 2.0
        let preds = Matrix::from_rows(&[&[2.0], &[4.0], &[6.0]]);
        let targets = Matrix::from_rows(&[&[1.0], &[5.0], &[4.0]]);
        let loss = mse(&preds, &targets);
        assert!((loss - 2.0).abs() < 1e-10, "expected MSE = 2.0, got {loss}");
    }

    // --- Gradient descent tests ---

    #[test]
    fn fit_learns_simple_line() {
        // y = 3x + 1, no noise. GD should converge close.
        let x = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
        let y = Matrix::from_rows(&[&[4.0], &[7.0], &[10.0], &[13.0], &[16.0]]);

        let model = fit(&x, &y, 0.01, 1000);

        let w = model.weights.get(0, 0);
        let b = model.bias;
        assert!(
            (w - 3.0).abs() < 0.1,
            "expected weight ≈ 3.0, got {w}"
        );
        assert!(
            (b - 1.0).abs() < 0.5,
            "expected bias ≈ 1.0, got {b}"
        );
    }

    #[test]
    fn fit_loss_decreases() {
        // Just verify that training actually reduces the loss.
        let x = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
        let y = Matrix::from_rows(&[&[2.0], &[4.0], &[6.0], &[8.0]]);

        // Before training: w=0, b=0 → predictions are all 0
        let zero_model = LinearRegression {
            weights: Matrix::zeros(1, 1),
            bias: 0.0,
        };
        let initial_preds = zero_model.predict(&x);
        let initial_loss = mse(&initial_preds, &y);

        // After training
        let trained = fit(&x, &y, 0.01, 500);
        let final_preds = trained.predict(&x);
        let final_loss = mse(&final_preds, &y);

        assert!(
            final_loss < initial_loss * 0.01,
            "loss should drop dramatically: {initial_loss} → {final_loss}"
        );
    }

    #[test]
    fn fit_multivariate() {
        // y = 2·x1 + 3·x2 + 1
        let x = Matrix::from_rows(&[
            &[1.0, 1.0],
            &[2.0, 1.0],
            &[1.0, 2.0],
            &[3.0, 2.0],
            &[2.0, 3.0],
        ]);
        let y = Matrix::from_rows(&[&[6.0], &[8.0], &[9.0], &[13.0], &[14.0]]);

        let model = fit(&x, &y, 0.01, 2000);

        let w0 = model.weights.get(0, 0);
        let w1 = model.weights.get(1, 0);
        let b = model.bias;
        assert!(
            (w0 - 2.0).abs() < 0.3,
            "expected w0 ≈ 2.0, got {w0}"
        );
        assert!(
            (w1 - 3.0).abs() < 0.3,
            "expected w1 ≈ 3.0, got {w1}"
        );
        assert!(
            (b - 1.0).abs() < 0.5,
            "expected bias ≈ 1.0, got {b}"
        );
    }
}
