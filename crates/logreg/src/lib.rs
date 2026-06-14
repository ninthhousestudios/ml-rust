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
    /// EXERCISE: Implement the forward pass.
    pub fn predict(&self, x: &Matrix) -> Matrix {
        // unimplemented!("Implement predict using matrix multiplication, bias addition, and your sigmoid function")
        assert_eq!(
            x.cols, self.weights.rows,
            "inner dimensions of X and w must be the same"
        );
        let mut xdotw = x.matmul(&self.weights);
        xdotw.broadcast(self.bias);
        sigmoid(&xdotw)
    }

    pub fn update(&mut self, new_w: Matrix, new_b: f64) {
        self.weights = new_w;
        self.bias = new_b;
    }

    /// Classify: 1.0 if P(class=1) > threshold, else 0.0.
    /// EXERCISE: Implement the hard thresholding decision boundary.
    pub fn classify(&self, x: &Matrix, threshold: f64) -> Matrix {
        // unimplemented!("Implement classification by thresholding your predicted probabilities")
        let predictions = self.predict(&x);
        let mut out = Matrix::zeros(x.rows, 1);
        for r in 0..x.rows {
            for c in 0..x.cols {
                dbg!(predictions.get(r, c));
                if predictions.get(r, c) > threshold {
                    out.set(r, c, 1.0)
                } else {
                    out.set(r, c, 0.0)
                }
            }
        }
        out
    }
}

/// Apply the sigmoid function element-wise: σ(z) = 1 / (1 + e⁻ᶻ).
/// EXERCISE: Implement this. Input and output are both (n × 1) matrices.
pub fn sigmoid(z: &Matrix) -> Matrix {
    assert_eq!(z.cols, 1, "z must by an nx1 matrix");
    let mut out = Matrix::zeros(z.rows, 1);
    for r in 0..z.rows {
        out.set(r, 0, sig(z.get(r, 0)));
    }
    out
}

fn sig(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Compute binary cross-entropy loss.
/// L = −(1/n) Σ [ y_i · log(ŷ_i) + (1 − y_i) · log(1 − ŷ_i) ]
/// EXERCISE: Implement log-loss with stable clamping to prevent log(0).
pub fn cross_entropy(yh: &Matrix, y: &Matrix) -> f64 {
    assert_eq!(yh.rows, y.rows);
    assert_eq!(yh.cols, 1);
    assert_eq!(y.cols, 1);
    let mut sum: f64 = 0.0;
    let n = yh.rows;

    for r in 0..yh.rows {
        sum += (y.get(r, 0) * (yh.get(r, 0).clamp(1e-15, 1.0 - 1e-15).ln()))
            + ((1.0 - y.get(r, 0)) * (1.0 - yh.get(r, 0)).clamp(1e-15, 1.0 - 1e-15).ln())
    }

    (-sum) / (n as f64)
}

/// Train a logistic regression model using batch gradient descent.
/// EXERCISE: Implement the core training loop.
/// training loop: predict, find gradients, update weights and bias, do it again epochs number of times
pub fn fit(x: &Matrix, y: &Matrix, lr: f64, epochs: usize) -> LogisticRegression {
    assert_eq!(x.rows, y.rows);
    assert_eq!(y.cols, 1);
    // Initialize w to zeros (n_features × 1) and b to 0.0
    let mut logreg: LogisticRegression = LogisticRegression {
        weights: Matrix::zeros(x.cols, 1),
        bias: 0.0,
    };
    let xt = x.transpose();
    let n = x.rows;

    for _ in 0..epochs {
        let yh = logreg.predict(&x);
        let errors = yh.subtract(&y);
        let dw = xt.matmul(&errors).scamul(1.0 / n as f64);
        let db = (1.0 / n as f64) * sum_e(&errors);
        logreg.update(
            logreg.weights.subtract(&dw.scamul(lr)),
            logreg.bias - lr * db,
        );
    }
    logreg
}

fn sum_e(e: &Matrix) -> f64 {
    assert_eq!(e.cols, 1, "errors should only have 1 column");
    let mut sum: f64 = 0.0;
    for r in 0..e.rows {
        sum += e.get(r, 0);
    }
    sum
}

/// Fraction of predictions matching targets.
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
            s_pos,
            s_neg,
            s_pos + s_neg
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
            &[1.0],
            &[1.5],
            &[2.0],
            &[2.5],
            &[4.0],
            &[4.5],
            &[5.0],
            &[5.5],
        ]);
        let y = Matrix::from_rows(&[
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0],
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
            &[1.0, 0.5],
            &[1.5, 1.0],
            &[2.0, 0.8],
            &[4.0, 3.5],
            &[4.5, 4.0],
            &[5.0, 3.8],
        ]);
        let y = Matrix::from_rows(&[&[0.0], &[0.0], &[0.0], &[1.0], &[1.0], &[1.0]]);

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
