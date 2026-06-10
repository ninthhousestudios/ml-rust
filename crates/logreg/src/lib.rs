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
        // Tip: You can inline the bias addition or write your own helper.
        // Don't rely on pre-baked versions if you want to internalize it.
        unimplemented!("Implement predict using matrix multiplication, bias addition, and your sigmoid function")
    }

    /// Classify: 1.0 if P(class=1) > threshold, else 0.0.
    /// EXERCISE: Implement the hard thresholding decision boundary.
    pub fn classify(&self, x: &Matrix, threshold: f64) -> Matrix {
        unimplemented!("Implement classification by thresholding your predicted probabilities")
    }
}

/// Apply the sigmoid function element-wise: σ(z) = 1 / (1 + e⁻ᶻ).
/// EXERCISE: Implement this. Input and output are both (n × 1) matrices.
pub fn sigmoid(z: &Matrix) -> Matrix {
    unimplemented!("derive σ'(z) = σ(z)·(1 − σ(z)) on paper, then implement element-wise sigmoid")
}

/// Compute binary cross-entropy loss.
/// L = −(1/n) Σ [ y_i · log(ŷ_i) + (1 − y_i) · log(1 − ŷ_i) ]
/// EXERCISE: Implement log-loss with stable clamping to prevent log(0).
pub fn cross_entropy(predictions: &Matrix, targets: &Matrix) -> f64 {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, 1);
    assert_eq!(targets.cols, 1);

    unimplemented!("Implement binary cross-entropy loss")
}

/// Train a logistic regression model using batch gradient descent.
/// EXERCISE: Implement the core training loop.
pub fn fit(x: &Matrix, y: &Matrix, lr: f64, epochs: usize) -> LogisticRegression {
    assert_eq!(x.rows, y.rows);
    assert_eq!(y.cols, 1);

    // Initialize w to zeros (n_features × 1) and b to 0.0
    unimplemented!("Implement the gradient descent loop using your new predict and math steps")
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
