//! Milestone 3: Decision tree classifier — entropy, information gain, and recursive splitting.
//!
//! ## What we're building
//!
//! A binary classifier that learns by recursively partitioning the feature space. Unlike
//! logistic regression (which finds a linear decision boundary), a decision tree carves out
//! axis-aligned rectangular regions — it can learn non-linear patterns like XOR that no
//! linear model can capture.
//!
//! ## The core concepts
//!
//! **Entropy** measures how "mixed" a set of labels is:
//! ```text
//!   H(S) = −Σ p_i · log₂(p_i)
//! ```
//! H = 0 for a pure set (all one class); H = 1 for a maximally mixed binary set (50/50).
//! Convention: 0 · log₂(0) = 0.
//!
//! **Information gain** measures how much a split reduces entropy:
//! ```text
//!   IG = H(parent) − (|left|/|total|)·H(left) − (|right|/|total|)·H(right)
//! ```
//! Higher IG = better split. The tree algorithm picks the split with highest IG at each node.
//!
//! ## The CART algorithm
//!
//! At each node:
//! 1. If the node is pure (H = 0) or max depth reached, create a `Leaf` with the majority class.
//! 2. Otherwise, for each feature, for each candidate threshold (midpoints between sorted
//!    unique values), compute IG.
//! 3. Pick the (feature, threshold) with highest IG.
//! 4. Split the data: left = rows where x[feature] ≤ threshold, right = the rest.
//! 5. Recurse to build left and right children.
//!
//! ## Implementation order
//!
//! 1. `entropy` — start here, smallest function
//! 2. `information_gain` — uses entropy
//! 3. `precision` and `recall` — evaluation metrics, self-contained
//! 4. `train_test_split` — honest evaluation
//! 5. `predict_one` — tree traversal via pattern matching on `Node`
//! 6. `predict` — calls predict_one per row
//! 7. `fit` — the main event: recursive tree construction
//!
//! See `lessons/0003-decision-trees-ml-landscape.html` for the full conceptual walkthrough.

use linalg::Matrix;

/// A node in the decision tree. This is the recursive data structure you'll build in `fit`.
///
/// `Leaf` holds a class prediction. `Split` holds the decision rule and its two children.
/// `Box<Node>` is how Rust handles recursive types — each child is heap-allocated so the
/// compiler knows the size of `Node` at compile time.
pub enum Node {
    Leaf {
        class: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// A trained decision tree. Wraps the root node.
pub struct DecisionTree {
    pub root: Node,
}

// ---------------------------------------------------------------------------
// Core ML functions — implement these in order
// ---------------------------------------------------------------------------

/// Compute the entropy of a set of binary labels (0.0 or 1.0).
///
/// H(S) = −Σ p_i · log₂(p_i)
///
/// Use `f64::log2()`. Handle the 0·log₂(0) = 0 convention by skipping classes
/// with zero probability.
pub fn entropy(labels: &[f64]) -> f64 {
    unimplemented!("Compute entropy of the label distribution")
}

/// Compute information gain for a binary split.
///
/// IG = H(parent) − (|left|/|total|)·H(left) − (|right|/|total|)·H(right)
pub fn information_gain(parent: &[f64], left: &[f64], right: &[f64]) -> f64 {
    unimplemented!("Compute IG using entropy of parent and weighted entropy of children")
}

/// Train a decision tree using the CART algorithm.
///
/// `x` is (n_samples × n_features), `y` is (n_samples × 1) with binary labels.
/// `max_depth` limits tree depth to prevent overfitting.
///
/// You'll need a recursive helper function — `fit` itself sets up the initial call,
/// and the helper builds subtrees. The helper should take:
/// - The data (x, y)
/// - Which rows belong to this node (a `&[usize]` of row indices)
/// - The current depth
/// - The max depth
///
/// At each level: try every feature × every threshold, compute IG, pick the best.
/// To get candidate thresholds: collect the unique values of a feature for the current
/// rows, sort them, and use midpoints between adjacent values.
///
/// Return `Node::Leaf` when pure, at max depth, or when no split improves IG.
/// Return `Node::Split` with `Box::new(left_child)` and `Box::new(right_child)`.
pub fn fit(x: &Matrix, y: &Matrix, max_depth: usize) -> DecisionTree {
    unimplemented!("Build a decision tree by recursively finding the best splits")
}

impl DecisionTree {
    /// Predict the class for a single sample.
    ///
    /// Walk the tree from root to leaf: at each `Split`, go left if
    /// `features[feature] <= threshold`, else right. At a `Leaf`, return the class.
    ///
    /// Hint: `match` on `&self.root` works, but you'll need a way to recurse
    /// into children. A free function taking `&Node` is one approach; a loop
    /// with a mutable reference (`let mut node = &self.root`) is another.
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        unimplemented!("Walk the tree from root to leaf using the feature values")
    }

    /// Predict classes for all rows in a matrix.
    ///
    /// Extract each row as a `Vec<f64>`, call `predict_one`, collect results.
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        unimplemented!("Call predict_one for each row of x")
    }
}

// ---------------------------------------------------------------------------
// Evaluation metrics — completing the thread from Milestones 1-3
// ---------------------------------------------------------------------------

/// Precision: of everything predicted positive, how many actually were?
///
/// precision = TP / (TP + FP)
/// TP = predicted 1.0 AND actual 1.0
/// FP = predicted 1.0 AND actual 0.0
pub fn precision(predicted: &[f64], actual: &[f64]) -> f64 {
    unimplemented!("Count true positives and false positives, compute the ratio")
}

/// Recall: of everything actually positive, how many did we catch?
///
/// recall = TP / (TP + FN)
/// FN = predicted 0.0 AND actual 1.0
pub fn recall(predicted: &[f64], actual: &[f64]) -> f64 {
    unimplemented!("Count true positives and false negatives, compute the ratio")
}

/// Split data into training and test sets.
///
/// Takes the first `floor(n * train_ratio)` rows as training, the rest as test.
/// Returns (x_train, y_train, x_test, y_test).
///
/// Deterministic (no shuffling) — keeps it simple and testable.
/// Real ML shuffles first; this is the conceptual version.
pub fn train_test_split(
    x: &Matrix,
    y: &Matrix,
    train_ratio: f64,
) -> (Matrix, Matrix, Matrix, Matrix) {
    unimplemented!("Split x and y into train/test portions based on train_ratio")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Entropy tests ---

    #[test]
    fn entropy_pure_set() {
        assert!(
            (entropy(&[1.0, 1.0, 1.0]) - 0.0).abs() < 1e-10,
            "all same class should have entropy 0"
        );
    }

    #[test]
    fn entropy_maximum_uncertainty() {
        assert!(
            (entropy(&[0.0, 1.0, 0.0, 1.0]) - 1.0).abs() < 1e-10,
            "50/50 binary split should have entropy 1.0"
        );
    }

    #[test]
    fn entropy_skewed() {
        // 75% class 0, 25% class 1 → H ≈ 0.8113
        let h = entropy(&[0.0, 0.0, 0.0, 1.0]);
        assert!((h - 0.8113).abs() < 0.001, "expected ~0.8113, got {h}");
    }

    // --- Information gain tests ---

    #[test]
    fn information_gain_perfect_split() {
        let parent = &[0.0, 0.0, 1.0, 1.0];
        let left = &[0.0, 0.0];
        let right = &[1.0, 1.0];
        let ig = information_gain(parent, left, right);
        assert!(
            (ig - 1.0).abs() < 1e-10,
            "perfect split should have IG = 1.0, got {ig}"
        );
    }

    #[test]
    fn information_gain_no_improvement() {
        let parent = &[0.0, 1.0, 0.0, 1.0];
        let left = &[0.0, 1.0];
        let right = &[0.0, 1.0];
        let ig = information_gain(parent, left, right);
        assert!(
            ig.abs() < 1e-10,
            "useless split should have IG ~ 0, got {ig}"
        );
    }

    // --- Decision tree tests ---

    #[test]
    fn fit_learns_and_gate() {
        let x = Matrix::from_rows(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
        let y = Matrix::from_rows(&[&[0.0], &[0.0], &[0.0], &[1.0]]);
        let tree = fit(&x, &y, 3);
        let preds = tree.predict(&x);
        for i in 0..4 {
            assert_eq!(
                preds[i],
                y.get(i, 0),
                "AND gate: wrong prediction for row {i}"
            );
        }
    }

    #[test]
    fn fit_learns_xor() {
        // XOR is NOT linearly separable — logistic regression cannot learn this.
        // A decision tree can. This is the gate insight.
        let x = Matrix::from_rows(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
        let y = Matrix::from_rows(&[&[0.0], &[1.0], &[1.0], &[0.0]]);
        let tree = fit(&x, &y, 3);
        let preds = tree.predict(&x);
        for i in 0..4 {
            assert_eq!(preds[i], y.get(i, 0), "XOR: wrong prediction for row {i}");
        }
    }

    // --- Evaluation tests ---

    #[test]
    fn precision_and_recall_basic() {
        // TP=2 (indices 0,1), FP=1 (index 2), FN=1 (index 3)
        let pred = &[1.0, 1.0, 1.0, 0.0];
        let actual = &[1.0, 1.0, 0.0, 1.0];
        assert!(
            (precision(pred, actual) - 2.0 / 3.0).abs() < 1e-10,
            "precision should be 2/3"
        );
        assert!(
            (recall(pred, actual) - 2.0 / 3.0).abs() < 1e-10,
            "recall should be 2/3"
        );
    }

    #[test]
    fn train_test_split_preserves_sizes() {
        let x = Matrix::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]]);
        let y = Matrix::from_rows(&[&[0.0], &[0.0], &[1.0], &[1.0], &[1.0]]);
        let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, 0.6);
        assert_eq!(x_train.rows, 3, "60% of 5 rows = 3 train rows");
        assert_eq!(x_test.rows, 2, "remaining 2 test rows");
        assert_eq!(y_train.rows, 3);
        assert_eq!(y_test.rows, 2);
        assert_eq!(x_train.cols, 1);
        assert_eq!(x_test.cols, 1);
    }
}
