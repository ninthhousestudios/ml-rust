//! Phase 0: the numerical substrate.
//!
//! Everything in this learning program — gradient descent, backprop, attention — is matrix
//! arithmetic underneath. Before we can build any of it, we need a matrix type we fully
//! understand. So we hand-roll one. No `ndarray`, no `nalgebra`. Just a `Vec<f64>` and the
//! arithmetic of indexing into it.
//!
//! ## The one idea in this file: row-major storage
//!
//! A matrix is conceptually 2D, but memory is 1D. We store the rows back-to-back in a flat
//! `Vec<f64>`. A 2x3 matrix
//!
//! ```text
//!   [ a b c ]
//!   [ d e f ]
//! ```
//!
//! lives in memory as `[a, b, c, d, e, f]`. To find element `(r, c)` we skip `r` whole rows
//! (each `cols` long) then walk `c` into the row:
//!
//! ```text
//!   flat_index = r * cols + c
//! ```
//!
//! That single formula is the whole trick. Internalize it and the rest is bookkeeping.

/// A dense matrix of `f64`, stored row-major in a flat vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Length is always `rows * cols`. Element (r, c) is at `data[r * cols + c]`.
    data: Vec<f64>,
}

impl Matrix {
    /// Build from a flat row-major vector. Panics if `data.len() != rows * cols` — the type's
    /// core invariant, so we check it loudly.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "expected {rows}x{cols} = {} elements, got {}",
            rows * cols,
            data.len()
        );
        Matrix { rows, cols, data }
    }

    /// An all-zeros matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build from nested rows, e.g. `Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]])`.
    /// Convenient for writing tests and examples by hand.
    pub fn from_rows(rows_data: &[&[f64]]) -> Self {
        let rows = rows_data.len();
        let cols = if rows == 0 { 0 } else { rows_data[0].len() };
        let mut data = Vec::with_capacity(rows * cols);
        for row in rows_data {
            assert_eq!(row.len(), cols, "all rows must have the same length");
            data.extend_from_slice(row);
        }
        Matrix { rows, cols, data }
    }

    /// Read element (r, c). This is where the row-major formula earns its keep.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Write element (r, c).
    pub fn set(&mut self, r: usize, c: usize, value: f64) {
        self.data[r * self.cols + c] = value;
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.rows, other.rows,
            "matrices must have the same number of rows"
        );
        assert_eq!(
            self.cols, other.cols,
            "matrices must have the same number of columns"
        );
        let mut out = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(r, c, self.get(r, c) - other.get(r, c));
            }
        }
        out
    }

    pub fn scamul(&self, scalar: f64) -> Matrix {
        let mut out = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(r, c, self.get(r, c) * scalar);
            }
        }
        out
    }

    // broadcast bias b to each row of self
    pub fn broadcast(&mut self, bias: f64) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.set(r, c, self.get(r, c) + bias);
            }
        }
    }

    /// Transpose: the (r, c) of the result is the (c, r) of the original. A worked example of
    /// using the indexing formula — read it, then write `matmul` in the same spirit.
    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Matrix multiply: `self` (m x n) times `other` (n x p) → (m x p).
    ///
    /// EXERCISE (Milestone 0). Implement this. The math:
    ///
    /// ```text
    ///   result(i, j) = sum over k of  self(i, k) * other(k, j)
    /// ```
    ///
    /// i.e. row `i` of `self` dotted with column `j` of `other`. Steps:
    ///   1. The inner dimensions must agree: `self.cols == other.rows`. Assert it.
    ///   2. The result is `self.rows` x `other.cols`. Start with `Matrix::zeros(...)`.
    ///   3. Triple loop over i, j, k accumulating the sum, then `set(i, j, sum)`.
    ///
    /// When `cargo test` goes green, the gate is cleared.

    // this is the naive implementation
    /*
    Right now, your loops are structured as R -> C -> K. Inside the innermost loop, you are calling
    other.get(k, c).

    If your matrix stores its data in a flat Vec row-by-row (Row-Major order, which is standard),
    changing k while keeping c constant means you are jumping down columns in memory.

    This causes CPU cache misses. The CPU wants to read continuous memory, but you are forcing it to
    skip ahead by an entire row's length on every iteration of k.

    */
    // next, implement this new version that is more efficient
    /*
    The Fix: Swap the loops to R -> K -> C If you change the loop order, you can accumulate values
    across a row of other continuously in memory. CPUs love this, and for larger matrices, this small
    change can make your code 5x to 10x faster without changing the math logic.
    */
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "inner dimensions must be equal");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                for c in 0..other.cols {
                    out.set(r, c, out.get(r, c) + (self.get(r, k) * other.get(k, c)));
                }
            }
        }
        out
    }

    pub fn matmul_naive(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "inner dimensions must be equal");
        let mut out = Matrix::zeros(self.rows, other.cols);
        let mut sum: f64 = 0.0;
        for r in 0..self.rows {
            for c in 0..other.cols {
                for k in 0..self.cols {
                    // get value at (r,k) and (k,c), multiply these values, then add
                    sum += self.get(r, k) * other.get(k, c);
                }
                out.set(r, c, sum);
                sum = 0.0;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_and_indexing() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        assert_eq!((m.rows, m.cols), (2, 3));
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 6.0); // flat index 1*3 + 2 = 5
    }

    #[test]
    fn set_writes_the_right_cell() {
        let mut m = Matrix::zeros(2, 2);
        m.set(1, 0, 9.0);
        assert_eq!(m.get(1, 0), 9.0);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn transpose_swaps_axes() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!((t.rows, t.cols), (3, 2));
        assert_eq!(
            t,
            Matrix::from_rows(&[&[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]])
        );
    }

    // --- The exercise. These fail until you implement `matmul`. ---

    #[test]
    fn matmul_identity_is_a_noop() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let id = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        assert_eq!(a.matmul(&id), a);
    }

    #[test]
    fn matmul_known_product() {
        // [1 2 3]   [ 7  8 ]   [ 58  64 ]
        // [4 5 6] x [ 9 10 ] = [139 154]
        //           [11 12]
        let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Matrix::from_rows(&[&[7.0, 8.0], &[9.0, 10.0], &[11.0, 12.0]]);
        let expected = Matrix::from_rows(&[&[58.0, 64.0], &[139.0, 154.0]]);
        assert_eq!(a.matmul(&b), expected);
    }

    #[test]
    fn matmul_naive_identity_is_a_noop() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let id = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        assert_eq!(a.matmul_naive(&id), a);
    }

    #[test]
    fn matmul_naive_known_product() {
        // [1 2 3]   [ 7  8 ]   [ 58  64 ]
        // [4 5 6] x [ 9 10 ] = [139 154]
        //           [11 12]
        let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Matrix::from_rows(&[&[7.0, 8.0], &[9.0, 10.0], &[11.0, 12.0]]);
        let expected = Matrix::from_rows(&[&[58.0, 64.0], &[139.0, 154.0]]);
        assert_eq!(a.matmul_naive(&b), expected);
    }
}
