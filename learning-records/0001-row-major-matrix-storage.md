# Row-major matrix storage and the indexing formula

Josh built a `Matrix` type backed by a flat `Vec<f64>` and demonstrated understanding of
row-major layout: element (r, c) lives at `data[r * cols + c]`. He implemented `get`, `set`,
constructors, and transpose using this formula. This is foundational — every future milestone
builds on this mental model of how 2D data lives in 1D memory.

## Evidence

Implemented in `crates/linalg/src/lib.rs`. All indexing tests pass. He can explain why
`r * cols + c` works and what it means for memory layout.
