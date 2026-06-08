# Matrix multiplication and cache-friendly loop ordering

Josh implemented matmul two ways: a naive R→C→K loop and a cache-friendly R→K→C reordering.
He understands that in row-major storage, iterating over `k` with fixed `c` (the naive way)
causes cache misses because it jumps down columns of the `other` matrix. Swapping to R→K→C
keeps memory access sequential. He benchmarked both and saw the performance difference.

This insight — that the same mathematical operation can have wildly different performance
depending on memory access patterns — is important context for why real ML frameworks care
so much about data layout.

## Evidence

Both `matmul` and `matmul_naive` implemented and tested in `crates/linalg/src/lib.rs`.
Wrote a benchmark comparing the two (in `crates/linalg/src/main.rs`).

## Implications

When we get to autograd (Milestone 4) and attention (Milestone 7), memory layout decisions
will matter again. This is the first time Josh has seen it; it won't be the last.
