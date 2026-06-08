# Elementwise matrix operations on the Matrix type

Josh added `subtract` and `scamul` (scalar multiplication) to the Matrix type while working on
Milestone 1 scaffolding. These follow the same flat-vector pattern as the existing ops: assert
dimensions match, loop over elements, apply the operation. He chose to add them as methods on
Matrix rather than doing inline get/set loops in the linreg code — the cleaner path.

This is prior knowledge for Milestone 1: the matrix arithmetic needed for gradient descent
(computing errors, scaling gradients) is now available.
