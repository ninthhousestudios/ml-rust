    /* extra credit
       2. The Idiomatic Level-Up: Ditch the Indices for Iterators
    In Rust, manual indexing (0..self.rows) and manual getters (self.get(r, k)) have a hidden cost: Bounds Checking. Every time you call get(), Rust secretly checks if the index is out of bounds to prevent crashes.

    Idiomatic Rust prefers Iterators. When you use iterators, the compiler proves safely at compile-time that you won't overflow the array, so it strips out all bounds checking entirely, leading to massive speedups.

    If your Matrix struct exposes row or column iterators, you can rewrite the core logic using standard functional tools like .zip(), .map(), and .sum().

    For example, calculating a single cell value by taking the dot product of a row and a column can look like this:

    Rust


    // A hypothetical idiomatic dot product for a single cell
    let cell_value: f64 = self.get_row(r)
        .zip(_other.get_col(c))
        .map(|(x, y)| x * y)
        .sum();
    Summary of Next Steps
    Easiest Win: Swap your loops to R -> K -> C and see if you can benchmark the speed difference.

    Rustaceous Win: See if you can implement methods on your Matrix struct like row_iter(&self, row: usize) that return slices or iterators, then use those to clean up the indexing math.

    How is your Matrix data stored under the hood (e.g., a 1D Vec<f64> or a 2D Vec<Vec<f64>>)? That will dictate exactly how you can implement those faster iterators!
    */

