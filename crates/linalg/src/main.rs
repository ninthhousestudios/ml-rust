use linalg::Matrix;
use std::time::Instant;

fn main() {
    // 1. Setup two 500x500 matrices
    // (Adjust this initialization to match how your Matrix struct actually builds random/dummy data)
    let size = 1000;
    let mut m1 = Matrix::zeros(size, size);
    let mut m2 = Matrix::zeros(size, size);

    // Fill them with some arbitrary data so the CPU actually does the math
    for r in 0..size {
        for c in 0..size {
            m1.set(r, c, (r + c) as f64 * 0.1);
            m2.set(r, c, (r - c) as f64 * 0.1);
        }
    }

    println!(
        "Running benchmarks for {}x{} matrix multiplication...",
        size, size
    );
    println!("---------------------------------------------------------");

    // 2. Benchmark the Naive Implementation (r -> c -> k)
    let start_naive = Instant::now();
    let _res_naive = m1.matmul_naive(&m2);
    let duration_naive = start_naive.elapsed();

    println!("Naive Implementation:   {:?}", duration_naive);

    // 3. Benchmark your Optimized Implementation (r -> k -> c)
    let start_opt = Instant::now();
    let _res_opt = m1.matmul(&m2); // Note: Make sure matmul takes &Matrix if you updated it!
    let duration_opt = start_opt.elapsed();

    println!("Optimized Cache-Friendly: {:?}", duration_opt);
    println!("---------------------------------------------------------");

    // 4. Calculate the speedup factor
    let speedup = duration_naive.as_secs_f64() / duration_opt.as_secs_f64();
    println!("Your optimization made the code {:.2}x faster!", speedup);
}
