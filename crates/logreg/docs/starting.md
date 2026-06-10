## 📐 Your Milestone 2 Checklist

To tackle this milestone completely on your own terms, follow the breakdown laid out in
your lesson file:

### 1. The Paper Derivation (Load-Bearing Derivation #2)

Before writing a single line of Rust, prove to yourself on paper how the derivative of
the cross-entropy loss combines with the derivative of the sigmoid. Watch how the messy
fractions elegantly collapse into that beautifully clean error term:


$$\hat{y} - y$$

### 2. Element-wise Sigmoid

Implement `sigmoid`. Since you're working with your own custom `Matrix` substrate,
decide how you want to handle mutating or allocating elementwise operations. Rust's
`f64::exp()` is your tool here.

### 3. Log-Loss with Epsilon Clamping

When implementing `cross_entropy`, remember the lesson's warning about the extreme
penalty of log-loss. If your model predicts exactly `0.0` or `1.0`, `f64::ln()` will
yield $\infty$, destroying your gradients. Use `.clamp(1e-15, 1.0 - 1e-15)` on your
predictions vector before computing the logs.

### 4. The Training Loop (`fit`)

Reuse your structural insights from your linear regression milestone. The matrix
dimensions match up exactly, but you will drop the factor of 2 when computing `dw` and
`db`:

* `dw = (1/n) * xt.matmul(&errors)`
* `db = (1/n) * sum_e(&errors)`

---

When you've cleared out the file and are ready to map your manual derivations to the
code implementation, where do you want to dig in first—the element-wise matrix
transformation for `sigmoid`, or setting up the log-loss equation?
