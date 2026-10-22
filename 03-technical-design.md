# 03 - Technical Design

## 1. Overview and Core Philosophy

`stackgate` is a pure-Python, zero-dependency statistical regression detection engine for sampling profiles. It bridges the gap between raw sampling profiles (such as those from Python 3.15's `profiling.sampling` Tachyon, `py-spy`, Austin, or folded stack generators) and CI performance gates.

Sampling profilers are inherently stochastic: each run is a random sample drawn from the program's execution time distribution. Comparing a single baseline run against a single candidate run produces false positives due to sampling noise and runner variance.

`stackgate` solves this by:
1. Collecting multiple runs ($K$ baseline, $M$ candidate runs).
2. Decomposing profile trees into per-function exclusive (self) and inclusive (cumulative) sample times, as well as exact call paths.
3. Conducting exact or Monte Carlo permutation tests and Welch's t-tests to evaluate the probability of regression.
4. Applying False Discovery Rate (FDR) control via the Benjamini-Hochberg (BH) procedure across all profiled functions to prevent false discoveries.
5. Decomposing the total regression into additive, function-level attributions.
6. Providing an A/A calibration workflow to measure the environment noise floor and determine recommended sample sizes.

## 2. Profile Data Model

### 2.1 Raw Input Formats
`stackgate` supports:
- **Collapsed / Folded Stacks**: The standard format introduced by Brendan Gregg:
  ```text
  main;process_request;parse_json 142
  main;process_request;validate 45
  main;worker_thread;fetch_db 320
  ```
- **Directory Ingestion**: A directory containing multiple `.folded` or `.txt` run profiles:
  ```text
  baseline/
    run_01.folded
    run_02.folded
    run_03.folded
    ...
  candidate/
    run_01.folded
    run_02.folded
    run_03.folded
    ...
  ```
- **Tachyon / Custom JSON**: JSON-serialized profile runs containing metadata and stack sample dictionaries.

### 2.2 In-Memory Representations
- **`StackFrame`**: Represents a single frame: function name, filename, and line number.
- **`CallPath`**: An immutable tuple of frames representing a root-to-leaf call path.
- **`Profile`**: A single run profile containing:
  - Total samples and estimated duration.
  - Mapping of `CallPath -> sample_count`.
  - Aggregated self-time (exclusive samples where frame is the leaf).
  - Aggregated inclusive-time (cumulative samples where frame appears anywhere in stack).
  - Metadata (run ID, sampling rate in Hz, total wall-clock duration).
- **`ProfileSet`**: A collection of $N$ independent runs for a single build/branch.
  - Aggregates distributions of samples across runs for each function and call path.
  - Computes run-to-run summary statistics: mean, standard deviation, variance, coefficient of variation (CV).

## 3. Statistical Methodology

### 3.1 Hypothesis Testing for Total Regression
Let $X = (x_1, \dots, x_K)$ be total run times (or sample counts) for the baseline, and $Y = (y_1, \dots, y_M)$ be those for the candidate.
- Null Hypothesis $H_0$: The candidate execution time distribution is identical to or faster than baseline ($\mu_Y \le \mu_X$).
- Alternative Hypothesis $H_1$: The candidate execution time distribution is slower than baseline ($\mu_Y > \mu_X$).

#### Permutation Test (Default)
Permutation testing is non-parametric and makes no normality assumptions.
- If $K + M \le 16$, exact permutation enumerates all $\binom{K+M}{K}$ assignments.
- If $K + M > 16$, a Monte Carlo permutation test with $B = 10,000$ iterations is performed with a Laplace-corrected p-value:
  $$p = \frac{\sum_{b=1}^B \mathbb{I}(T_b \ge T_{\text{obs}}) + 1}{B + 1}$$

#### Welch's t-test
For fast parametric testing with unequal sample variances:
$$t = \frac{\bar{Y} - \bar{X}}{\sqrt{\frac{s_X^2}{K} + \frac{s_Y^2}{M}}}$$
with Welch–Satterthwaite degrees of freedom $\nu$. Cumulative distribution function calculated via the regularized incomplete beta function $I_x(a, b)$.

### 3.2 Confidence Intervals
Bootstrap confidence intervals (default 95% BCa or percentile bootstrap with 10,000 resamples) compute:
- Absolute difference $\Delta = \bar{Y} - \bar{X}$ (with $[\Delta_{\text{lower}}, \Delta_{\text{upper}}]$)
- Relative percentage change $\% \Delta = \frac{\bar{Y} - \bar{X}}{\bar{X}} \times 100\%$ (with $[\% \Delta_{\text{lower}}, \% \Delta_{\text{upper}}]$)

### 3.3 Multiple-Comparison Control (Function Attribution)
When testing hundreds of functions simultaneously, raw $p$-values will produce false discoveries. `stackgate` controls the False Discovery Rate (FDR) at level $q$ (default 0.05) using the Benjamini-Hochberg (BH) step-up procedure:
1. Sort raw $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Compute adjusted $q$-values:
   $$q_{(i)} = \min_{j \ge i} \left( \frac{m}{j} p_{(j)} \right)$$
3. Flag function $i$ as a significant contributor if $q_{(i)} \le \alpha_{\text{FDR}}$ and effect exceeds noise floor.

### 3.4 Additive Regression Attribution
For each regressed function $f$:
$$\Delta_f = \bar{Y}_f - \bar{X}_f$$
Attribution percentage:
$$\text{Attribution}_f = \frac{\Delta_f}{\Delta_{\text{total}}} \times 100\%$$
This attributes exactly what fraction of the slowdown is explained by each specific function.

### 3.5 Noise Floor & A/A Calibration
An A/A test repeatedly splits a set of baseline runs into two random partitions and runs the regression detector.
- Measured False Positive Rate (FPR) must be $\le \alpha$.
- The 95th percentile of pseudo-differences establishes the runner's empirical **noise floor**.
- Any observed candidate regression smaller than this noise floor is marked INCONCLUSIVE or PASS rather than a false alarm.

## 4. Verdict Decision Matrix

| Total Time Stat. Sig.? | Effect Size vs Threshold | Function Attribution Found? | Verdict |
|---|---|---|---|
| Yes ($p < \alpha$) | $\Delta\% > \text{threshold}$ and $>$ noise floor | Yes | **FAIL** (Regression) |
| Yes ($p < \alpha$) | $\Delta\% < -\text{threshold}$ | Speedup | **IMPROVEMENT** |
| No ($p \ge \alpha$) | Within noise floor | No | **PASS** |
| High Variance / Low Power | $K < 3$ or CI spans $\pm \text{threshold}$ | Ambiguous | **INCONCLUSIVE** |
