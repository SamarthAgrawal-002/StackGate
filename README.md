# StackGate 🚦

[![PyPI version](https://img.shields.io/pypi/v/stackgate.svg?color=blue)](https://pypi.org/project/stackgate/)
[![Python versions](https://img.shields.io/pypi/pyversions/stackgate.svg)](https://pypi.org/project/stackgate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Zero Runtime Dependencies](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-brightgreen.svg)](#features)
[![CI](https://github.com/samarthagrawal/stackgate/actions/workflows/ci.yml/badge.svg)](https://github.com/samarthagrawal/stackgate/actions)

> **Statistically rigorous, function-level performance-regression detection from sampling profiles, for Python.**

Python 3.15 introduces `profiling.sampling` ("Tachyon") in the standard library, enabling low-overhead sampling profiles and differential flame graphs. But comparing a single baseline run to a single candidate run produces false alarms: sampling noise and CI runner jitter easily swamp small 1–5% differences.

**StackGate closes this gap**:
1. Takes **multiple runs per side** ($K$ baseline runs vs $M$ candidate runs).
2. Models run-to-run noise and calculates a runner **noise floor**.
3. Conducts **permutation tests** to determine whether the program got slower with rigorous statistical significance ($p$-value and confidence interval).
4. Controls false discoveries across hundreds of functions using the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure.
5. Attributes the regression **additively to specific functions and call paths** (e.g. `parse_message: +0.28s, 80% of regression, q=0.0004`).
6. Outputs automated **PASS / FAIL / INCONCLUSIVE** verdicts, Markdown PR comments for GitHub Actions, and structured JSON for CI gates.

---

## What a user sees

```text
$ stackgate compare profiles/base/ profiles/head/ --threshold 2.0

Verdict: FAIL (Performance Regression Detected)
Total wall time:   4.120s -> 4.470s (+8.50%, 95% CI +5.9% .. +11.2%, p=0.0003, K=8 vs 8)
Runner noise floor: 1.45% | Threshold: 2.00% | Baseline CV: 3.2%

Where it went (self time, FDR q < 0.05):
  parse_message                  0.610s -> 0.890s  +0.280s  (80.0% of regression)  q=0.0004
  json/decoder.py:raw_decode     0.150s -> 0.210s  +0.060s  (17.1% of regression)  q=0.0018

New call paths (1):
  + main;handle_request;validate_schema;re.compile     +0.050s

Not significant / below noise floor: 212 functions
```

---

## Key Features

- 🔬 **Statistically Rigorous**: Exact or Monte Carlo two-sample permutation tests, Welch's t-test, and bootstrap confidence intervals.
- 🎯 **Function-Level Attribution**: Pinpoints the exact functions driving the slowdown and calculates each function's share of the total regression.
- 🛡️ **Multiple-Comparison Protection**: Applies the Benjamini-Hochberg (BH) procedure to control False Discovery Rate (FDR) when testing hundreds of functions simultaneously.
- 🎛️ **Empirical A/A Calibration**: Measures your CI runner's natural jitter and determines the recommended sample size $K$ to reliably detect a given effect size.
- ⚡ **Zero External Dependencies**: Pure Python standard library core (`math`, `statistics`, `random`, `argparse`). Installs anywhere in milliseconds with zero supply-chain risk.
- 🧩 **Universal Profile Ingestion**: Ingests Brendan Gregg collapsed/folded stacks, Python 3.15 Tachyon profiles, `py-spy` profiles, Austin, and custom JSON.
- 🤖 **CI-First Automation**: Built-in Markdown PR comment generator, color terminal reports, and exit codes (0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE).
- 🧪 **Pytest Integration**: Includes a `stackgate_benchmark` fixture and `assert_no_regression` assertion helper.

---

## Installation

```bash
pip install stackgate
```

*StackGate requires Python 3.10+ and has **zero** required third-party dependencies.*

---

## Quickstart

### 1. Command-Line Interface (CLI)

#### Comparing Profile Directories
Collect baseline runs (e.g., from `main`) and candidate runs (from your PR), then compare them:

```bash
# Compare directories of folded profiles
stackgate compare profiles/baseline/ profiles/candidate/ --threshold 2.0

# Generate a GitHub PR comment markdown report
stackgate compare profiles/base/ profiles/head/ --format markdown -o pr_comment.md

# Output machine-readable JSON for CI pipelines
stackgate compare profiles/base/ profiles/head/ --format json -o result.json
```

#### Empirical A/A Calibration
Measure your test runner's noise floor and determine how many runs ($K$) you need:

```bash
stackgate calibrate profiles/baseline/
```
Output:
```text
A/A Calibration on 10 runs (100 splits):
  Empirical False Positive Rate: 4.0% (target: 5.0%)
  Run-to-run noise floor (95th pct): 1.62%
  Duration CV: 2.84%
  Recommended sample sizes for 80% power: [2.0%: K=16, 3.0%: K=7, 5.0%: K=3, 10.0%: K=3]
```

#### Recording Profiles Directly
StackGate can run your benchmark or CLI command multiple times and record folded profiles:

```bash
stackgate record --runs 5 --out profiles/candidate -- python my_script.py --benchmark
```

---

### 2. Python Programmatic API

```python
import stackgate

# Ingest profile sets from directory or files
baseline = stackgate.parse_directory("profiles/baseline/")
candidate = stackgate.parse_directory("profiles/candidate/")

# Perform noise-aware comparison
result = stackgate.compare(
    baseline=baseline,
    candidate=candidate,
    threshold=2.5,  # 2.5% regression threshold
    alpha=0.05,
    metric="self",
)

print(f"Verdict: {result.verdict}")
print(f"Total Change: {result.total_pct_change:+.2f}% (p={result.total_p_value:.4f})")
print(f"95% CI: [{result.ci_lower:+.1f}%, {result.ci_upper:+.1f}%]")

for fn in result.regressed_functions:
    print(f"  {fn.function}: +{fn.delta:.3f}s ({fn.attribution_pct:.1f}% share, q={fn.adjusted_p_value:.4f})")

# Render GitHub PR comment
pr_markdown = result.to_markdown()

# Programmatic CI assertion
stackgate.assert_no_regression(baseline, candidate, threshold=2.5)
```

---

### 3. Pytest Integration

```python
import pytest
from stackgate import assert_no_regression

def test_pipeline_throughput(stackgate_benchmark):
    # Runs the callable 5 times under the sampling profiler
    candidate_profile = stackgate_benchmark(run_heavy_computation, runs=5)
    
    # Assert against golden baseline
    assert_no_regression(
        baseline="baselines/heavy_computation/",
        candidate=candidate_profile,
        threshold=3.0, # Fail if >3% slower with p < 0.05
    )
```

---

## GitHub Actions CI Workflow

Add this to `.github/workflows/perf-gate.yml`:

```yaml
name: Performance Gate

on:
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          pip install stackgate .

      - name: Profile Candidate Branch
        run: |
          stackgate record --runs 8 --out profiles/candidate -- python -m myapp.benchmark

      - name: Checkout Baseline
        run: |
          git checkout origin/main

      - name: Profile Baseline
        run: |
          stackgate record --runs 8 --out profiles/baseline -- python -m myapp.benchmark

      - name: Run StackGate Comparison
        id: gate
        run: |
          stackgate compare profiles/baseline profiles/candidate \
            --threshold 3.0 \
            --format markdown \
            --output comment.md

      - name: Comment PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            if (fs.existsSync('comment.md')) {
              const body = fs.readFileSync('comment.md', 'utf8');
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: body
              });
            }
```

---

## Statistical Methodology

| Component | Technique | Rationale |
|---|---|---|
| **Total Regression Test** | Non-parametric Permutation Test (exact for $N \le 16$, Monte Carlo $B=10,000$ for $N > 16$) | Profile durations violate normality under JIT/GC pauses. Permutation tests make zero distributional assumptions. |
| **Confidence Interval** | Bootstrap Resampling ($B=5,000$) | Delivers realistic error bounds on relative percentage change $\% \Delta$. |
| **Function Attribution** | Benjamini-Hochberg (BH) FDR Procedure | Profiling inspects hundreds of functions. Standard $p < 0.05$ produces multiple false positives. BH controls false discoveries at $q < 0.05$. |
| **Additive Decomposition** | Leaf-frame self-time delta partitioning | Attributions cleanly sum: $\sum_f \Delta_f \approx \Delta_{\text{total}}$. |
| **Noise Floor Calibration** | Empirical A/A permutation split | Prevents alerting when observed differences are within CI runner background jitter. |

---

## Supported Formats

- **Brendan Gregg Collapsed/Folded Stacks**:
  ```text
  main;handle_request;parse_json 142
  main;handle_request;validate 45
  main;worker;fetch_sql 320
  ```
- **Python 3.15 Tachyon**: `profiling.sampling` record/replay output.
- **`py-spy`**: Folded output via `py-spy record -f folded`.
- **Austin**: Folded format (`austin -P ...`).
- **StackGate JSON Profiles**: Serialized JSON representation.

---

## Comparison with Existing Tools

| Tool | Granularity | Noise Model | Verdict / CI Gate | Function Attribution | Zero Deps? |
|---|---|---|---|---|---|
| **Tachyon `--diff-flamegraph`** | Function/Stack | None (1 run) | ❌ No | Visual only | Standard Library |
| **Brendan Gregg `difffolded.pl`** | Stack | None (1 run) | ❌ No | Visual only | Perl script |
| **pytest-benchmark / pyperf** | Benchmark | Yes | Varies | ❌ Benchmark only | Needs external packages |
| **Continuous Profilers (Pyroscope)** | Flamegraph | None | ❌ Dashboard only | Visual UI | Hosted / Go backend |
| **StackGate** | **Function + Path** | **Yes (Multi-Run)** | **✅ Automated Gate** | **✅ Additive FDR** | **✅ Pure Python** |

---

## Author & Contributing

- **Author**: Samarth Agrawal ([agrawalsamarth0206@gmail.com](mailto:agrawalsamarth0206@gmail.com))
- **Contributions**: Pull requests, bug reports, and suggestions are welcome! Please open an issue on GitHub.

## License

Distributed under the [MIT License](LICENSE). Copyright &copy; 2026 Samarth Agrawal.
