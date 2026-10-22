# 04 - API and CLI Specification

## 1. CLI Commands

### 1.1 `stackgate compare`
Compares a set of baseline profiles against candidate profiles.

```bash
stackgate compare <baseline_dir_or_files...> <candidate_dir_or_files...> [OPTIONS]
```

**Options**:
- `--threshold FLOAT`: Relative regression threshold percentage (e.g. `2.0` or `5.0`). Default: `2.0`.
- `--alpha FLOAT`: Significance level for hypothesis testing. Default: `0.05`.
- `--fdr-q FLOAT`: False discovery rate q-value threshold. Default: `0.05`.
- `--method [permutation|welch|mann-whitney]`: Statistical test method. Default: `permutation`.
- `--metric [self|inclusive]`: Metric used for function ranking. Default: `self`.
- `--min-effect-pct FLOAT`: Minimum effect percentage to consider significant. Default: `1.0`.
- `--min-effect-ms FLOAT`: Minimum absolute time delta in ms. Default: `5.0`.
- `--noise-floor FLOAT`: Custom noise floor override percentage.
- `--ci FLOAT`: Confidence interval level. Default: `0.95`.
- `--permutations INT`: Number of Monte Carlo permutations. Default: `10000`.
- `--format [terminal|markdown|json|html]`: Output report format. Default: `terminal`.
- `--output PATH`: File path to write the output report.
- `--strict`: If set, treats `INCONCLUSIVE` as failure (exit code 2).
- `--fail-on-regression`: Exit with non-zero code on regression (default: True).

**Exit Codes**:
- `0`: PASS or IMPROVEMENT (no regression detected above threshold).
- `1`: FAIL (regression detected with statistical significance).
- `2`: INCONCLUSIVE (under `--strict` mode, or insufficient sample power).
- `3`: Error (invalid arguments, unparseable input files, corrupted profiles).

### 1.2 `stackgate calibrate`
Runs an empirical A/A calibration on a directory of baseline runs to calculate runner noise floor and recommended sample count $K$.

```bash
stackgate calibrate <runs_dir> [OPTIONS]
```

**Options**:
- `--alpha FLOAT`: Target false positive rate (default: `0.05`).
- `--splits INT`: Number of random A/A partitions (default: `100`).
- `--target-mde FLOAT`: Target minimum detectable effect percentage (default: `5.0`).
- `--format [terminal|json]`: Output format. Default: `terminal`.

### 1.3 `stackgate record`
Profiles an arbitrary command $N$ times and generates folded stack profiles.

```bash
stackgate record --runs INT --out DIR [OPTIONS] -- COMMAND...
```

**Options**:
- `--runs INT`: Number of runs to execute (required, recommended $\ge 5$).
- `--out DIR`: Directory to store generated profiles (required).
- `--rate INT`: Sampling frequency in Hz (default: `100`).
- `--warmup INT`: Number of warmup runs to discard (default: `1`).
- `--engine [auto|builtin|pyspy|tachyon]`: Sampling engine to use (default: `auto`).

### 1.4 `stackgate report`
Re-renders a previously saved JSON comparison result into Markdown or HTML.

```bash
stackgate report <input_json> --format [markdown|html|terminal] [--output PATH]
```

## 2. Python Programmatic API

```python
import stackgate

# 1. Parsing profiles
baseline_runs = stackgate.parse_directory("profiles/base/")
candidate_runs = stackgate.parse_directory("profiles/head/")

# 2. Running comparison
result = stackgate.compare(
    baseline=baseline_runs,
    candidate=candidate_runs,
    threshold=0.03,  # 3% threshold
    alpha=0.05,
    metric="self"
)

# 3. Inspecting result
print(result.verdict)  # Verdict.FAIL, Verdict.PASS, etc.
print(f"Total change: {result.total_pct_change:+.2f}% (p={result.total_p_value:.4f})")
print(f"95% CI: [{result.ci_lower:+.2f}%, {result.ci_upper:+.2f}%]")

for fn in result.regressed_functions:
    print(f"  {fn.function}: {fn.baseline_mean:.2f}s -> {fn.candidate_mean:.2f}s ({fn.attribution_pct:.1f}% of regression, q={fn.adjusted_p_value:.4f})")

# 4. Generate reports
markdown_comment = result.to_markdown()
json_payload = result.to_json()

# 5. Programmatic assertion for CI / tests
stackgate.assert_no_regression(baseline_runs, candidate_runs, threshold=0.03)
```

## 3. Pytest Integration

```python
import pytest
from stackgate.pytest_plugin import assert_no_regression

def test_json_pipeline_performance(stackgate_benchmark):
    # Benchmark runs 5 iterations under the profiler
    profile = stackgate_benchmark(run_heavy_json_pipeline, runs=5)
    # Compare against cached baseline
    assert_no_regression("baselines/json_heavy.folded", profile, threshold=0.05)
```
