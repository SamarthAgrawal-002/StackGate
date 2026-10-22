# 02 - Problem and Value

## The problem

Performance regressions in Python projects are usually discovered late (by users, or by someone eyeballing a flame graph). The tools in common use leave a gap:

1. **Benchmark-level gates** say "the benchmark got 6% slower" but not *where*.
2. **Profilers** say where time goes, but a single profile has no error bars, and two profiles differ even when the code did not change.
3. **Differential flame graphs** (Tachyon built-in, Gregg-style) show where two profiles differ, but cannot say whether the difference is real, and cannot fail a CI job.

Result: developers either ignore profile diffs (too noisy) or hand-interpret them (slow, inconsistent), and CI performance gates are either absent or flaky.

## The solution in one sentence

Treat each profile as a **sample from a noisy process**, collect several per side, and apply proper statistics to produce a verdict and an attribution.

## What a user sees

```text
$ stackgate compare base/ head/

Verdict: FAIL  (regression, high confidence)
Total wall time:   4.12 s -> 4.47 s   (+8.5%, 95% CI +5.9% .. +11.2%, p=0.0003, K=8 vs 8)

Where it went (self time, FDR q<0.05):
  parse_message      0.61 s -> 0.89 s   +0.28 s   (80% of regression)  p=0.0004
  json/decoder.py:raw_decode   ...
New call path (not in baseline):
  main;handle_request;validate_schema;re.compile   +0.05 s

Not significant / below noise floor: 212 functions
```

## Target users

| User | Why they care |
|---|---|
| Library maintainers | Want to block PRs that slow hot paths without flaky benchmark jobs |
| Application teams | Want a pre-merge perf check on API handlers, CLIs, data pipelines |
| CPython and ecosystem contributors | Evaluating interpreter/JIT/specialization changes needs noise-aware comparisons |
| Performance engineers | Want an evidence-based "is this optimization real?" answer |

## Why now

- Python 3.15 (released 1 Oct 2026) puts a production-safe sampling profiler in the stdlib, with a record/replay binary workflow and a differential flame graph. Adoption will create demand for the next layer: *automated judgement* on top of profiles.
- Free-threaded builds and the JIT make performance more variable and more important; noise-aware tooling matters more than before.
- CI budgets are tight; fewer, better-targeted runs are valuable.

## Differentiation

| Tool | Granularity | Noise model | Verdict / CI gate | Function-level attribution |
|---|---|---|---|---|
| Tachyon `--diff-flamegraph` | Function/stack | None | No | Visual only |
| Gregg `difffolded.pl` | Stack | None | No | Visual only |
| Benchmark runners (pyperf, pytest-benchmark, asv)* | Benchmark | Yes | Varies | No |
| **`stackgate`** | Function + call path | Yes (multi-run) | Yes | Yes, additive decomposition |

\*Background knowledge, not re-verified in this research.

## What makes it a genuine open-source contribution

- **Fills a documented gap** (the stdlib docs themselves flag the noise problem).
- **Complements, does not fork**: builds on stdlib Tachyon, py-spy, and the open collapsed-stack format.
- **Reusable core**: a small, dependency-free statistics-and-parsing library that other tools (IDE plugins, dashboards, pytest plugins) can import.
- **Transparent methods**: documented tests, published false-positive and power measurements, reproducible synthetic benchmarks.
- **Possible upstream value**: findings about noise on real CI runners and a calibration methodology are useful to CPython performance discussions even if the package itself stays third-party.

## Non-goals (to keep scope honest)

- Not a profiler (it consumes profiles from Tachyon, py-spy, etc.).
- Not a benchmark framework (it can orchestrate runs, but does not define benchmarks).
- Not a hosted service or dashboard.
- Not instruction-count or hardware-counter measurement.
