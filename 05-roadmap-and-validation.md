# 05 - Roadmap, Validation and OSS Hygiene

## 1. Validation Strategy

### 1.1 Empirical Accuracy Testing
- **Synthetic Profiles with Known Injected Regressions**:
  - Null hypothesis calibration (A/A test): 1,000 synthetic A/A comparisons with identical distributions. Verify measured false positive rate $\le \alpha$ (nominal 5%).
  - Power analysis (A/B test): Inject known regressions of +2%, +5%, +10%, and +20% into individual leaf functions with varying sample noise ($\sigma = 1\%, 3\%, 5\%$).
  - Measure true positive rate (statistical power) across sample sizes $K \in \{3, 5, 8, 12, 20\}$.
- **Multiple Comparison False Discovery Control**:
  - Test profile trees with 1,000+ distinct functions.
  - Verify that Benjamini-Hochberg FDR control maintains empirical false discoveries $\le q$ even under massive candidate function spaces.

### 1.2 Robustness & Stress Testing
- **Malformed & Edge-Case Input Handling**:
  - Empty files, zero sample counts, non-numeric counts.
  - Corrupt UTF-8, Windows CRLF vs Unix LF line endings.
  - Recursion, cycle loops, deeply nested call stacks (> 1000 frames).
  - Single run comparisons (gracefully handling $K=1$ by switching to informative noise warnings).
  - Zero-time functions and completely unchanged profiles.
- **Scale Stress**:
  - Ingesting profiles with 50,000+ call paths and millions of samples in under 1 second.
  - Deterministic permutations with fixed seeds for reproducibility.

## 2. OSS Hygiene Checklist
- Modern `pyproject.toml` with PEP 621 metadata.
- Author: Samarth Agrawal (`agrawalsamarth0206@gmail.com`).
- MIT License.
- Complete type annotations with `py.typed` marker (PEP 561).
- Zero mandatory external dependencies (pure Python standard library core).
- Full automated test suite with `pytest`.
- Comprehensive GitHub Actions CI workflow (lint, typecheck, multi-OS tests).
- Professional `README.md` with badges, diagrams, quickstart, CLI reference, and architecture guide.
