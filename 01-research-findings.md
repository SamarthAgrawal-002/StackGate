# 01 - Research Findings

## Method

I searched PyPI pages, GitHub, python.org docs/PEPs, and technical articles (Oct 2026) for gaps in the Python ecosystem. For each candidate idea I searched for existing packages before keeping or discarding it. Main lesson: **the obvious "AI-era" tooling ideas are already crowded**, mostly by small, recently published packages.

## Candidate ideas examined and rejected

| Candidate | What I found already exists | Verdict |
|---|---|---|
| Record/replay testing for AI agents (VCR for LLM calls) | `agent-replay-py`, `AgentTape`, `agentverify`, `pytest-agentreplay`, `replay-agent` (TraceOps), `agentloop`, `agentcassette`, `agentcheck-pytest-plugin` | Saturated |
| Race/thread-safety tooling for free-threaded Python | `threadcheck` (static + runtime race detector), `interleave-test` (deterministic interleaving fuzzer), `ftcheck` (TSan-based, Rust-focused) | Saturated; also covered by the py-free-threading guide's TSan workflow |
| Tooling for PEP 810 lazy imports (Python 3.15) | `Lifeguard for Lazy Imports` (static side-effect analysis), `importcost` (measure, audit, apply, CI budget), `flake8-lazy` | Saturated |
| Slopsquatting / hallucinated-dependency detection | `slopsquat-detector`, `depmirage`, `vibescan`, `slopguard-cli`, `Slopstop`, `slopspotter`, `slopgate` | Saturated |
| Lockfile review / dependency-PR diffing (uv.lock, pylock.toml) | `trustcheck` (trust diff across lockfile formats), `uv-lock-report`; `pip-tools` is adding PEP 751 output | Mostly covered |
| Generic "top libraries" lists | Polars, DuckDB, uv, Ruff, Textual, Litestar, etc. | Not gaps, just popular tools |

## Where a gap remained

**Statistical regression detection on sampling profiles (Python 3.15 `profiling.sampling`).**

What exists:

- Tachyon ships `--diff-flamegraph baseline.bin`. Per the docs it compares a run to a baseline and colours frames red/blue by change in *direct* time.
- Brendan Gregg's `difffolded.pl` + `flamegraph.pl` do the same for collapsed stacks (2014 technique).
- A research prototype (FlameGraphDiff) generates differential flame graphs from two profiles.
- General statistical-gate packages exist for *LLM evals* (e.g. `sigeval`, using two-proportion tests), which shows the approach is valued, but it does not apply to profiles.

What the Tachyon docs themselves say that creates the opening:

- Time values are *estimates from sample counts*, and results "will vary slightly between runs".
- For differences of only 1-2%, "sampling noise can obscure real differences".
- Differential flame graphs are described as colour-coded change, with no significance, confidence interval, or pass/fail.
- A third-party summary says differential flame graphs are "useful for regression hunting in CI", but there is no built-in gate.

What I did **not** find: a Python tool that takes multiple runs per side, estimates run-to-run noise, runs a proper significance test, controls false discoveries across functions, and returns a verdict with function-level attribution.

## Background-knowledge neighbours (not re-verified in this research)

These are from my own knowledge and should be checked before you write the "related work" section:

- `pyperf` / `pyperformance`: rigorous benchmark timing, but at benchmark level, not function-level attribution from profiles.
- `pytest-benchmark`, `asv` (airspeed velocity): benchmark timing and history tracking.
- CodSpeed: a hosted commercial service for CI performance tracking.
- Continuous-profiling backends (Pyroscope, Parca) offer profile comparison in a UI.

`stackgate` would complement these rather than replace them: they answer "did the benchmark number move?", `stackgate` answers "which function moved, and are we sure?".

## Honest limits of this research

- Web search cannot prove a package does not exist. PyPI and GitHub were reached only through search snippets, so name availability (`stackgate` is a placeholder) is **unchecked**.
- The Tachyon binary format is not documented as a stable spec in the docs I read. Collapsed-stack output *is* documented, so the design ingests collapsed stacks and shells out to `replay` for binaries.
- I did not run any code or benchmarks. Statistical claims in `03-technical-design.md` are standard methods applied to this problem; they need empirical validation (see `05-roadmap-and-validation.md`).

## Sources consulted

- Python 3.15 docs, `profiling.sampling`: https://docs.python.org/3.15/library/profiling.sampling.html
- Real Python, sampling profiler overview: https://realpython.com/python315-sampling-profiler/
- Talk Python 565 (Tachyon): https://talkpython.fm/episodes/show/565/tachyon-python-3.15s-built-in-sampling-profiler
- Python 3.15 overview (Anaconda): https://www.anaconda.com/blog/python-3-15
- Brendan Gregg, differential flame graphs: https://www.brendangregg.com/blog/2014-11-09/differential-flame-graphs.html
- Differential flame graphs research paper: https://www.researchgate.net/publication/282681970_Understanding_software_performance_regressions_using_differential_flame_graphs
- Free-threading guide: https://py-free-threading.github.io/
- Rejected-idea packages (PyPI): `threadcheck`, `interleave-test`, `ftcheck`, `lifeguard-lazy-imports`, `importcost`, `slopgate`, `slopstop`, `slopspotter`, `depmirage`, `trustcheck`, `uv-lock-report`, `agent-replay-py`, `agenttape`, `agentverify`, `pytest-agentreplay`, `agentcassette`
- PEP 810: https://peps.python.org/pep-0810/
