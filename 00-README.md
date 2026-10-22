# Package Idea: `stackgate` (working name)

> **Statistically rigorous, function-level performance-regression detection from sampling profiles, for Python.**

Research date: 22 Oct 2026 (Python 3.15 was released 1 Oct 2026).

## One-paragraph pitch

Python 3.15 ships a sampling profiler in the standard library (`profiling.sampling`, "Tachyon") that can record a profile with near-zero overhead and draw a *differential flame graph* against a baseline. But the comparison is **visual and pairwise**: one baseline run, one candidate run, red/blue colours, no notion of noise, no verdict, no CI gate. Tachyon's own docs warn that results vary run to run and that 1-2% differences can be swamped by sampling noise. `stackgate` closes that gap: it takes **several runs per side**, models run-to-run noise, tells you **whether the program got slower (with a confidence interval)**, and **attributes the change to specific functions and call paths** with multiple-comparison control. It outputs a PASS / FAIL / INCONCLUSIVE verdict, a Markdown PR comment, and JSON for CI.

## Files in this folder

| File | What it contains |
|---|---|
| `00-README.md` | This overview |
| `01-research-findings.md` | What was searched, which ideas were rejected and why, evidence for the gap, honest limits of the research |
| `02-problem-and-value.md` | Problem statement, target users, why now, differentiation, what "genuine contribution" means here |
| `03-technical-design.md` | Data model, statistics, algorithms, architecture, module layout |
| `04-api-and-cli-spec.md` | Public functions, CLI, config file, report formats, exit codes, pytest plugin |
| `05-roadmap-and-validation.md` | Pre-build verification checklist, milestones, test strategy, risks, OSS hygiene |

## Why this idea (short version)

1. **Timely**: the stdlib profiler is new (3.15), so tooling around it barely exists yet.
2. **Real pain**: performance regressions are found late; existing gates compare a single number per benchmark and cannot say *which function* caused it.
3. **Hard enough to be worth it**: the value is in the statistics (small-K permutation tests, multiple-comparison control, A/A calibration), not in a thin wrapper. That also makes it harder to clone in an afternoon.
4. **Scoped**: a pure-Python, stdlib-only core is realistic for one person to build and maintain.

## Read this caveat first

I could not find an existing tool that does exactly this, but a web search cannot *prove* absence. Section 1 of `05-roadmap-and-validation.md` is a checklist to run before writing code. Several other plausible ideas I examined turned out to be heavily crowded already, so verify this one the same way.
