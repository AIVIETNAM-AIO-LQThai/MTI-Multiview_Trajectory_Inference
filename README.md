# MTI — Multi-View Trajectory Inference

Comparing different conditional views and inference routes over trajectory evidence.

## Research question

When a model learned from clean trajectories meets possibly contaminated trajectory evidence, how do different query structures
(full record, sign-repaired candidates, sign-folded magnitude-retaining observers) estimate a decision-relevant belief under finite
data and compute — and what does composing inference at test time cost relative to training directly for a declared corruption
channel? Scope is provisional and is narrowed by evidence. See [docs/research_charter.md](docs/research_charter.md).

Three questions are kept separate: **information** (how much decision value awareness can recover under the declared channel),
**estimation** (do learned routes differ at finite data/compute), **policy relevance** (does belief error matter downstream).

## Current evidence status (2026-10-04)

The *information* question (exact oracle) and a first *estimation* pilot have been measured, on synthetic data from the declared switching-mode scalar model ("S2") with a
single-flip action-sign channel and a one-step quadratic decision. Policy relevance beyond the one-step cost has not been tested.

- Exact reference implemented and validated (enumeration cross-checks, mutation audit; the full current suite has 85 tests, all passing); candidate and folded exact routes agree to ≤ 1.5e-14.
- Primary cell (L=8, q_w=0.5, β=0.2, η=0.05): V_2 (value of awareness) = 0.0712 ± 0.0031, irrecoverable loss I_loss = 0.2614 ± 0.0069,
  naive excess 0.331, clean-history value 4.01 — awareness is worth ~1.8% of the clean history's decision value here; 13% at L=8, q_w=0.02, β=0.5; < 1.6% at L=32.
- Learned pilot (5 seeds, N=10³/10⁵, two physics, five priors; [report](docs/reports/learned_pilot_report.md)): the exact-composition arms with learned components (shared masked-belief model queried by folded or candidate routes, density scorer, fitted HMM)
  recover 93–100% (lead physics q_w=0.02) and 77–100% (q_w=0.5) of the aware oracle's opportunity at N=10⁵ (A2 at P1; higher at shifted priors); no general advantage of folded over candidate routes was established (14 equivalent / 4 inconclusive / 2 material). Caveats: grid-edge selection at N=10⁵, A4/A5 families contain the truth.
- E1 (matched backbone and loss family): composition vs density-direct estimation gave 6 material improvements, 4 equivalences, 4 inconclusive and 0 material disadvantages; this is not dominance. Precision is limited and the benefit is regime-dependent; a fitted HMM is near-oracle ([E1 report §7](docs/reports/e1_report.md)).
- E2 (channel misspecification, branch `exp/channel-misspecification`): [docs/reports/e2_report.md](docs/reports/e2_report.md) — composing with a wrong declared prior can be worse than ignoring corruption, and the harm tracks over-declared recent-lag corruption. Under prior uncertainty the mean prior is Bayes-optimal, but there is no free robust prior ([E2b report](docs/reports/e2b_report.md)).
- E3 (channel identification, branch `exp/channel-identification`): the channel is identified from unlabeled prefixes for L ≥ 3 (not at L=1/2); likelihood adaptation removes declared-prior risk given data, but clean-law error is absorbed into inferred corruption ([E3 report](docs/reports/e3_report.md)).
- **Consolidated claims and limits: [docs/findings_summary.md](docs/findings_summary.md).**
- Not established (oracle results only): any learned-model advantage of one route over the other, that localisation is necessary, any poisoning defence or policy robustness.

Report: [docs/reports/oracle_v1_report.md](docs/reports/oracle_v1_report.md) (tables: [oracle_v1_tables.md](docs/reports/oracle_v1_tables.md)).
Math: [docs/mathematical_specification.md](docs/mathematical_specification.md). Learned pilot: [docs/reports/learned_pilot_report.md](docs/reports/learned_pilot_report.md), protocol [docs/learned_pilot_protocol.md](docs/learned_pilot_protocol.md).
Current phase / handoff: [docs/research_status.md](docs/research_status.md).

## Install and run

Python ≥ 3.11 (results here were produced with 3.14.7), numpy, scipy; pytest and matplotlib for development.

```bash
uv venv .venv
uv pip install -e ".[dev]"
.venv/Scripts/python -m pytest -v                          # 85 tests, ~15 s (Linux/macOS: .venv/bin/python)
.venv/Scripts/python tests/mutation_audit.py               # shows the tests fail on 6 deliberate bugs
.venv/Scripts/python experiments/run_oracle.py --out results/oracle_v1   # all 10 cells, ~2.5 min on 18 workers
.venv/Scripts/python experiments/make_report.py            # tables + figure in docs/reports/
```

Single cell / smoke test: `--cells primary_L8_qw0.5_b0.2 --scale 0.1 --out results/scratch`.

## Layout

| path | content |
|---|---|
| `src/mti/simulate.py` | S2 simulator with named random streams; `Prefix` is the only permitted inference input |
| `src/mti/inference.py` | log-domain HMM, candidate route, folded route (computed by marginalisation, not by the identity) |
| `src/mti/decision.py` | one-step cost, κ, a*, conditional cost |
| `src/mti/experiment.py`, `metrics.py`, `recal.py` | chunk workers (RB estimator, independent check, profile), CIs, recalibration diagnostic |
| `src/mti/enumerate.py` | brute-force mode-path / hypothesis enumeration used only by tests |
| `configs/oracle_v1.toml` | all cells, seeds, sample-size rules, chunk sizes |
| `results/oracle_v1/` | per-cell JSON, `summary.csv`, test transcripts and maximum errors |
