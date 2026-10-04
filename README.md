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

Only the *information* question has been measured, on synthetic data from the declared switching-mode scalar model ("S2") with a
single-flip action-sign channel and a one-step quadratic decision. No learned model has been trained.

- Exact reference implemented and validated (68 tests, enumeration cross-checks, mutation audit); candidate and folded exact routes agree to ≤ 1.5e-14.
- Primary cell (L=8, q_w=0.5, β=0.2, η=0.05): V_2 (value of awareness) = 0.0712 ± 0.0031, irrecoverable loss I_loss = 0.2614 ± 0.0069,
  naive excess 0.331, clean-history value 4.01 — awareness is worth ~1.8% of the clean history's decision value here; 13% at L=8, q_w=0.02, β=0.5; < 1.6% at L=32.
- Not established: any learned-model advantage, that localisation is necessary, any poisoning defence or policy robustness.

Report: [docs/reports/oracle_v1_report.md](docs/reports/oracle_v1_report.md) (tables: [oracle_v1_tables.md](docs/reports/oracle_v1_tables.md)).
Math: [docs/mathematical_specification.md](docs/mathematical_specification.md). Next experiment (not run): [docs/learned_pilot_protocol.md](docs/learned_pilot_protocol.md).
Current phase / handoff: [docs/research_status.md](docs/research_status.md).

## Install and run

Python ≥ 3.11 (results here were produced with 3.14.7), numpy, scipy; pytest and matplotlib for development.

```bash
uv venv .venv
uv pip install -e ".[dev]"
.venv/Scripts/python -m pytest -v                          # 68 tests, ~6 s (Linux/macOS: .venv/bin/python)
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
