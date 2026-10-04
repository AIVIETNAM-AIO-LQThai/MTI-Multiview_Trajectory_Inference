# Research status (replace in place)

**Phase:** PLAN complete → **next: CODE / EXECUTE on Sonnet 5.5** (user selects model, sends `CONTINUE CODE`).
**Updated:** 2026-10-04 by Opus 5.5 (`claude-opus-5-5`, observed via host session metadata).

## Provenance
Repo `AIVIETNAM-AIO-LQThai/MTI-Multiview_Trajectory_Inference`, branch `exp/oracle-evidence` from `c250658`
(stub README only). Docs written in this pass; no code yet. Host: Windows 11, Python 3.12.10, numpy 2.5.1,
scipy 1.18.1, sklearn 1.9.1, torch 2.10 CPU, 20 cores, uv 0.12.5; **pytest not installed**.

## Completed vs unrun
Done: [charter](research_charter.md), [math spec](mathematical_specification.md) (audit notes A1–A8; no
target changed), [learned protocol draft](learned_pilot_protocol.md). Unrun: all code, tests, experiments.

## Approved decisions (see spec)
Independent folded route via neutral-emission filter/smoother (§5); RB-over-θ estimator with prefix as unit
(§11); pilot→fixed-N main→independent check; hidden-mode strata use κ(û−m_L)² (§8); recalibration via
binned g with one-sided bounds (§9); tolerances §13 (do not loosen).

## Next task (Sonnet): implement, validate, run
1. Env: `uv venv .venv && uv pip install -e .[dev]` (numpy, scipy, pytest; no torch needed). Configs in TOML
   (stdlib `tomllib`).
2. Layout [impl free within spirit]: `pyproject.toml`; `src/mti/{params,simulate,channel,hmm,inference,decision,metrics,enumerate}.py`
   (`enumerate.py` = brute-force references for tests only); `experiments/run_oracle.py`;
   `configs/oracle_v1.toml` (all cells, seeds, chunk size); `tests/test_*.py` covering T1–T16.
   Inference API takes only `Prefix(s, a)`; probe/hidden/decision fields live in separate containers.
3. Run `pytest -q` → save output to `results/oracle_v1/tests.txt`; all T1–T14, T16 must pass.
4. Run primary cell (pilot → main → check), controls, then the 8-cell map. Save
   `results/oracle_v1/{cell}.json` (estimates, sd, N, half-widths, z-checks, seeds, runtime, git commit +
   dirty flag, package versions, config hash) and `results/oracle_v1/summary.csv`. Do not commit bulk samples.
5. Write `docs/reports/oracle_v1_report.md` covering briefing §7 items 1–6 + recalibration + map; separate
   analytic / verified / measured / not-established. Update README (question, install/run, evidence status),
   claim ledger rows C1–C6, and this file (handoff to Opus `REVIEW EVIDENCE`).
6. Commit locally on `exp/oracle-evidence` in logical commits; **do not push**.

## Acceptance criteria
- All tests pass at spec tolerances; max errors recorded.
- Primary V_2 95% half-width ≤ 5% of V_2 (or absolute rule §11); T15 |z| ≤ 4.
- Every map cell reported (or explicitly marked unrun with reason).
- Report states commands, seeds, N, runtime, provenance.

## Escalate to Opus (CONTINUE ANALYZE) only if
An independent oracle check still disagrees after focused debugging (send minimal failing case + exact
discrepancies); a fix would change a [target] item; or results need nontrivial interpretation before
proceeding. Keep unaffected work moving.

## Open scientific questions (for evidence review)
Second learned-pilot cell; Δ_min in raw units; whether L=32 is worth learned-arm cost.
