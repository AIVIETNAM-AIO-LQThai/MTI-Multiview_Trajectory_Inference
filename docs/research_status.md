# Research status (replace in place)

**Phase:** learned pilot v1 complete → awaiting user commit, then **Opus 5.5 `REVIEW EVIDENCE`** on [learned_pilot_report.md](reports/learned_pilot_report.md).
**Updated:** 2026-10-04 by Sonnet 5.5 (`claude-sonnet-5-5`).

## Provenance
Branch `exp/oracle-evidence`. Pilot code commit `09621bb` (clean); results in `results/learned_pilot/` (tuning, 20 units, analysis.json), `results/learned_gate/`. Report, tables, README/charter edits are uncommitted until the user commits. The user makes all git commits; never push.

## Done
- Pilot per protocol v2: 192 tuning trainings (1,177 s) + 20 units (2,839 s total). Tests: 81 pass.
- Headline (lead physics, N=10⁵, E = E[κ(μ̂−μ_2)²]): A5 ≈ 0, A4 0.0008–0.0011, A2-G/M 0.006–0.011, A6p-broad 0.010–0.098 (0.53 at P4), A6 and A6p-narrow degrade at shifts, A3 ensemble and learned naive A1 ≈ no better than exact naive.
  G vs M equivalent in 14/20 cells; A4/A5 beat A2 at N=10³ in all cells.
- Diagnostics: query error grows 4–6× with contamination count; route-gap identity error ≤ 2.2e-15; natural-switch / clean-record costs reproduce the oracle's for composition arms.

## Flags for Opus (details: report §2 and §4)
1. Grid-edge selection at N=10⁵ (all 12 selections at the 12-epoch cap) → maybe not converged; optional extended-epoch rerun of N=10⁵ units (≈1.3 CPU-h).
2. 5 seeds do not meet the Δ_min/2 power rule in many cells (inconclusive verdicts); extend to 10 seeds (≈ +2.7 CPU-h) if needed.
3. A4/A5 families contain the truth; A1 is a weak baseline (trained on ¾N, worse than exact naive).
4. Interpretation of H2/H3/H5 and whether mask-diversity / separate-encoder / distillation experiments are now motivated.

## Next
Opus review → decide: (a) extended-epoch rerun, (b) 10 seeds, (c) any deferred experiment. Do not start any without that decision.
