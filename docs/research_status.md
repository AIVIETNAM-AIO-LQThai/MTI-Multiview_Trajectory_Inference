# Research status (replace in place)

**Phase:** E1 complete → awaiting user commit, then **Opus 5.5 `REVIEW EVIDENCE`** on [e1_report.md](reports/e1_report.md).
**Updated:** 2026-10-04 by Sonnet 5.5 (`claude-sonnet-5-5`).

## Provenance
Branch `exp/oracle-evidence`. E1 code commit `0ac6b34` (clean); results in `results/e1/` (tuning, 20 units, analysis.json). Report/tables are uncommitted until the user commits. The user makes all git commits; never push.

## Done
- E1 per protocol §E1: A6d (fixed P1) and A6pd-broad (prior-conditioned) density-direct arms; 64 tuning trainings + 20 units (466 s); 85 tests pass.
- Headline (lead physics): in-family N=10³ A4 (clean density + composition) beats the matched-supervision direct A6pd at P2/P5 (ratio 0.18–0.23); at N=10⁵ A4 and A6pd are equivalent in Δ_min terms (A4 2–20× lower in ratio);
  A6pd is brittle at the P3/P4 vertices (worse than probe-trained A6p-broad at N=10⁵). Supervision type matters: A6pd beats A6p-broad in-family at N=10³ (ratio 0.38–0.57).

## For Opus (details: report §4–5)
1. Which predeclared outcome applies: outcome 1 at N=10⁵, outcome 2 at N=10³ on the lead physics only; the secondary physics is inconclusive.
2. Residual confound: A4 is trained on clean prefixes (1 view), A6pd/A6d on corrupted records (2 views).
3. Whether the thesis can now claim an in-family estimation advantage for composition at small N, or only flexibility.
4. Whether any further experiment is justified (candidate: a clean-law direct comparison, or stop).

## Next
Opus review; no further experiment starts without the user's authorization.
