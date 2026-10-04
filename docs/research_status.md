# Research status (replace in place)

**Phase:** E2 + E2b (channel misspecification) complete and reviewed (Opus 5.5, 2026-10-04). **Experimental work stopped** pending a user research decision.
**Updated:** 2026-10-04 by Opus 5.5 (`claude-opus-5-5`). Branch `exp/channel-misspecification`.

## Provenance
- Committed: `a7a0591` (phase-1 freeze), `4b9380f` (E2).
- Uncommitted: E2b code/tests/results/report and this review. The E2b run recorded base `4b9380f` plus exactly these dirty files, with no code edits after the run.
- The user makes all git commits; never push.

## Where things stand
Claims and limits are in [findings_summary.md](findings_summary.md) v4 (C1–C19, H-loc).
- Composition with a declared channel is flexible and Bayes-optimal under prior uncertainty.
- Its value is bounded by the declared prior. Over-declared recent-lag corruption harms, and there is no free robust prior.
- Learned-route claims are scoped; a fitted HMM is near-oracle.

## Options (each needs a user decision; none started)
1. **Empirical-Bayes estimation of q from many corrupted records (recommended).** Same exogenous channel; tests whether repeated deployment removes the misspecification trade-off.
2. **Mode-dependent placement.** First derive p(m,θ|S) ∝ p_0(T_θS, m)·Pr(θ | T_θS, m) and validate it by enumeration.
3. **The original MTM question.** Does multi-view consistency flag harmful trajectories beyond residual/rarity/dynamics signals?

## On resume
Read this file and findings_summary.md. Do not rerun completed work.
