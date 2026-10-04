# Research status (replace in place)

**Phase:** learned pilot reviewed (Opus 5.5, 2026-10-04) → **awaiting user decision on E1**. If authorized: Sonnet 5.5 `CONTINUE CODE`.
**Updated:** 2026-10-04 by Opus 5.5 (`claude-opus-5-5`).

## Provenance
Branch `exp/oracle-evidence`. Pilot code `09621bb`, results/report `6ad73fa`. The review edits (report §3.3 sign fix, §5; protocol §E1; charter; this file) are uncommitted until the user commits them.
The user makes all git commits; never push.

## Review verdict ([learned report §5](reports/learned_pilot_report.md))
- **H2 (route form) not supported:** folded ≈ candidate given shared heads. Component learning and Bayesian weighting are the levers.
- **H3 partially supported:** prior flexibility is real; in-family accuracy is no better than a broad prior-conditioned direct estimator.
- **H5 not motivated.**
- **Qualifications:**
  - Δ_min saturates at N=10⁵.
  - Learners are physics-informed.
  - Supervision is confounded with route.
  - Contamination-error growth is descriptive only.
- **Thesis narrowed** to composition-as-flexible-estimator, accuracy governed by component learning.

## Next task (Sonnet, only after the user authorizes E1)
- Implement A6d and A6pd per protocol §E1.
  - Training: a density NLL on channel-simulated records plus the original probe; A6pd is q-conditioned over the broad family.
  - Reuse the pilot's training prefixes, test sets and seeds, so all comparisons are paired. Do not retrain existing arms; read their results from `results/learned_pilot/`.
- Tune on seed 0 (8-config grid), then run seeds 1–5 for both physics and both N.
- Extend `analyze_learned.py` with comparisons C-a…C-d and a relative-difference (E_x/E_y) column.
- Tests: an access test (no oracle, label or probe leakage into inputs), and check that the belief read-out equals tanh(logit_L/2).
- Report against the predeclared interpretation rules. Hand back for commits.

## Escalate to Opus if
A result falls outside the three predeclared outcomes, or density-direct training is unstable.
