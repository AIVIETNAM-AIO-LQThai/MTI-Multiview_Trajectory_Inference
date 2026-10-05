# Research status (replace in place)

**Phase:** E4 approved by the user (2026-10-05). Stage 0 run and **a stop rule fired literally: STOPPED, awaiting Opus.** Stage 1 not started.
**Updated:** by Sonnet 5.5 (`claude-sonnet-5-5`). Branch `exp/joint-identification` (from `329d7c3`).

## Stage-0 stop (protocol section 3)
- Rule: "At L = 3 on T3, if KL along the ridge exceeds 1e-6 ... the derivation or the code is wrong. Stop."
- Observed (`results/e4/stage0/summary.md`): KL = -4.3e-6 (lead) and -6.6e-6 (secondary) at every eta' from 0.01 to 0.08, identical to the digits shown; I_prof/I_eta = 1.4e-14. At eta' = 0.12, KL = +3.9e-3 (the ridge ends where the channel leaves the simplex).
- Reading, to be confirmed by Opus: the constant negative offset is the in-sample likelihood gain of the fitted (eta', q*) over the true parameters (about d/(2n) = 3.75e-6 for 3 free weights at n = 2e5), not ridge curvature. The rule as written is violated in sign-blind form (KL > 1e-6 in absolute value); the ridge itself is exact. No threshold was changed by Sonnet.
- Other stop rules: none fired (L=8 KL(0.03) 1.1e-2 to 2.4e-2; I_prof/I_eta 0.61-0.81 at L=8, 0.21-0.51 at L=4).
- Stage-0 prediction recorded before stage 1: predicted SE(eta_hat) at n = 1e4 is 0.0011-0.0016 (L=8), 0.0025-0.0064 (L=4), none above 0.01 -> H-E4 is **not** predicted to fail at 1e4.

## Provenance
Uncommitted:
- `docs/literature_review.md` and `docs/e4_protocol.md`;
- updated findings_summary (v6), charter, README and this file;
- the derivation check `experiments/check_joint_identification.py`, with output in `results/derivation_checks/joint_identification.json`. It runs no estimators and no training (about 1 min).

The user makes all git commits. Pushing is done only when the user asks.

## Where things stand
- Claims: [findings_summary.md](findings_summary.md) v6 (C1–C26).
- Positioning: [literature_review.md](literature_review.md). Most aggregate results are instances of known theory measured at decision level.
- The candidate contribution is C25 (joint identification threshold L = 4) together with E4's sample costs.
- The per-record MTM question is deprioritised: the route gap is a finite-learner diagnostic (C26).

## Next (needs user approval)
**E4: joint identification of clean persistence and channel.** Protocol: [e4_protocol.md](e4_protocol.md).
- Stage 0 is an exact-law diagnostic.
- Stage 1 compares deployable estimators against known-η, absorption and clean-validation baselines.
- No neural training.
- Hand to Sonnet 5.5 on a new branch `exp/joint-identification` after approval.

## On resume
Read this file, findings_summary.md and literature_review.md. Do not rerun completed work.
