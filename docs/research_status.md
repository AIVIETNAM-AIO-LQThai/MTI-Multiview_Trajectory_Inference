# Research status (replace in place)

**Phase:** literature positioning complete (Opus 5.5, 2026-10-05). **E4 proposed, awaiting user approval**; no E4 code or runs exist.
**Updated:** 2026-10-05 by Opus 5.5 (`claude-opus-5-5`). Branch `exp/channel-identification` at `3d20dac` (= origin).

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
