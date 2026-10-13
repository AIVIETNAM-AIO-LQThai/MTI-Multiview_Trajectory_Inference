# Research status (replace in place)

**Phase:** decision-identification closeout **complete** (Opus Part I, Sonnet Part II, Opus Part III: [decision_identification_closeout.md](decision_identification_closeout.md)). The identification chapter is closed. **Next: thesis write-up** from [research_synthesis.md](research_synthesis.md). No experiments scheduled.
**Updated:** 2026-10-12 by Opus 5.5 (`claude-opus-5-5`).

## Current repository state
- Branch `exp/joint-identification`, HEAD `08dd58e` (the consolidation, committed).
- Uncommitted (closeout):
  - new: `docs/decision_identification_closeout.md`, `docs/decision_identification_discussion.md` (imported, with a provenance header), `experiments/decision_id/`, `results/decision_id/`, `tests/test_decision_id.py`;
  - edited: `tests/conftest.py`, `experiments/check_e4_eval_se.py`, README, findings_summary (v9), research_synthesis (§4.4 qualifications, new §4.6, §5.4–§6 corrections), consolidation_review, literature_review, research_charter, e4_report, e4_protocol, this file.
- The user makes all git commits. Pushing is done only when the user asks.

## Historical run provenance (unchanged)
- E4 stage 0: `3f868c5`. E4 stage 1: run at `ea8bca4` + the A1 patch; committed in `7b95b9d`. Opus E4 review: `64e2234`. Consolidation: `08dd58e`.
- Closeout computations: run on the working tree at `08dd58e` (provenance in each `results/decision_id/*.json` shows a dirty tree; rerun after committing to refresh).
- Received verification archive (2026-10-11): `results/decision_id/archive/`, all nine scripts as received.

## Where things stand
- Claims: [findings_summary.md](findings_summary.md) v9 (C1–C31); final thesis claim table: closeout III.3; claim-to-evidence map: [consolidation_review.md](consolidation_review.md) §4.
- Out of scope for this thesis: a population L = 3 minimax theorem, an unknown-physics plug-in experiment, finite-sample identified-set estimation, family-violation dose–response, learned detectors, real data.

## Reproduce
`python experiments/decision_id/{l3_certify,l2_geometry,l3_beta_cap,se_e4,kflip,nuisance}.py`; `python -m pytest -q` (123 passing; `MTI_WRITE_TEST_RECORD=1` to refresh the committed record).

## On resume
Read this file, decision_identification_closeout.md (Part III) and research_synthesis.md. Do not rerun completed work.
