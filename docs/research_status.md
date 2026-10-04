# Research status (replace in place)

**Phase:** E2 reviewed (Opus 5.5, 2026-10-04) → **E2b (robust declared priors) specified**, [e2_protocol.md](e2_protocol.md) §10. Next: **Sonnet 5.5 `CONTINUE CODE`** (after the user's commits), unless the user stops here.
**Updated:** 2026-10-04 by Opus 5.5 (`claude-opus-5-5`). Branch `exp/channel-misspecification`.

## Provenance / pending commits
HEAD is still `dc1abb8`; Batch A, E2 and this review are uncommitted. The two commits requested by Sonnet are still needed. The review edits (e2_report §6, protocol §10, findings_summary v3, charter, this file) can go into the second one.
The user makes all git commits; never push. E2 runs recorded base `dc1abb8` + patch `19d381a3…`.

## Review verdict
See [e2_report.md](reports/e2_report.md) §6 and [findings_summary.md](findings_summary.md) v3 (C15–C18, H-loc).
- Composition's value is bounded by the accuracy of the declared prior.
- Understatement and flattening are safe; overstating a weak channel, or concentrating where corruption is diffuse, harms.
- Better estimators implement a wrong prior faithfully; direct-arm "robustness" is attenuation.
- The coverage confound explains the large-N but not the small-N advantage.

## Next task (Sonnet, E2b; no training)
Implement protocol §10:
- `experiments/run_e2b.py`: exact selection on the screening set (master 8101), then confirmation (master 8201) with exact and saved A4 checkpoints (`results/e2/checkpoints/N*_s*_A4.pt`).
- `experiments/analyze_e2b.py`: K1/K2/K3, harm/improvement counts, retained-benefit shares, and the lag-0 odds-ratio scatter.
- `tests/test_e2b.py`: T-E2b-1…3.
- `docs/reports/e2b_report.md`.

Acceptance: tests pass; selection and confirmation on disjoint prefixes; all 30 truths reported for every selected prior; provenance = commit + patch.

## Return to Opus when
E2b is done, or T-E2b-2 fails.
