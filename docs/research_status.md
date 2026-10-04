# Research status (replace in place)

**Phase:** E3 (channel identification and adaptive composition) complete and reviewed (Opus 5.5, 2026-10-05). **Experimental work stopped** pending a user research decision.
**Updated:** 2026-10-05 by Opus 5.5 (`claude-opus-5-5`). Branch `exp/channel-identification` (from `fc74aef`).

## Provenance
- Uncommitted: E3 protocol, code, tests, results, report and review, plus doc updates.
- The E3 run recorded base `fc74aef` with untracked-file hashes. Run code matches; only `test_e3.py` changed afterwards (a tolerance tightened to the protocol value).
- The user makes all git commits. Pushing is done only when the user asks.

## Where things stand
Claims and limits are in [findings_summary.md](findings_summary.md) v5 (C1–C24).
- Composition with a declared channel is prior-flexible and Bayes-optimal under prior uncertainty, but only as good as its prior.
- For L ≥ 3 the prior is identifiable from unlabeled records, and likelihood adaptation removes the declared-prior risk given data. Clean-law error is absorbed into the inferred channel and is not removed by data.
- Multi-view routes show no general advantage; in aggregate, residual moments carry the channel information.
- No novelty is claimed.

## Options (each needs a user decision; none started)
1. **Per-record MTM detection benchmark.** This is the original question. Does multi-view consistency (route gap, compatibility residuals) of a learned shared model flag *decision-harmful* records better than:
   - the exact posterior (the ceiling),
   - residual/likelihood-ratio scores,
   - rarity (log-likelihood)?
   Requires retraining an A2-type model (no checkpoints exist), i.e. new neural training.
2. **Joint identification of clean persistence and channel** from corrupted records. This is plausible from the distance structure of the cross-moments for L ≥ 4, and attacks absorption (C22) directly. Exact law first; cheap.
3. **Literature positioning** against label-noise rate estimation, mixture-proportion estimation and robust/empirical Bayes, before any further experiment.
4. **Consolidate and stop:** a final synthesis from the existing record.

## On resume
Read this file and findings_summary.md. Do not rerun completed work.
