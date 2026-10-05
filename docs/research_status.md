# Research status (replace in place)

**Phase:** E4 (joint identification of clean persistence and channel) complete and reviewed by Opus 5.5 (2026-10-05). **Experimental work stopped.**
**Updated:** 2026-10-05 by Opus 5.5 (`claude-opus-5-5`). Branch `exp/joint-identification` at `7b95b9d`.

## Provenance
- E4 stage 0: `3f868c5`.
- Stage 1: run at `ea8bca4` + the A1 patch (`results/e4/stage1/code_patch.diff`), committed in `7b95b9d`.
- Uncommitted: the Opus review (e4_report §9), findings_summary v7, charter, README and this file.
- The user makes all git commits. Pushing is done only when the user asks.

## Where things stand
- Claims: [findings_summary.md](findings_summary.md) v7 (C1–C28). Positioning: [literature_review.md](literature_review.md).
- E4 result: within-family clean-law (η) error is removable from corrupted records alone for L ≥ 4, at measured cost. L = 3 is not. Family error is absorbed persistently. A few hundred clean prefixes do as well as joint estimation.

## Recommended next step (needs user decision)
**Consolidation:** a synthesis of phases 1–E4 against the revised contribution statement, with no new experiments.
- Optional, only if the thesis needs it: a family-violation dose–response (absorbed β̂ and regret against the excess short-excursion rate), exact-law first, cheap.

## On resume
Read this file, findings_summary.md, literature_review.md and reports/e4_report.md §9. Do not rerun completed work.
