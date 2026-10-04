# Research status (replace in place)

**Phase:** phase 1 complete (oracle, shifted priors, learned pilot, E1), reviewed by Opus 5.5 on 2026-10-04. **Experimental work is stopped** pending a new authorization from the user.
**Updated:** 2026-10-04 by Opus 5.5 (`claude-opus-5-5`).

## Provenance
Branch `exp/oracle-evidence`; latest results commit `abd83b7`. The final review edits are uncommitted until the user commits them: E1 report §6, `findings_summary.md`, charter, README, this file.
The user makes all git commits; never push.

## Where things stand
- Claims, evidence and limits: [findings_summary.md](findings_summary.md).
- Supported thesis: exact inference-time composition with the declared channel is a prior-flexible estimator of the aware belief. Under matched supervision it is never worse than direct training, better off-family, and better in-family at small N at low noise.
- The multi-view route (folded vs candidate) does not matter.

## Possible next phases (each needs user authorization; see findings_summary.md)
1. External validity of C12 with generic learners and unknown physics.
2. Channel misspecification sensitivity. Recommended first: this is the main practical risk of composition.
3. Closed-loop policy relevance.

## On resume
Read this file and findings_summary.md. Do not restart completed work.
