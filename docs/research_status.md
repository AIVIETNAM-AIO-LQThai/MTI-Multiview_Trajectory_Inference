# Research status (replace in place)

**Phase:** Task 0 (oracle at shifted priors) complete → **next: Opus 5.5 `REVIEW EVIDENCE`** on the Task 0 addendum, then user authorization for the learned pilot.
**Updated:** 2026-10-04 by Sonnet 5.5 (`claude-sonnet-5-5`, set via `/model`).

## Provenance
Branch `exp/oracle-evidence`. Oracle v1 results: code `0f1b6d1`. Task 0 results: code `2168bee` (clean code tree), `configs/oracle_v1_priors.toml`.
Docs/results from this phase are uncommitted until the user commits them. The user makes all git commits.

## Done
- Task 0: [addendum](reports/oracle_v1_priors_addendum.md) and [report §11](reports/oracle_v1_report.md).
  - Four cells (L=8, π = point mass on lag 0, β∈{0.2,0.5}, q_w∈{0.02,0.5}), N=100,000 each.
  - Tests: 69 pass. Checks: max |z| ≤ 2.08; route equality ≤ 4e-15.
- Protocol v2 decision 4 now carries the full Δ_min table for all 8 (physics, prior) pairs (0.0080 … 0.4926; the floor binds only at q_w=0.5, P1).

## Key results
- Under lag-0 priors V_2 is 0.108 … 4.93 (27–159% of VOI_clean) and V_2 is 64–89% of the naive excess.
- At β=0.5 with lag-0 corruption the naive action costs more than the zero action.
- The aware oracle's natural-switch penalty grows: D = −3.1 … −6.3.
- The (s_L, μ̃) recalibrator attains 16–24% of V_2 at P3 and 35–52% at P4 (lower bounds).
- Escalation conditions were not triggered: no T4 contradiction, and V_2 is above the floor at P3/P4.

## Questions for Opus
1. Δ_min = 10% V_2 at P3/P4 is 4–40% of the clean-oracle cost. Keep the rule, or add a cost-based cap?
2. P3/P4 are a point mass on lag 0. Is that the intended "prior shift" for H3, or should an intermediate prior (e.g. mass on lags 0–1) be added?
3. Learned-pilot go-ahead is the user's decision.

## Next task (after Opus review and user authorization)
Implement the learned arms per [learned_pilot_protocol.md](learned_pilot_protocol.md) v2, starting with the feasibility gate (decision 9).
Do not start before the review and the user's go-ahead.
