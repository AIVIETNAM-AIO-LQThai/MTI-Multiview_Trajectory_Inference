# Research status (replace in place)

**Phase:** Task 0 reviewed (Opus 5.5, 2026-10-04) → **awaiting user authorization for the learned pilot**. If authorized: Sonnet 5.5 `CONTINUE CODE`.
**Updated:** 2026-10-04 by Opus 5.5 (`claude-opus-5-5`, set via `/model`).

## Provenance
Branch `exp/oracle-evidence`:
- Oracle v1: code `0f1b6d1`.
- Task 0: code `2168bee`, results `262f48c`.
- Review-2 edits (protocol decisions 1, 2, 4, 6; report §12; charter C4; this file) are uncommitted until the user commits them.

The user makes all git commits; never push.

## Review-2 decisions ([protocol v2](learned_pilot_protocol.md))
- Δ_min = clip(0.10·V_2, 0.002·VOI_clean, 0.02·VOI_clean). The cap binds at P3/P4.
- New test prior P5 (β=0.2, π_j ∝ 2^{−lag}): in-family for A6p-broad, extrapolation for A6p-narrow.
- Secondary physics (q_w=0.5): clean-trained arms are evaluated at all priors.
- Predeclared P4 caution: Ĝ collapses to ψ̂_{L−1}.
- The uniform-prior "small V_2" conclusion is prior-specific.

## Next task (Sonnet, only after the user authorizes the learned pilot)
1. **Task 0b (oracle, about 1 min):** add `pi = "recent"` (π_j ∝ 2^{−(L−1−j)}) to `make_channel`, with a test. Run P5 for both physics (`cell_id` 14–15) into
   `results/oracle_v1_priors/`. Fill the two P5 entries of the Δ_min table and add them to the addendum.
2. **Feasibility gate** (protocol decision 9):
   - Implement the shared backbone and one arm (A1).
   - Time one seed at N=10⁵ on the lead physics.
   - Project the total, and cut the tuning grid to 4 configurations only if the projection exceeds about 8 CPU-hours.
3. **Learned pilot** per protocol v2 (arms A0–A6p; physics q_w ∈ {0.02, 0.5}; priors P1–P5; N ∈ {10³, 10⁵}; 5 seeds, up to 10 for power).
   - Same test discipline as the oracle.
   - Access tests: no oracle targets, labels or probes in any input.
   - Verify numerically the route-gap identity Ĝ − M̂ = Ẑ⁻¹Σ q_j Ĉ_j/(1−r̂_j).
   - Hand back for commits at each logical milestone.

**Acceptance:** every (arm, physics, prior, N) cell is reported, including flexibility failures. Paired seed-level CIs; equivalence judged against Δ_min. Required
diagnostics: natural-switch stratum E_{L−1}, clean-record harm, query-specific errors by contamination count.

## Escalate to Opus if
A learned result needs interpretation before the next run; the route-gap identity fails numerically; a fix would change access or supervision; or results
materially contradict H2/H3 predictions.
