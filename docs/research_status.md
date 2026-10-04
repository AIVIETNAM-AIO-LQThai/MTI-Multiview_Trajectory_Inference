# Research status (replace in place)

**Phase:** oracle v1 reviewed (Opus 5.5, 2026-10-04) → **next: Sonnet 5.5 `CONTINUE CODE`, Task 0 only** (oracle at shifted priors).
The learned pilot itself needs **explicit user authorization** (it was outside the first assignment).

## Provenance
Branch `exp/oracle-evidence`. Oracle results: code `0f1b6d1`, results/report commit `5fc4383`. Review edits (report §§3, 4, 10; protocol v2;
charter; this file) are uncommitted until the user commits them. The user makes all git commits; never push.

## Review verdict (details: [report §10](reports/oracle_v1_report.md))
- Supported within S2 + exogenous single-flip channel + one-step rule:
  - C1 route equivalence and C2 decomposition (verified).
  - C3/C4: V_2 is small in the primary cell (1.8% of VOI_clean) and 13% at L=8, q_w=0.02, β=0.5. I_loss ≥ V_2 in 7 of 8 cells.
  - C10: the aware oracle is worse on clean records and in natural-switch states.
  - C5 holds as a lower bound only; C6 is descriptive only.
- Qualifications added:
  - The L=8 vs L=32 contrast is confounded with the per-position flip rate β/L.
  - The localisation sentence was reworded (top-1 by lag was not measured).
- Open: H2, H3, H5, and external validity.

## Approved decisions
[learned_pilot_protocol.md](learned_pilot_protocol.md) v2:
- Lead physics q_w=0.02, L=8; secondary physics q_w=0.5, L=8 at P1/P2; L=32 dropped.
- Test priors P1–P4: β∈{0.2,0.5} × π∈{uniform, δ at lag 0}. Arms trained on clean data are trained once per physics.
- Δ_min = max(0.1·V_2, 0.002·VOI_clean) per (physics, prior).
- Reference lines: the exact naive filter (𝓔 = V_2) and the learned naive A1.
- A1r refit-per-prior gets its own row.
- Feasibility gate: about 8 CPU-hours.

## Next task (Sonnet, Task 0): oracle at shifted priors
1. Config `configs/oracle_v1_priors.toml`. Add an optional cell field `pi = "uniform" | "lag0"` (`ChannelSpec.point_mass(L, β, L−1)`) to
   `make_cells`; default `uniform`, so the existing config is unchanged.
   - Cells: P3 and P4 for both physics (q_w ∈ {0.02, 0.5}, L=8, η=0.05).
   - New `cell_id` values ≥ 10, so streams do not overlap with oracle_v1 cells 0–9.
   - Same seeds and rules as oracle_v1.
2. Tests:
   - Add a test that `pi="lag0"` builds the intended channel.
   - Run the full suite: all must pass; T4 already covers point-mass priors.
3. Run → `results/oracle_v1_priors/`.
   - Under a point-mass π the contamination profile and recalibration still run. The "attacked" stratum then has a single location; that is fine.
4. Add an addendum to the oracle report: V_2, I_loss, X_naive, VOI_clean, D in the natural-switch stratum, and **Δ_min for all 8 (physics, prior) pairs**. Copy the Δ_min table into protocol v2 decision 4.
5. Hand back to the user for commits; do **not** start learned arms.

**Acceptance:** all tests pass; V_2 half-width ≤ 5% of V_2 (or the absolute rule); main-vs-check |z| ≤ 4; provenance shows a clean code commit.

## Escalate to Opus only if
A P3/P4 result contradicts a T4 identity, or the Δ_min values make the lead physics uninformative (e.g. V_2 at P3/P4 below the floor).
