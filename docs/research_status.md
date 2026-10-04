# Research status (replace in place)

**Phase:** learned pilot, milestone 1 (infrastructure + feasibility gate) done → awaiting user commit; **next: milestone 2 (pilot driver + tuning + run)** on Sonnet 5.5.
**Updated:** 2026-10-04 by Sonnet 5.5 (`claude-sonnet-5-5`). Learned pilot authorized by the user ("CONTINUE CODE" after the authorization question).

## Provenance
Branch `exp/oracle-evidence`. Oracle results: code `0f1b6d1` / `f89ff2c`. Milestone-1 code (src/mti/learned, tests, gate script, P5 docs) is uncommitted until the user commits. The user makes all git commits; never push.

## Done
- Task 0b: P5 oracle cells; capped Δ_min table complete (report §13, [addendum](reports/oracle_v1_priors_addendum.md)).
- Learned infrastructure (`src/mti/learned/`): features, MaskedBelief (BiGRU) and CausalDensity models, composition (G, M, avg, ensemble), training loops for A1, A1r, A2/A3, A4, A6, A6p, HMM-EM (A5),
  test-set evaluator against the exact oracle (RB over views; E, E_none, E_cor, D vs exact naive, natural-switch strata).
- Tests: 81 pass (11 new learned tests: access control, folded features carry no sign, query construction = direct model calls, composition reproduces μ_2 and the route-gap identity, EM recovers η, probe-loss minimiser).
- Feasibility gate: [learned_gate.md](reports/learned_gate.md): ≈5.6 CPU-h projected (8-config grid, 5 seeds) → grid stays at 8.
- Fixed: provenance filter bug (`run_oracle.py`); `test_max_errors.json` is now only rewritten by full-suite runs; recalibrator exact oddness at f=0.

## Smoke observations (N=2×10⁴, 6 epochs, q_w=0.02, P1; not results)
Exact naive E=0.0786; A1 0.111 (undertrained); A2-G 0.039, A2-M 0.042, A2-avg 0.040; A3-ens 0.081; A4 0.0085; A5 ≈0 (η̂=0.0507); A6 0.040. Gap identity error 1e-15.

## Next (milestone 2)
1. Pilot driver `experiments/run_learned.py`: per (physics, N, seed) train all arms (A1 on ¾N + A1r per prior on ¼N; A2/A3; A4; A5; A6 at P1; A6p-narrow/broad), evaluate on a 20,000-prefix test set at P1–P5; write JSON per run.
2. Tuning on seed 0 over an 8-config grid (lr × wd × epochs) per arm, selected on the deployable validation criterion (clean arms: clean probe/likelihood loss; channel-trained arms: channel-simulated probe loss at P1).
3. Run 5 seeds (extend to 10 if the paired CI is wider than Δ_min/2), N∈{10³,10⁵}, physics q_w∈{0.02,0.5}; report every (arm, physics, prior, N) cell; equivalence judged against Δ_min.
4. Required diagnostics: natural-switch stratum, clean-record harm (D vs exact naive), query-specific errors by contamination count, signed route gaps, Ĉ_j distribution.
5. Hand back for commits at each milestone; escalate to Opus for interpretation before any follow-up experiment.
