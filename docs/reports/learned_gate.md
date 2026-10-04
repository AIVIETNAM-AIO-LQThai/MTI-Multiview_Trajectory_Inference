# Learned pilot — feasibility gate (protocol v2 decision 9)

Measured by `experiments/run_learned_gate.py` (lead physics L=8, q_w=0.02, η=0.05; N=10⁵ training prefixes; 1 CPU thread per process; torch 2.14.1+cpu, Windows 11,
20-core host). Raw output: `results/learned_gate/gate.json`.

## Measured (single thread)

| arm | model | params | s / epoch at N=10⁵ (390 steps, batch 256) | view-passes / epoch |
|---|---|---|---|---|
| A1 naive filter | BiGRU(64) | 44,290 | 6.5 | 99,840 |
| A2/A3 shared fixed-view (1 full + 2 folded views) | BiGRU(64) | 44,290 | 15.9 | 299,520 |
| A4 causal mixture density | GRU(64) | 17,666 | 2.0 | 99,840 |
| A6 channel-trained direct (2 corrupted views) | BiGRU(64) | 44,290 | 7.6 | 199,680 |
| A6p prior-conditioned direct (2 corrupted views) | BiGRU(64)+q | 47,362 | 7.7 | 199,680 |
| A5 HMM EM (η̂ = 0.0500 at N=10⁵) | — | 1 | 0.9 total | — |

Test set (20,000 independent prefixes × 9 corruption views = 180,000 records): building one prior's oracle 0.8 s; querying A2 (2L+1 = 17 sequences per record) 70 s;
A4 (L+1 sequences) 16 s; a full-view-only model 4 s. Network outputs on the views are prior-independent, so clean-trained arms are queried once and composed under all five priors.

## Projection (assumptions stated; not yet measured end to end)

Assumptions: ≤12 epochs at N=10⁵ and ≤150 epochs at N=10³ (early stopping can only reduce this); A1r refit once per prior on ¼N calibration prefixes (about 8 s each);
5 seeds × 2 physics; tuning on seed 0 only, for both physics and both N, over the grid; evaluation on every prior.

| item | CPU-hours |
|---|---|
| training, per (physics, seed): N=10⁵ ≈ 590 s, N=10³ ≈ 75 s | 0.18 |
| evaluation, per (physics, seed): two N × (A2 70 s + A4 16 s + other arms ≈ 80 s) | 0.09 |
| main grid, 2 physics × 5 seeds | 2.7 |
| tuning, 8-config grid (seed 0, 2 physics, 2 N, train-only) | 2.9 |
| tuning, 4-config grid | 1.5 |
| **total, 8-config grid, 5 seeds** | **≈ 5.6** |
| total, 8-config grid, 10 seeds | ≈ 8.3 |
| total, 4-config grid, 10 seeds | ≈ 6.9 |

With about 18 parallel single-thread processes the 5-seed total is under one hour of wall-clock time.

## Decision

The projected total for the 8-config grid at 5 seeds is **≈ 5.6 CPU-hours, below the 8-hour trigger**, so **the grid stays at 8 configurations** (all arms equally). If the power rule
(paired-seed half-width ≤ Δ_min/2) requires 10 seeds, the projection reaches ≈ 8.3 CPU-hours, at the trigger; the grid would then be cut to 4 for all arms and the cut recorded.
Test set size: 20,000 prefixes (RB over views). Test-set Monte Carlo error is reported separately from training-seed uncertainty.

## Caveats

- The numbers are single-process timings on an otherwise idle machine; parallel runs may be slower per process.
- Early stopping and the 12-epoch cap are assumptions, not measurements.
- A3 and A2 share a model (compute counted once).
