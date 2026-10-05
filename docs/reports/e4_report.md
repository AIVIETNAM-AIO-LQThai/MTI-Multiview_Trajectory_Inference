# E4 report: joint identification of clean persistence and channel

Sonnet 5.5 (`claude-sonnet-5-5`), 2026-10-05, branch `exp/joint-identification`. Protocol: [e4_protocol.md](../e4_protocol.md) with amendment A1 (§9).
This report states results and their direct reading. The claim ledger and the thesis-level interpretation are left to the Opus review.

## 1. What was run

**Provenance.**
- Stage 1: base commit `ea8bca4`, with the A1 changes to `run_e4.py` and `test_e4.py` uncommitted. Their diff is `results/e4/stage1/code_patch.diff`, sha 795bf052662646b4. Python 3.14.7, numpy 2.5.3.
- Stage 0: `results/e4/stage0/` (`3f868c5` content; see the stage-0 provenance file).
- Tests: the full suite passes, including `tests/test_e4.py` (T-E4-0 … T-E4-8).

**Jobs.**
- 500 jobs, all completed, nothing dropped: 480 main (2 physics × L ∈ {3, 4, 8} × 4 truths × 20 replicates) and 20 F1.
- Replicates 0–9 ran to n = 10⁵; 10–19 ran to n = 3 × 10⁴, as in the protocol.

**Deviation: budget.**
- The protocol's cap is 4 h. The run took **15,430 s (4.29 h)** on 18 workers.
  - Isolated, a heavy L = 8 job took 198 s. Under 18-way contention it took 0.8–1.6 ks, and secondary-physics L = 4 jobs took about 1.4 ks.
  - At the 3.5 h check the projection was about 4.5 h.
- The §7 reduction order would have saved little at that point, since the L = 3 n = 10⁵ jobs were mostly done. I did **not** reduce, so no cell was dropped and no design element changed.
- This is a budget overrun, recorded here, not a design change.

**Streams.** Masters 8501 (adaptation), 8601 (evaluation), 8701 (clean validation), 8901 (F1 second persistence stream), 8801 (stage 0). The F1 cells follow A1 (adaptation task id + 300, clean + 500, evaluation cells 2 and 3; test T-E4-8).

## 2. Primary result (P1): H-E4 holds at L = 8, n = 10⁴, in both physics

Regret differences in units of VOI_clean, 90% paired t-interval over R = 20 replicates. The margin is Δ = 0.0020 VOI on T0.

| physics | J-MLE − EX-MLE (equivalence) | ETA⁻−MLE − J-MLE (superiority) | ETA⁺−MLE − J-MLE | J-MLE η̂ mean (sd) | β̂ (truth 0) |
|---|---|---|---|---|---|
| lead, L = 8, T0 | +0.0001 [0.0000, 0.0001] → **equivalent** | +0.0283 [0.0270, 0.0296] → **J better** | +0.0055 [0.0054, 0.0056] → J better | 0.0495 (0.0010) | 0.004 |
| secondary, L = 8, T0 | +0.0001 [0.0000, 0.0002] → **equivalent** | +0.0283 [0.0270, 0.0296] → **J better** | +0.0127 [0.0125, 0.0130] → J better | 0.0496 (0.0007) | 0.008 |

Both conditions of H-E4 are met in both physics, and the stage-0 prediction (H-E4 should hold at 10⁴) was correct. The full grid is in `results/e4/stage1/summary.md`.

- **L = 8, all four truths, both physics.**
  - J-MLE is equivalent to EX-MLE at n = 10⁴ in all 16 cells at L ∈ {4, 8} (CI half-width ≤ 0.0006 VOI). At L = 8 the mean η̂ is 0.0492–0.0505 (truth 0.05); at L = 4 it is 0.0459–0.0505.
  - It is better than the η̂ = 0.03 arm (ETA⁻) by 0.012–0.028 VOI. That is "better" in 5 of 8 cells and "equivalent" in the 3 lead corrupted cells only because Δ is wider there (0.02 VOI).
- **L = 4.** The same pattern at n = 10⁴, with η̂ sd 0.002–0.007. J-MLE is equivalent to EX-MLE in all 8 cells; it is better than ETA⁻ in the two T0 cells (where Δ is small), equivalent in 5 cells and inconclusive in 1.
- **L = 3: the negative control behaves as derived.** See §3.

## 3. Crossover and the L = 3 control (P2, P3)

**Crossover n** (smallest n at which J-MLE stays equivalent to EX-MLE; classes in `summary.md`):

| | T0 (β = 0) | T3, T4, T6 |
|---|---|---|
| L = 8, lead | 3,000 | 300 (the smallest n run) |
| L = 8, secondary | 1,000 | 300 |
| L = 4, lead | 10,000 | 300 |
| L = 4, secondary | 10,000 | 1,000–3,000 |

- For corrupted truths the joint estimator is equivalent to the known-η estimator at every n run. The margin there is 0.012–0.02 VOI.
- The absorption control (T0) needs 1,000–10,000 records, because the margin is only 0.002 VOI there.
- At n ≤ 1,000 on T0, L = 8, the harm rates against the pipeline's own naive output are:

| | n | J-MLE | EX-MLE | ETA⁻ |
|---|---|---|---|---|
| lead | 300 | 0.45 | 0.35 | 0.85 |
| lead | 1,000 | 0.35 | 0.25 | 1.00 |
| secondary | 300 | 0.50 | 0.50 | 0.80 |
| secondary | 1,000 | 0.25 | 0.25 | 0.95 |

  At n = 10⁴ they are 0.00–0.05 for J-MLE and EX-MLE and 1.00 for ETA⁻. These are the E3 boundary-bias harms, and J-MLE is slightly above EX-MLE at small n.

**L = 3 (P3).**
- J-MLE is worse than EX-MLE at every n, in every truth, in both physics, except secondary T3 at n ≥ 10⁴ (inconclusive). It is also worse than the fixed ETA⁻/ETA⁺ arms.
- The estimate runs along the ridge toward small η:
  - median η̂ at n = 10⁴: lead T0 0.0039, lead T3 0.0050, secondary T0 0.0015;
  - β̂ settles at 0.26 on the T3 truth whose β is 0.2.
- Regret does not vanish:
  - lead T3 at n = 10³ / 10⁴ / 10⁵: 0.054 / 0.058 / 0.056 VOI, against Δ = 0.02;
  - secondary T3 falls 0.035 → 0.019 → 0.014, against Δ = 0.014;
  - the other L = 3 cells stay at 0.04–0.17 VOI.
- **Reading note for P3.** The spread of η̂ does shrink somewhat with n (e.g. lead T3 SD 0.0062 → 0.0032). This is because η̂ is pinned near the lower end of the search bracket, not because η is identified. The criterion that holds is "regret does not vanish", not "spread does not shrink". The ridge was found to be one-sided or two-sided as derived; at secondary T3 it is partly resolved by the simplex boundary at large n (β̂ 0.255 → 0.237).

## 4. Practical baseline (P4): a small clean set does as well

CV-n_c (η̂ from n_c clean prefixes, then EM over q) relative to J-MLE at n = 10⁴, the 16 cells at L ∈ {4, 8} (the class says whether CV is worse than J-MLE beyond Δ; verified by recomputation):

| n_c | equivalent | inconclusive | CV worse |
|---|---|---|---|
| 30 | 4 | 5 | 7 |
| 100 | 10 | 3 | 3 |
| 1,000 | 15 | 1 | 0 |

- CV-1000 is equivalent to J-MLE in 15 of 16 cells (secondary L = 4 T0 is inconclusive) and never worse.
- CV-100 is equivalent in 10 of 16 cells and CV-30 in 4 of 16.
- sd(η̂_cv) over replicates is 0.012–0.040 at n_c = 30, 0.006–0.025 at 100 and 0.002–0.007 at 1,000.
- At L = 3 the CV arms beat J-MLE (clean data resolve the ridge).
- So where separate clean prefixes exist, a few hundred are as good as joint estimation at n = 10⁴, and a clean set beats J-MLE at L = 3. Joint estimation is the option when no clean records exist.

## 5. Estimator comparison and calibration (P5, P7)

- **P5, J-MOM.**
  - At L ≥ 4 and n = 10⁴ its regret is 4–65 times J-MLE's (median 17×; up to 109× at n = 10⁵), and the sd of its η̂ is 0.002–0.02.
  - At L = 3 the ratios are below 1 or near 1 only because both estimators fail.
  - It is a cheap diagnostic, not a competitive estimator.
- **P7, measured SD(η̂) vs the stage-0 Fisher prediction** (interior truths T3 and T6, L ≥ 4, n = 10⁴, 3 × 10⁴, 10⁵):
  - 23 of 24 ratios are in the band [0.7, 1.4], with the 10⁴ ratios 0.81–1.15.
  - The exception is lead L = 8 T6 at n = 10⁵ (ratio 1.68). It rests on 10 replicates, for which the SD itself has about 23% relative error. I did not investigate further.
  - The Fisher prediction is therefore a usable planning tool at n ≥ 10⁴.

## 6. F1 (descriptive): record-heterogeneous persistence, η ∈ {0.02, 0.08}

Truths T0, T3 (L = 8, lead), R = 10, n = 10³, 10⁴, 3 × 10⁴. The oracle is the exact mixture posterior over (persistence class, θ).
- J-MLE estimates η̂ = 0.046–0.048 and β̂ = 0.010 on T0 (truth 0), with regret 0.0007 VOI at n ≥ 10⁴.
- CV-1000 (clean set drawn from the same mixture) behaves the same: η̂ 0.049, β̂ 0.010, regret 0.0009. EX-MLE (η = 0.05) is 0.0007.
- ETA⁻ shows the known absorption (β̂ 0.055, harm rate 1.00); ETA⁺ is harmless (β̂ 0.000), but with regret 0.0068.
- T3: J-MLE regret 0.0015 at n = 10⁴, indistinguishable from EX-MLE (0.0016).
- **Reading.** This particular heterogeneity (η_r ∈ {0.02, 0.08}, marginal 0.05) caused **no visible absorption**, and the single-η marginal model remains near-oracle. So F1 does not show family-violation harm; it shows only that this mild non-Markov departure is tolerated. It does not establish that no family violation harms. Spurious β̂ was ≈ 0.01, well below the E3 absorption of 0.054.

## 7. Limits (Sonnet's view; Opus to adjudicate)

- All results: S2 simulator, physics (ρ, c, q_a, q_w, λ) supplied, η = 0.05, one clean family (first-order symmetric Markov), the exogenous single-flip channel, one-step regret. No neural models.
- The profile-likelihood estimator searches η ∈ [0.001, 0.499], with the coarse grid 0.005–0.15. The truth is inside the grid.
- The CV baseline's clean HMM is correctly specified, so it shares the family assumption with J-MLE.
- No claim about family validation, other channels, other L, trained policies or real data.

## 8. Artifacts

- `results/e4/stage0/` (24 units + `summary.md`).
- `results/e4/stage1/units/` (500 units), `summary.md`, `provenance.json`, `code_patch.diff`; log `results/e4_stage1_log.txt`.
- Code: `src/mti/joint.py`, `experiments/run_e4.py`, `run_e4_stage0.py`, `analyze_e4.py`, `analyze_e4_stage0.py`; `tests/test_e4.py`.
