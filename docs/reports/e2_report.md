# E2 report: channel misspecification (Sonnet 5.5; interpretation pending Opus review)

Protocol, fixed before any E2 data: [e2_protocol.md](../e2_protocol.md). Full tables and figures: [e2_tables.md](e2_tables.md), `fig_e2_screen_phys{0,1}.png`, `fig_e2_learned.png`.
All outcomes are measured on synthetic S2 data under the **true** channel; the physical model and the exogenous single-flip mechanism are unchanged.

## 1. What was run

- **Branch `exp/channel-misspecification`**, base `dc1abb8`; code differences are recorded in `results/e2/code_patch.diff` + `provenance.json` (patch sha256 in the tables header). Tests: 93 pass (8 new: T-E2-1…6).
- **Exact screening** (exploratory; exact clean law): 169 (q_true, q_assumed) pairs × 2 physics (q_w 0.02 and 0.5), 100,000 prefixes each (master 8101), 6 min per physics.
- **Learned confirmation** (lead physics q_w=0.02): 14 predeclared cells × N ∈ {10³, 10⁵} × seeds 1–10, independent 20,000-prefix test set (master 8201). Arms A4, A5 (fitted HMM), A6pd-broad, A6pd-cov (coverage control), each with its own "ignore the channel" version; exact R1 reference. 20 units, 590 s on 18 processes; checkpoints in `results/e2/checkpoints/`.
- **Not run:** learned arms on the full screening grid (optional in the protocol); mode-dependent placement; secondary physics for learned arms.

## 2. Validation

| check | result |
|---|---|
| T-E2-1 convex identity μ₂(S;β) = (1−w)μ(S) + w m̄(S), w = oA/(1+oA), both physics, α ∈ {0, .5, 1}, β ∈ {0,…,.5} | max error ≤ 1e-10 (tests) |
| T-E2-2 no per-record loss to naive when the location prior is correct and (β_a ≤ β_t or odds(β_a) ≤ 2 odds(β_t)) | **0 violations** in 83 predicted-safe pairs per physics × 900,000 records (screening) and in the tests; the test has power (violations do occur outside the region) |
| T-E2-3 q_a = q_t ⇒ R1 ≡ R0; β_a = 0 ⇒ R1 = naive | exactly 0 / exactly equal |
| T-E2-4 RB estimate = single-draw Monte Carlo | \|z\| ≤ 4 |
| T-E2-5 π_α and the coverage family | P(π_last > 0.9): coverage family > 0.03, broad family < 1e-3 |
| T-E2-6 L + C + X = E_true per sample | ≤ 1e-12 (≤ 1e-14 in the runs) |
| Reproduction: retrained A4 and A6pd-broad (seeds 1–5, N=10³/10⁵) vs the pilot/E1 values at P1 and P3 | **bit-identical** (difference 0.0) |

No stop condition of the protocol was triggered. The sign of every learned benefit at the matched controls (β_a = β_t, α_a = α_t) agrees with the exact reference.

## 3. Exact screening (details in tables)

**Rate axis, location prior correct.** B_exact = E_true(naive) − E_true(exact composition with β_a); positive = accounting for the wrong β still beats ignoring it.
- **β_t = 0.2:** B_exact > 0 for every β_a tested (0.01 … 0.5) at both physics and every α. It peaks near β_a ≈ β_t. At α=0 (uniform), overstating to β_a = 0.5 keeps only 0.030 of 0.094 (q_w 0.02; 32%); at α=1 (last position) it keeps 1.63 of 1.70 (96%).
- **β_t = 0.05 (a weak true channel):** B_exact turns **negative** (overestimation harms in aggregate) between β_a = 0.1 and 0.15 at α=0 (both physics), between 0.4 and 0.5 at α=1 for q_w=0.02, and between 0.2 and 0.25 at α=1 for q_w=0.5. The break-even depends on α and on the noise level and is not at a single odds ratio. Underestimation (β_a ≤ β_t) never harmed.
- **β_t = 0:** any β_a > 0 harms where evidence is ambiguous (prediction iv); the screening tables list it.
- **The record-level safe region is narrower than the aggregate safe region:** beyond odds ratio 2, a large share of individual records is harmed (q_w=0.02, β_t = 0.05: 25–72% of records at β_a = 0.1 and 54–83% at β_a = 0.5, rising with α), yet the aggregate benefit is often still positive (per-pair counts in `results/e2/screen/*.json`).

**Location axis (β = 0.2 both sides)**, B_exact; rows true α, columns assumed α (q_w=0.02):

| α_true \ α_assumed | 0 | 0.5 | 1 |
|---|---|---|---|
| 0 (uniform) | +0.094 | −0.030 | **−0.295** |
| 1 (last action) | **+1.051** | +1.628 | +1.704 |

Asymmetric: assuming a uniform location when the truth is the last action still gains 1.05 (62% of the matched 1.70), while assuming the last action when the truth is uniform loses 0.295 relative to ignoring the channel (q_w=0.5: +0.53 and −0.26).

**Strata** (exact composition vs naive; realised hidden-mode costs, D positive = costlier). Composition always costs on clean records and after natural mode switches and gains on corrupted records, even when the aggregate benefit is positive.
Examples (q_w=0.02, α=0, β_t = 0.2, β_a = β_t): clean records +0.069 ± 0.006, corrupted −0.729 ± 0.008, natural-switch +1.036 ± 0.096. With overstated β_a = 0.5: +0.235, −1.054, +3.442 (aggregate B +0.030). With a weak true channel (β_t = 0.05, β_a = 0.2): clean +0.069, corrupted −0.729, switch +1.183, aggregate −0.028.
At α = 1: clean +0.14…+0.61 and natural-switch +2.1…+9.5 against a corrupted-record gain of −7.6…−10.5. The aggregate hides these mixture-dependent transfers.

## 4. Learned confirmation (lead physics; Δ = clip(0.10 V₂,true, 0.002·VOI, 0.02·VOI); b positive = good; per-cell values in the tables)

**Tallies over 14 cells** (material improvement / equivalent / noninferior / inconclusive / material harm):

| comparison | N=10³ | N=10⁵ |
|---|---|---|
| Q1 B_pipe(A4) (A4 vs the same model with no channel) | 13 / 0 / 0 / 0 / **1** | 13 / 0 / 0 / 0 / **1** |
| Q2 A6pd-broad − A4 (b = E(A6pd) − E(A4)) | 8 / 2 / 2 / 1 / **1** | 6 / 5 / 2 / 0 / **1** |
| Q3 A6pd-cov − A4 | 8 / 2 / 1 / 2 / **1** | 2 / 7 / 1 / 3 / **1** |
| Q4 B_exact(R1), the exact reference | 13 / 0 / 0 / 0 / **1** | 13 / 0 / 0 / 0 / **1** |

- **The single "material harm" cell is the same in every comparison:** true α = 0 (uniform), assumed α = 1 (last action), β = 0.2. There the exact reference loses to naive by −0.299 (CI ±0.015), A4 by −0.317 (N=10³) / −0.300 (N=10⁵), A5 −0.297/−0.300. The learned composition faithfully reproduces the exact reference's harm; it adds nothing.
  In that cell **A6pd and A6c lose much less** than composition (Q2/Q3: A6pd better than A4 by 0.39 at N=10³ and 0.28 at N=10⁵; A6pd's own B_pipe −0.0075 / −0.049; A6c −0.040 / −0.195): the direct arms respond weakly to the (wrong) conditioning.
- **Rate axis** (β_t=0.2): A4's B_pipe positive for every β_a ∈ {0.05,…,0.5} at both α and both N. It follows the exact curve closely at N=10⁵ (N=10⁵; e.g. α=0, β_a = 0.5: A4 +0.028 vs exact +0.030; α=1, β_a = 0.2: +1.713 vs +1.721).
- **Coverage control (Q3 vs Q2):** at α_t = 1 and N=10⁵, A6pd-cov is equivalent to A4 for β_a ≥ 0.2 (Q3 +0.026, +0.010, −0.023) where A6pd-broad was materially worse (Q2 +0.245, +0.289, +0.459). At β_a = 0.05/0.1 A4 stays materially better than A6pd-cov (+0.260, +0.093). At N=10³ the control does **not** close the gap (Q3 +0.26 … +0.53, material).
  In the cells with α_t ≤ 0.5 other than the harm cell at N=10⁵, A6pd-broad is equivalent to A4 in 5 and noninferior in 2 (Q2 tally); A6pd-cov is equivalent in 7, noninferior in 1, inconclusive in 3 over all 14 cells (Q3 tally).
- **Error decomposition** (seed means; E_true = learning + channel + cross). For A4 and A5 the cross term is small (median |cross| 0.0014 and 0.0002; maximum 0.087 and 0.0045): channel and learning errors are close to additive. For the direct arms the cross term is large and **negative**: in the harm cell A6pd has learning 0.373, channel 0.390, cross **−0.713** (N=10³); A6c 0.324, 0.390, −0.630; at N=10⁵: A6pd 0.257, 0.390, −0.540. The direct arms' learning error cancels most of the channel error.
- **Fitted HMM (A5):** B_exact is within 0.005 of the exact reference in all cells at N=10³ (learning term ≤ 0.0004) and within 0.0005 at N=10⁵ (learning term ≤ 6e-6).
- **Strata for A4** match the exact pattern (e.g. rate_a1_ba0.2, N=10⁵: clean +0.381, corrupted −10.261, natural-switch +5.749; harm cell: clean +0.381, corrupted −0.139, natural-switch +6.908).

**Precision:** the column `prec.` of the confirmatory table marks Q1/Q2/Q3/Q4 half-width ≤ Δ/2 (y/n); the harm cell and most α=1 cells are precise; several α=0 Q2/Q3 cells at N=10³ are not (inconclusive/noninferior). Intervals condition on the shared test set; the test-set SE is shown next to each interval; its 95% half-width is smaller than the seed-level half-width in 36 of 42 Q1–Q3 intervals at N=10³ and in 25 of 42 at N=10⁵ (where seed variation is small).

## 5. Flags for the Opus review (no interpretation is made here)

1. **The harm cell is a prior-direction effect**, not an estimation effect: the exact reference itself harms. Its sharper implication is that composition applies a wrong declared prior *faithfully*, while a direct model's weak response happens to protect. Is "composition is only as good as the declared prior" now the main caveat of the thesis, and does it change C8?
2. **The negative cross term** of the direct arms means "learning error + channel error" does not add; direct training can appear robust through cancellation. Whether that is a real property or an artefact of under-trained conditioning is open.
3. **The coverage control** reproduces the review's concern: much of E1's concentrated-prior advantage at N=10⁵ is training-coverage. At N=10³ it persists.
4. **Break-even regions** are (α, β_t, noise)-dependent; the aggregate benefit hides large costs on clean records and natural-switch states, which grow with α and with overstatement.
5. **Scope of what was run:** one physics for learned arms, one true β (0.2) for the learned rate cells, 10 seeds, N=10³/10⁵, the tuned configs of E1/pilot (no new tuning), exact clean-law-dependent arms (A5) near-oracle.

## 6. Opus evidence review (2026-10-04, `claude-opus-5-5`)

Checked: provenance (base `dc1abb8` + patch `19d381a3…`; bit-identical reproduction of the pilot/E1 arms), the validation outputs, the harm-cell and coverage-control numbers, and the full location matrices in `results/e2/screen/*.json`. **The measurements are sound and the protocol was followed.** No stop condition applied: the sign of the exact reference is reproduced everywhere, and the harm cell was predicted by the exact screening, not produced by learning.

**What E2 establishes** (S2, exogenous single-flip channel, supplied physics, one-step regret):
1. **A declared prior can make channel-aware inference worse than ignoring corruption**, and the direction of the error matters more than its size.
   - *Rate error:* understatement never hurt. This is proved per record when the location prior is right, and holds empirically for every tested pair. Overstatement was safe per record up to twice the true odds. In aggregate it stayed beneficial for β_t = 0.2 through β_a = 0.5, but turned harmful for a weak true channel (β_t = 0.05) at roughly 2–3× overstatement under a uniform location prior.
   - *Location error:* harm occurred **only** when the true location prior is diffuse (α_t = 0) and the declared one concentrated (α_a ≥ 0.5): −0.030 to −0.295 at q_w=0.02, −0.047 to −0.263 at q_w=0.5. Flattened declarations (α_a ≤ 0.25) never lost to naive anywhere on the grid, and kept 88% / 81% of the matched benefit when the truth was fully concentrated.
   - *Working explanation, a hypothesis consistent with both axes:* harm tracks **overstatement of the corruption rate at the decision-relevant recent lags**. In the harm cell the declared lag-0 rate is 8× the true one. α_a = 0.25 (2.75× at lag 0) still gains, while α_a = 0.5 (4.5×) slightly loses. This matches the 2–3× aggregate break-even on the rate axis.
2. **Better estimators implement a wrong prior more faithfully.** A4 and the fitted HMM reproduce the exact harm in their pipeline-matched benefit (B_pipe −0.317 / −0.297 at N=10³ and −0.300 / −0.300 at N=10⁵, vs exact −0.299). The direct arms' smaller loss in the harm cell is **attenuated responsiveness to the conditioning prior**, not robustness: the cross term of −0.54 to −0.71 cancels learning against channel error.
   The coverage-trained A6pd-cov learns the conditioning better: its learning term in the harm cell is 0.055 vs A6pd's 0.257 (N=10⁵), and it consequently loses more there (B_pipe −0.195 vs −0.049). At the matching concentrated cell, the same attenuation costs A6pd a learning term of 0.246 vs A6pd-cov's 0.027.
3. **The coverage confound is real at large N and absent at small N.**
   - At N=10⁵, most of E1's concentrated-prior advantage disappears against a coverage-trained direct estimator: equivalent at α=1 for β_a ≥ 0.2, and at the matched uniform cell (Q3 +0.003).
   - At N=10³ composition stays materially better than the coverage control in every α=1 rate cell (+0.26 to +0.53) and at the matched uniform cell (+0.039).
   - So the small-N finite-data advantage survives the review's alternative explanation; the large-N concentrated-prior advantage does not.
4. **Aggregate regret hides transfers.** Composition always costs on clean records and after natural mode switches, and recovers more on corrupted records. Overstating the prior enlarges the costs: at α=1 the clean-record cost goes from +0.14 to +0.61 as β_a rises from 0.05 to 0.5. A risk-sensitive user would not read the aggregate alone.

**Analytic addition (for E2b; derivation in the protocol §10).** If the true q is unknown but independent of the record, with uncertainty weights w over candidate channels, the compound law has channel prior Σ_q w_q q, because the channel enters linearly. The Bayes estimator under that uncertainty is therefore exactly composition with **the mean prior E_w[q]**. It minimises the w-weighted true-law regret over *all* S-measurable estimators, not merely among plug-ins.
Robust composition is consequently a choice of declared prior, and the right choice depends on the stated criterion:
- E[q] for Bayes risk;
- a flattened, low-rate prior for "never worse than naive";
- an intermediate prior for minimax regret. On the screened location axis that is α_a ≈ 0.5: max regret 0.124 vs 0.202 at α_a = 0.25 and 0.389 at α_a = 1 (exploratory).

**Not established:** behaviour under other channels (mode-dependent placement), unknown physics or generic learners, multi-step control, and whether q can be estimated from corrupted data well enough to remove misspecification in repeated deployment.

**Decision:** E2's question is answered within its scope. One bounded follow-up is justified because it turns the negative result into a usable rule with its own fresh confirmation: **E2b, robust declared priors under prior uncertainty** (protocol §10; exact-dominated; reuses the A4 checkpoints, no retraining). Everything else stays out of scope.
