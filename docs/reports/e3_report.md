# E3 report: channel identification and adaptive composition (Sonnet 5.5; interpretation pending Opus review)

Protocol, fixed before any data: [e3_protocol.md](../e3_protocol.md). Full tables: [e3_tables.md](e3_tables.md); figure `fig_e3_curves.png`. All results are measured on synthetic S2 data; physics and the exogenous single-flip mechanism are unchanged; **no training**.

## 1. What was run

- **Branch `exp/channel-identification`**, base `fc74aef`. Runs record the base commit plus a code patch (`results/e3/code_patch.diff`, hash in the tables header). Tests: **105 pass** (8 new: T-E3-0…7).
- **188 tasks** (clean-law source × true channel), 698 s on 18 processes.
  - Sources: exact law (EX); too-persistent and too-flexible exact laws (ETA−: η̂ = 0.03, ETA+: 0.08); fitted HMM from E2's recorded η̂ (N=10³ and 10⁵, seeds 1–5); saved A4 checkpoints (N=10³ and 10⁵, seeds 1–5).
  - EX, ETA±, and HMM seed 1: n ∈ {10, 30, 100, 300, 10³, 3·10³, 10⁴}, 20 replicates; HMM seeds 2–5 and A4: n ∈ {100, 10³, 10⁴}, 4 replicates per seed.
  - 8 truths on L=8 (T0 β=0; T1 β=.02 α=0; T2 β=.05 α=.25; T3 β=.2 α=.5; T4 β=.35 α=1; T5/T6/T7 off the π_α family) and 4 identification-control truths (L=1, L=2).
- **Estimators:** fixed E2b declarations (F-qbar, F-noharm, F-minimax, naive); MLE (EM over the whole simplex); MAP (κ₀ = 30 toward q̄); PP (posterior predictive over the 110-point E2b grid); MOM (pair-sum moments + NNLS); ORACLE-q. Adaptation data: (s, a) only; evaluation: 20,000 disjoint prefixes (master 8401) scored under the true channel with the exact clean law.
- **Differences from the protocol:** sets of size n are nested within a replicate (first n of 10,000 records); for A4 the moment estimator uses the same-seed HMM η̂. No other deviations.

## 2. Validation and identification controls

| check | result |
|---|---|
| T-E3-0 fast composition (two matrix products) = log-domain route | ≤ 1e-10 |
| T-E3-1 L=1 and L=2 exact identities | pass (L=2 belief gap between allocations 1.49) |
| T-E3-2 cross-moment formula per pair; MOM recovery of q at L=8 | formula max \|z\| = 2.1 (tolerance 4, per-pair standard errors); recovery < 0.01 |
| T-E3-3 EM log-likelihood monotone; exact-law recovery, L=8, n=20,000 | passes; \|β̂ − β\| < 0.02 |
| T-E3-4 / 5 / 6 / 7 PP concentration; MAP limits; absorption sign of MOM; access and disjoint streams | pass |

**Predictions vs outcomes (exact law, 20 replicates; protocol stop conditions not triggered):**
- **L=1 (negative control):** nothing is learned about β. The MLE returns its starting value (β̂ = 0.500 at every n), giving regret 1.877 (truth β=0.1) and 0.469 (β=0.3), worse than ignoring corruption (0.117, 1.056) in the first case. PP stays at the grid mean (β̂ 0.199); MOM has no pairs and returns naive. The regret gap does not shrink with n.
- **L=2, allocation (1,0) (negative control):** β̂ converges (0.199 at n=10⁴) but TV(π̂, π) stays 0.50 for MLE and the MLE regret stays 0.425 (naive 0.058). PP: TV 0.75, regret 0.96. MOM happens to reach regret 0.0002 here, but that is the NNLS tie-break (it returns the vertex q = (0.2, 0)); on allocation (½,½) the same tie-break gives TV 0.50 and regret 0.426, while the MLE converges to regret 0.0000 (TV 0). The L=2 outcomes therefore depend on how the unidentified split is resolved.
- **L=8 (positive control):** MLE β̂ = 0.1984 at n=10⁴ (replicate sd 0.0064; truth 0.2); regret 0.0003 vs naive 0.809.
- **Artifact to read the L=1 MAP rows with:** the MAP-EM on a flat likelihood moves toward q̄ only at the EM rate, so at large n the 5,000-iteration cap leaves it near its starting value. This is an optimiser artifact of the unidentified case, not a statistical result.

## 3. Results

**3.1 Adaptation vs fixed declarations (exact law; Q1).**
- At n = 10, MLE/MAP/PP are materially better than F-qbar on the weak and off-family truths T0, T1, T2, T5 (F-qbar harms those four truths); on T4 from n = 30.
- On T3, T6, T7, where q̄ is already close to the truth (F-qbar regret ≤ 0.042), no adaptive estimator is ever materially better. MLE is materially worse at n = 10 (T3) and n = 10–30 (T6).
- With n = 10⁴ the MLE retains ≥ 99% of the achievable benefit on T2–T7 (T2 97.9%); on T1 (β = 0.02, where naive regret is only 0.0012) it retains 12%.
- MOM is materially worse than F-qbar for T3, T6 (n ≤ 300) and T7 (n ≤ 100), and its regret exceeds the MLE's by 1.6–41× at the tabulated n (Q4).
- MAP is materially better than MLE only at n = 10 on T3 and n = 10–30 on T6 (inconclusive on T4, T7). It is **worse** on T0–T2 and on the off-family T5, where the prior toward q̄ is wrong (e.g. T0: MAP − MLE = +0.148 at n = 30, +0.062 at n = 300; T5: +0.24 at n = 10, +0.13 at n = 100).

**3.2 Harm (Q2).** Harm = B_pipe < −Δ_t; at T0, Δ_t is the floor (0.002·VOI), so harm there is small in absolute terms.
- *Exact law, worst truth (T0 unless noted), MLE:* 0.55 at n = 10 (T1), 0.40 at n = 30–100, 0.35 at n = 1,000, **0.05 at n = 10,000**. MAP: 1.00 up to n = 1,000, 0.20 at n = 10⁴. MOM: 0.70, 0.65, 0.70, 0.45, 0.25. PP: 1.00 up to n = 100, then **0.45–0.50 at T5 for n ≥ 1,000**.
- *PP under an off-grid truth does not recover:* at T5 its regret stays 0.097 at n = 10⁴ (naive 0.087), i.e. it is worse than ignoring corruption; MLE reaches 0.0008. For T6/T7 PP plateaus at 0.030 / 0.023 (F-qbar 0.030 / 0.042).
- *Fixed declarations:* F-qbar and F-minimax harm T0, T1, T2, T5; F-noharm harms only T0 (as in E2b).

**3.3 Absorption of clean-law error (Q5, truth T0 β = 0).** Predicted sign confirmed.
- **Too persistent clean law (ETA−, η̂ = 0.03):** β̂ does not vanish: MLE 0.0660 / 0.0609 / **0.0542** at n = 100 / 1,000 / 10⁴, harm rate 0.80 → 1.00 → **1.00**; MOM raw (unprojected) β̂ = +0.2226 at n = 10⁴, against −0.348 for ETA+ and +0.003 for the exact law (protocol §2 A form). The error is absorbed into inferred corruption and **does not diminish with more adaptation data**.
- **Too flexible clean law (ETA+, 0.08):** β̂ → 0 (MLE 0.0000 at n = 10⁴; harm 0.00).
- **Exact law and fitted HMM:** MLE β̂ 0.0042 (EX), 0.0049 (HMM-10⁵, mean of 5 seeds), 0.0046 (HMM-10³) at n = 10⁴, harm ≤ 0.05. Persistence fits are good (η̂ 0.0478–0.0531 at N=10³, 0.0499–0.0502 at N=10⁵).
- **Saved A4 as clean law:** A4-10⁵: β̂ 0.0059, harm 0.00; **A4-10³: β̂ 0.0204 (seed range 0.008–0.028), harm 0.55**, mean B_pipe −0.012: the clean-law error of the small-N network shows up as spurious corruption at β = 0, again not shrinking at n = 10⁴ (T1 estimate under A4-10³: 0.0345 vs truth 0.02; under HMM and A4-10⁵: 0.0193–0.0208).

**3.4 Learned clean laws (§8 of the tables).** With HMM and A4-10⁵ the adaptive curves track the exact-law curves (e.g. T3 MLE regret at n = 10⁴: 0.0005 HMM-10⁵, 0.0012 A4-10⁵, vs 0.0003 exact), recovering β̂ (mean over seeds) within 0.002 at T3 for HMM-10⁵ and A4-10⁵ and within 0.003 for A4-10³ (individual A4-10³ seeds up to 0.011 off). A4-10³ is the clean law that degrades the weak truths (T0/T1 above); HMM seed spread is small.

**3.5 Off-family truths (Q6).** MLE handles T5–T7 (regret ≤ 0.0008 at n = 10⁴) and MAP does at T6–T7 (≤ 0.0010; T5: 0.0053); MOM does at large n (≤ 0.009) but is noisy below n = 1,000; PP is limited by the grid (above).

## 4. Limits and flags for the Opus review

1. **Identification does not make adaptation harmless at small n:** with the exact law and n ≤ 1,000, the worst-truth harm rate of every adaptive estimator is 0.35–1.00 (mostly at β = 0, where the absolute loss is small), against 0 for naive. Exact-law MLE reaches 0.05 only at n = 10⁴.
2. **The absorbed clean-law error is a bias that data cannot remove** (ETA−, A4-10³). A learned clean law must be validated independently of the channel before channel inference is trusted.
3. **The moment baseline works** where identification holds and the clean law is right (T4–T7 at n ≥ 1,000; ≈ MLE regret at T5 and n = 10⁴), but with 2–28× the MLE regret at moderate n. It shows that, in this simulator, aggregate channel information is available from residual–action cross-moments alone. It says nothing about per-record detection (the MTM question).
4. **Scope:** one physics (q_w = 0.02, η = 0.05, L=8); adaptation data from one stable channel; nested sets within a replicate; the A4 and HMM seed analyses use 4 replicates (HMM seed 1: 20); harm uses the point estimate of B_pipe on one shared evaluation set; no multiplicity correction; MAP κ₀ = 30 was fixed a priori, not tuned.
5. The L=2 MOM and L=1 MAP outcomes are tie-break / optimiser artifacts (above); they are controls, not claims.


## 5. Opus evidence review (2026-10-05, `claude-opus-5-5`)

Checked:
- **Provenance.** Base `fc74aef`. All E3 code was untracked, so the recorded patch is empty, and provenance rests on the untracked-file hashes. `run_e3.py` and `adapt.py` match the files that ran; `test_e3.py` changed afterwards only by tightening a tolerance to the protocol's value.
- **The identification controls.**
- **Harm magnitudes,** recomputed: at β = 0 the MLE's mean loss is 0.070 → 0.036 → 0.010 → 0.001 at n = 10 / 100 / 1,000 / 10⁴, against VOI_clean = 3.17.
- **The report numbers,** re-verified by Sonnet against the saved data with corrections applied.

**The results are sound.**

**Supported** (S2, exogenous single-flip channel stable across records, supplied physics, prefixes without the probe):
1. **Identification (C20, verified three ways: analytically, by exact or Monte Carlo checks, and by the adaptation controls).**
   - At L = 1 the rate is unidentified.
   - At L = 2 the location allocation is unidentified.
   - Both change decisions, and estimators confronted with them behave arbitrarily (start values, tie-breaks, grid means): their regret does not shrink with data.
   - For L ≥ 3 with η < ½, the channel is identified from residual–action cross-moments, and the likelihood recovers it (β̂ = 0.198 for β = 0.2 at n = 10⁴).
2. **Adaptive composition can remove the declared-prior risk (C21), given data and a correct clean law.**
   - The full-simplex MLE reaches near-oracle regret on every L = 8 truth by n = 10⁴, including off-family channels, retaining ≥ 98% of the achievable benefit wherever that benefit is not negligible.
   - It beats the fixed mean-prior declaration from n = 10 exactly where that declaration was harmful (weak, clean or off-family channels), and never where the declaration was already close.
   - Its cost is a small-n bias at the β ≥ 0 boundary: a clean channel is read as slightly corrupted (mean loss 2% of VOI_clean at n = 10, 0.03% at n = 10⁴).
   - The headline worst-truth harm rates (0.35–1.00 at n ≤ 1,000) are dominated by this small, boundary-driven loss at β = 0 and should be read with these magnitudes.
3. **Clean-law error is absorbed into the inferred channel, and data do not remove it (C22).** This is the most consequential new finding.
   - It has the predicted direction: a too-persistent clean model (η̂ = 0.03) settles at β̂ ≈ 0.054 when there is no corruption, harming every replicate at n = 10⁴.
   - The saved small-data A4 (N = 10³) does the same at smaller scale (β̂ 0.020, harm 0.55).
   - Identification is therefore *relative to the clean law*. An inferred corruption rate is not evidence of corruption unless the clean law has been validated separately.
4. **Bayesian adaptation over a restricted family is not robust off-family (C23).** The grid posterior stays worse than ignoring corruption on the lag-1 point-mass truth even at n = 10⁴. Shrinkage toward q̄ helps only when q̄ is near the truth.
5. **Aggregate channel information is available from plain residual–action cross-moments (C24).** The moment estimator identifies the channel, though with 1.6–41× the MLE's regret at moderate n.

**What E3 adds to the thesis.**
- E2 established that composition is only as good as its declared prior.
- E3 shows when that prior can be learned (L ≥ 3, η < ½, a stable channel), at what sample cost, with which estimator (full-simplex likelihood, not a restricted Bayesian family), and under which precondition: a clean law validated separately from the channel.
- That turns composition from "flexible but fragile" into "flexible and self-correcting, conditional on a validated clean law."

**What it leaves unanswered about the MTM question.**
- E3 works entirely in aggregate. It does not test whether multi-view consistency flags *individual* harmful records beyond residual, rarity or dynamics signals.
- In this simulator, aggregate corruption information is already present in simple residual statistics (C24).
- Combined with phase 1 (learned folded and candidate routes are mostly equivalent), the evidence so far gives **no indication** that multi-view structure carries unique information here. It also does not rule out a per-record advantage, which has never been measured.

**Honest positioning.** The tools used (Bayesian mixture posteriors, EM for mixture weights, method of moments, empirical Bayes, robust declarations) are standard. Phases 1–E3 form a careful, decision-focused case study of known principles in a controlled setting. Novelty relative to the label-noise, mixture-proportion and robust-Bayes literatures has **not** been assessed and must not be claimed.

**Decision.** E3 answers its predeclared question; **experimental work stops here.** The next step needs a user research decision (see the status file):
- (a) the per-record MTM detection benchmark, which needs new neural training;
- (b) joint identification of clean persistence and channel, which targets absorption directly and is cheap and exact-law first;
- (c) a literature-positioning review;
- (d) consolidate and stop.
