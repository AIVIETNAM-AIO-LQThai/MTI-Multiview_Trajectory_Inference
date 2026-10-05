# Literature positioning and next research decision

> **Status note (2026-10-05, consolidation).** This review precedes E4. Its recommendation was carried out. The proof of the L = 4 threshold and the classification of each contribution (known principle / structural corollary / empirical / possibly new) are now in [research_synthesis.md](research_synthesis.md) §4 and [consolidation_review.md](consolidation_review.md) §5. Where they differ, those documents supersede this one.

Written by Opus 5.5 (`claude-opus-5-5`) on 2026-10-05. Branch `exp/channel-identification`, base `3d20dac`.
Scope:
- The evidence through E3 ([findings_summary.md](findings_summary.md) v5, C1–C24).
- One derivation check: [check_joint_identification.py](../experiments/check_joint_identification.py), output in `results/derivation_checks/joint_identification.json`. It does no training and runs no estimators.

How to read source status:
- **Read** means the cited sections and theorems were read in full text.
- **Located** means the source was identified and its abstract or summary was read, but not its full text. No claim below depends on a located source beyond what its abstract states.

## 1. Recommendation (one direction)

**Joint identification of clean persistence and channel (E4), exact-law first, no neural training.** The protocol is in [e4_protocol.md](e4_protocol.md).

Why this direction:
- C22 (clean-law error absorbed into the inferred channel) is the finding that most limits every other positive result. It decides whether "inferred corruption" means anything.
- Its within-family part now has an exact answer that can be tested cheaply:
  - with η unknown, (η, q) is **not identified for L ≤ 3**: there is an exact, decision-relevant ridge;
  - it is separated at the pattern level for **L ≥ 4** on every tested truth (§4.3).
- What remains open is quantitative and falsifiable: how many records the separation costs, and whether joint estimation removes the C22 harm at practical n. A strong, cheap baseline already exists: validating η on a small clean set.

Why not the per-record MTM detection benchmark (§4.1–4.2):
- Under a learner that converges, the learned route gap vanishes, so it carries no population information.
- The harm-optimal score is a known function of the exact posterior, which a fitted HMM already approximates near-oracle (C14).
- A positive result would therefore show only that an imperfect learner's inconsistency correlates with harm, and it would need new neural training to show even that.
- That benchmark stays possible later. It is not the best next use of evidence.

Why not synthesis alone: the identification question has a clean, bounded test whose outcome would change C22's wording either way.

## 2. Comparison table

| Our claim / question | Closest prior result | Matching / differing assumptions | What we add | Evidence still needed |
|---|---|---|---|---|
| C20: channel identified from unlabeled prefixes for L ≥ 3 (η known), not at L = 1 (rate) or L = 2 (allocation) | Finite-mixture identifiability: Teicher 1963; Yakowitz & Spragins 1968 (a mixture of **known** components is identified iff the components are linearly independent). MPE: Ramaswamy, Scott & Tewari 2016, §2–3 (irreducibility, after Blanchard, Lee & Scott 2010 and Scott 2015; anchor set Def. 8; separability Def. 9) | **Differs from MPE.** Our components p₀(T_θ·) are fully known given the clean law, so no irreducibility or anchor condition is needed; the issue is linear dependence created by the sign symmetry (p₀(T₀S) = p₀(T₁S) at L = 2). In MPE one component is unknown. Matches the classical known-component case | The concrete dependence structure for the sign channel (L = 1, 2 counterexamples that matter for decisions), and the residual–action moment construction | None for the aggregate claim. Relabel C20 as an instance of known-component identifiability |
| C21: full-simplex MLE adaptation reaches near-oracle decisions | EM/NPMLE for mixing weights over known components: Laird 1978 (JASA); Redner & Walker 1984 (SIAM Rev.); empirical Bayes (Robbins 1956). Confident learning estimates a noise joint, then cleans (Northcutt, Jiang & Chuang 2021, JAIR) | Matches: known components, weights estimated by likelihood. Differs: our target is decision regret, not weight error | Measured sample cost to decision-level near-oracle (n ≈ 10⁴ at L = 8); boundary bias at β = 0 measured in regret units | None. Standard-method claim |
| C22: clean-law error absorbed into the inferred channel | Misspecified MLE converges to the KL-closest (pseudo-true) parameter (White 1982, Econometrica, Thms 2.2–3.2). Time-series outliers: misspecified dynamics and outlier effects are confounded and must be estimated jointly (Chen & Liu 1993, JASA; additive vs innovational outliers, Fox 1972, JRSS-B) | Matches: a pseudo-true channel compensates for a wrong clean law. Differs: the confounded parameters here are switching persistence and a structured flip channel, not ARMA coefficients and outlier sizes | Exact absorption sign (η̂ < η gives spurious β̂ > 0), and that within the Markov family the confound is an **exact ridge for L ≤ 3** and removable in principle for L ≥ 4 (§4.3) | E4: sample cost of joint estimation; behaviour under a clean law outside the family |
| Joint (η, q) identification (new, §4.3) | Noisy population recovery with **unknown** noise rate (Lovett & Zhang 2017, COLT; known-noise case: Dvir et al. 2012; Moitra & Saks 2013). HMM identifiability from consecutive observations (Allman, Matias & Rhodes 2009, AoS, HMM section; Alexandrovich, Holzmann & Leister 2016, Biometrika: full-rank transitions, distinct emissions) | Our apparent edge pattern = iid Bern(η) noise ⊕ a draw from a structured distribution supported on L+1 footprints. That is population recovery with unknown noise **and** a known small support. Lovett & Zhang treat unknown noise with general support (located; abstract only). The HMM results assume emissions differ across states; ours differ only through a sign that the channel can also flip | An exact minimum L (= 4), the reason (isolated interior edges cannot come from one flip), and a decision-relevant ridge at L = 3 | Record-level (not only pattern-level) separation and its Fisher information; finite-sample behaviour (E4) |
| C18: mean prior is Bayes-optimal under prior uncertainty; C19: no free robust prior | Bayes decision under a prior over priors reduces to the mixture prior; Γ-minimax and robust Bayes (Berger 1985, *Statistical Decision Theory and Bayesian Analysis*, §4.7) | Matches exactly (linearity of the posterior in the channel prior) | Measured transfer pattern: which channels the Bayes prior harms (over-declared lag-0 mass) | None. Relabel C18 as a known result |
| C23: restricted-grid posterior fails off-family | Bayesian posteriors under misspecification concentrate at the KL-closest family member (Berk 1966, AMS; Kleijn & van der Vaart 2012, EJS, Thm 2.1) | Matches | A decision-level size: worse than ignoring corruption at n = 10⁴ on the lag-1 point mass | None. Relabel as a known result. Correctly specified PP is Bayes-optimal; only the restricted grid fails |
| C24: aggregate channel information is in residual–action moments | Method of moments for mixtures and HMMs (spectral and moment methods; e.g. the Allman et al. 2009 Kruskal-based argument uses low-order joint moments) | Matches in spirit | The explicit pair formula E[z̃_i z̃_j] = c²q_a²(1−2η)^{\|i−j\|}(1−2(q_i+q_j)) | None |
| C1: exact candidate/folded route equivalence | MTM (Wu et al. 2023, ICML, §3: random masking; §4.4–4.6: one model queried under different masks for prediction, representation and control). MTM does **not** consider corruption, anomaly detection or cross-mask consistency (read) | MTM motivates querying one model under several conditionals; it makes no consistency claim | The exact identity G = M under the declared law, and therefore that consistency carries **no population information** (§4.1) | None |
| Original MTM question: does multi-view consistency flag decision-harmful records? | Robust offline RL under data corruption: RDT (Xu et al., ICLR 2025, arXiv 2407.04285; §3.2.1 embedding dropout, §3.2.3 iterative data correction with a z-score threshold, Gaussian-weighted learning; evaluation by return only). RIQL (Yang et al., ICLR 2024). Corruption-robust offline RL bounds, Ω(dε) (Zhang et al., AISTATS 2022). Per-example noise detection: confident learning (Northcutt et al. 2021). Masked and multi-view anomaly detection (located only: arXiv 2605.30046, 2607.19032) | RDT detects and corrects corrupted **actions** using the model's own prediction error. That is a per-record residual score, without a harm target and without validation against an oracle. None of these define decision harm per record or compare against an exact posterior | A harm oracle h*(S) (§4.2), and the proof that the route gap vanishes at the population level | Untested. Would need new neural training (§4.1–4.2); deprioritised |

## 3. What the literature changes in our claims

- C18, C20, C21, C23 and C24 are **instances of known theory**:
  - known-component mixture identifiability (C20);
  - EM/NPMLE for mixing weights (C21);
  - Bayes decisions under prior uncertainty (C18);
  - misspecified-posterior concentration (C23);
  - moment identification (C24).
  
  What they contribute is the decision-level measurement in this setting, not the principle. The ledger is relabeled accordingly.
- C22 is an instance of the pseudo-true parameter, and of the classical model/outlier confound in time series. The exact form for the switching sign channel (direction, ridge at L ≤ 3, minimum L = 4 for joint identification) was not found in the sources read. It is a *possible* small contribution, conditional on a fuller search of the switching-model outlier literature.
- RDT is the closest application predecessor for per-record detection. It already uses model prediction error to flag corrupted actions. Any MTM detection claim must therefore beat a residual score of that kind, not only rarity.

## 4. Conceptual issues

### 4.1 Exact consistency vs learned detection

Under the declared law the routes are equal (C1): every compatibility residual C_j is exactly 0. For learned heads the gap decomposes as

  Ĝ − M̂ = Ẑ⁻¹ Σ_j q_j Ĉ_j / (1 − r̂_j),

where Ĉ_j is the learned compatibility residual of view j. What this means:
- **What a nonzero gap measures.** The gap is the learner's internal inconsistency between heads that should agree. It is a property of the **estimator**, not of the record's law. As data and compute grow and the heads converge, Ĉ_j → 0 and the gap vanishes, for clean and corrupted records alike.
- **Falsifiable mechanism.** Corrupted records are off the training distribution, so heads extrapolate differently there. If so, the gap correlates with corruption or harm *at finite N*, and the correlation should **decay** with N and model capacity.
  - Falsifier: the gap's incremental detection value (beyond the baselines) does not shrink from N = 10³ to 10⁵. That would indicate a systematic architectural bias rather than an estimation-error signal.
  - Either way the signal is bounded by what an exact posterior already provides.
- **Behaviour with data and compute.** The gap's value is largest exactly where the learner is worst. At N = 10⁵ the fitted HMM is already near-oracle (C14), so the window in which the gap could be useful is small data or a misspecified learner. That is a statement about learners, not about multi-view information.

### 4.2 Detection target and oracle

Let μ(S) be the naive belief, μ(T_θS) the clean-prefix belief under hypothesis θ, and μ₂(S) = Σ_θ p(θ|S) μ(T_θS) the aware belief. In the one-step quadratic decision, a belief b(S) has excess cost κ(b − μ_H)² against the clean-history belief μ_H.

- **Realised harm** of naive processing: h(S, θ) = κ(μ(S) − μ(T_θS))². It is 0 when θ = none, and needs θ, so it can be used for evaluation only.
- **Expected-harm oracle:**

  h*(S) = E[h | S] = κ Σ_θ p(θ|S)(μ(S) − μ(T_θS))² = κ(μ(S) − μ₂(S))² + κ Var_θ(μ(T_θS) | S).

  - **Recoverable part:** g(S) = κ(μ(S) − μ₂(S))². It is exactly the expected benefit of switching that record to the aware belief, since E[h − κ(μ₂ − μ(T_θS))² | S] = g(S).
  - **Irrecoverable part:** κVar(·|S). No processing of S removes it.
  - E[g] = V₂ and E[κVar] = I_loss, so the decomposition reproduces C2 per record.
- **What a detector should rank by.** If the action triggered by detection is "use the aware belief", the decision-relevant target is g(S), not the corruption posterior p(θ ≠ none | S). A record can be confidently corrupted at a position that does not move the belief, and then it is harmless.
- **Avoiding circularity.**
  - g is computed from the same model a detector would be compared against, so it serves only as the **ceiling**.
  - Detectors are evaluated against the **realised** benefit h(S, θ) − κ(μ₂(S) − μ(T_θS))². That uses θ from the held-out evaluation set only, and is an unbiased estimate of g(S).
  - Natural switches near the end of the prefix, and difficult clean records with large residuals, are the controls that separate "harmful corruption" from "surprising but clean" (C10, C17).

### 4.3 Clean-model/channel confounding

**Existing theory.**
- Misspecified likelihoods converge to the pseudo-true parameter (White 1982), and posteriors concentrate there (Kleijn & van der Vaart 2012).
- Outlier effects and model parameters must be estimated jointly (Chen & Liu 1993).
- None of this says *when* joint estimation is identified for a given structure.

**Reduction.** Under S2 with the exogenous single-flip channel, a flip at position θ is observationally the same as a mode excursion of length one at θ. Hence the record law depends on (η, q) only through the law of the **apparent edge pattern** E ⊕ F_θ, where:
- E is iid Bern(η) over the L−1 mode edges;
- F_θ is the footprint of the flip: none → no edges; an end flip → one edge; an interior flip j → the adjacent pair (j−1, j).

Characters: E[(−1)^{Σ_A}] = (1−2η)^{|A|} · (q_none + Σ_j q_j (−1)^{|A ∩ F_j|}). The record law is a garbling of the pattern law, so equal pattern laws imply equal record laws.

**Counterexamples and conditions (η unknown, physics known, η < ½):**
- **L = 2:** one edge, so only η ⊕ β is identified.
- **L = 3:** every pattern in {0,1}² is some footprint, so for any η' < η the channel F' = Bern(ν)^{⊗2} ∗ F, with (1−2η) = (1−2η')(1−2ν), is a valid channel giving the same record law.
  - This is an **exact ridge**: the clean model is more persistent, and the inferred corruption is larger.
  - Verified at the record level for η = 0.05 → η' = 0.03, (β = 0.2, α = 0.5), q_w = 0.5:
    - max record log-likelihood difference 5e-15;
    - β' = 0.230;
    - beliefs differ by up to 0.21;
    - mean regret of the ridge belief 0.028, which is **5% of V₂** = 0.565.
  - Confounding at L = 3 is therefore real.
  - *Correction after E4 stage 0 (Opus, 2026-10-05).* Whether it matters for decisions depends on the truth:
    - **β = 0 (T0):** at η' = 0.03 the ridge costs 3.0% (lead) and 1.6% (secondary) of VOI_clean, against a margin of 0.2%. It is material: this is the absorption case.
    - **β = 0.2–0.35:** the cost is 0.8–1.4% of VOI_clean, which is **below** the margin (10% of V₂, clipped at 1.2–2% of VOI_clean).
    - The "5% of V₂" above is likewise below that margin.
- **L ≥ 4:** global identification holds. See the proof in [research_synthesis.md](research_synthesis.md) §4 (Proposition (ii)): a persistence η' < η would force a channel B_ν ∗ F with full support, which a footprint law cannot have once 2^{L−1} > L + 1. *(This replaces the earlier "generic root" argument and the grid-based pattern check, which were weaker; the LP in `experiments/check_joint_identification.py` remains a numerical consistency check.)*
  - **Minimum L = 4:** the single-flip footprints fill the edge-pattern cube exactly when L ≤ 3.

**Weak information.**
- Near η, the pattern-level distance grows slowly. At |η' − η| = 0.005: TV 4.5e-3 at L = 4, 1.9e-2 at L = 8.
- This is an *upper* bound on record-level distinguishability (the noise q_w garbles patterns).
- So L = 4 is identified but weakly. Finite-sample stability must be measured, not assumed (E4 stage 0: profile Fisher information).

**Does joint estimation resolve the confound, or shift it?** It **shifts** it.
- Joint (η, q) estimation removes the η-error part of C22 within the first-order symmetric Markov family.
- It cannot validate the family. Any clean law that produces excess length-one excursions is observationally equivalent to interior corruption. Examples: a non-geometric dwell time, or record-heterogeneous persistence.
- The general counterexample: if the clean family contains G ∗ F for the true G and F, the channel is unidentified.
- So the condition moves from "η known" to "the clean switch process is first-order Markov with the supplied physics". This is **within-family correction, not family validation**. Family validation needs information that corrupted records cannot provide, such as a clean validation set or external knowledge of the dynamics.

### 4.4 Bayesian and estimator claims, stated precisely

- **PP correctly specified vs restricted grid.**
  - A posterior predictive over a prior that contains the truth is Bayes-optimal for that prior (C18 logic).
  - E3's PP failure (C23) is a failure of the **restricted grid**, i.e. misspecification, not of Bayesian adaptation as such.
- **MLE is not universally harmless.**
  - It has a small-n boundary bias at β = 0 (C21).
  - It faithfully reproduces any clean-law misfit as spurious corruption (C22).
  - At L ≤ 2 it returns an arbitrary point on a non-identified set.
- **Mixture-average vs per-channel harm.**
  - Bayes optimality (C18) is a statement about the average over the prior. Per-channel harm (C19, E3 Q2) is a different criterion: the Bayes rule can harm specific channels while minimising the average.
  - Reports must keep the two apart. "Near-oracle" claims are per truth; "Bayes-optimal" claims are average.

## 5. Revised contribution statement

> In a controlled switching-dynamics model with an exogenous action-sign corruption channel, we characterise when composing a clean-trajectory model with a corruption channel improves one-step decisions:
>
> 1. **Exact multi-view routes are equivalent**, so multi-view consistency carries no population information. Any detection value of learned inconsistency is a finite-learner property.
> 2. Composition's value is **bounded by the declared prior**. The harm is predictable (over-declared recent corruption).
> 3. The channel is identified from unlabeled corrupted records for L ≥ 3 when the clean law is known, and adaptation then removes the prior risk. This is an instance of known-component mixture identifiability, measured at decision level.
> 4. **Clean-law error is absorbed into inferred corruption.** With unknown persistence, clean persistence and channel are jointly unidentified for L ≤ 3 (an exact, decision-relevant ridge) and separated for L ≥ 4 within the Markov family. Attribution to corruption is not identified against an allowed clean family that can reproduce the corrupted law (consolidation review D1; synthesis Proposition (v)).
>
> Items 1, 3 and the general form of 4 follow from known theory; the contribution is the exact decision-level characterisation in this setting and its measured sample costs. Item 4's L-threshold and the E4 sample-cost results are the candidates for novelty. That is pending a fuller search of the switching-model outlier literature.

Not claimed:
- a detection method;
- a poisoning defence;
- policy robustness;
- results on real data;
- a novel estimator.

## 6. Sources

Read:
- Ramaswamy, Scott & Tewari (2016). *Mixture proportion estimation via kernel embeddings of distributions.* ICML, PMLR 48. https://proceedings.mlr.press/v48/ramaswamy16.html. Sections read: §2–4, Defs. 8–9, Thms. 10–11.
- Wu, Majumdar, Stone, Lin, Mordatch, Abbeel & Rajeswaran (2023). *Masked trajectory models for prediction, representation, and control.* ICML. https://arxiv.org/abs/2305.02968. Sections read: §3, §4.4–4.6.
- Xu et al. (2025). *Robust Decision Transformer: tackling data corruption in offline RL via sequence modeling.* ICLR. https://arxiv.org/abs/2407.04285. Sections read: §3.2.1, §3.2.3 and the evaluation.

Located (abstract or summary only):
- Lovett & Zhang (2017). *Noisy population recovery from unknown noise.* COLT, PMLR 65. https://proceedings.mlr.press/v65/lovett17a.html
- Moitra & Saks (2013). *A polynomial time algorithm for lossy population recovery.* FOCS. Also Dvir, Rao, Wigderson & Yehudayoff (2012), *Restriction access*, ITCS.
- Allman, Matias & Rhodes (2009). *Identifiability of parameters in latent structure models with many observed variables.* Ann. Statist. https://arxiv.org/abs/0809.5032
- Alexandrovich, Holzmann & Leister (2016). *Nonparametric identification and maximum likelihood estimation for hidden Markov models.* Biometrika. https://arxiv.org/abs/1404.4210
- Kleijn & van der Vaart (2012). *The Bernstein–von Mises theorem under misspecification.* Electron. J. Statist.
- White (1982). *Maximum likelihood estimation of misspecified models.* Econometrica 50(1).
- Zhang et al. (2022). *Corruption-robust offline reinforcement learning.* AISTATS. https://arxiv.org/abs/2106.06630
- Yang et al. (2024). *Towards robust offline RL under diverse data corruption* (RIQL). ICLR. https://arxiv.org/abs/2310.12955
- Patrini et al. (2017), CVPR; Xia et al. (2019), NeurIPS (T-Revision); Li et al. (2021), ICML, https://arxiv.org/abs/2102.02400. These are transition-matrix estimation for label noise; anchor points vs no anchors.
- Northcutt, Jiang & Chuang (2021). *Confident learning.* JAIR. https://arxiv.org/abs/1911.00068
- Teicher (1963), Ann. Math. Statist.; Yakowitz & Spragins (1968), Ann. Math. Statist.

Classical, cited from standard knowledge and not re-read in this step:
- Laird (1978), JASA; Redner & Walker (1984), SIAM Rev.
- Robbins (1956); Berger (1985); Berk (1966).
- Fox (1972), JRSS-B; Chen & Liu (1993), JASA.
- Blanchard, Lee & Scott (2010), JMLR; Scott (2015), AISTATS.

Masked and multi-view anomaly detection: arXiv 2605.30046 and 2607.19032 were located but not read. They concern detection by reconstruction or cross-view disagreement in other domains, and no claim here depends on them.
