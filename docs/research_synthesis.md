# Identifiability and Decision Costs of Action-Sign Corruption in Switching Trajectories

*MTI research synthesis: first thesis draft.* Opus 5.5 wrote the abstract, the contribution statement and §4. Sonnet 5.5 assembled §1, §3 and §5–9 from Opus's outline; Opus 5.5 reviewed the assembled draft (2026-10-05). Draft dated 2026-10-05, branch `exp/joint-identification`.
Claim-to-evidence map and response to the external review: [consolidation_review.md](consolidation_review.md).

## Abstract

Trajectory records used for learning and control can contain corrupted actions. In a switching system a corrupted action sign is hard to tell apart from a genuine change of regime. We study this confound in a controlled scalar switching model (S2) in which an exogenous channel flips at most one recorded action sign per record, and the target is the belief about the current regime that enters a one-step quadratic decision.

**Information and identification.**
- Exact inference routes that query a clean model through different conditional views coincide. Multi-view consistency therefore carries no population information beyond the declared corruption prior.
- The value of corruption awareness is strongly prior-dependent: from under 2% to 159% of the clean-history decision value across the priors and record lengths studied.
- We prove that, from unlabeled records with the physics known, the corruption channel is identified iff the record has L ≥ 3 actions when switching persistence is known. Persistence and channel are jointly identified iff L ≥ 4.
- Where identification fails (L ≤ 2 with persistence known; L ≤ 3 with it unknown), observationally equivalent parameters generally imply different decisions.
- Attribution to corruption is always conditional on the clean-dynamics family: a family that can reproduce the corrupted law makes it unidentifiable.

**Estimation.**
- A misspecified declared prior can be worse than ignoring corruption. No fixed prior is both Bayes-good and harmless.
- Given a correct clean law, full-simplex likelihood adaptation reaches near-oracle decisions with about 10⁴ unlabeled records.
- A clean-law error at fixed persistence is absorbed as spurious corruption. Jointly estimating persistence removes this absorption at L = 8 with 10⁴ records, where it is equivalent to knowing persistence.
- Supplying 1,000 extra clean records instead gives an equivalent result in 15 of 16 tested cells (more information, not an equal budget).
- A mild violation of the clean family left a small spurious corruption estimate that persisted over the measured sample sizes.

The methods are standard and the identification result is a structural specialisation of noisy population recovery. The contribution is the exact, decision-level characterisation and its measured costs.

## 1. Motivation and how the question changed

The project began from masked trajectory models (MTM, Wu et al. 2023), which answer many conditional queries about a trajectory with one model. The hope was that *consistency between views*, for example a repaired view and a sign-marginalised view of the same record, would flag corrupted records that harm a downstream decision.

The first result removed the population-level basis for that hope.
- **Exact route equivalence (C1).** Under the declared law, the candidate route (repair each possible flip) and the folded route (marginalise the sign) give the same aware belief μ₂ to ≤ 1.5e-14, and all compatibility residuals are exactly zero.
- **What agreement does and does not mean.** Agreement holds *because* both routes use the declared law. It does not certify that law.
- **What a learned gap measures.** A nonzero gap between learned heads measures the learners' inconsistency. For a converged learner it vanishes (C26). Whatever multi-view processing gains at the population level therefore reduces to the corruption prior that is declared or learned.

The work therefore followed that prior:
- how much it is worth (V₂, I_loss; §5.1);
- what a wrong one costs (§5.2);
- whether it can be learned from unlabeled records (§4, §5.3);
- how it is confounded with the clean dynamics (§4, §5.4–5.6).

The strongest completed evidence now concerns **identifiability and the decision costs of corrupted trajectory inference**, not detection. The per-record detection question that motivated the project was **not tested**. §3 defines its decision-relevant target, g(S), and §7 states it as an open boundary.

## 2. Central question and bounded contribution

**Question.** Under which assumptions can recorded action-sign corruption be distinguished from genuine switching dynamics, and what data and model knowledge are needed to estimate a belief that improves a specified downstream decision?

**Contribution** (bounded; each item is mapped to evidence in [consolidation_review.md](consolidation_review.md) §4):
1. **Information.**
   - Exact route equivalence.
   - The decomposition of naive excess cost into a recoverable part V₂ and an irrecoverable part I_loss.
   - V₂ is prior-specific.
2. **Identification** (§4).
   - A complete proposition: L ≥ 3 with persistence known; L ≥ 4 jointly.
   - Explicit, decision-relevant non-identification at L ≤ 3.
   - Attribution conditional on the clean family.
3. **Estimation costs** (§5).
   - When declared priors harm.
   - The sample cost of likelihood adaptation.
   - Absorption of clean-law error, and its removal by joint estimation.
   - What extra clean records buy.
   - One measured instance of family-error absorption.

**Not claimed:**
- a detector of harmful records;
- a poisoning defence;
- trained-policy or closed-loop benefit;
- unknown physics, other channels or real data;
- novel estimators.

## 3. Setting

**Model (S2).** For j = 0, …, L−1,

  s_{j+1} = ρ s_j + c m_j a_j + w_j,

- m_j ∈ {−1, +1} is a hidden symmetric Markov chain: m₀ uniform, Pr(m_{k+1} ≠ m_k) = η.
- Actions a_j are iid N(0, q_a) and process noise w_j iid N(0, q_w).
- s₀ is drawn from the stationary law N(0, (c²q_a + q_w)/(1 − ρ²)).
- Modes, actions, noise and s₀ are mutually independent. The decision uses the next mode m_L ([mathematical_specification.md](mathematical_specification.md) §1).
- Lead physics: ρ = 0.9, c = 1, q_a = 1, q_w = 0.02, λ = 0.1, η = 0.05, L = 8. Secondary physics: q_w = 0.5.

**Channel.**
- θ ∈ {none, 0, …, L−1}, with Pr(θ = j) = q_j, β = Σ q_j and Pr(none) = 1 − β.
- θ is independent of the modes, the actions and the noise.
- T_θ negates the recorded sign of action θ only. The physical trajectory is unchanged.
- A record is S = T_θ H, where H is the clean prefix.

**Access.** Estimators see records (s, a) and the physics (ρ, c, q_a, q_w, λ). η is supplied or estimated, as stated for each result. Hidden modes, θ and the probe are evaluation-only fields.

**Target and decision loss.**
- Target: μ₂(S) = E[m_L | S] under the true observation law.
- One-step quadratic decision. The regret of a belief μ̂ is κ(μ̂ − μ₂)², with κ = ρ²c²s_L²/(c² + λ).
- VOI_clean is the decision value of the clean history.
- Margin Δ = clip(0.1 V₂, 0.002 VOI_clean, 0.02 VOI_clean).

**Oracle and comparator hierarchy.** Results in §5 name their comparator explicitly.

| level | knows | role |
|---|---|---|
| clean-history oracle | the clean prefix H (extra information) | defines VOI_clean and I_loss; not attainable from S |
| true-law oracle | true η and q | reference for regret (regret 0) |
| EX-MLE | true η; estimates q | known-persistence estimator; the comparator for J-MLE |
| J-MLE | neither; estimates η and q jointly | joint estimator |
| fixed declarations | a declared prior | prior-risk comparisons (§5.2) |
| naive | nothing about corruption (β = 0) | reference for harm |

Equivalence to EX-MLE is **not** closeness to the true-law oracle; absolute regret is reported alongside every such comparison.

**Per-record harm.** For naive processing of record S, the expected harm is
h*(S) = κ Σ_θ p(θ|S)(μ(S) − μ(T_θS))² = g(S) + κ Var_θ(μ(T_θS) | S), with g(S) = κ(μ(S) − μ₂(S))².
- g(S) is the **recoverable** part: the expected benefit of replacing the naive belief by the aware belief.
- The variance term is **irrecoverable**: no processing of S removes it.
- E[g] = V₂ and E[κ Var] = I_loss, so naive excess = V₂ + I_loss.
- g(S) is not the corruption probability, and not the total clean-history harm.
- All statements are one-step.

## 4. Identification

### 4.1 Assumptions and the apparent edge pattern

Assume S2 with:
- c ≠ 0, q_a > 0, q_w > 0 and |ρ| < 1, with (ρ, c, q_a, q_w) known;
- m₀ ~ Unif{±1};
- mode edges Eₖ = 1{m_{k+1} ≠ m_k}, k = 0, …, L−2, iid Bern(η) with η ∈ [0, ½];
- s₀, the actions, the noise and the mode path mutually independent;
- θ independent of all of them, with q in the simplex Δ_L = {(q_none, q_0, …, q_{L−1}) ≥ 0, Σ = 1}.

Let d = L − 1 be the number of edges that enter the record. The transitions use m₀ … m_{L−1}. The final step to m_L enters only the target.

**Definitions.**
- Write σ ∈ {±1}^L for the flip vector (σ_θ = −1, all other coordinates +1; σ ≡ 1 if θ = none).
- The recorded actions are ã = σ ⊙ a, and the **apparent mode path** is m̃ = σ ⊙ m.
- The apparent edges are Ẽₖ = 1{m̃_{k+1} ≠ m̃_k} = Eₖ ⊕ F_{θ,k}, where the **footprint** F_θ ∈ {0,1}^d is:
  - none → 0;
  - θ = 0 → e₀;
  - θ = L−1 → e_{L−2};
  - interior j → e_{j−1} + e_j.
- Write F_q for the law of F_θ under q, and B_η for the iid Bern(η)^{⊗d} law.
- The apparent edge law is the XOR convolution P_{η,q} = B_η ∗ F_q.

### 4.2 Lemma 1: the record law identifies the apparent edge law

**Claim.** Under §4.1, the joint law of a record (s₀, ã, s₁, …, s_L) determines P_{η,q}. Conversely, it depends on (η, q) only through P_{η,q}.

**Proof.**
1. *Independence of the recorded actions.* Given θ, σ is fixed and ã = σ ⊙ a is iid N(0, q_a) because a is symmetric. So ã is independent of θ. Since a is also independent of m, ã is independent of (m̃, θ), and of s₀.
2. *Initial apparent mode.* m̃₀ = σ₀ m₀ is uniform and independent of (E, θ), because m₀ is uniform and independent of them.
3. *Residuals.* The residuals δ_k = s_{k+1} − ρ s_k are computable from the record (ρ known). Since c m_k a_k = c m̃_k ã_k:

   p(δ | s₀, ã) = Σ_{m̃ ∈ {±1}^L} P̃(m̃) Π_k N(δ_k; c m̃_k ã_k, q_w),

   with P̃(m̃) = ½ P_{η,q}(edges(m̃)), by steps 1–2.
4. *Identifiability.* For ã with all coordinates nonzero (probability 1), the 2^L mean vectors c m̃ ⊙ ã are pairwise distinct and the covariance q_w I is common. Finite mixtures of multivariate normal distributions are identifiable (Yakowitz & Spragins 1968). So the conditional law of δ given ã determines P̃, and hence P_{η,q}.
5. *Converse.* The law of (s₀, ã) does not depend on (η, q), so the record law depends on the parameters only through P̃.

**Degenerate cases.**
- c = 0 or q_a = 0: all components coincide and nothing is identified.
- q_w = 0: the components become distinct point masses and P̃ is still identified.
- The physics must be supplied. With unknown (ρ, c, q_w), Lemma 1 would need a joint mixture-identifiability argument that is not made here. ∎

### 4.3 Lemma 2: footprint injectivity

θ ↦ F_θ is injective iff L ≥ 3.
- For L ≥ 3 the footprints are 0, the two end singletons, and L − 2 distinct adjacent pairs, all distinct.
- For L = 2 (d = 1), both flips have footprint e₀.
- For L = 1 (d = 0), all footprints are empty. ∎

### 4.4 Proposition (identification of channel and persistence)

Under §4.1:

**(i) Persistence known, η < ½.** q is identified iff L ≥ 3.
- At L = 2 only β = F_q(e₀) is identified: the allocation between the two flips is not.
- At L = 1 nothing is identified: β is not.

**(ii) Persistence unknown, L ≥ 4.** (η, q) is identified on [0, ½) × Δ_L: if P_{η,q} = P_{η',q'} then η' = η and q' = q.

**(iii) L = 3.** For every η ∈ (0, ½), every q and every η' ∈ [0, η), define ν by 1 − 2η = (1 − 2η')(1 − 2ν). Then (η', q') with F_{q'} = B_ν ∗ F_q is a valid parameter, and P_{η',q'} = P_{η,q}.
- Their beliefs μ₂ differ in general.
- For L = 2 and L = 1, (η, q) is likewise not identified.

**(iv) η = ½.** P_{½,q} is uniform for every q. Nothing about the channel is identified, and V₂ = 0 because μ₂ ≡ 0.

**(v) Clean family not fixed.** Let 𝒢 be the allowed family of apparent-edge laws for the clean dynamics. If some G' ∈ 𝒢 equals G ∗ F_q for the true clean law G, then (G', none) and (G, q) are observationally equivalent.
- So attribution to corruption is identified only relative to a family that excludes such G'.
- This does **not** imply that departures from 𝒢 are undetectable: a lack of fit to every (η, q) can be visible in P.

**Proof.**
- **Fourier facts.** B_η has Walsh–Fourier coefficients B̂_η(A) = (1 − 2η)^{|A|} for A ⊆ [d], and convolution multiplies coefficients. For η < ½ all coefficients are nonzero, so convolution with B_η is invertible.

- **(i).** Given η < ½, invert B_η to recover F_q. By Lemma 2, F_q determines q iff L ≥ 3.
  - L = 2: F_q(e₀) = q₀ + q₁ = β and F_q(0) = 1 − β; the split q₀ : q₁ is free.
  - L = 1: F_q ≡ δ₀, so β is free.

- **(ii).** Suppose B_η ∗ F = B_{η'} ∗ F', with F = F_q and F' = F_{q'}, and η ≠ η'. Swapping the labels if needed, take η' < η < ½.
  1. Define ν ∈ (0, ½) by 1 − 2η = (1 − 2η')(1 − 2ν). Then B_η = B_{η'} ∗ B_ν, since the coefficients multiply.
  2. Hence B_{η'} ∗ (B_ν ∗ F) = B_{η'} ∗ F'. Inverting B_{η'} gives F' = B_ν ∗ F.
  3. Since 0 < ν < ½, B_ν gives positive mass to every pattern in {0,1}^d. So B_ν ∗ F, a mixture of shifted copies of B_ν, has full support.
  4. But F' is a footprint law, supported on at most L + 1 patterns, and 2^{L−1} > L + 1 for L ≥ 4. This is a contradiction, so η' = η.
  5. Then (i) gives q' = q.

- **(iii).** At L = 3 the footprints are 00, 10, 11 and 01: a bijection onto {0,1}². So every distribution on {0,1}² is F_{q'} for exactly one q'.
  - The construction in (iii) therefore gives a valid q'.
  - B_{η'} ∗ B_ν ∗ F_q = B_η ∗ F_q.
  - The beliefs differ in general: μ₂ = E[m_L | S] = (1 − 2η) E[m̃_{L−1} σ_{L−1} | S], and the posterior split of apparent edges into switches E and footprint F differs between the two parameterisations.
  - Numerically (`experiments/check_joint_identification.py`; η = 0.05 → 0.03, (β, α) = (0.2, 0.5), q_w = 0.5): the record log-likelihoods agree to 5e-15 and the beliefs differ by up to 0.21.
  - L = 2 and L = 1 follow from (i) together with the freedom in η.
  - Not every point is ambiguous: at η = 0 there is no η' < η branch.

- **(iv).** B̂_½(A) = 0 for every A ≠ ∅, so P is uniform. Also μ₂ = (1 − 2η)(…) = 0.

- **(v).** This is immediate from Lemma 1: the record law depends on the clean dynamics and the channel only through the apparent-edge law. ∎

### 4.5 What is proved, what is local, what is numerical

| statement | status |
|---|---|
| Lemma 1, Lemma 2, Proposition (i)–(v) | **analytic, proof complete** (assumptions of §4.1) |
| Profile Fisher information for η: 0.21–0.51 of I_η at L = 4, 0.61–0.81 at L = 8, ≤ 3e-13 at L = 3 (E4 stage 0) | *local*: precision of estimation near the truth, not part of the proof |
| Pattern-level LP and record-level KL separation at the tested parameters (stage 0) | *numerical*: consistent with the proposition |
| Sample costs of estimators (§5) | *measured* |

**Why the threshold is L = 4.** The single-flip footprints fill the edge-pattern cube exactly when L ≤ 3, because L + 1 ≥ 2^{L−1}. A pattern that no single flip can produce exists exactly when L ≥ 4, for example an isolated interior edge. Its frequency pins the persistence.

**Relation to prior work.**
- Lovett & Zhang (2017, §1.1) note that with an unknown noise rate the parameters of noisy population recovery are not identifiable in general. Their single-bit example is our L = 2 case. Their Theorem 1 returns a statistically close proper representation, not the true parameters.
- With the support *known and proper*, the unknown rate is identified, by the argument in (ii). That is the specialisation used here.
- HMM identifiability theorems cover the clean switching model but not a superposed record-level channel. Examples: Alexandrovich et al. 2016, Theorem 1 (full-rank ergodic transitions, distinct emissions, 2K + 1 observations); Allman et al. 2009, Theorem 6 (generic).

## 5. Empirical results

All results: S2, supplied physics, the exogenous single-flip channel, one-step regret. Margins and comparators are as in §3. Sources are listed in [consolidation_review.md](consolidation_review.md) §4. Uncertainty is stated separately for replicates and for evaluation Monte Carlo.

### 5.1 The value of awareness is prior-specific (oracle; C2–C4, C10)
- Naive excess = V₂ + I_loss exactly (C2).
- In the primary cell (L = 8, q_w = 0.5, β = 0.2), V₂ = 0.0712 ± 0.0031 against VOI_clean = 4.007 ± 0.039: **1.8%**. At L = 8, q_w = 0.02, β = 0.5 it is 13%, and at L = 32 below 1.6% (the L contrast is confounded with the per-position rate β/L). With corruption concentrated on the last action it rises to 27–159% ([oracle report](reports/oracle_v1_report.md) §3–4, §11–13).
- Awareness costs on clean records and after natural mode switches (C10). The net gain is on the mixture.

### 5.2 Prior risk (E2, E2b; C15–C19, H-loc)
- **Per truth.** A wrong declared prior can be worse than ignoring corruption (C15).
  - Understating the rate never hurts: proved per record when the location prior is right, with 0 violations found.
  - Flatter location priors never hurt on the screened grid.
  - Overstating a weak channel beyond about 2–3×, or concentrating location mass where corruption is diffuse, does hurt.
  - Harm tracks over-declared lag-0 mass (AUC 0.99, descriptive; H-loc).
- **Average over a prior** (E2b, a 30-channel uncertainty set). Composition with the mean prior is the Bayes rule (C18, a known result).
  - It cut mean regret by 80–87%, but lost to ignoring corruption on 9–10 of the 30 channels.
  - A prior that harms no channel keeps about ⅓ of the gain.
  - There is no free robust prior (C19). Average and per-channel statements are different criteria.
- Better estimators implement a wrong prior more faithfully (C16), and aggregate regret hides transfers between clean and corrupted records (C17).

### 5.3 Learning the channel (E3; C20–C24)
- **Identification controls.** Regret does not shrink with n at L = 1 (rate unidentified) or L = 2 (allocation unidentified), as the Proposition predicts. At L = 8 with the true η known, β̂ = 0.198 for β = 0.2 at n = 10⁴.
- **Full-simplex likelihood adaptation (C21).** It is near-oracle by n = 10⁴ on every L = 8 truth, retaining ≥ 98% of the achievable benefit wherever that benefit is not negligible. It beats the fixed mean-prior declaration from n = 10 where that declaration harmed. Its cost is a small boundary bias at β = 0 (mean loss 2% of VOI_clean at n = 10, 0.03% at 10⁴).
- **Restricted grids (C23).** A grid posterior stays worse than ignoring corruption on an off-grid truth even at n = 10⁴. This is misspecification, not a failure of Bayesian adaptation as such.
- **Moments (C24).** Plain residual–action moments identify the channel, at 1.6–41× the MLE regret.
- **Absorption (C22).** With η fixed at a wrong 0.03 (truth 0.05) and no corruption, the inferred channel settles at β̂ ≈ 0.054 and every replicate is harmed at n = 10⁴. More data do not remove it.

### 5.4 Joint estimation removes within-family absorption (E4; C27)
Comparators: EX-MLE (η known, q estimated) and the η = 0.03 arm. Replicates: R = 20 for n ≤ 3 × 10⁴; **R = 10 at n = 10⁵**. Replicate intervals are 90% paired t-intervals on one fixed evaluation set (20,000 prefixes).

**Table A. Primary contrasts at n = 10⁴ (regret in units of VOI_clean; true-law oracle regret = 0; R = 20)**

| physics | L | truth | Δ | EX-MLE | J-MLE | ETA⁻ (η = 0.03) | J − EX [90% CI] | ETA⁻ − J [90% CI] | J vs EX | ETA⁻ vs J |
|---|---|---|---|---|---|---|---|---|---|---|
| lead | 8 | T0 | 0.0020 | 0.00008 | 0.00015 | 0.0285 | +0.0001 [+0.0000, +0.0001] | +0.0283 [+0.0270, +0.0296] | equivalent | J better |
| lead | 8 | T3 | 0.0200 | 0.00008 | 0.00015 | 0.0129 | +0.0001 [+0.0000, +0.0001] | +0.0127 [+0.0124, +0.0130] | equivalent | equivalent |
| lead | 8 | T4 | 0.0200 | 0.00025 | 0.00037 | 0.0173 | +0.0001 [+0.0000, +0.0002] | +0.0170 [+0.0162, +0.0178] | equivalent | equivalent |
| lead | 8 | T6 | 0.0200 | 0.00010 | 0.00011 | 0.0138 | +0.0000 [-0.0000, +0.0000] | +0.0137 [+0.0133, +0.0141] | equivalent | equivalent |
| lead | 4 | T0 | 0.0020 | 0.00012 | 0.00063 | 0.0293 | +0.0005 [+0.0002, +0.0008] | +0.0286 [+0.0274, +0.0299] | equivalent | J better |
| lead | 4 | T3 | 0.0200 | 0.00005 | 0.00032 | 0.0126 | +0.0003 [+0.0002, +0.0004] | +0.0123 [+0.0120, +0.0125] | equivalent | equivalent |
| lead | 4 | T4 | 0.0200 | 0.00018 | 0.00034 | 0.0171 | +0.0002 [+0.0000, +0.0003] | +0.0168 [+0.0161, +0.0175] | equivalent | equivalent |
| lead | 4 | T6 | 0.0200 | 0.00005 | 0.00026 | 0.0132 | +0.0002 [+0.0001, +0.0004] | +0.0129 [+0.0126, +0.0132] | equivalent | equivalent |
| sec | 8 | T0 | 0.0020 | 0.00026 | 0.00038 | 0.0287 | +0.0001 [+0.0000, +0.0002] | +0.0283 [+0.0270, +0.0296] | equivalent | J better |
| sec | 8 | T3 | 0.0120 | 0.00032 | 0.00043 | 0.0199 | +0.0001 [+0.0000, +0.0002] | +0.0195 [+0.0190, +0.0201] | equivalent | J better |
| sec | 8 | T4 | 0.0200 | 0.00016 | 0.00024 | 0.0232 | +0.0001 [+0.0000, +0.0002] | +0.0230 [+0.0223, +0.0236] | equivalent | J better |
| sec | 8 | T6 | 0.0113 | 0.00017 | 0.00028 | 0.0213 | +0.0001 [+0.0001, +0.0002] | +0.0210 [+0.0207, +0.0214] | equivalent | J better |
| sec | 4 | T0 | 0.0020 | 0.00014 | 0.00077 | 0.0229 | +0.0006 [+0.0002, +0.0011] | +0.0222 [+0.0212, +0.0231] | equivalent | J better |
| sec | 4 | T3 | 0.0137 | 0.00021 | 0.00101 | 0.0124 | +0.0008 [+0.0005, +0.0011] | +0.0114 [+0.0110, +0.0119] | equivalent | equivalent |
| sec | 4 | T4 | 0.0200 | 0.00029 | 0.00136 | 0.0136 | +0.0011 [+0.0006, +0.0015] | +0.0123 [+0.0115, +0.0130] | equivalent | equivalent |
| sec | 4 | T6 | 0.0116 | 0.00021 | 0.00154 | 0.0132 | +0.0013 [+0.0008, +0.0019] | +0.0117 [+0.0109, +0.0125] | equivalent | inconclusive |

- **Primary contrast (P1).** At L = 8, n = 10⁴, β = 0 (truth T0), J-MLE minus EX-MLE is +0.00007 [0.0000, 0.00014] VOI (lead) and +0.00011 [0.00002, 0.00020] (secondary), against Δ = 0.002. J-MLE beats the η = 0.03 arm by 0.0283 VOI [0.0270, 0.0296] in both physics.
- **Evaluation-set uncertainty** (recomputed in `experiments/check_e4_eval_se.py`; recomputed mean regrets equal the saved ones exactly):
  - the prefix-level SE of J − EX is at most 4 × 10⁻⁵ VOI (lead) and 7 × 10⁻⁵ (secondary) per replicate, so two orders below Δ;
  - the SE of ETA⁻ − J is 0.0012–0.0019 VOI per replicate, so about 15 times smaller than the 0.0283 effect.
- **L = 3 (negative control).** J-MLE is worse than EX-MLE at every n in every cell except secondary T3 at n ≥ 10⁴ (inconclusive). η̂ drifts below 0.05 along the flat ridge (median at n = 10⁴: 0.004 lead T0, 0.005 lead T3, 0.0015 secondary T0), and β̂ settles near 0.26 on lead T3 (truth 0.2). Regret does not vanish: lead T3 is 0.054, 0.058, 0.056 VOI at n = 10³, 10⁴, 10⁵ against Δ = 0.02. This matches Proposition (iii).
- **Crossover** (smallest n from which J-MLE stays equivalent to EX-MLE). For corrupted truths it is 300 (the smallest n run) except secondary L = 4 (1,000–3,000). With no corruption it is 3,000 (lead L = 8), 1,000 (secondary L = 8) and 10⁴ (L = 4), because the margin is only 0.002 VOI there.
- **Equivalence is not oracle accuracy.** Mean regret relative to Δ (Table B): at n = 300 the secondary-physics L = 4 cells have J-MLE regret of 1.8–3.9 Δ, against 0.3–1.8 Δ for EX-MLE; the lead cells are 0.2–0.5 Δ. By n = 10⁴ J-MLE is at about 0.13 Δ or less in every cell (largest 0.132).
- **Calibration.** Stage-0 profile Fisher information predicted the measured SD of η̂ on the interior truths T3 and T6: 23 of 24 ratios lie in [0.7, 1.4]. The exception is lead L = 8 T6 at n = 10⁵ (ratio 1.68, R = 10). This is a planning aid, not a finite-sample guarantee.

**Figure.** [fig_e4_regret.png](reports/fig_e4_regret.png) shows regret against n for EX-MLE, J-MLE and the η = 0.03 arm.

**Table B.** (J-MLE and EX-MLE regret relative to Δ.)

| physics | L | truth | J-MLE n=300 | EX-MLE n=300 | J-MLE n=10⁴ | EX-MLE n=10⁴ |
|---|---|---|---|---|---|---|
| lead | 4 | T3 | 0.41 | 0.22 | 0.016 | 0.003 |
| lead | 4 | T4 | 0.26 | 0.05 | 0.017 | 0.009 |
| lead | 4 | T6 | 0.49 | 0.13 | 0.013 | 0.003 |
| lead | 8 | T3 | 0.34 | 0.24 | 0.007 | 0.004 |
| lead | 8 | T4 | 0.18 | 0.10 | 0.019 | 0.013 |
| lead | 8 | T6 | 0.24 | 0.22 | 0.005 | 0.005 |
| sec | 4 | T3 | 2.73 | 0.88 | 0.074 | 0.016 |
| sec | 4 | T4 | 1.83 | 0.28 | 0.068 | 0.014 |
| sec | 4 | T6 | 3.90 | 1.75 | 0.132 | 0.018 |
| sec | 8 | T3 | 1.11 | 0.78 | 0.036 | 0.027 |
| sec | 8 | T4 | 0.44 | 0.24 | 0.012 | 0.008 |
| sec | 8 | T6 | 0.87 | 0.48 | 0.024 | 0.015 |

### 5.5 Extra clean data (E4; CV arms)
The CV arm estimates η from n_c separate **clean** prefixes and then fits the channel on the same corrupted records. It therefore has *more information*, and this is not an equal-budget comparison.

**Table C. Clean-set comparison at n = 10⁴ (cells with L ∈ {4, 8}: 16 cells; class of CV relative to J-MLE)**

| clean prefixes n_c | equivalent | inconclusive | CV worse |
|---|---|---|---|
| 30 | 4 | 5 | 7 |
| 100 | 10 | 3 | 3 |
| 1000 | 15 | 1 | 0 |

- CV-1000 is equivalent to J-MLE in 15 of 16 cells (one inconclusive: secondary L = 4 T0) and never worse. CV-100 is equivalent in 10 of 16 and CV-30 in 4 of 16. The spread of η̂_cv over replicates is 0.012–0.040 (n_c = 30), 0.006–0.025 (100) and 0.002–0.007 (1,000).
- At L = 3 the CV arms beat J-MLE, because clean records have no channel confound.
- Joint estimation therefore matters when no clean records exist (L ≥ 4). When clean prefixes exist they add information.
- Estimating η from clean data is **parameter estimation within an assumed family**. It does not validate the family (§6).

### 5.6 One family violation (E4 F1; C28)
Record-heterogeneous persistence: η_r ∈ {0.02, 0.08} with equal probability (marginal rate 0.05), L = 8, lead physics, no corruption. The oracle is the exact mixture posterior.

**Table D. F1 (L = 8, lead physics, no corruption; R = 10) against the in-family control (R = 20): inferred β̂ and regret / VOI_clean**

| arm | n | F1 β̂ | in-family β̂ | F1 regret/VOI | in-family regret/VOI | Δ/VOI |
|---|---|---|---|---|---|---|
| EX-MLE | 10000 | 0.0071 | 0.0036 | 0.0007 | 0.00008 | 0.0020 |
| EX-MLE | 30000 | 0.0062 | 0.0020 | 0.0007 | 0.00005 | 0.0020 |
| J-MLE | 10000 | 0.0102 | 0.0045 | 0.0007 | 0.00015 | 0.0020 |
| J-MLE | 30000 | 0.0099 | 0.0026 | 0.0007 | 0.00009 | 0.0020 |

CV-1000 on F1-T0 (clean prefixes from the same mixture): n=10000: β̂ 0.0102, regret/VOI 0.0009; n=30000: β̂ 0.0100, regret/VOI 0.0009

- On this truth, β̂ is about 0.010 for J-MLE (and for CV-1000) at n = 10⁴ and 3 × 10⁴. The in-family value falls from 0.0045 to 0.0026 over the same n. So the estimate is persistent over the measured n, and clean prefixes drawn from the same mixture do not remove it.
- The regret, 0.0007 VOI, is below Δ = 0.002.
- A proposed explanation is the excess rate of adjacent mode-edge pairs, Var(η_r) = 0.0009, which is exactly what an interior flip produces. This explanation was **not tested**. F1 does not establish asymptotic bias, a dose–response law, or failure under general non-Markov dynamics.

### 5.7 Learned estimators (pilot, E1, E2; C7, C12–C14)
- **Routes.** Shared learned models queried by folded or candidate routes show no general advantage of either (14 equivalent / 4 inconclusive / 2 material).
- **Composition vs direct estimation.** At a matched backbone and loss family, composition vs density-direct was 6 better / 4 equivalent / 4 inconclusive / 0 worse (14 cells; 8 meet the precision target). The concentrated-prior advantage was confounded with sparse training coverage. In E2, a coverage-trained direct estimator matched composition at N = 10⁵ (C12b withdrawn), while composition stayed materially better at N = 10³.
- **Structured estimators.** A fitted, correctly structured HMM is near-oracle (C14), so nothing here shows a need for neural inference.
- **Scope.** All learned results use physics-informed features at L = 8 (C11).

## 6. What inferred corruption means

- **Relative to a clean model.** An estimated corruption rate is a statement about the data *given a clean-dynamics family*.
  - If the family is right, the channel is identified (L ≥ 3 with η known; L ≥ 4 jointly; Proposition (i)–(ii)).
  - If the allowed family can reproduce footprints, attribution is not identified (Proposition (v)).
  - The decision cost of the L ≤ 3 ambiguity depends on the truth: material at β = 0 (1.6–3.0% of VOI_clean against Δ = 0.2%), below the margin at β = 0.2–0.35.
- **Detection versus attribution.** That attribution is not identified does not mean every family violation is undetectable.
  - A violation can change the apparent-edge law in a way no (η, q) pair reproduces, and is then visible to a goodness-of-fit test on pattern statistics.
  - What cannot be done without more information is to say *which* mechanism, dynamics or corruption, produced a misfit that some (η, q) can reproduce.
  - No goodness-of-fit test was run, so detectability of F1 is untested.
- **Clean data.** Clean records carry no channel confound. They are the right input for estimating persistence and for checking the family.
  - Fitting a wrong family more precisely does not repair it: in F1, CV-1000 also absorbs (β̂ ≈ 0.010).
  - Extra clean prefixes matched J-MLE in 4 / 10 / 15 of 16 cells for n_c = 30 / 100 / 1,000 (Table C). This is a parameter-estimation result within the assumed family, with more information than J-MLE.
- **Practical reading for this model class** (each item inherits the scope above):
  1. Understate rather than overstate corruption. Do not over-declare corruption of the most recent actions.
  2. With unlabeled corrupted records (L ≥ 3), estimate the channel by full-simplex likelihood rather than a restricted Bayesian grid.
  3. With L ≥ 4 and no clean records, estimate persistence jointly. With clean records, use them for persistence *and* for checking the model family.
  4. Treat an inferred corruption rate as evidence of corruption only relative to a clean family that cannot mimic footprints.
  5. Do not expect consistency between views to detect harm beyond what the posterior and residual statistics provide.

## 7. Limitations and open questions

**Scope.**
- S2, supplied physics, the exogenous single-flip channel, one-step quadratic decisions.
- The first-order symmetric Markov clean family (except one mild F1 control).
- L ∈ {3, 4, 8} in E4, η = 0.05, two physics settings. Learned results: physics-informed features, L = 8 only.
- Fixed channel across records.

**Not established:**
- per-record detection of decision-harmful records by multi-view consistency (the original MTM question). The route gap is a finite-learner diagnostic (C26) and no experiment compared it with posterior, residual or dynamics baselines;
- a poisoning defence;
- trained-policy, closed-loop or real-data benefit;
- unknown physics, other corruption channels, other L, dose–response for family violations;
- novelty of any method.

**Open questions, in the order they would change the conclusions:**
1. *Family-violation dose–response:* how absorbed corruption and regret grow with the excess rate of short excursions. This is needed for any quantitative statement about the size of family-error absorption.
2. *Unknown physics:* whether the identification survives estimating (ρ, c, q_w). Lemma 1 would then need joint mixture identifiability.
3. *Per-record harm detection:* target g(S), with posterior, residual, rarity and dynamics baselines, and natural-switch controls.
4. *Goodness-of-fit* detectability of family violations.
5. *Policy relevance:* multi-step and closed-loop decisions, other channels, real data.

**Evidence gaps** (see [consolidation_review.md](consolidation_review.md) §6): Chen & Liu (1993) and Balsells-Rodas et al. (2024) were not read in full; the novelty search is limited.

## 8. References

Status: **full** = full text read; **abstract** = abstract or summary only; **standard** = standard result cited from general knowledge.

- Allman, E. S., Matias, C., Rhodes, J. A. (2009). Identifiability of parameters in latent structure models with many observed variables. *Ann. Statist.* 37(6A). arXiv:0809.5032. §6.1, Theorem 6. **full (§6.1)**.
- Alexandrovich, G., Holzmann, H., Leister, A. (2016). Nonparametric identification and maximum likelihood estimation for hidden Markov models. *Biometrika* 103(2). arXiv:1404.4210. Theorems 1–2. **full (§2)**.
- Balsells-Rodas, C., Wang, Y., Li, Y. (2024). On the identifiability of switching dynamical systems. *ICML*. arXiv:2305.15925. **abstract**.
- Berger, J. O. (1985). *Statistical Decision Theory and Bayesian Analysis*. **standard**.
- Chen, C., Liu, L.-M. (1993). Joint estimation of model parameters and outlier effects in time series. *J. Amer. Statist. Assoc.* 88(421):284–297. **abstract**.
- Kleijn, B. J. K., van der Vaart, A. W. (2012). The Bernstein–von Mises theorem under misspecification. *Electron. J. Stat.* **abstract**.
- Lovett, S., Zhang, J. (2017). Noisy population recovery from unknown noise. *COLT*, PMLR 65:1417–1431. §1.1, Theorems 1–3. **full**.
- Ramaswamy, H., Scott, C., Tewari, A. (2016). Mixture proportion estimation via kernel embeddings of distributions. *ICML*, PMLR 48. **full**.
- Teicher, H. (1963). Identifiability of finite mixtures. *Ann. Math. Statist.* 34. **standard**.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica* 50(1). **standard**.
- Wu, P. et al. (2023). Masked trajectory models for prediction, representation, and control. *ICML*. arXiv:2305.02968. **full (§3, §4.4–4.6)**.
- Xu, J. et al. (2025). Robust Decision Transformer. *ICLR*. arXiv:2407.04285. **full (§3.2)**.
- Yakowitz, S. J., Spragins, J. D. (1968). On the identifiability of finite mixtures. *Ann. Math. Statist.* 39. **standard**.

Further sources (label-noise transition estimation, offline-RL corruption robustness, masked and multi-view anomaly detection) are listed in [literature_review.md](literature_review.md) §6 with their read status.

## 9. Map to thesis chapters

| chapter | content | sections here |
|---|---|---|
| 1 | Introduction: motivation, central question, contribution | §1–2 |
| 2 | Setting and decision-theoretic targets | §3 |
| 3 | Information and identification | §4, §5.1 |
| 4 | Prior risk and adaptation | §5.2–5.3 |
| 5 | Clean-law / channel confounding | §5.4–5.6, §6 |
| 6 | Learned estimators | §5.7 |
| 7 | Limitations and future work | §7 |

**Appendices.**
- Proofs: §4.
- Protocols and amendments: `docs/e2_protocol.md`, `e3_protocol.md`, `e4_protocol.md` (including A1).
- Reproducibility: every run records its commit, dirty diff and untracked-file hashes (`results/*/provenance.json`); stage-1 E4 ran at `ea8bca4` plus the A1 patch.
