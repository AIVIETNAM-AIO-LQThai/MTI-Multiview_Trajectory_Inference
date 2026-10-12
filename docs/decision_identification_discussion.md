> *Provenance: written by Opus 5.5 on 2026-10-11 in a separate clone of `08dd58e`; imported unchanged on 2026-10-12 (body sha256 prefix 91c7dc4a412ce31a before this header). Superseded where [decision_identification_closeout.md](decision_identification_closeout.md) differs: the "V\*" column of Table B is the endpoint-pair lower bound ¼·diam² (certified equal to the minimax value on the evaluation measure only by the closeout), Q1 is restated around target-set geometry (L = 2 has a three-point least-favourable prior), Proposition B is an exact nesting statement, the crossed SE formula double-counts the interaction, and clean calibration collapses the ambiguity only at L ≥ 3.*

# Decision identification under the L ≤ 3 ambiguity: repository review and discussion with Codex

Opus 5.5 (`claude-opus-5-5`), 2026-10-11. Read-only review. This file is the only change to the working tree; nothing is committed, pushed or merged.
Companions: [research_synthesis.md](research_synthesis.md), [consolidation_review.md](consolidation_review.md).

## 0. Snapshot, checks and what was rerun

**Snapshot.**
- Reviewed `origin/exp/joint-identification` at `08dd58e86ed4d127004e356c3329b389356b1548` (parent `64e2234`). This is Codex's snapshot; the branch has no later commits.
- The review used a fresh clone, so local uncommitted work on your machine was not visible to me.

**Definitions against the code: no mismatch with the brief.**
- κ = ρ²c²s_L²/(c²+λ) is `decision.kappa`, and a* = −ρc s_L μ/(c²+λ) is `decision.a_star`.
- μ = (1−2η)·tanh(λ_L/2) includes the final m_{L−1} → m_L step (`inference.belief_from_tilde`), so μ₂(S) = E[m_L | S].
- Regret is Σ_θ w_θ κ(μ̂ − μ₂)² per clean prefix, averaged over 20,000 prefixes with all L+1 views (`run_e4.run_job`). It is therefore an exact channel expectation per prefix: **one prefix with its views is one evaluation unit.**
- One clarification that matters below: VOI_clean = mean κμ_H² is computed on clean prefixes of the *reference* latent law. It is not a function of the record law P.

**Reproduced (evaluation only, from saved estimates or exact laws; nothing fitted or trained).**
- Codex's ridge table, to all reported digits.
- Codex's shared-prefix SEs, to all reported digits.
- The saved P1 regrets, to a maximum relative difference of 1.9e-12.

**New exact calculations in this review.**
- The complete L = 3 identified sets for the E4 truths.
- The decision diameter, the minimax floor and the least-favourable prior.
- An η = ½ counterexample.
- Channel-invariant nuisance moments, and a trusted-outcome check.
- k-flip thresholds, by LP for k ≤ 3 and L ≤ 9.

**Tests.**
- I ran the torch-free test files: 84 passed. The 3 access tests need torch and were not run, and the reported full count of 114 was not re-run.
- `tests/conftest.py` rewrites `results/oracle_v1/test_max_errors.json` on every session. My run changed it, and I restored it with `git checkout` (correction 12).

**Scripts.** The verification scripts live outside the repository (sent alongside this document). Whether to add them under `experiments/` is your decision.

## 1. Independent audit of the progression

| stage | central conclusion | status | comment |
|---|---|---|---|
| Oracle (C1) | Candidate and folded routes give the same μ₂ under the declared law | algebraic identity, verified ≤ 1.5e-14 | Correctly read: multi-view consistency carries no population information beyond the corruption law. |
| Oracle (C2–C4, C10) | Naive excess = V₂ + I_loss; V₂ is 1.8%–159% of VOI_clean depending on the prior | identity + exact-law numbers in tested cells | Sound. |
| Learned pilot, E1, E2 (C7, C12–C14) | No general route advantage; composition vs direct is regime-dependent; fitted HMM is near-oracle | numerical, tested cells (L = 8, physics-informed features, 5 seeds) | Sound and correctly bounded; C12b withdrawal is right. |
| E2, E2b (C15–C19) | Wrong declared priors can harm; no free robust prior | numerical + per-record proof for rate understatement; C18 is known | Sound; average and per-channel criteria are kept apart. |
| E3 (C20–C24) | Channel identified iff L ≥ 3 with η known; full-simplex MLE near-oracle at 10⁴; absorption at fixed wrong η | C20 is a theorem (Prop. (i)); the rest numerical | Sound. |
| Identification §4 | Lemma 1, Lemma 2, Prop. (i)–(v) | theorem | I re-derived Lemma 1 (needs ã_j ≠ 0, a.s. true), Lemma 2, (i), (ii) and (v): correct. **(iii) is correct but incomplete** (§3). **(iv) needs a qualification** (§5.1). |
| E4 stage 1 (C27, C27b–c, CV) | J-MLE ≡ EX-MLE at L = 8, n = 10⁴; removes ETA⁻ absorption | numerical (P1 holds) | Sound. The L = 3 behaviour is structural, not drift (§4.5). |
| E4 F1 (C28) | One mild family violation absorbed, persistent over measured n | descriptive, one instance | Correctly bounded. |
| Consolidation | Bounded thesis: information, identification, estimation costs | — | Agreed, except the L ≤ 3 decision-materiality statement (§2–4). |

**Framing.** I agree with Codex. The strongest story is identification and decision costs under recorded-action corruption. A practical harmful-record detector, a poisoning defence and closed-loop gains are not established.

**Overlooked results.**
1. The L ≤ 3 non-identification has a *quantifiable decision floor*. It is material at corrupted truths, but the documents currently report it as immaterial.
2. The E4 L = 3 J-MLE regrets come from a deterministic selection of the bottom of the identified set (§4.5). They do not describe joint likelihood as such.

## 2. Verdicts on Codex's qualifications, and my additions

| point | verdict | reasoning / evidence | consequence |
|---|---|---|---|
| **R1.** "Material only at β = 0" holds only for the η′ = 0.03 comparison | **Accept, and strengthen** | Codex's table reproduces exactly: β′ = 0.23019 / 0.26377 / 0.27 and regret 0.01046 / 0.05720 / 0.07173 VOI at η′ = 0.03 / 0.005 / 0. Record laws agree to ≤ 5.3e-15 at both arc endpoints (stable log-sum-exp). The full identified set is **two-sided** for interior channels (§3). Its decision floor exceeds the margin in 5 of 6 corrupted cells (§4). | C25 wording and every copy of it must change (correction 1). Historical stage-0 numbers stay, labelled as the η′ = 0.03 point. |
| **R2.** The ridge counterexample | **Accept** | Equation B_{η′} ∗ F_{q′} = B_{0.05} ∗ F_q, access (records plus supplied physics) and normalisation (reference VOI_clean of the truth) are all as in E4. Only the normaliser needs a caveat: VOI_clean differs between compatible latent laws (§4.1). | Report raw cost or E_P[κ] alongside. |
| **R3.** literature_review §4.3, "excess length-one excursions ⇒ equivalence" | **Accept** | Equality of record laws requires an allowed clean law G′ with G′ = G ∗ F_q (Prop. (v)). F1 produces excess short excursions but is absorbed only as a misspecified fit (β̂ ≈ 0.010); exact equivalence is neither claimed nor shown. | Correction 5. |
| **S1.** Shared evaluation prefixes | **Accept** | Reproduced. Shared-prefix SE of the replicate average with fits and VOI fixed: J−EX 3.95e-6 (lead) and 7.71e-6 (sec); ETA⁻−J 0.00158 and 0.00139 VOI. The independent-set figures are 2.45e-6, 3.98e-6, 0.000356 and 0.000316, ratios 1.6 / 1.9 / 4.4 / 4.4. | Corrections 7–8. No classification changes. |
| **S2.** The independence figure is "a lower bound" | **Accept, and sharpen** | SE²_shared = SE²_indep + (R²N)⁻¹ Σ_{r≠r′} Ĉov_i(D_r, D_r′); the sign is set by cross-replicate covariances. Mean cross-replicate correlation is 0.23 / 0.08 (J−EX) and 0.997 / 0.987 (ETA⁻−J), but individual J−EX pairs reach −0.9998. It is no bound in general. | Relabel as "independent-evaluation-set approximation". |
| **S3.** η search domain ≠ theorem domain | **Accept** | Grid 0.005–0.15, then a bounded search in [best ± 0.005] ∩ [0.001, 0.499]. The effective domain is [0.001, 0.155], whereas e4_report §5 says "[0.001, 0.499]". The 0.001 bound is attained at L = 3 (45% of lead T3 replicates at n = 10⁴). | Correction 6. |
| **S4.** Status files stale | **Accept** | research_status and the consolidation_review header still say "uncommitted, HEAD 64e2234". findings_summary "scope of every statement" says "declared and correct prior". | Corrections 10–11. |
| **A1** (mine). Prop. (iii) is incomplete | new | The L = 3 identified set is an arc η′ ∈ [0, η_hi], with η_hi > η exactly when all four footprint masses are positive (Prop. A). Stage 0 already showed this: |KL(0.08)| ≤ 2e-5 on T3 and T6, against 1e-3 to 1e-2 on T0 and T4. | Correction 2. |
| **A2** (mine). The L = 3 "drift" of η̂ | new | At L ≤ 3 the sample profile likelihood is exactly non-increasing in η (Prop. B). J-MLE therefore returns the bottom of the arc up to tie-breaking, and its regret equals that of a near-bottom point. | Correction 4. |
| **A3** (mine). Prop. (iv), η = ½ | new | "μ₂ ≡ 0, so the decision is identified" requires η known or L ≥ 4. At L = 3 with η unknown, (η′ < ½, uniform footprint law) is observationally equivalent and has \|μ\| ≈ 0.71 on average (§5.1). | Correction 3. |

## 3. The complete L = 3 identified set

**Proposition A** (L = 3; Lemma 1 assumptions; full channel simplex). Let F = F_q on {0,1}², with Walsh–Fourier coefficients F̂(A), and set t = (1−2η)/(1−2η′). Then (η′, q′) reproduces the record law iff F′ = F_{q′} satisfies F̂′(A) = t^{|A|} F̂(A) and F′ ≥ 0. Every distribution on {0,1}² is a footprint law at L = 3, so there is no further constraint. Consequently:

- **(a) Lower branch.** Every η′ ∈ [0, η] is admissible: there t ≤ 1, and F′ = B_ν ∗ F with 1 − 2ν = t.
- **(b) Upper branch.** Some η′ > η is admissible iff F has full support, i.e. q_none, q₀, q₁, q₂ > 0. The largest is η_hi = ½(1 − (1−2η)/t_hi), where t_hi is the smallest root above 1 of the four quadratics F′(x; t) = 0.
- **(c) The set and when it is a point.** Ξ(P) = {(η′, q′(η′)) : 0 ≤ η′ ≤ η_hi} is an arc. (η, q) is identified at L = 3 **iff η = 0 and F has a zero**.

*Proof of (b).*
1. dF′(x; t)/dt at t = 1 equals ¼ Σ_A |A| F̂(A) χ_A(x) = −½ (ΔF)(x), where ΔF(x) = Σ_i [F(x ⊕ eᵢ) − F(x)].
2. At a zero x of F, this derivative is −½ Σ_i F(x ⊕ eᵢ) < 0, unless both neighbours of x also vanish.
3. If both neighbours vanish, F = δ_y at the antipode y, and F′ at a neighbour of y is ¼(1 − t²) < 0 for t > 1.
4. So any zero of F blocks t > 1. Under full support, continuity in t gives t_hi > 1. ∎

**Table A.** L = 3 arcs for the E4 truths. Channel weights are [q_none, q₀, q₁, q₂]. The arc does not depend on physics. Record log-likelihoods agree to ≤ 5.3e-15 at both endpoints in every cell and both physics.

| truth | channel q | full support? | arc η′ | q′ at η′ = 0 (β′) | q′ at η_hi (β′) |
|---|---|---|---|---|---|
| T0 | [1, 0, 0, 0] | no | [0, 0.05] | [0.9025, 0.0475, 0.0025, 0.0475] (0.0975) | the truth |
| T3 | [0.8, 0.0333, 0.0333, 0.1333] | yes | [0, 0.0848] | [0.73, 0.07, 0.04, 0.16] (0.270) | [0.8613, 0, 0.0303, 0.1084] (0.139) |
| T4 | [0.65, 0, 0, 0.35] | no | [0, 0.05] | [0.6033, 0.0318, 0.0183, 0.3468] (0.397) | the truth |
| T6 | [0.8, 0.0286, 0.0571, 0.1143] | yes | [0, 0.0792] | [0.7289, 0.0668, 0.0604, 0.1439] (0.271) | [0.8513, 0, 0.0570, 0.0917] (0.149) |

**The saved data agree.** At lead T3, replicate 0, n = 10⁴, the saved J-MLE profile is flat to within 2e-9 per record over η ∈ 0.005–0.080 and drops first at 0.085 (by 6e-6). That is η_hi = 0.0848.

## 4. Decision consequences: diameter, floor and the minimax rule

### 4.1 A common scale

For ξ ∈ Ξ(P) the record S has law P, so the excess one-step cost of a belief rule b against ξ's oracle is

  R_ξ(b) = E_P[κ (b − μ_ξ)²].

This is on a common physical scale for every ξ. VOI_clean(ξ) is not: it depends on the latent clean prefixes under ξ.

There are three usable normalisers:
- raw cost;
- E_P[κ], which is identified: 3.97 (lead) and 5.86 (sec) at L = 3;
- the reference truth's VOI_clean, used below for continuity with the documents and labelled as such.

### 4.2 Lower bound (Codex item 3): verified

Pointwise, (b − μ₁)² + (b − μ₂)² ≥ ½(μ₁ − μ₂)². Hence

  max{R₁, R₂} ≥ ½(R₁ + R₂) ≥ ¼ E_P[κ(μ₁ − μ₂)²],

with equality at b = (μ₁ + μ₂)/2.

**Consequence.** Define diam²(P) = sup_{ξ,ξ′ ∈ Ξ(P)} E_P[κ(μ_ξ − μ_ξ′)²]. No estimator, with any amount of data from P, can be uniformly within ε of the oracle over Ξ(P) when ε < ¼ diam²(P). At L ≤ 3, "near-oracle" is therefore possible only as a truth-specific statement. A positive floor need not exceed the practical margin; that is checked in Table B.

### 4.3 Global minimax versus pointwise conservatism (Codex item 4)

The global criterion inf_b sup_ξ R_ξ(b) uses one ξ for all records.

- **Global minimax.** R_ξ(b) is convex in b, and ξ ranges over a compact arc. The record carries no information about ξ within Ξ(P), so a prior π on Ξ is not updated by S. The value is

    V*(P) = max_π E_P[κ Var_π(μ_ξ(S))],

  and a minimax rule is b*(S) = E_{π*}[μ_ξ(S)]: a fixed mixture of compatible beliefs (Γ-minimax with a least-favourable prior; Berger 1985).
- **Pointwise rule.** b_pw(S) = ½(min_ξ μ_ξ(S) + max_ξ μ_ξ(S)) solves a different problem, inf_b E_P[κ sup_ξ (b − μ_ξ)²], in which nature picks ξ per record. It upper-bounds the global problem.
- **When they coincide.** If μ_ξ(S) is monotone along the arc for P-a.e. S, then:
  - the endpoint midpoint b̄ = ½(μ_{η′=0} + μ_{η_hi}) equals b_pw;
  - |b̄ − μ_ξ| ≤ ½|μ₀ − μ_hi| pointwise for every ξ;
  - so b̄ is globally minimax, with V* = ¼ diam².
- **This is not a theorem here.** Monotonicity held for 82% (lead T0) to 99% of record views.

**Numerically**, on a 41-point arc grid with Frank–Wolfe:
- the least-favourable prior was ½/½ on the two endpoints in all 8 cells;
- b_pw was within 0.4% of V* at lead T0 (0.05046 vs 0.05028), 0.15% at secondary T0, and ≤ 0.01% in the other six cells.

**Table B.** L = 3, E4 evaluation prefixes (master 8601, cell 0, 20,000 prefixes). Regrets are in units of the reference VOI_clean. The SE of V* is the prefix-level SE with VOI fixed; the ratio (delta-method) SE is equal or smaller.

| physics | truth | Δ/VOI | diam²/VOI | V*/VOI (SE) | V* raw | V* vs Δ | far endpoint at truth | midpoint at truth | E4 J-MLE at n = 10⁴ (mean, R = 20) |
|---|---|---|---|---|---|---|---|---|---|
| lead | T0 | 0.002 | 0.201 | 0.0503 (0.0027) | 0.157 | material | 0.201 | 0.0503 | 0.169 |
| lead | T3 | 0.020 | 0.186 | 0.0465 (0.0006) | 0.146 | material | 0.0717 | 0.0029 | 0.058 |
| lead | T4 | 0.020 | 0.097 | 0.0244 (0.0008) | 0.076 | material (≈ 5 SE) | 0.0975 | 0.0244 | 0.077 |
| lead | T6 | 0.020 | 0.174 | 0.0436 (0.0006) | 0.137 | material | 0.0755 | 0.0044 | 0.063 |
| sec | T0 | 0.002 | 0.107 | 0.0266 (0.0018) | 0.103 | material | 0.107 | 0.0266 | 0.098 |
| sec | T3 | 0.014 | 0.129 | 0.0323 (0.0005) | 0.125 | material | 0.0491 | 0.0018 | 0.019 |
| sec | T4 | 0.020 | 0.055 | 0.0136 (0.0004) | 0.053 | **below** | 0.0546 | 0.0137 | 0.043 |
| sec | T6 | 0.012 | 0.118 | 0.0295 (0.0005) | 0.114 | material | 0.0505 | 0.0029 | 0.039 |

### 4.4 Reading

- **"Material only at β = 0" is wrong as a statement about the ambiguity.**
  - At β = 0.2–0.35 the minimax floor exceeds the margin in 5 of 6 cells; the exception is secondary T4.
  - In all 6 corrupted cells the farthest compatible point exceeds the margin.
- **The truth is at an arc end for boundary channels.** At T0 and T4 (η_hi = η) the midpoint rule pays the full floor even at the truth.
- **At interior truths the midpoint is close to the truth** (0.0029 VOI at lead T3). That reflects where the truth sits on the arc; the guarantee is only V*.
- **Bottom selection is expensive in the worst case.** Its worst case is the full diameter, 0.186 VOI at lead T3, four times the floor.

### 4.5 J-MLE at L ≤ 3 is a deterministic bottom selection

**Proposition B.** Take L ≤ 3, Lemma 1 assumptions and the full simplex.
- For η₁ < η₂, the map q ↦ B_ν ∗ F_q with 1 − 2η₂ = (1 − 2η₁)(1 − 2ν) sends every channel at η₂ to a valid channel at η₁, because every distribution on the cube is a footprint law.
- It preserves the density of **every record**: by the converse of Lemma 1, a record's density depends only on B_η ∗ F_q.
- Hence the sample profile log-likelihood satisfies ℓ_n(η₁) ≥ ℓ_n(η₂) exactly, and its maximiser set contains the lower end of the search domain. ∎

**The saved units agree.**
- On interior truths the profile is flat to 4e-11 to 5e-10 per record over 0.005–0.08.
- On boundary truths (T0, T4), η = 0.005 beats η = 0.05 by 0.08–1.5 nats in total, where the truth's simplex boundary binds.
- J-MLE's L = 3 estimates sit at the bottom: lead median η̂ 0.003–0.007, with 45% of lead T3 replicates at the 0.001 search bound at n = 10⁴.
- Its regret matches near-bottom arc points: lead T3 0.058 against 0.057 at η′ = 0.005, and lead T4 0.077 against 0.077.

**Consequence.** The reported L = 3 regret measures one selection rule on a flat set, not joint likelihood as a set-valued estimator.

## 5. Codex's further questions

### 5.1 Decision identification versus parameter identification

**Definition.** The decision target is the map ξ ↦ μ_ξ ∈ L²(P, κ). It is identified at P iff μ_ξ = μ_ξ′ P-a.s. for all ξ, ξ′ ∈ Ξ(P), i.e. iff diam(P) = 0. If the sign of c is unknown, replace μ by γ = cμ (§5.6). Parameter identification implies decision identification; the converse fails.

**The η = ½ example, qualified.** P is uniform.
- *With η known, or L ≥ 4:* every ξ ∈ Ξ has η′ = ½. If η′ < ½, then (1−2η′)^{|A|} F̂′(A) = 0 for all A ≠ ∅ forces F′ to be uniform on the cube, which a footprint law cannot be when L ≥ 4. So μ ≡ 0 and the decision is identified while q is not. This is Codex's example.
- *At L = 3 with η unknown:* (η′, q′ = (¼, ¼, ¼, ¼)), i.e. β′ = ¾, reproduces P for every η′ < ½, and its beliefs are far from zero. On 50,000 records the maximum log-density difference is 1.8e-15, and the mean |μ| is 0.71. The decision is not identified. At L = 4 the same alternative differs by up to 1.9 nats per record.

**A functional that is identified at L ≤ 3.** The apparent-path belief E[m̃_{L−1} | S] depends only on P̃, so it is identified. The decision instead needs E[σ_{L−1} m̃_{L−1} | S] and η, and those are not identified.

### 5.2 What information removes the ambiguity (Codex item 5)

| information | effect on Ξ(P) | effect on the decision | status |
|---|---|---|---|
| More deterministic views of the same prefix | none | none | data processing; C1 |
| Longer records (L ≥ 4) | point | identified | Prop. (ii). The channel over L positions is a different object, so this is meaningful only if longer records exist. |
| Clean calibration prefixes | η identified (clean HMM), hence q at L = 3 (Prop. (i)) | identified at L = 3; at L = 2 only β; nothing at L = 1 | known theorems; finite-n cost measured in E4 (CV arms) |
| Bound on β or on η | sub-arc; β′ falls monotonically along the arc in all 8 cells (T3: 0.270 → 0.139) | floor = ¼ × diameter of the sub-arc (under monotonicity) | exact computation, not yet run |
| Trusted outcomes on the corrupted prefixes | reduced to the decision-equivalence class | identified | §5.3 |
| A genuinely new observation (action or mode sensor) | depends on the sensor | — | not studied |

### 5.3 Trusted outcomes

**The identity holds:** E[u(y − ρ s_L) | S]/q_u = c μ₂(S) = γ(S). It needs:
- (A1) the outcome equation y = ρ s_L + c m_L u + ε, with the recorded s_L intact (true in S2);
- (A2) u ⊥ (S, m_L), or the weaker u ⊥ m_L | S with E[u² | S] > 0, normalising by E[u² | S];
- (A3) E[ε | S, u, m_L] = 0;
- (A4) calibration and deployment share the law of (S, m_L): the same clean process and the same channel.

**No Markov or clean-family assumption is needed.** The identity is a conditional moment of whatever law generated (S, m_L). With Gaussian ε of known variance, y | S, u is a two-component mixture with weights (1 ± μ₂)/2, so outcomes identify μ₂ through the likelihood as well.

**Short-prefix example (L = 1).** The record law does not depend on (η, β) at all, and μ₂(S) = (1−2η)(1−2β) tanh(c ã₀ δ₀ / q_w). The outcome regression identifies the product (1−2η)(1−2β), and with it a*, while neither factor is identified.

**At L = 3 (lead T3, 4 × 10⁵ records).** Regressing u(y − ρ s_L)/q_u on candidate beliefs gives slope 1.000 at the truth, against 0.81 and 1.11 at the arc endpoints. Outcomes select the decision-equivalence class.

**What is already established.**
- `probe_loss` with A6/A6d is exactly this regression under a *declared, simulated* channel, on clean prefixes with the original clean probe kept.
- The pilot, E1 and E2 measured its finite-sample behaviour (L = 8, physics-informed).

**What the calibration access regime would add.**
- It needs no clean source and no declared channel.
- It is the only regime here in which μ₂ is identified at L ≤ 3 and under family violations such as F1, because it never attributes.
- It is **not** a fair-information competitor to prefix-only J-MLE. The fair comparator is J-MLE given the same outcomes, which can add the outcome term to its likelihood.
- The open question would be the sample cost of a noisy regression target against a structured likelihood.

**Failure modes.**
- *Probe sign corrupted independently at rate p_u:* the estimate is attenuated by (1 − 2p_u), its sign reverses if p_u > ½, and the scale is unidentified unless p_u is known.
- *Probe flip coupled to prefix corruption* (e.g. at most one flip over L + 1 positions): the bias depends on S.
- *u depending on m_L* (an expert or closed-loop controller with extra state information): the identity fails.
- *u = g(S):* still identified wherever g(S) ≠ 0, using E[u² | S].
- *Outcome-dependent selection of calibration records:* biased.
- *Channel shift between calibration and deployment:* γ_cal ≠ γ_dep.

### 5.4 Weakening the supplied physics within S2 (idea A)

**The identities hold.**
- ρ = E[s_j s_{j+1}]/E[s_j²] uses E[a_j] = 0 and a_j, w_j ⊥ (s_j, m_j).
- q_a = E[ã²], because a flip preserves the magnitude.
- E[δ² | ã²] = c²ã² + q_w uses m² = 1 and E[w | a, m] = 0.
- Var(a²) = 2q_a² > 0 for Gaussian actions.
- All four are channel-invariant because states are intact.

**Numerically**, on 2 × 10⁵ records with β = 0.35 concentrated on the last action:

| physics | ρ̂ | q̂_a | ĉ² | q̂_w (true) |
|---|---|---|---|---|
| lead | 0.9004 | 1.0003 | 0.9998 | 0.0201 (0.02) |
| sec | 0.9002 | 1.0003 | 0.9993 | 0.5002 (0.5) |

**Sign symmetry.** The map (c, m) ↦ (−c, −m) preserves the record law (symmetric chain, uniform m₀). It leaves γ = cμ₂ and a* = −ρ s_L γ/(c² + λ) unchanged, so the decision is identified although the sign of c is not. λ is a task parameter supplied by the decision maker.

**Corollary.** Lemma 1 and Prop. (i)–(v) hold with (ρ, c², q_a, q_w) identified rather than supplied, within S2: exogenous zero-mean actions, intact states and magnitudes, and the stated noise structure. This closes the population half of synthesis §7 question 2. The plug-in finite-sample cost is unmeasured, and the result says nothing about unknown physics outside S2.

### 5.5 At most k flips (idea B): confirmed

"Iff" here is over the full simplex of allowed-subset channels.

**Setting.** σ = (−1)^b with b ∈ A_k = {|b| ≤ k}, under an exogenous channel law over A_k, with footprint Db. Lemma 1 extends verbatim: it uses only that σ is independent of (m, a, w, s₀) and that a is symmetric.

1. **Injectivity.** ker D = {0, 1}. D is injective on A_k iff there is no b with |b| ≤ k and L − |b| ≤ k, i.e. iff L ≥ 2k + 1.
2. **Image.** D(A_k) is the whole cube iff L ≤ 2k + 1.
   - Each pattern has preimages b and b ⊕ 1, with weights w and L − w, and min(w, L − w) ≤ ⌊L/2⌋.
   - For L ≥ 2k + 2, the all-ones pattern needs an alternating b of weight ≥ k + 1, so it is not in the image.
3. **Known η < ½.** F is identified by deconvolution, and the channel is identified iff L ≥ 2k + 1. For L ≤ 2k, complementary pairs are confounded, and **this is decision-relevant**: b and b ⊕ 1 have opposite σ_{L−1}, hence opposite-sign contributions to μ₂. (k = 1, L = 2 is the existing allocation ambiguity.)
4. **Unknown common η.** The swap and full-support argument of Prop. (ii) applies whenever D(A_k) ≠ cube. With injectivity, (η, channel) is jointly identified for L ≥ 2k + 2 on [0, ½) × Δ(A_k). For L ≤ 2k + 1 the lower branch exists for every η > 0, so joint identification fails. At η = 0 it can still hold at channels with zeros, as at k = 1, L = 3.
5. **LP check.** I computed the minimum total variation over channels at η′ ∈ {0.02, 0.035, 0.065, 0.08}, for a random interior channel with η = 0.05, k ∈ {1, 2, 3} and L = 2, …, 9. All 24 cases agree with items 1–4. At L = 2k + 1 the lower alternatives are exact and the upper one is bounded, as for k = 1.

**Scope.** A common η is needed (B_η = B_{η′} ∗ B_ν uses a single rate), so heterogeneous edge rates break the argument. This is an extension of the structural argument, not a new principle.

**Practical limit.** The simplex has Σ_{i≤k} C(L, i) − 1 free parameters (92 at L = 8, k = 3), and exact inference enumerates that many hypotheses per record. Identification says nothing about the sample size needed. Independent per-label flips form a different, parametric family that this argument does not cover.

## 6. The strongest completed contribution, and whether the thesis is sufficient

**In plain language.** In a controlled switching model where a recorded action's sign may be flipped, the project shows:
- exactly when unlabeled records can separate corruption from genuine regime changes: L ≥ 3 with persistence known and L ≥ 4 without, with physics supplied (now population-identifiable within S2);
- what that separation is worth for a one-step decision;
- what a wrong corruption prior costs;
- how many records standard estimators need;
- that multi-view consistency adds nothing at the population level beyond the corruption law.

**Limits.**
- a scalar model and one channel family;
- one-step quadratic decisions;
- standard methods;
- no detector, defence, closed-loop or real-data result.

**Sufficient for a bounded thesis? Yes, after one indispensable repair.**
- The abstract says that observationally equivalent parameters "generally imply different decisions".
- The only evidence offered for materiality is one interior ridge point, and the documents draw the wrong conclusion from it.
- The repair is exact and analytic, and is essentially done in §3–4: the identified set, the decision floor and the minimax rule.
- Trusted outcomes (§5.3), nuisance identification (§5.4) and k flips (§5.5) strengthen the identification chapter but are not required.

## 7. Recommended next step: one bounded theory section, then the write-up

**Direction.** Decision identification under partial identification in S2, written as the closing section of the identification chapter. This is Codex's direction, scoped so that it does not become a new project: no estimators, no training, no new data.

**Falsifiable questions.**
- **Q1.** At L ≤ 3, does the least-favourable prior over Ξ(P) put all its mass on the two arc endpoints, so that the endpoint midpoint is globally minimax? It is refuted by a least-favourable prior with interior mass, or by a b_pw worst case materially above V*. Current evidence: endpoints in 8 of 8 cells, with b_pw within 0.4% of V*.
- **Q2.** Which single restriction is the weakest that brings V* below Δ: β ≤ β̄, a persistence bound, a clean calibration set, or trusted outcomes?

**Assumptions.** Lemma 1 conditions; physics supplied or moment-identified (§5.4); an exogenous single-flip channel; a one-step quadratic decision.

**Closest primary literature** (standard; not re-read for this note):
- Manski, *Partial Identification of Probability Distributions* (2003);
- Manski, J. Econometrics 95 (2000), decisions under ambiguity;
- Stoye, J. Econometrics (2009; 2012), minimax-regret treatment choice under partial identification;
- Tamer, Annu. Rev. Econ. 2 (2010), review;
- Berger (1985), Γ-minimax and least-favourable priors;
- Lovett & Zhang (2017), for the non-identification itself.

Under quadratic loss with a fixed ξ, the minimax problem is a variance maximisation over priors on a one-dimensional arc. The contribution would be the exact characterisation for this channel, not a method.

**Exact-law protocol.**
1. **Cells.** L ∈ {1, 2, 3}, with η known and unknown; truths T0, T3, T4, T6; both physics; an η = ½ control at L = 3 and L = 4.
2. **Identified sets in closed form.** L = 3 by Prop. A; L = 2 by a free allocation plus the η ⊕ β arc; at L = 1 every parameter is compatible.
3. **Quantities on the E4 evaluation prefixes** (master 8601, cell 0, 20,000; one prefix with its views is one unit):
   - diam², V* and the least-favourable prior on a 41-point arc grid (Frank–Wolfe);
   - worst cases of the endpoint midpoint, b_pw, the bottom endpoint and naive;
   - the per-record monotonicity fraction.
4. **Side-information sweep.** β̄ and the persistence bound over the arc range; V* of each restricted arc; trusted outcomes and clean calibration as the collapse cases.
5. **Comparator.** The existing E4 J-MLE L = 3 units, read as bottom selection (§4.5).
6. **Normalisation.** Raw cost, E_P[κ], and the reference VOI_clean, labelled as such.
7. **Uncertainty.** Prefix-level SE of diam² and V*, with VOI fixed and by the delta method (currently 0.0004–0.0027 VOI). Grid convergence from 21 to 41 to 81 points.

**Stopping rule.** Stop when Q1 and Q2 are answered for these cells. A finite-sample set estimator is a separate user decision, because inference at the simplex boundary and η′ = 0 is non-regular (no Wilks).

**What would change a conclusion.**
- The C25 wording is already changed by this review.
- If Q1 fails, the conservative rule becomes a least-favourable mixture instead of the endpoint midpoint.
- If a modest β̄ suffices, the practical reading becomes "a corruption-rate cap removes the L = 3 decision ambiguity".

**When to narrow or abandon.**
- If Q2 reduces to monotone algebra along the arc (likely), narrow it to one paragraph.
- If any step needs a set estimated from finite data, stop and write up.

**Why not the alternatives as next projects.**
- The trusted-outcome comparison is mostly a reframing of A6 plus an access remark. It is worth an experiment only if a target application actually provides trusted outcomes.
- Nuisance identification and k flips are corollaries; an appendix paragraph each.

## 8. Message to Codex

**I agree with:**
- your materiality point and ridge counterexample (reproduced exactly);
- the §4.3 sentence correction;
- the shared-prefix SE (reproduced exactly) and its non-bound status;
- the search-domain and status corrections;
- your framing of the contribution and its limits.

**I would change four things.**
1. Your ridge table is the lower half of one side of the set. For interior channels the arc is two-sided (Prop. A). A regret-to-the-truth table should be replaced by the diameter and the minimax floor on a common cost scale (§4).
2. J-MLE's L = 3 regret is a bottom selection forced by the exact monotonicity of the sample profile (Prop. B). It is not drift.
3. Decision identification at η = ½ needs η known or L ≥ 4 (§5.1).
4. The independent-set SE is no bound in either direction: individual J−EX pair correlations reach −0.9998.

**What I would ask you to investigate next.**
1. Check Propositions A and B and the η = ½ counterexample independently.
2. Test Q1 on a finer grid and at L = 2.

**Questions whose answers would change the choice.**
1. *Is the L ≤ 3 regime a claimed result of the thesis (the abstract sentence), or only a negative control?* Provisional judgment: claimed, so §3–4 is required. If it is only a control, a corrected sentence suffices and we go straight to the write-up.
2. *Does an intended application provide trusted outcomes or a cap on the corruption rate?* Provisional judgment: none is specified, so these stay propositions, not experiments.

## 9. Concrete corrections (documentation and SE reporting)

Historical results and numbers are preserved; only wording and labels change.

1. **"Material only at β = 0".** Locations: findings_summary.md, *Literature positioning* (the L = 3 ridge bullet); consolidation_review.md §4 row C25; research_synthesis.md §6, first bullet; literature_review.md §4.3, *Correction after E4 stage 0*. Replace with:
   > At L = 3 the identified set is an arc η′ ∈ [0, η_hi]: η_hi = η for the boundary channels T0 and T4, and ≈ 0.085 and 0.079 for the interior channels T3 and T6. The stage-0 comparison at η′ = 0.03 (0.8–1.4% of VOI_clean at β = 0.2–0.35; 1.6–3.0% at β = 0) is one interior point. Over the arc, the farthest compatible point costs 4.9–9.7% of VOI_clean at β = 0.2–0.35 and 10.7–20.1% at β = 0. No belief rule can guarantee less than the minimax floor ¼·diam²: 1.4–4.6% at β = 0.2–0.35 (above the margin in 5 of 6 corrupted cells, below it only for secondary T4) and 2.7–5.0% at β = 0. Reference VOI_clean of the truth; E4 evaluation prefixes; prefix SE ≤ 0.003 VOI.
2. **Prop. (iii) and its summaries** (research_synthesis §4.4; consolidation_review §3). Add the upper branch, and replace "Not every point is ambiguous: at η = 0 there is no η′ < η branch" with:
   > (η, q) is identified at L = 3 iff η = 0 and some footprint mass is zero; with full support an upper branch η′ ∈ (η, η_hi] also exists (Proposition A).
3. **Prop. (iv).** Append:
   > with η known, or L ≥ 4. At L = 3 with η unknown, (η′ < ½, uniform footprint law, β′ = ¾) is observationally equivalent and its beliefs are not zero.
4. **The L = 3 estimator reading** (research_synthesis §5.4 L = 3 bullet; e4_report §9 item 2; findings_summary C27). Replace "η̂ drifts below 0.05 along the flat ridge" and the "fitted gain of about 1e-6" explanation with:
   > At L ≤ 3 the sample profile log-likelihood is exactly non-increasing in η (the ridge map preserves every record's likelihood), so its maximiser set contains the lower end of the search domain. The reported η̂ are tie-breaks on a flat set, and the reported regret is that of a near-bottom arc point (e.g. 0.057 VOI at η′ = 0.005 on lead T3), not an intrinsic property of joint likelihood.
5. **literature_review §4.3**, the sentence "Any clean law that produces excess length-one excursions is observationally equivalent to interior corruption". Replace with:
   > A clean law whose apparent-edge law equals G ∗ F_q for the true G and some channel q is observationally equivalent to that channel (Proposition (v)). Excess length-one excursions alone do not imply equal record laws: F1 produces them and is absorbed as a misspecified fit (β̂ ≈ 0.010), without any claim of exact equivalence.
6. **Search domain** (e4_report §5, "searches η ∈ [0.001, 0.499]"; also the synthesis wherever J-MLE is defined). Replace with:
   > Coarse grid 0.005–0.15, then bounded refinement within ±0.005 of the best grid point, clipped to [0.001, 0.499]: effective domain [0.001, 0.155]. This is the tested procedure, not the theorem domain [0, ½). At L = 3 the bound 0.001 is attained (45% of lead T3 replicates at n = 10⁴).
7. **`experiments/check_e4_eval_se.py`** (docstring item 2 and the key `se_of_replicate_mean_independent_over_voi`). Relabel as "independent-evaluation-set approximation (not a bound)". Add:
   - `se_shared_prefix_fixed_fits_over_voi` = sd_i(D̄_i)/√N;
   - the mean and minimum cross-replicate correlation;
   - a delta-method SE with VOI_clean estimated on the same prefixes.

   Expected values (× VOI):

   | contrast | physics | shared-prefix SE | delta-method SE |
   |---|---|---|---|
   | J − EX | lead | 3.95e-6 | 3.89e-6 |
   | J − EX | sec | 7.71e-6 | 7.66e-6 |
   | ETA⁻ − J | lead | 0.00158 | 0.00155 |
   | ETA⁻ − J | sec | 0.00139 | 0.00138 |

   This is a recommendation; the code is not edited here.
8. **Reporting text** (research_synthesis §5.4 "Evaluation-set uncertainty"; consolidation_review §1 and G1; research_status "Checks"). Label the Table A intervals "replicate intervals conditional on the fixed evaluation set (fit uncertainty only)". Add:
   > Evaluation-prefix SE of the replicate-average contrast (fits and VOI_clean fixed): J − EX 4.0e-6 (lead) and 7.7e-6 (sec); ETA⁻ − J 0.0016 and 0.0014 VOI. A crossed (replicate + prefix) SE for population regret is 4.0e-5 and 5.2e-5 for J − EX, and 0.0017 and 0.0016 VOI for ETA⁻ − J. Conclusions are unchanged: ETA⁻ − J ≈ 0.0283 is about 15 crossed SEs above Δ, and J − EX stays within ±Δ.

   State SE²_shared = SE²_indep + (R²N)⁻¹ Σ_{r≠r′} Ĉov_i(D_r, D_r′), so the independent figure is not a lower bound in general.
9. **Normaliser.** Wherever compatible latent laws are compared (C25, stage 0, §4 here), state that VOI_clean depends on the latent law, and give raw cost or E_P[κ] alongside.
10. **findings_summary "Scope of every statement"**: "a declared and correct prior". Make it stage-specific:
    > oracle and pilot: declared, correct prior; E2/E2b: declared, possibly wrong prior; E3/E4: channel learned from unlabeled records, within the assumed clean family unless stated (F1).
11. **research_status.md and the consolidation_review.md header.** HEAD is `08dd58e`, and the consolidation is committed. Remove the "uncommitted" lists.
12. **`tests/conftest.py`** writes `results/oracle_v1/test_max_errors.json` on every pytest session, including partial runs, so a subset run overwrites a committed artifact. Write to a scratch path unless a full-suite flag is set.
13. **Contribution wording** (abstract; §2 item 2). Once §4 is adopted, change "explicit, decision-relevant non-identification at L ≤ 3" to "… with a quantified minimax decision floor".

## Appendix: how the new numbers were computed

**Arc channels (L = 3).**
- Walsh–Hadamard transform of F_q on {0,1}², scaled by t^{|A|}, then inverted.
- η_hi by bisection on min_x F′(x; t) ≥ 0.
- Footprint order follows `mti.joint.footprints`: none → 00, flip 0 → 10, flip 1 → 11, flip 2 → 01.

**Beliefs and regrets.**
- `ExactRecords(ts, phys.with_(eta=η′))` and `Precomputed.build(...).compose(q′)`.
- The evaluation set is a numpy-only replica of `TestSet.__init__` (identical `simulate(phys, 20000, 8601, 0, 0)` call; the replica avoids the torch import).
- Regret uses the true channel's view weights.

**Likelihood checks.** log p(S) = logsumexp_θ(ll_θ + log w_θ) on every record view.

**Least-favourable prior.** Maximise π·d − πᵀMπ over the simplex on the arc grid, with M_ab = E_P[κ μ_a μ_b] and d = diag(M), by Frank–Wolfe with exact line search.

**Shared SE.** Saved (η̂, ŵ) for J-MLE, EX-MLE and ETA⁻-MLE at n = 10⁴, L = 8, T0, both physics, R = 20. Per-prefix regrets D_ri, then sd_i(D̄_i)/√N; crossed SE² = s²(replicate means)/R + s²(prefix means)/N.

**Nuisance moments and trusted outcomes.** `simulate` + `apply_channel` with master 1234 (nuisance) and 4321 (trusted-outcome check), unused by E4. The trusted-outcome target is z = u(y − ρ s_L)/q_u, using the simulator's clean probe.

**k flips.** The LP over allowed-subset channels mirrors `mti.joint.pattern_tv`.
