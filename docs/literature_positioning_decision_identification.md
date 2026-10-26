# Literature positioning of the decision-identification section

Opus 5.5, 2026-10-27. Companion to [decision_identification_chapter.md](decision_identification_chapter.md). It supersedes the novelty wording of the chapter's §7 where the two differ, and it extends [literature_review.md](literature_review.md).

**Reading status.**
- *Full text*: the cited sections were read in this pass or an earlier one.
- *Partial*: some sections read, others truncated or unavailable.
- *Abstract*: only the abstract or an index entry was read.
- *Standard*: a textbook result, cited without re-reading.

Inaccessible items are listed in §4. Reading the sources below cannot establish novelty: it only rules out the closest matches found.

## 1. Comparison with the two flagged papers

### 1.1 Chen & Liu (1993), *Joint estimation of model parameters and outlier effects in time series*, JASA 88(421):284–297 [full text]

| aspect | Chen & Liu | this work |
|---|---|---|
| observation model | Univariate ARIMA Y_t with observed Y*_t = Y_t + ω [A(B)/(G(B)H(B))] I_t(t₁) (§1, eq. 2); multiple interventions in eq. 19 | Scalar switching dynamics s_{j+1} = ρ s_j + c m_j a_j + w_j with a latent regime m and an observed regressor a |
| contamination | Deterministic outlier effects at unknown times, of four types: innovational (IO), additive (AO), level shift (LS), temporary change (TC) (eqs. 3–6) | A random, exogenous sign flip of one **recorded regressor** (action); the physical trajectory is unchanged |
| target | Model parameters and outlier locations, types and effects | The channel q and persistence η; the decision belief μ₂ and its worst-case cost |
| method / result | An iterative three-stage detection–estimation procedure (§2.1, Stages I–III) on residual test statistics (eq. 20), with joint estimation of the outlier effects (§1.3, eq. 17) to avoid masking and spurious outliers. The evidence is simulation (§3; critical value C ≈ 3.0 for series of length 100–200) and an example (§4). **No identifiability theorem.** | Exact identification results by horizon (Propositions 1–4), the decision-target geometry (Propositions 8–10) and decision costs |

**Relation.** The closest conceptual point is the distinction between an effect confined to the *recorded* data and one that propagates through the dynamics:
- In Chen & Liu, "except for the case of an IO, the effects of outliers on the observed series are independent of the model" (§1.1). Types are told apart by their dynamic signature under an estimated ARMA model.
- In S2, a recorded-sign flip is "AO-like" on the regressor, while a genuine regime switch changes the dynamics.

Chen & Liu's masking and spurious-outlier effects are finite-sample and procedural phenomena. Our Proposition 3 shows something different in kind: at L = 3 the two explanations are **exactly** observationally equivalent along an arc, so no procedure can separate them.

Their remark (§2.2) that under non-Gaussian noise the procedure "cannot distinguish such extreme values as outliers or regular observations" is an informal analogue of our Proposition 5 (attribution relative to the clean family).

**Classification.** Conceptual background for absorption and the outlier/dynamics confound (C22 in [findings_summary.md](findings_summary.md)). Neither predecessor nor competitor for the identification or decision results.

### 1.2 Balsells-Rodas, Wang & Li (2024), *On the identifiability of switching dynamical systems*, ICML; arXiv:2305.15925v4 [full text, incl. App. A–D]

| aspect | Balsells-Rodas et al. | this work |
|---|---|---|
| observation model | Markov switching model (MSM) on a latent or observed continuous z_t with regime-dependent non-linear Gaussian transitions p(z_t \| z_{t−1}, s_t) (§2.2, eq. 8). Switching dynamical system x_t = f(z_t) + ε_t (§3.2) | An observed state s and observed actions, with regime-dependent linear-Gaussian transitions driven by the action |
| contamination | None. Gaussian observation noise with a known distribution (§3.2). No inputs or controls | Exogenous sign corruption of the recorded input |
| target | The MSM up to permutations (Def. 3.1); the transition matrix Q and initial distribution up to permutations (App. C, Cor. C.1); the SDS up to affine maps (Thm. 3.5) | (η, q) exactly, or the identified set; the decision belief and its cost |
| key assumptions | (a1) unique indexing of the regime transitions (eqs. 9, 11); (a2) analytic means and covariances (Thm. 3.2). Q identified when the regime path is a stationary first-order Markov chain (Cor. C.1) | A1–A5 of the chapter: physics supplied (or identified within S2), η ∈ [0, ½], exogenous single-flip channel |
| method | Finite-mixture identifiability through linear independence: Prop. A.3 (Yakowitz & Spragins 1968), Lemma B.3, induction over T (Thm. B.4) | The same device in Lemma 1, followed by an exact deconvolution analysis on the edge cube |

**Relation.** Their Theorem 3.2 "holds independently of the choice of p(s_{1:T})" (§3.1). Applied to our record law, the same argument identifies the law of the **apparent** regime path, which is our Lemma 1 (a structural special case: linear Gaussian means with an input, known noise).

Their Corollary C.1 recovers Q only when that path is a stationary Markov chain. Under corruption, the apparent path has edge law B_η ∗ F_q, which is in general not Markov. Separating B_η from F_q is exactly what our Propositions 2–4 resolve, and it is outside their scope.

Their "up to permutations" equivalence corresponds to our sign symmetry (c, m) ↦ (−c, −m) (Corollary 12).

**Classification.** Background for Lemma 1 and for clean-model identification. It does not address corrupted inputs, the B_η/F_q decomposition, or decisions.

## 2. Partial identification and minimax decisions

| source | status | what it establishes | consequence for our claims |
|---|---|---|---|
| Eldar, Beck & Teboulle (2008), *A minimax Chebyshev estimator for bounded error estimation*, IEEE TSP 56(4):1388–1397 | full text, §II | The estimator minimising the worst-case squared error over a feasible parameter set is its **Chebyshev center**, and the squared radius is the minimax value (§II, eqs. 5–6). Computing it is hard in general, hence a semidefinite relaxation | Theorem 7's identity V* = squared Chebyshev radius of T(P) is **established background** under squared loss. Our use applies it to the set of compatible decision targets. |
| Christensen, Moon & Schorfheide, *Optimal decision rules when payoffs are partially identified*, arXiv:2204.11748v4 (Rev. Econ. Stud.) | partial (§§1–3.5 of v4; later sections truncated) | **Discrete** decisions with payoffs depending on a set-identified θ ∈ Θ₀(μ). Conditional on the point-identified μ, actions are scored by the worst case over Θ₀(μ) (eq. 9), averaged over μ (eqs. 10–13). The least-favourable conditional prior is a point mass at a maximiser (§3.5) | The same decision-theoretic stance (worst case over the identified set given the identified reduced form). Ours is the population, squared-loss, continuous-target case, where the least-favourable prior can need **several** support points (the L = 2 triangle). Background framework, different loss and action space. |
| Giacomini & Kitagawa (2021), *Robust Bayesian inference for set-identified models*, Econometrica 89(4):1519–1556 | abstract (UCL full text returned HTTP 403) | Multiple-prior inference. The prior for the structural parameter given the reduced form "cannot be revised by data" | Background for Theorem 7(i): priors on Ξ(P) are not updated by records. |
| Berger (1985); Jung (1901); Manski (2000), J. Econometrics 95:415–442 | standard / not re-read | Γ-minimax and least-favourable priors; Jung's diameter–radius bound; decisions under ambiguity with partial identification | Background for Theorem 7(ii)–(iii). |
| Lovett & Zhang (2017), COLT, PMLR 65 | full text (earlier pass) | Unknown-noise population recovery: the true parameters are not identifiable in general (§1.1; their one-bit example is our L = 2 case); Theorem 1 gives a statistically close proper representation | Propositions 2–4 are a **structural specialisation**: a known, proper support identifies the noise rate. |
| Allman, Matias & Rhodes (2009), Thm. 6; Alexandrovich, Holzmann & Leister (2016), Thm. 1 | full text (earlier pass) | Identifiability of HMMs (generic; or under full-rank transitions and distinct emissions) | Background for the clean switching model only. |

## 3. Classification of the section's results

| result | classification | reason |
|---|---|---|
| Lemma 1 (record law ↔ apparent edge law) | established background (structural instance) | finite-mixture linear independence (Yakowitz & Spragins; the device used by Balsells-Rodas et al., App. A–B) |
| Lemma 2; Proposition 1 (η known) | established background / elementary | footprint injectivity plus deconvolution on the cube with known noise |
| Propositions 2–4 (horizon thresholds, identified sets incl. the L = 3 arc, singleton case, uniform stratum) | **structural specialisation** | unknown-noise population recovery (Lovett & Zhang) with a known, proper support; the explicit sets for this channel were not found in the sources read |
| Proposition 5 (attribution relative to the clean family) | established principle | model-relative attribution; cf. Chen & Liu §2.2 |
| Theorem 7 (weak duality, Chebyshev radius, Jung bound, endpoint-ball criterion) | **established background** | Chebyshev-center minimax (Eldar et al. §II); Γ-minimax (Berger); Jung |
| Proposition 6 (decision vs parameter identification, incl. η = ½ and ρ = 0) | structural specialisation | definitions are standard; the cases are specific to this channel |
| Propositions 8–10 (L = 2 coefficient and decision triangle; three-point least-favourable prior; ρ → 1 acuteness; L = 3 four-function representation and certificate reduction) | **potentially original, provisional** | decision-target geometry specific to recorded-action sign corruption in switching dynamics; not found in the sources read |
| Proposition 11 (profile nesting at L ≤ 3) | structural specialisation | a direct consequence of the identified-set structure |
| Numerical decision costs (Tables 2–4) and their certificates | **potentially original, provisional** (empirical / computational) | model-specific measurements |
| Corollaries 12–14 | structural corollaries | moment identification, a trusted-outcome identity, and a k-flip extension of the same argument |

**Revised novelty statement.** The general tools are established: finite-mixture identification, unknown-noise population recovery, Chebyshev-center minimax and Γ-minimax. The candidate contribution is narrower than "a decision-identification theory". It is:
- the exact identified sets for recorded-action sign corruption in switching dynamics;
- the geometry of the resulting decision-target sets (segment, triangle, arc), including the multi-point least-favourable prior;
- the certified decision costs in the tested cells.

Its novelty remains **provisional**.

## 4. Unresolved and inaccessible

- **Giacomini & Kitagawa (2021):** full text not accessible (HTTP 403). Only the abstract was read.
- **Christensen, Moon & Schorfheide:** sections after §4.4 of v4 not read (truncation).
- **Set-membership estimation** (Milanese & Vicino 1991, Automatica 27(6); related "central" or Chebyshev-center estimators): bibliographic record only. This literature likely contains further Chebyshev-center results, which would strengthen the "background" classification of Theorem 7 but cannot touch Propositions 8–10.
- **Not searched systematically:**
  - Markov-switching models with **corrupted or misrecorded exogenous inputs**, including errors-in-variables in regime-switching regressions;
  - robust offline-RL work on action-label noise with decision-level identification.

  These are the most likely places for a closer predecessor. A targeted search there is the remaining step before any novelty claim is made in print.
