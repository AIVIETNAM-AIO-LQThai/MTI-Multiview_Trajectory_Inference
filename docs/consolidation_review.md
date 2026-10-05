# Consolidation review: response to the external (ChatGPT/Codex) review of E4 and the claim-to-evidence map

Opus 5.5 (`claude-opus-5-5`), 2026-10-05. Branch `exp/joint-identification`, HEAD `64e2234`; this review and the synthesis are uncommitted. Final review (O3) of the assembled synthesis: §8.
Companion: [research_synthesis.md](research_synthesis.md) (the write-up). Superseded: the earlier draft `synthesis.md`, which was never committed.

## 1. Verdict in brief

**The external review is sound.** I accept points A, C, D2, D3, E and F as stated, and B, D1 and D4 with refinements. I reject none.

On one point the reviewer is stronger than our own work. **B (the proof route) is better than our argument.**
- `literature_review.md` §4.3 excluded η' > η only "generically", supported by finite-grid checks.
- The reviewer's swap argument gives **global** identification of η for L ≥ 4.
- I verified it, and closed the step the reviewer flagged: that the noisy record law identifies the apparent-pattern law.
- The completed proposition and proof are in research_synthesis §4.

**One defect neither review caught.** The E4 protocol (§5) promised that "evaluation-set SE is reported separately". The E4 report does not do this, and saved units hold only mean regrets. It has now been fixed by an evaluation-only recomputation from the saved estimates (`experiments/check_e4_eval_se.py` → `results/e4/stage1/eval_se_check.json`). The recomputed mean regrets equal the saved values exactly. The prefix-level SE of J − EX is ≤ 7 × 10⁻⁵ VOI per replicate, against Δ = 0.002; that of ETA⁻ − J is 0.0010–0.0019, against an effect of 0.0283. No classification changes.

## 2. Point-by-point response

| point | response | reason / evidence |
|---|---|---|
| **A.** E4 headline and scope | **Accepted** | Independently recomputed from the raw units in the E4 review: J-MLE − EX-MLE = +0.00007 [0.0000, 0.00014] (lead) and +0.00011 [0.00002, 0.00020] (secondary) VOI_clean, margin 0.002, R = 20, n = 10⁴, L = 8, T0. The ETA⁻ − J-MLE difference is 0.0283 [0.0270, 0.0296]. No universal sample requirement, optimizer consistency, unknown-physics or other-channel claim is made. |
| **B.** Finish the identification argument | **Accepted and completed** | Proof in research_synthesis §4 (outline in §3 below). Status labels are kept apart: global identification (analytic), local profile Fisher information, numerical LP/KL separation at tested parameters, and measured finite-sample costs. |
| **C.** Comparator ≠ oracle; absolute regret; R; Fisher scope | **Accepted** | Every headline in the synthesis names its comparator, absolute regret/VOI, margin, CI and R. R = 10 at n = 10⁵. Fisher calibration was measured on the interior truths T3 and T6 only (23/24 ratios in [0.7, 1.4]) and is not a finite-sample guarantee. Per-truth results are kept apart from prior averages (E2b), and primary comparisons from descriptive grids. |
| **D1.** "Never separated outside the family" | **Accepted, refined** | The C25 ledger row was already conditional ("never identified against clean families containing the corrupted law"). The overbroad sentences are findings_summary §"Literature positioning" and literature_review §5; both are corrected. Precise statement (Proposition (v)): if the allowed clean family contains a law equal to the corrupted law, attribution is not identified. A specified larger family can still be identifiable. |
| **D2.** Lack-of-fit detection ≠ attribution | **Accepted** (an insight we had missed) | A family violation can change the pattern law in ways no (η, q) pair reproduces, and is then detectable by a goodness-of-fit test. Non-identification concerns only *attributing* a misfit to dynamics versus corruption. No goodness-of-fit test was run, so detectability of F1 is untested. |
| **D3.** F1 proves no asymptotic bias or dose–response | **Accepted** | Wording: "β̂ ≈ 0.010 at n = 10⁴ and 3 × 10⁴, while the in-family value falls 0.0045 → 0.0026: persistent over the measured n". Var(η_r) = 0.0009, the excess adjacent-edge-pair rate, is a proposed mechanism and was not tested. |
| **D4.** Clean records do not repair a wrong family | **Accepted, refined** | In F1, CV-1000 also absorbs (β̂ ≈ 0.010). Clean data matter because they carry no channel confound, which makes them the right input for checking the family. Estimating the wrong family's η more precisely does not repair the family. Our earlier wording ("validate the clean law with a few hundred clean records") conflated parameter estimation with family validation, and is withdrawn. |
| **E.** Clean-set comparison | **Accepted** | Exact counts at n = 10⁴, L ∈ {4, 8}, 16 cells: CV-30 / CV-100 / CV-1000 are equivalent to J-MLE in 4 / 10 / 15 cells (CV-1000: 1 inconclusive, 0 worse). This is a comparison of information access, not an equal budget: CV uses the corrupted set *plus* the clean prefixes. "A few hundred suffice" is withdrawn wherever it appeared. |
| **F.** MTM limits | **Accepted** | Route agreement holds under the declared law and does not certify it. The learned gap is a finite-learner diagnostic (C26). The recoverable score g(S) = κ(μ(S) − μ₂(S))² is kept distinct from the corruption probability and from total harm. One-step scope. Learned results are physics-informed, with the E1 coverage confound resolved by the E2 control (C12b withdrawn). |
| **§4.** Literature before claiming novelty | **Accepted; done (focused)** | See §5 below. Outcome: the L = 4 threshold is a **structural corollary** of an elementary noisy-population-recovery argument, specialised to this channel. It is not a new general principle. |

## 3. The identification result (summary; full proof in research_synthesis §4)

**Assumptions.** S2 as in the spec §1–2:
- c ≠ 0, q_a > 0, q_w > 0, |ρ| < 1, with (ρ, c, q_a, q_w) supplied;
- m₀ ~ Unif{±1} and edges iid Bern(η), η ∈ [0, ½];
- actions iid N(0, q_a), and s₀, actions, noise and modes mutually independent;
- θ independent of everything, with channel q on the simplex.

**Results.**
- **Lemma 1.** The record law identifies the law of the apparent edge pattern P = B_η ∗ F_q, where F_q is the footprint distribution of θ.
  - Mechanism: conditionally on the recorded actions, the residuals are a finite mixture of distinct Gaussians N(c m̃ ⊙ ã, q_w I).
  - Finite normal mixtures are identifiable (Yakowitz & Spragins 1968).
- **Lemma 2.** The footprint map θ ↦ F_θ is injective iff L ≥ 3.
- **Proposition.**
  - **(i) η known, η < ½:** q is identified iff L ≥ 3. At L = 2 only β is identified; at L = 1 nothing is.
  - **(ii) η unknown, L ≥ 4:** (η, q) is globally identified on [0, ½) × simplex.
  - **(iii) L = 3:** for every η ∈ (0, ½) and every q, the pairs (η', B_ν ∗ F) with η' ∈ [0, η) are observationally equivalent, and their beliefs generally differ. L = 2 and L = 1 are likewise unidentified.
  - **(iv) η = ½:** no channel information at all.
  - **(v) Clean family unrestricted:** if an allowed clean law reproduces the corrupted law, attribution is not identified.
- **Why the threshold is L = 4.** The L + 1 footprints fill {0,1}^{L−1} exactly when L ≤ 3, and leave a non-footprint pattern when L ≥ 4.

**Status.**
- Analytic, with the proof complete.
- Earlier numerical checks (pattern LP, record-level ridge, stage-0 profile Fisher information) are consistent with it.
- Profile Fisher information remains a *local* statement about estimation precision, not part of the proof.

## 4. Claim-to-evidence map

Margin Δ = clip(0.1 V₂, 0.002 VOI, 0.02 VOI). All claims hold for S2, the exogenous single-flip channel and one-step regret.

| # | exact statement (strongest justified wording) | key assumptions | status | evidence | comparator / margin |
|---|---|---|---|---|---|
| C1 | Candidate and folded exact routes give the same aware belief μ₂ under the declared law; route agreement does not certify the law | declared law correct | verified (≤ 1.5e-14) | `tests/test_corruption.py` T6–T8; [oracle §2](reports/oracle_v1_report.md) | — |
| C2–C4 | Naive excess = V₂ + I_loss. V₂ is prior-specific: 1.8% of VOI_clean (primary cell) to 27–159% (lag-0-concentrated priors) | true channel known | verified / measured | oracle §3–4, §11–13; `results/oracle_v1` | true-law oracle |
| C7, C12–C14 | No general folded-vs-candidate advantage (14 eq / 4 inc / 2 material). Composition vs density-direct: 6 / 4 / 4 / 0, regime-dependent; coverage confound resolved (C12b withdrawn). A fitted HMM is near-oracle | physics-informed features, L = 8, 5 seeds | measured | [pilot §6](reports/learned_pilot_report.md); [E1 §7](reports/e1_report.md); [E2 §6](reports/e2_report.md) | matched backbone; per-cell margins |
| C15–C17 | A wrong declared prior can be worse than ignoring corruption; rate understatement never hurts (proved per record when the location prior is right) | per-truth; declared vs true prior | measured + partly proved | [E2](reports/e2_report.md) §3, §6 | naive (same pipeline), Δ_t |
| C18–C19, H-loc | Mean prior = Bayes rule under prior uncertainty (known). Under it, 9–10 of 30 channels are harmed; a harmless prior keeps ≈ ⅓ of the gain; harm tracks over-declared lag-0 mass (descriptive) | prior over 30 channels; *average vs per-channel kept apart* | analytic + measured | [E2b](reports/e2b_report.md) | naive, Δ_t |
| C20 | Channel identified from unlabeled prefixes iff L ≥ 3 (η known) = Proposition (i) | Lemma 1 assumptions | analytic + verified | research_synthesis §4; `tests/test_e3.py` TE3_1/2 | — |
| C21, C23, C24 | Full-simplex MLE near-oracle by n = 10⁴; restricted-grid PP fails off-grid (misspecification); moment estimator identifies, at 1.6–41× MLE regret | correct clean law | measured | [E3](reports/e3_report.md) §3, §5 | true-law oracle; fixed q̄; Δ_t |
| C22 | At a fixed wrong η̂, clean-law error is absorbed as spurious corruption (η̂ = 0.03 → β̂ ≈ 0.054 at β = 0; harm rate 1.00 at n = 10⁴) | in-family η error | measured; sign predicted | E3 §3.3 | naive, Δ = 0.002 VOI |
| C25 | Joint identification = Proposition (ii)–(iv): global for L ≥ 4, explicit ridge at L = 3 (material only at β = 0: 1.6–3.0% of VOI vs 0.2%) | Lemma 1 assumptions; Markov family | **analytic (proof complete)**; numerically consistent | research_synthesis §4; `experiments/check_joint_identification.py`; `results/e4/stage0/` | — |
| C27 | J-MLE from corrupted prefixes alone is equivalent to known-η EX-MLE at L = 8, n = 10⁴, β = 0, in both physics, and better than η̂ = 0.03 by 0.0283 VOI. L = 4 likewise. L = 3 fails at every n | Markov family; physics supplied; R = 20 (R = 10 at n = 10⁵) | measured (primary P1) | [E4](reports/e4_report.md) §2, §9; `results/e4/stage1/units/` | EX-MLE (**not** the oracle), Δ = 0.002 VOI, paired 90% CI |
| C27b | Equivalence to EX-MLE ≠ near-oracle: at n = 300, J-MLE regret is 0.2–0.5 Δ (lead) and 0.4–3.9 Δ (secondary) on corrupted truths; ≤ 0.13 Δ by n = 10⁴ | — | measured (descriptive) | E4 §9 | true-law oracle (regret 0) |
| C27c | Stage-0 Fisher SE predicted the measured SD(η̂): 23/24 ratios in [0.7, 1.4] | interior truths T3, T6; n ≥ 10⁴ | measured | E4 §5 | — |
| C28 | Family error, one mild instance: record-heterogeneous persistence {0.02, 0.08} gives β̂ ≈ 0.010 at n = 10⁴ and 3 × 10⁴ (in-family 0.0045 → 0.0026); regret 0.0007 VOI < Δ. CV-1000 also absorbs | one violation; R = 10 | measured (descriptive) | E4 §6, §9 | EX-MLE; Δ = 0.002 VOI |
| CV | Clean set + corrupted set vs J-MLE at n = 10⁴, L ∈ {4, 8}: CV-30 / 100 / 1,000 equivalent in 4 / 10 / 15 of 16 cells; at L = 3 the CV arms beat J-MLE | extra information (not an equal budget) | measured | E4 §4 | J-MLE, Δ_t |
| C26 | Harm oracle h* = κ(μ − μ₂)² + κVar(μ(T_θS) \| S); recoverable target g(S) = κ(μ − μ₂)²; learned route gap vanishes for a converged learner | one-step decision | analytic | literature_review §4.1–4.2 | — |
| — | Multi-view consistency detects harmful records beyond posterior/residual/dynamics baselines | — | **not tested** | — | — |

## 5. Focused literature pass: what each source does and does not say

| source | read | what it states | consequence for us |
|---|---|---|---|
| Lovett & Zhang, COLT 2017 (PMLR 65:1417–1431) | full text | §1.1: with unknown noise, "it is impossible in general to recover the 'true' parameters π, µ". Example n = 1: only (2p−1)µ is identified. **Theorem 1** returns a proper (π̂, µ̂) whose noisy distribution is ε-close in statistical distance, in quasi-polynomial time; Theorems 2–3 cover noise-candidate lists and robustness. No identifiability theorem | Our setting is a special case: n = L−1 bits, equal noise, **known** support (the L + 1 footprints). Their n = 1 example is our L = 2 case. The elementary fact that unknown noise is identified when the known support is a proper subset of the cube is not stated there, and our proof supplies it. Classification: **structural corollary / specialisation**, not a new principle. |
| Allman, Matias & Rhodes, AoS 2009 | full text, §6.1 | **Theorem 6**: HMM parameters with r hidden and κ observed states are *generically* identifiable from 2k + 1 consecutive variables, with C(k+κ−1, κ−1) ≥ r | Concerns HMMs with fixed emissions. Our corrupted law is a record-level mixture over HMMs with one perturbed emission position, not an HMM. Not applicable directly; contrasts generic with our global result. |
| Alexandrovich, Holzmann & Leister, Biometrika 2016 | full text, §2 | **Theorem 1**: stationary K-state HMM, transition matrix of full rank and ergodic (A1), distinct state distributions (A2); 2K + 1 consecutive observations identify the parameters up to label swapping. Theorem 2 covers a general initial distribution | Identifies the *clean* switching model. For K = 2, full rank means η ≠ ½, matching our η = ½ degeneracy. It does not address a superposed corruption channel. |
| Balsells-Rodas, Wang & Li, ICML 2024 (arXiv 2305.15925v4) | abstract | Identifiability of Markov switching models (non-linear Gaussian transitions) and of switching dynamical systems, up to affine transformations | No corruption or outlier channel in the abstract. **Unresolved** at full-text level; no claim depends on it. |
| Chen & Liu, JASA 1993, 88(421):284–297 | abstract | Iterative joint estimation of ARMA parameters and outlier effects for four outlier types; discusses "spurious and masking effects" | A direct conceptual predecessor of *absorption*, where model misfit creates spurious outliers. A procedure, not an identification theorem. Full text not read (**unresolved**). |
| Teicher 1963; Yakowitz & Spragins 1968 | standard results | Identifiability of finite mixtures; finite mixtures of multivariate normals are identifiable | Used in Lemma 1 and Proposition (i): **known principle**. |
| Ramaswamy et al. 2016; Wu et al. 2023 (MTM); Xu et al. 2025 (RDT) | full text (earlier step) | MPE needs irreducibility or anchors because one component is unknown. MTM has no detection. RDT corrects corrupted actions by residuals and is evaluated by return only | Unchanged from literature_review §2. |

**Contribution classification.**
- **Known principles:** C1 (Bayes identity), C18, C20, C21, C23, C24, Lemma 1.
- **Structural corollary / specialisation:** the Proposition, including the L = 4 threshold and the L = 3 ridge, which applies an elementary population-recovery argument to the switching sign-flip channel.
- **Empirical characterisations:** C2–C4, C15–C17, C19, C22, C27, C28 and the CV comparison, including sample costs and decision materiality.
- **Potentially new:**
  - the *decision-level* account joining these pieces (which ambiguities matter for decisions, and at what sample cost they are resolved);
  - the explicit statement that persistence and a single-flip channel are jointly identified iff L ≥ 4.
  Neither was found in the sources read. That is a limited search outcome, not proof of novelty.

## 6. Remaining gaps

| id | gap | blocks a claim? |
|---|---|---|
| G1 | E4 evaluation-set SE not reported, contrary to protocol §5 | **Resolved.** `results/e4/stage1/eval_se_check.json`: saved regrets reproduce exactly; evaluation SE negligible against the margins |
| G2 | Chen & Liu and Balsells-Rodas et al. not read in full | Limits positioning only; no claim depends on them |
| G3 | Wider search of the switching-model / outlier literature | Limits the novelty statement (§5), which is kept conditional |
| G4 | Family-violation dose–response untested (one mild instance) | Limits external validity of C28. **Blocks** any quantitative claim about family-error size |
| G5 | Goodness-of-fit detectability of family violations untested (D2) | External validity; no claim made |
| G6 | Unknown physics, other channels, multi-step decisions, real data | External validity |
| G7 | Learned results only for physics-informed features at L = 8 | Scope qualification (C11) |

## 7. Verdict

**Does the completed evidence support a coherent thesis contribution? Yes, a bounded one.**

The contribution is an exact, decision-level analysis of *when action-sign corruption can be distinguished from genuine switching*, and what that distinction is worth for a downstream decision. It has three parts:
1. **Information.** Exact route equivalence, and the prior-specific value of awareness.
2. **Identification.** A complete proposition: the channel is identified iff L ≥ 3 with persistence known, and jointly iff L ≥ 4. The non-identification is explicit and decision-relevant at L ≤ 3, and attribution is conditional on the clean family.
3. **Estimation costs.** Measured: when declared priors harm; likelihood adaptation near-oracle at about 10⁴ records; joint estimation equivalent to known persistence at L = 8, n = 10⁴; what extra clean data buy; and one measured instance of family-error absorption.

The methods are standard and the identification result is a structural corollary. The value lies in the exact characterisation and the decision-level measurements, not in new estimators.

**Dose–response.**
- *Unnecessary* for this bounded claim, which states family-error attribution as conditional non-identification plus one mild measured instance.
- *Necessary* for any stronger, quantitative claim, such as "absorbed corruption scales with the excess short-excursion rate" or "the method is robust to realistic non-Markov dynamics".

## 8. Final review of the assembled synthesis (O3, Opus 5.5)

**Checked.**
- Every number in research_synthesis §5.4–5.6 comes from `experiments/tables_e4_synthesis.py`, which reads the saved units. P1 matches my earlier independent recomputation.
- The evaluation-SE script reproduces the saved regrets exactly.
- The figure is built from saved units only.
- The §5.1–5.3 and §5.7 numbers match findings_summary and the cited reports.
- Citations: theorem numbers only for the sources read in full (Lovett–Zhang §1.1 and Thms 1–3; Allman et al. §6.1 Thm 6; Alexandrovich et al. Thms 1–2). The others are marked abstract or standard.
- Full pytest: 114 passing; test record unchanged.

**Corrected in the draft.**
1. §6 had reintroduced "a few hundred clean prefixes matched J-MLE". Replaced by the exact counts, with the information caveat.
2. §5.1 now notes that the L = 32 contrast is confounded with the per-position rate.
3. Abstract:
   - "near-oracle with about 10⁴ records" now states its precondition, a correct clean law;
   - the non-identification sentence now names both regimes (L ≤ 2 with persistence known, L ≤ 3 with it unknown);
   - the clean-set sentence states that it uses more information.
4. A spurious "in odds" qualifier was removed from the C15 summary.

**Verdict.** The draft is consistent with the evidence and with the accepted qualifications of the external review. The proposition is proved under the stated assumptions, and every empirical headline names its comparator, margin, interval and R. No claim exceeds its evidence. The remaining gaps (G2–G7) limit positioning and external validity and block no stated claim. **The consolidation is complete.** The optional dose–response remains unrun, and is needed only for a quantitative family-error claim.
