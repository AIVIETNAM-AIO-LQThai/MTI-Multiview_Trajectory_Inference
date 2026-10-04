# E2 protocol: channel misspecification (exogenous single-flip channel; prior error only)

Settled by Opus 5.5, 2026-10-04, **before any E2 data were generated**. Branch: `exp/channel-misspecification`.
Notation follows [mathematical_specification.md](mathematical_specification.md). Physics and the exogenous single-flip mechanism are unchanged.
A mode-dependent placement channel is **out of scope** (§9).

## 1. Question

When the channel prior supplied to an estimator is wrong, when does accounting for the channel still improve one-step decisions, and when is ignoring corruption safer?

- q_true = (β_t, π_t) generates the evaluation records.
- q_assumed = (β_a, π_a) is supplied to the estimator.

## 2. Channel parameterisation

- **Location axis:** π_α = (1−α)·Uniform(L) + α·δ_{L−1}, α ∈ [0,1]. α=0 is uniform; α=1 puts all mass on the last recorded action.
- **Rate axis:** β ∈ [0, ½].
- With (β=0.2, α=0) the channel is the pilot's P1; with (β=0.2, α=1) it is P3.

## 3. Analytic result for rate-only error (Opus derivation; must be verified numerically, T-E2-1)

Fix π_a = π_t = π. For a record S let ℓ_j be the clean-law likelihood ratios, A(S) = Σ_j π_j ℓ_j, and m̄(S) = Σ_j π_j ℓ_j μ(T_jS)/A.
Then exactly

  μ_2(S; β) = (1 − w_β) μ(S) + w_β m̄(S),  w_β = o_β A / (1 + o_β A),  o_β = β/(1−β).

A and m̄ do not depend on β. Under the true law, per record:

  regret(β_a) = κ (w_a − w_t)² (m̄ − μ)²,  naive regret = κ w_t² (m̄ − μ)²  ⇒  regret(β_a)/naive = (1 − w_a/w_t)².

**Consequences (predictions):**
- (i) **Underestimating β never hurts relative to ignoring the channel.** For 0 ≤ β_a ≤ β_t the exact composed belief is no worse than naive on *every* record.
- (ii) **Overestimation is safe on every record when o_a ≤ 2·o_t**, because w_a/w_t ≤ o_a/o_t. For β_t = 0.2 this means β_a ≤ 1/3; for β_t = 0.05, β_a ≤ 0.095.
- (iii) Beyond o_a = 2 o_t, records with weak corruption evidence (A small, w_a/w_t → o_a/o_t) are harmed and records with strong evidence are not. The aggregate break-even therefore lies above odds ratio 2 and depends on the distribution of A. It must be measured, not assumed.
- (iv) If β_t = 0, any β_a > 0 is strictly harmful wherever m̄ ≠ μ.

No such ordering exists for location error (π_a ≠ π_t): it is measured empirically.

## 4. Arms and information access

| arm | role | gets | does not get |
|---|---|---|---|
| R0 true-channel aware oracle μ_2(S; q_true) | evaluation reference (𝓔_true ≡ 0) | exact clean law, q_true | — |
| R1 exact composition μ_2(S; q_assumed) | isolates channel error | exact clean law (true η), q_assumed | q_true |
| R2 exact naive μ(S) (= R1 with β_a=0) | ignore-corruption reference | exact clean law | any channel |
| A5(q_a) / A5-naive | fitted HMM (η̂ by EM on clean data with probe) composed with q_a / with β=0 | physics, clean data, q_assumed | η, modes, labels, q_true |
| A4(q_a) / A4-naive | learned clean causal density (pilot arm) + candidate composition with q_a / with β=0 | physics, clean data, q_assumed | same |
| A6pd(q_a) / A6pd(0) | density-direct, prior-conditioned, trained on the broad Dirichlet(1) family (E1 arm), conditioned on q_a / on q=0 | physics, clean data + its own simulated channel family, q_assumed as input | same |
| **A6pd-cov(q_a) / A6pd-cov(0)** | coverage control: as A6pd, but the training family is the 50/50 mixture {β~U[0,½], π~Dirichlet(1)} ∪ {β~U[0,½], π=π_α with α~U[0,1]}, explicitly covering the location axis including the last-position vertex | same as A6pd | same |

All deployable arms receive the same q_assumed. None receives q_true, η, modes or corruption labels.
The probe (a_L, s_{L+1}) is a training target only.
A6pd-cov reuses A6pd's tuned configuration; no new tuning (declared).

## 5. Estimands (all under the TRUE law, Rao–Blackwellised over q_true's corruption views, prefix as the sampling unit)

- **Regret:** 𝓔_true(μ̂; q_a) = E_{q_true}[κ(S)(μ̂(S; q_a) − μ_2(S; q_true))²].
- **Benefit over ignoring the channel:**
  - *Primary for each arm:* pipeline-matched, B_pipe = 𝓔_true(same arm with β_a = 0) − 𝓔_true(arm with q_a). This is the deployable choice "compose or not" for a given model.
  - *Also reported:* B_exact = 𝓔_true(R2) − 𝓔_true(arm), using the exact naive filter as in the brief.
  - Positive values mean improvement.
- **Decomposition (no additivity assumed):** 𝓔_true(μ̂) = L + C + X, where
  - L = E[κ(μ̂ − μ_2(q_a))²] is the learning term,
  - C = E[κ(μ_2(q_a) − μ_2(q_t))²] = 𝓔_true(R1) is the channel term,
  - X = 2E[κ(μ̂ − μ_2(q_a))(μ_2(q_a) − μ_2(q_t))] is the cross term.
  All three are reported.
- **Raw decision cost:** E_true[ρ²s_L² + q_w − κμ_H² + κ(μ̂ − μ_H)²]. μ_H is the conditional mean given (H, θ) under the exogenous channel.
- **Strata, using realised hidden-mode costs:** D = κ[(μ̂ − m_L)² − (μ̂_naive − m_L)²] against the same pipeline's naive output, positive = costlier. Reported on:
  - clean records (θ = none under q_true),
  - corrupted records,
  - the natural-switch stratum E_{L−1} (m_{L−2} ≠ m_{L−1}).

  Strata show where the aggregate benefit or harm comes from (e.g. overstated β pays on clean records). Hidden-mode strata use m_L, never μ_H.

## 6. Design

**Exact screening** (exploratory; both physics q_w ∈ {0.02, 0.5}, L=8, η=0.05; 100,000 fresh prefixes, master 8101):
- **Rate axis:** α ∈ {0, 0.5, 1} (α_a = α_t) × β_t ∈ {0, 0.05, 0.2, 0.5} × β_a ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5}.
- **Location axis:** β_a = β_t = 0.2 × α_t ∈ {0, 0.25, 0.5, 0.75, 1} × α_a ∈ {0, 0.25, 0.5, 0.75, 1}.
- Exact-match diagonals are the controls (R1 = R0 exactly).

**Learned confirmation** (lead physics q_w=0.02 only; N ∈ {10³, 10⁵}; training seeds 1–10, where seeds 1–5 reuse the pilot's training prefixes; 20,000-prefix confirmation test set, master 8201, independent of screening).
The **confirmatory cells are fixed now**, not selected from screening:
- *Rate axis* (β_t = 0.2): α_t = α_a ∈ {0, 1} × β_a ∈ {0.05, 0.1, 0.2, 0.3, 0.5} — 10 cells. These include the matched controls and cover prediction (ii): 0.3 is safe and 0.5 is beyond odds ratio 2.
- *Location axis* (β = 0.2 on both sides): (α_t, α_a) ∈ {(0,1), (1,0), (0.5,0), (0.5,1)} — 4 cells.

Learned composition arms may also be evaluated on the whole screening grid as **exploratory** output, labelled as such.

## 7. Comparisons, margins, uncertainty

**Confirmatory (per cell × N):**
- Q1: B_pipe(A4).
- Q2: 𝓔_true(A4) − 𝓔_true(A6pd).
- Q3: 𝓔_true(A4) − 𝓔_true(A6pd-cov).
- Q4: B_exact(R1). This is the exact reference, also reported on the confirmation set.

**Secondary:** B_pipe for A5, A6pd and A6pd-cov; B_exact for all arms; the L/C/X decomposition; strata; raw costs.

**Margin:** Δ = clip(0.10·V_2,true, 0.002·VOI_clean, 0.02·VOI_clean), where V_2,true = 𝓔_true(R2) on the confirmation set.

**Verdict for a benefit-type quantity b (positive = good):**

| verdict | condition |
|---|---|
| material improvement | CI_lo > Δ |
| material harm | CI_hi < −Δ |
| equivalent | CI ⊂ (−Δ, Δ) |
| noninferior | CI_lo > −Δ, otherwise |
| inconclusive | otherwise |

"No evidence of harm" (CI_hi ≥ 0) is not reported as noninferiority.

**Uncertainty:**
- Exact arms: per-prefix paired values, normal 95% CI (prefix-level). Views of a prefix are averaged with q_true weights before the CI.
- Learned arms: 95% t-interval over the 10 training seeds of paired seed-level differences, **conditional on the shared finite confirmation test set**. The test-set Monte Carlo error is reported separately: the prefix-level SE of the seed-averaged difference.
- Precision target: CI half-width ≤ Δ/2. Report achieved precision; do not add seeds after looking at learned outcomes.

**Multiplicity:** 14 cells × 2 N × 3 confirmatory comparisons are all reported, without correction. Conclusions rest on patterns across cells, not on single cells.

## 8. Validation (before any reported result) and stopping / escalation

**Tests:**
- T-E2-1: the convex-combination identity of §3 per record, to 1e-10.
- T-E2-2: per-record inequality checks. For β_a ≤ β_t, and for o_a ≤ 2 o_t, R1 regret ≤ naive regret on every record (zero violations allowed beyond 1e-12).
- T-E2-3: q_a = q_t ⇒ R1 regret ≡ 0; β_a = 0 ⇒ R1 = R2 exactly.
- T-E2-4: 𝓔_true via RB weights equals a Monte Carlo estimate with θ sampled from q_true (|z| ≤ 4).
- T-E2-5: π_α properties and the coverage-family sampler.
- T-E2-6: the L + C + X decomposition is exact per sample.

**Reproduction check:** A4 and A6pd with seeds 1–5 at the matched cells P1/P3 must reproduce the pilot/E1 values within seed noise. Bit-identical is preferred; if not bit-identical, explain the difference.

**Stop and return to Opus if:** a validation test fails; reproduction fails without explanation; the exact screening contradicts §3 (i)–(ii); or a learned result reverses the exact reference's sign at the matched controls.
Otherwise run screening → learned → analysis without pausing, and return to Opus with the results.

**Budget:**
- Screening: minutes (queries are computed once per record and composed under every prior).
- Learned training: 3 density arms × 2 N × 10 seeds = 60 trainings, ≤ 2 CPU-hours.
- Evaluation: A6pd/A6pd-cov need one forward pass per q_assumed.

Checkpoints are saved to `results/e2/checkpoints/` (state_dict + config + data seeds + η̂). Provenance: base commit plus a patch of any uncommitted code changes, saved with the results.

## 9. Not in E2

- **Mode-dependent placement.** It needs a new reference, p(m,θ|S) ∝ p_0(T_θS, m)·Pr(θ | T_θS, m), derived and validated by enumeration first.
- Estimating q from corrupted data (an adaptive arm).
- Unknown physics; generic learners; compatibility training; closed-loop control; MTM/DT poisoning.

## 10. E2b — robust declared priors under prior uncertainty

Specified by Opus 5.5, 2026-10-04, after the E2 review and **before any E2b data**. Same physics and exogenous single-flip mechanism as E2.

**Question.** If the corruption prior is only known to lie in a set, which prior should be declared, and how much of the matched benefit survives?

**Analytic basis.** Let the true channel q be unknown, independent of the record, with uncertainty weights w over a set U. The compound law of a record is p(S) = Σ_q w_q Σ_θ q_θ p_0(T_θS) = Σ_θ q̄_θ p_0(T_θS), with q̄ = Σ_q w_q q. The channel enters linearly. Hence:
- E[m_L | S] under the compound law equals μ_2(S; q̄).
- Since true-law regret differs from expected cost by a term independent of the estimator, **composition with q̄ minimises the w-weighted average true-law regret over all S-measurable estimators** (population statement).
- With U = {β_t} × {α_t} and product weights, q̄ = (β̄, π_ᾱ) with β̄ = mean β_t and ᾱ = Σ w β α / Σ w β (= mean α_t for product-uniform w). So q̄ stays in the π_α family.

**Design.**
- Truth set U: β_t ∈ {0.02, 0.05, 0.1, 0.2, 0.35, 0.5} × α_t ∈ {0, 0.25, 0.5, 0.75, 1} (30 channels). Both physics. Uniform weights w over U are declared for the Bayes criterion.
- Candidates: β_a ∈ {0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5} × α_a ∈ {0, 0.125, …, 1} (9 values), plus q̄ exactly.
- Criteria, fixed now:
  - K1 Bayes risk = mean over U of 𝓔_true.
  - K2 no-harm = max over candidates of min over U of B_exact; report the set with min_U B ≥ −Δ_t, and the best K1 within it.
  - K3 minimax regret = min over candidates of max over U of 𝓔_true.
- **Selection** of one prior per criterion uses the E2 screening prefixes (master 8101, 100k). **Confirmation** uses the independent E2 confirmation set (master 8201, 20k), for:
  - the three selected priors and q̄,
  - two references: the naive filter, and "declare the single most plausible channel" (β_a = 0.2, α_a = 0.5),
  - evaluated over all 30 truths with the exact clean law and with the **saved A4 checkpoints** (N ∈ {10³, 10⁵}, seeds 1–10; no retraining; A4 queries are prior-independent).
- Report per prior: K1, K2, K3, the number of truths with material harm (CI_hi < −Δ_t) or material improvement (CI_lo > Δ_t), and the share of the matched benefit retained (B_exact(prior)/B_exact(q_true)) per truth.
- **Descriptive hypothesis check** (from the E2 review) on all E2 + E2b screening pairs: is B_exact < 0 confined to declared/true lag-0 odds ratios above a threshold? Report the scatter and the empirical threshold. This is descriptive only.

**Validation.**
- T-E2b-1: q̄ algebra. The mixture of the 30 channels' view weights equals q̄'s view weights exactly.
- T-E2b-2: Bayes optimality, statistical. On a large Monte Carlo set with q drawn from w per prefix, μ_2(S; q̄) has lower mean regret than every grid candidate, or is within 4 paired SE.
- T-E2b-3: when U is a singleton, K1 = K3 = 0 and the selected prior equals the truth.

**Uncertainty / stopping.**
- Exact quantities: prefix-level paired CIs. A4: seed-level t-intervals conditional on the shared confirmation set.
- Stop and return to Opus if T-E2b-2 fails materially (q̄ beaten by more than 4 SE). That would mean a broken reference or a wrong derivation.
- Budget: about 120 compositions per physics on the screening set; A4 evaluation 4 priors × 30 truths × 20 checkpoints from cached queries. Well under 1 CPU-hour; no training.

**Not in E2b:** estimating q from corrupted data (empirical Bayes across records), mode-dependent placement, new learners.
