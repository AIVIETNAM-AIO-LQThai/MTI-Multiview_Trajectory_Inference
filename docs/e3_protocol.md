# E3 protocol: channel identification and adaptive composition

Settled by Opus 5.5, 2026-10-05, **before any E3 data**. Branch `exp/channel-identification` (from `fc74aef`, synchronised with origin).
Notation: [mathematical_specification.md](mathematical_specification.md), [e2_protocol.md](e2_protocol.md). The physics and the exogenous single-flip channel are unchanged.

## 1. Question and contribution

**Question.** E2/E2b showed that composition is only as good as the declared prior, and that no fixed declaration is both Bayes-good and harmless. Can the shared channel be **identified from unlabeled corrupted records**? Does adaptive composition then remove the misspecification risk? How many records does that take, and what goes wrong when the clean law itself is estimated?

**What E3 can add to the thesis:**
- An identification result (positive and negative).
- The sample cost of turning "declare a prior" into "learn the prior".
- The small-sample harm of adaptation compared with fixed declarations.
- How clean-law error is absorbed into inferred corruption.

**What it cannot answer:**
- The original MTM question, whether multi-view consistency detects *harmful trajectories per record* beyond residual, rarity or dynamics signals, is **not** tested.
- E3 identifies the channel *in aggregate*, and one of its baselines does so from plain residual–action moments. If that baseline works, it shows that in this simulator aggregate corruption information is available from simple dynamics statistics. Any future multi-view detection claim would have to beat such statistics; E3 does not test one.

## 2. Identification (verified independently by Opus, 2026-10-05; ad-hoc script, to become tests T-E3-1/2)

Let z̃_k = ã_k δ_k with δ_k = s_{k+1} − ρ s_k (recorded action ã_k = σ_k a_k, where σ_k = −1 iff the flip is at k).

- **(N1) L = 1: β is not identified, but changes the decision.**
  - Because of the global symmetry (m, a) → (−m, −a), p₀(TS) = p₀(S). Hence p(S) = (1−β)p₀(S) + βp₀(TS) = p₀(S) for every β.
  - Yet μ₂(S; β) = (1−2β)μ(S).
  - Verified exactly: |ℓℓ(TS) − ℓℓ(S)| = 0; μ₂ − (1−2β)μ ≤ 1e-16.
- **(N2) L = 2: β is identified; the allocation π is not, and it changes the decision.**
  - Global symmetry gives p₀(T₁S) = p₀(T₀S) and μ(T₁S) = −μ(T₀S). So p(S) = p₀(S)[(1−β) + βℓ₀], which depends on q only through β.
  - Yet μ₂ ∝ (1−β)μ(S) + (q₀ − q₁)ℓ₀μ(T₀S).
  - Verified exactly: equal record laws for allocations (1,0), (0,1), (½,½), with beliefs differing by up to 1.49.
- **(P) L ≥ 3, η < ½, η known: q is identified.**
  - For i ≠ j: E[z̃_i z̃_j] = c² q_a² (1−2η)^{|i−j|} (1 − 2(q_i + q_j)). The derivation uses that δ_k does not depend on earlier noise and that the channel is independent of H.
  - Every pair sum q_i + q_j is identified, and q_i = ½[(q_i+q_j) + (q_i+q_k) − (q_j+q_k)].
  - Verified by Monte Carlo (400k prefixes): formula error ≤ 0.017 on a unit scale; recovered q within 0.0025 (L=4), 0.0035 (L=8, η=0.05) and 0.016 (L=8, η=0.2).

**Qualifications.**
- (i) The information in pair (i, j) scales as (1−2η)^{2|i−j|}: distant pairs, and η near ½, are weakly informative. At η = ½ nothing is identified, but then V₂ = 0 as well.
- (ii) These statements concern records **without** the logged probe. If adaptation data included the next clean transition, even L = 1 would become identified. E3 therefore uses prefixes only (§4).
- (iii) Identification uses the clean law (here η) as known. With an estimated clean law the identified object is the channel *relative to the fitted clean model*.

**(A) Absorption of clean-law error (analytic prediction for the moment estimator; qualitative for likelihood).** With true η and fitted η̂, the moment pair-sum estimate is

  ŝ_ij = ½ [1 − (1 − 2s_ij)·((1−2η)/(1−2η̂))^{|i−j|}].

So η̂ < η (a clean model that is **too persistent**) gives ŝ > s. Natural mode switches are read as sign inconsistencies, i.e. spurious corruption, and the bias grows with pair distance. η̂ > η gives the opposite. At β_true = 0 this predicts **spurious positive β̂ exactly when the clean model is too persistent**. The likelihood estimator can absorb any clean-law misfit the same way; E3 measures it.

## 3. Arms

**Clean-law sources** (used both to estimate q and to compose):

| source | what it is | role |
|---|---|---|
| EX | true clean law (true η) | diagnostic: isolates channel-estimation error |
| HMM-N | fitted HMM with η̂ from E2 unit JSON, N ∈ {10³, 10⁵}, seeds 1–5 | structured learned clean law; no new fitting needed |
| A4-N | saved A4 checkpoints, N ∈ {10³, 10⁵}, seeds 1–5 | learned neural clean law; **no new training** |
| ETA± | exact law with η̂ ∈ {0.03, 0.08} (true 0.05) | controlled absorption probe |

**Channel estimators** (given adaptation records R and a clean-law source):

| estimator | definition |
|---|---|
| F-q̄ / F-noharm / F-minimax / naive | fixed E2b declarations: (0.2033, α 0.5) / (0.075, α 0) / (0.15, α 0.875) / β=0 |
| MLE | EM for the mixture weights over {none, 0, …, L−1}: maximise Σ_r log Z_q(S_r), Z_q = q_none + Σ_j q_j ℓ_j(S_r); full simplex; start from uniform (½ none, ½/L each); stop at Δloglik < 1e-10 or 5,000 iterations |
| MAP-q̄ | the same EM with a Dirichlet prior of κ₀ = 30 pseudo-records centred on q̄ (M-step (Σ post + κ₀ q̄)/(n + κ₀)); shrinks toward the fixed Bayes declaration at small n |
| PP | posterior predictive over the 110-candidate E2b grid G with uniform prior: weights ∝ Π_r Z_g(S_r); predictive channel = Σ_g post_g q_g (by C18 this is the Bayes composition under the grid prior). **Off-grid truths are deliberately misspecified for PP** |
| MOM | moment baseline: pair sums from Σ_r z̃_i z̃_j / n with f_ij = c²q_a²(1−2η̂)^{|i−j|}; weighted least squares (weights f_ij²) for q, then projection onto {q ≥ 0, Σq ≤ 1} (NNLS). Needs only physics + η̂ (the A4 source uses the same-seed HMM η̂, declared) |
| ORACLE-q | composition with q_true (reference; regret 0 for EX) |

**Access:**
- Adaptation records contain (s, a) only: no probe, no modes, no θ.
- Evaluation uses a disjoint fixed held-out set.
- q_true is used only to generate data and to score.

## 4. Design

**Truths** (lead physics: ρ=0.9, c=1, q_a=1, q_w=0.02, λ=0.1, η=0.05, L=8):

| id | channel | why |
|---|---|---|
| T0 | β=0 | spurious-corruption / absorption control |
| T1 | (0.02, α=0) | weak and diffuse; harmed by q̄ in E2b |
| T2 | (0.05, α=0.25) | weak |
| T3 | (0.2, α=0.5) | mid; equals the "most plausible" declaration |
| T4 | (0.35, α=1) | strong, concentrated |
| T5 | β=0.125, point mass at lag 1 | **off-family**: not a π_α prior, not in U |
| T6 | β=0.2, π_j ∝ 2^{−lag} | **off-family** |
| T7 | β=0.3, half on lag 0, half on lag 7 | **off-family**, bimodal |

**Adaptation sets:**
- n ∈ {10, 30, 100, 300, 1,000, 3,000, 10,000} corrupted prefixes from q_true.
- R = 20 independent replicates per (truth, n): master 8301, cell = truth id, chunk = 1000·n_index + replicate.
- Evaluation: fixed 20,000 prefixes (master 8401), disjoint from adaptation, from E2's sets, and from training data.

**Identification controls** (EX only): (N1) L=1, (N2) L=2, (P) L=8; η = 0.05, q_w = 0.02, same seeds scheme.
- L=1: truths β ∈ {0.1, 0.3}.
- L=2: β=0.2 with allocations (1,0) and (½,½).
- n up to 10,000, R = 20.
- The extra control η = ½ (L=8) is optional, since it is trivially V₂ = 0.

**Clean-law × estimator matrix:**
- EX, HMM-10⁵ (seed 1), HMM-10³ (seed 1), ETA−, ETA+: all estimators, all truths, all n, R = 20.
- A4-10⁵ and A4-10³: seeds 1–5 × R = 4, n ∈ {100, 1,000, 10,000}, all truths, adaptive estimators + naive + F-q̄.
- HMM seeds 2–5 at n ∈ {100, 1,000, 10,000} as a seed-sensitivity check.

## 5. Estimands, comparisons, uncertainty

**Per (source, estimator, truth, n, replicate),** on the evaluation set under q_true:
- regret 𝓔_true;
- benefit over the same pipeline's naive output (B_pipe), and over exact naive (B_exact);
- harm indicator B_pipe < −Δ_t, with Δ_t = clip(0.10 V₂,t, 0.002 VOI, 0.02 VOI) and V₂,t from EX naive;
- channel error: |β̂ − β|, TV(π̂, π) for β > 0, and the lag-0 mass ratio q̂_{L−1}/q_{L−1};
- gap to ORACLE-q.

**Primary comparisons (predeclared):**
- **Q1 (crossover):** MLE vs F-q̄ regret per truth and n (replicate-paired). Report the smallest n where MLE is materially better (CI_lo > Δ_t) and any n where it is materially worse.
- **Q2 (safety):** harm rate (fraction of replicates with B_pipe < −Δ_t, with a Wilson 95% CI) for MLE, MAP-q̄, PP, MOM, F-q̄ at each n. The headline is the **largest harm rate over truths** per (estimator, n).
- **Q3 (shrinkage):** MAP-q̄ vs MLE at n ≤ 300.
- **Q4 (efficiency):** MOM vs MLE regret ratio at each n.
- **Q5 (absorption):** at T0, the distribution of β̂ and the harm rate for EX, ETA−, ETA+, HMM-10³/10⁵ and A4-10³/10⁵. Prediction (§2 A): ETA− (too persistent) gives β̂ > 0 and harm; ETA+ gives β̂ ≈ 0 after projection.
- **Q6 (off-family):** at T5–T7, PP (misspecified grid) vs MLE vs F-q̄.

**Identification controls:**
- At L=1 the gap to ORACLE-q must **not** vanish as n grows (β̂ does not converge to β). At L=2 β̂ must converge while TV(π̂, π) and the regret gap do not, for (1,0).
- At L=8 under EX, β̂ → β and the regret gap → 0 (positive control).
- These are tests of the theory: a failure is a stop condition.

**Uncertainty:**
- Replicate-level 95% t-intervals (R = 20), paired across estimators within a replicate, conditional on the shared evaluation set. Evaluation-set error is reported separately (prefix-level SE for one replicate).
- A4: average replicates within a seed, then a t-interval over 5 seeds.
- No multiplicity correction; conclusions rest on curves across n, not single cells.

## 6. Validation tests (before any reported result)

| test | content |
|---|---|
| T-E3-1 | N1 and N2 exact identities (L=1 law invariance and μ₂ = (1−2β)μ; L=2 equal laws for different allocations, beliefs differ) |
| T-E3-2 | cross-moment formula per pair (Monte Carlo, \|z\| ≤ 4 using per-pair SE) and recovery of q at L=8 within 0.01 |
| T-E3-3 | EM log-likelihood is non-decreasing; EX, L=8, n = 20,000, β = 0.2 gives \|β̂ − β\| < 0.02 |
| T-E3-4 | PP concentrates on the true grid point (posterior weight > 0.9) when truth ∈ G and n is large; predictive q = Σ post·q |
| T-E3-5 | MAP-q̄ → MLE as κ₀ → 0 and → q̄ as κ₀ → ∞ |
| T-E3-6 | absorption sign: MOM at β = 0 gives a pre-projection β̂ > 0 with η̂ < η and < 0 with η̂ > η (matches §2 A) |
| T-E3-7 | access: adaptation records are Prefix objects only; adaptation, evaluation and training masters are disjoint |

## 7. Stop / escalate, budget

- **Stop and return to Opus if:**
  - any validation test fails;
  - a positive control fails (EX, L=8: |β̂ − β| > 0.02 at n = 10,000, T3);
  - a negative control shows identification;
  - a result reverses the predicted absorption sign.
- Otherwise run everything and return to Opus with the report.
- **Budget:** no training. Roughly 4,500 compositions per clean-law source on the 180k evaluation records, plus A4 queries for 20 checkpoints; ≤ 2 CPU-hours on 18 processes. Cache clean-law queries per evaluation set and per adaptation set.

## 8. Not in E3

Per-record detection of harmful trajectories (the MTM question); mode-dependent placement; non-stationary channels; generic learners or unknown physics; new neural training; closed loop.
