# Learned pilot protocol (v2 — approved design, NOT RUN)

v1 drafted by the Opus planning pass; **v2 settled by the Opus evidence review, 2026-10-04**, from the oracle report
[oracle_v1_report.md](reports/oracle_v1_report.md). Running it requires the user's authorization: the first assignment excluded the learned suite.
Notation and exact targets: [mathematical_specification.md](mathematical_specification.md).

## v2 decisions (supersede v1 where they conflict)

1. **Lead physics:** ρ=0.9, c=1, q_a=1, λ=0.1, η=0.05, **L=8, q_w=0.02**.
   **Secondary physics:** the oracle primary (q_w=0.5, L=8). Clean-trained arms are evaluated at every prior (evaluation only, no extra training);
   channel-trained arms at P1/P2 only (review 2).
   **L=32 dropped:** V_2/VOI_clean ≤ 1.6% there, and the L contrast is confounded with the per-position flip rate β/L.
2. **Test priors are the prior-shift design:**

   | prior | β | π |
   |---|---|---|
   | P1 | 0.2 | uniform |
   | P2 | 0.5 | uniform |
   | P3 | 0.2 | δ at lag 0 |
   | P4 | 0.5 | δ at lag 0 |
   | P5 | 0.2 | recent-weighted: π_j ∝ 2^{−(L−1−j)} (lag 0 gets ≈0.50) — *added by review 2* |

   P1 → P5 → P3 is a path of increasing location concentration at fixed β=0.2:
   - P5 is in the interior of A6p-broad's Dirichlet(1) family but outside A6p-narrow's family.
   - P3/P4 are vertices of the simplex, i.e. limits that no Dirichlet(1) draw reaches.

   This separates in-family interpolation (P5) from extrapolation (P3/P4).

   Arms trained only on clean data (A1, A2, A3, A4, A5) are trained **once per physics** and evaluated at every prior, so β ∈ {0.2, 0.5} costs no
   extra training. Channel-trained arms (A1r, A6) are trained at P1. A6p is trained over its declared family.
3. **Task 0 (DONE 2026-10-04):** compute exact oracle V_2, I_loss and VOI_clean at P3 and P4 for both physics. P1/P2 already exist for
   q_w=0.02 (map cells), and for q_w=0.5 they are the primary cell and the L8/q_w0.5/β0.5 cell. Fix the Δ_min table from these numbers before training.
   Report the oracle P3/P4 results as an addendum to the oracle report.
4. **Δ_min(physics, prior) = clip(0.10·V_2, 0.002·VOI_clean, 0.02·VOI_clean)** (revised by review 2, before any learned run).
   - Rationale for 10% of V_2: the exact naive filter has endpoint 𝓔 = V_2, so 10% of V_2 is one tenth of the gap between ignoring the channel and
     knowing it exactly.
   - Floor: prevents instability when V_2 is small.
   - Cap (new): when V_2 is comparable to or larger than the whole decision value of clean history (P3/P4: V_2/VOI_clean = 0.27–1.59),
     "10% of V_2" would call cost differences of up to 40% of the clean-oracle cost immaterial, which is not credible. 2% of VOI_clean
     (≈0.062 at q_w=0.02, ≈5% of the clean-oracle cost; ≈0.080 at q_w=0.5, ≈2.7%) is a clearly noticeable decision effect in raw cost units.

   Raw values ([addendum](reports/oracle_v1_priors_addendum.md); VOI_clean from the same cells):

   | physics | P1 | P2 | P3 | P4 | P5 |
   |---|---|---|---|---|---|
   | q_w=0.02 | 0.0090 | 0.0416 | 0.0624 (cap) | 0.0620 (cap) | 0.0629 (cap) |
   | q_w=0.5 | 0.0080 (floor) | 0.0305 | 0.0794 (cap) | 0.0797 (cap) | 0.0441 (10% V_2) |

   Power is set separately (Δ_min/2 half-width rule). The cap makes P3/P4 demand more precision than v2 did, so if 10 seeds do not reach it,
   report the achieved precision rather than raising Δ_min.
5. **Reference lines in every table:** exact aware (𝓔=0), exact naive (𝓔=V_2), learned naive A1.
   An aware-belief estimator with 𝓔 ≥ V_2 is no better than ignoring the channel exactly.
6. **Structural notes for interpretation, not design changes:**
   - At β=½ the folded route gives weight (1−2β)=0 to μ(S), so G uses folded queries only; M and the average still use μ(S).
   - Under P3/P4 the candidate route queries S and T_{L−1}S, while the folded route queries S (unless β=½) and O_{L−1}. The inference-cost gap
     between routes therefore nearly vanishes, and H3 compute comparisons should be read per prior.
   - **P4 reduction (predeclared, review 2):** at β=½ with π=δ_{L−1}, μ_2 = ψ_{L−1} exactly (spec T4).
     - The folded route collapses to one masked-belief query: Ĝ = ψ̂_{L−1}. The sign head cancels.
     - The candidate route still needs μ̂(S), μ̂(T_{L−1}S) and r̂.
     - A G-over-M advantage at P4 is therefore partly a structural reduction. It is a valid estimation result for that prior, but must not be
       generalised to other priors without P1/P2/P3/P5 support. A3 is not equivalent to Ĝ there.
7. **Required diagnostics, added from the oracle:** natural-switch stratum E_{L−1} and clean-record (θ=none) harm for every arm. Use the
   hidden-mode realised-cost difference against the exact naive action.
8. **A1r per-prior refit:** gets its own row in the H3 tables, labelled "refit per prior", with refit data and compute disclosed. It is not
   counted as test-time composition.
9. **Feasibility gate:** before the full grid, time one arm × one seed at N=10⁵ on the lead physics. If the projected total exceeds about 8 CPU-hours,
   cut the tuning grid from 8 to 4 configurations for all arms equally, and record the decision.

## Purpose

Test H2 (finite-estimation differences between query routes) and H3 (cost/benefit of test-time
composition under prior shift) at matched data, information access and disclosed compute, in the physics and priors fixed above.

## Information access (all arms)

- Known to every learned arm: ρ, c, q_a, q_w, λ, L. Not known: η, modes, θ labels.
- Test channel prior (β, π ≡ q) is supplied to every arm able to consume it; a fixed-prior arm trained at
  P1 cannot consume a new prior — this is reported as a flexibility distinction, not a learning failure.
- Training supervision: clean prefixes H plus their logged probe `(a_L, s_{L+1})`. Probe loss
  `(δ_L − c a_L h(I))²`, `δ_L = s_{L+1} − ρ s_L`; I never contains the probe or any future transition.
  On clean inputs the minimiser is μ(H); on channel-simulated inputs with the **original clean probe**
  it is μ_2 under the simulated channel.
- Forbidden: oracle μ_2/ψ/r targets, hidden modes, θ labels as inputs or training labels; assigning the
  original probe target to a transformed record `T_jH` as if it were a clean sample.
- Allowed self-supervision on folded views: `O_j(H)` is a valid clean marginal query, so probe loss on
  O_j(H) targets ψ_j, and Bernoulli loss with label `1[a_j ≥ 0]` on O_j(H) targets p_j^+.

## Arms

| Id | Arm | Trained on | Test-time queries / record |
|---|---|---|---|
| A0 | Exact oracle μ_2 | — | — |
| A1 | Naive learned clean filter f(S) | clean H, probe loss | 1 |
| A1r | Restricted-summary recaliber g(|s_L|, f(S)), odd in f, fitted to **this** f realisation | channel-simulated calibration prefixes, probe loss | 1 |
| A2-G / A2-M / A2-avg | One shared fixed-view model F with belief head (full or folded query; mask flag; |a_j| retained) and sign head at the masked slot; folded composition G, candidate composition M with ℓ̂ = r̂/(1−r̂) and μ̂(T_jS) from the belief head, and (G+M)/2 | clean H and O_j(H) only | G: L+1; M: 2L+1; avg: 2L+1 |
| A3 | Unweighted masked ensemble `(μ̂(S)+Σ_jψ̂_j)/(L+1)` from F | (same F) | L+1 |
| A4 | Causal density scorer: same backbone, transition density `δ_k | past, a_k ~ w_k N(c a_k, q_w) + (1−w_k) N(−c a_k, q_w)` (mixture-capable; exact family contains the truth); μ̂ = (2w_L−1); candidate composition with density ratios | clean transitions incl. probe transition, max likelihood | L+1 |
| A5 | Fitted symmetric 2-state HMM, η̂ by Baum–Welch, emissions from known physics; exact composition (§4) with test prior | clean H incl. probe transition | O(L) analytic |
| A6 | Channel-trained direct regression h(S), fixed prior P1 | channel-simulated corruptions of training prefixes, probe loss | 1 |
| A6p | Prior-conditioned direct h(S, q) | as A6, priors drawn from a declared family | 1 |

Disclosures: A5 receives the correct structural prior (2 symmetric states, known emission form) — it is a
strong, structurally privileged baseline, not a "generic" learner. A4's mixture family also contains the
truth; a generic K=4 Gaussian MDN (means affine in a_k) is an optional ablation.

## Prior-shift comparison (H3)

Test priors: P1 β=0.2 uniform (primary); P2 β=0.5 uniform; P3 β=0.2 π=δ_{lag 0}; P4 β=0.5 π=δ_{lag 0}; P5 β=0.2 recent-weighted (decision 2).
A6p-narrow: train family β~U[0,0.3], π uniform (P2–P5 are extrapolations). A6p-broad: β~U[0,½],
π~Dirichlet(1_L) (P1, P2, P5 inside its support; P3/P4 are simplex vertices that no draw reaches, i.e. extrapolation limits). Composition arms (A2, A4, A5) and A1r
(refitted per prior using simulated calibration data — disclose refit cost) receive each test prior.
Report every (arm, prior) cell including flexibility failures.

## Budgets and matching

- Data: total N ∈ {10³, 10⁵} independent clean prefixes per arm for training+calibration (each arm discloses
  its split; A1r uses ¼ of N for calibration, A1 uses ¾ — the same split for A1/A1r realisations).
  Channel-simulated arms resample corruption per epoch on the same prefixes (views, not new prefixes).
  Validation: independent N/4 prefixes, selection by **deployable** criterion (probe loss on
  channel-simulated validation records at the test prior), identical for all arms. Test: 2×10⁵
  independent prefixes, RB over θ, exact μ_2 (evaluation only).
- Model: one backbone family for all neural arms (causal GRU, hidden 64, ≤1e5 params); heads small MLPs.
- Training compute: matched optimiser steps × batch; count view-forward-passes per prefix (A2 sees 1 full +
  K folded views per prefix per step; A6/A6p get the same number of corrupted views). Report wall time.
- Tuning: fixed 8-config grid (lr × weight decay × epochs) per arm, tuned on seed 0; then 5 training seeds
  at the selected config (increase to 10 if the power check below fails).
- Feasibility (to measure before committing): ~7 arms × 2 N × (8+4) runs ≈ 170 CPU runs; L=8 GRU on
  10⁵ prefixes is expected (not measured) to take minutes per run on the 20-core CPU host.

## Endpoints and analysis

- Primary: `𝓔(μ̂) = E[κ(μ̂−μ_2)²]` at each test prior, plus captured opportunity `1 − 𝓔/V_2` for estimators of
  the aware belief.
- Smallest meaningful difference: `Δ_min = max(0.10·V_2, 0.002·VOI_clean)` per (physics, prior); see v2 decision 4. Statistical power is
  decided separately: seeds are increased until the 95% CI half-width of paired seed-level differences is ≤ Δ_min/2.
- Comparisons are paired (same test prefixes, same training-seed index). Uncertainty = t-interval across
  training seeds of paired differences (test-set MC error reported separately).
- H2 supported in a cell if some route pair differs by ≥ Δ_min with CI excluding 0; "no material
  difference" requires the CI inside ±Δ_min (equivalence), not merely a non-significant gap.
- H3 supported if composition arms beat the best direct arm with matched prior access by ≥ Δ_min at a
  shifted prior while staying within Δ_min of it at P1; report inference-query cost alongside.
- Mechanism diagnostics (required, not headline): signed and squared errors by θ stratum (none / lag) and
  natural-switch strata (hidden-mode strata use κ[(μ̂−m_L)²−(μ_2−m_L)²], spec §8 A4); query-specific
  errors of μ̂(T_kS) vs μ(T_kS) by contamination count, ψ̂ vs ψ, r̂ log-loss; distribution of Ĉ_j.
- Route-gap identity: with shared heads, ℓ̂ = r̂/(1−r̂) and shared Ẑ, `Ĝ − M̂ = Ẑ⁻¹Σ_j q_jĈ_j/(1−r̂_j)` —
  verify numerically on the learned model. This is an identity about route **disagreement**, not an error
  bound: all-zero belief heads are mutually consistent and wrong; weighted residuals can cancel. A zero
  average gap does not dismiss a mechanism — inspect signed per-stratum gaps.

## Deferred (need a result from this pilot to justify)

Mask diversity; separate-encoder ablations; compatibility/distillation training. If distillation is later
tested: stop-gradient on teacher outputs does not freeze a shared teacher; require frozen teacher
parameters, a justified update scheme, or measured teacher drift and accuracy.

## Resolved review items (2026-10-04)

1. Cells: lead physics q_w=0.02, L=8; secondary q_w=0.5, L=8 at P1/P2; L=32 dropped (see v2 decision 1).
2. Δ_min: definition confirmed; raw values for P1/P2 listed in v2 decision 4; values for P3/P4 come from Task 0.
3. A1r per-prior refit: own row, not composition (v2 decision 8).
