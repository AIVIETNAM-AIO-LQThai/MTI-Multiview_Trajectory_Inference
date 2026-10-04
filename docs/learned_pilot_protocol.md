# Learned pilot protocol (draft v1 — NOT RUN)

Drafted by Opus planning pass 2026-10-04. Status: **pending Opus evidence review of the oracle report**;
numerical thresholds marked ⟨TBD-oracle⟩ are filled from the oracle results before any run. Notation and
exact targets: [mathematical_specification.md](mathematical_specification.md).

> **Update after oracle report (2026-10-04, pending Opus review):** oracle V_2 in the primary cell is 0.0712 (1.8% of VOI_clean 4.007), so
> Δ_min = max(0.10·V_2, 0.002·VOI_clean) = 0.0080 and the absolute floor binds (≈11% of V_2). The most discriminating cell is L=8, q_w=0.02, β=0.5
> (V_2 = 0.416, Δ_min = 0.042). L=32 has V_2/VOI ≤ 1.6%; low priority. Natural-switch states (aware oracle worse by 0.56 ± 0.10 in primary) must be in the diagnostics.

## Purpose

Test H2 (finite-estimation differences between query routes) and H3 (cost/benefit of test-time
composition under prior shift) at matched data, information access and disclosed compute. Primary cell:
the oracle primary cell (L=8, q_w=0.5, β=0.2, η=0.05, uniform π); a second cell is chosen after the
oracle map (preferably one with larger V_2, e.g. q_w=0.02 or β=0.5), not before.

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

Test priors: P1 β=0.2 uniform (primary); P2 β=0.5 uniform; P3 β=0.2 π=δ_{lag 0}; P4 β=0.5 π=δ_{lag 0}.
A6p-narrow: train family β~U[0,0.3], π uniform (P2–P4 are extrapolations). A6p-broad: β~U[0,½],
π~Dirichlet(1_L) (P1–P4 in support, P3/P4 near its boundary). Composition arms (A2, A4, A5) and A1r
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
- Smallest meaningful difference: `Δ_min = max(0.10·V_2, 0.002·VOI_clean)` per cell ⟨TBD-oracle: confirm
  values in raw cost units⟩. Rationale: 10% of the opportunity that awareness can buy in this cell; the
  absolute floor prevents instability when V_2 ≈ 0. Statistical power is decided separately: seeds are
  increased until the 95% CI half-width of paired seed-level differences is ≤ Δ_min/2.
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

## Open items for Opus review

1. Second cell choice and whether L=32 is worth the cost (after the oracle map).
2. Confirm Δ_min numbers in raw units.
3. Whether A1r's per-prior refit counts as "composition-like" flexibility in H3 tables (proposal: own row).
