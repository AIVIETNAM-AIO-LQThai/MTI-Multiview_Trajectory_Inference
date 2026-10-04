# E2b report: robust declared priors under prior uncertainty (Sonnet 5.5; interpretation pending Opus review)

Protocol fixed before any data: [e2_protocol.md](../e2_protocol.md) §10. Full tables: [e2b_tables.md](e2b_tables.md), `fig_e2b_hypothesis.png`. Same physics and exogenous single-flip mechanism as E2; no training (A4 uses the saved E2 checkpoints).

## 1. What was run

- Branch `exp/channel-misspecification`. The run records base `4b9380f` with a dirty-code list in `results/e2b/e2b.json` (misspec.py `alpha_of` + the new E2b files, committed afterwards). Tests: 97 pass (4 new: T-E2b-1…3 and an α round-trip regression).
- **Uncertainty set U:** β_t ∈ {0.02, 0.05, 0.1, 0.2, 0.35, 0.5} × α_t ∈ {0, 0.25, 0.5, 0.75, 1}: 30 channels, uniform weights. Mean prior q̄ = (β̄ = 0.2033, ᾱ = 0.5).
- **Selection** (screening prefixes, master 8101, 100k; both physics): 110 candidate priors (β_a ∈ 13 values × α_a ∈ 9, plus q̄) scored under all 30 truths with the exact clean law. Selected: grid Bayes (K1), no-harm (K2), minimax regret (K3).
- **Confirmation** (independent prefixes, master 8201, 20k): the selected priors, q̄, "most plausible" (β = 0.2, α = 0.5) and naive, under all 30 truths; exact clean law and A4 (lead physics, N = 10³ and 10⁵, seeds 1–10).
- **Bug caught and fixed before analysis:** a first run stored the *last-position mass* of π as "α" and rebuilt priors from it in the confirmation and A4 stages (q̄ would have been evaluated at α = 0.5625). I killed that run, added `alpha_of` and a round-trip regression test, and re-ran everything from scratch; all reported numbers are from the corrected run.

## 2. Validation

| check | result |
|---|---|
| T-E2b-1: mixture of the 30 channels' view weights = view weights of q̄; compound identity Z̄ μ₂(q̄) = Σ w Z_t μ₂(q_t) on every record | max relative error ≤ 1e-10 |
| T-E2b-2: no candidate has lower uniform-average regret than q̄ (and the compound-law formula for K1(c) − K1(q̄) agrees) | passes (|z| ≤ 4); **in the selection run the grid optimum coincides with q̄ (K1 0.12400 vs 0.12400; every candidate ≥ K1(q̄) − 3.5e-6 at q_w=0.02; at q_w=0.5 the grid optimum is q̄ itself)** |
| T-E2b-3: singleton set ⇒ q̄ = truth and zero regret | exact |

## 3. Results (both physics; confirmation set, exact clean law)

K1 = mean regret over the 30 truths, K2 = worst-truth benefit over naive (min_t B), K3 = worst-truth regret. Harm = CI_hi(B) < −Δ_t; improvement = CI_lo(B) > Δ_t.

**Lead physics (q_w = 0.02)**

| prior (β_a, α_a) | K1 | K2 | K3 | truths harmed / improved / other | median retained benefit |
|---|---|---|---|---|---|
| naive | 0.9477 | 0 | 5.049 | 0 / 0 / 30 | 0 |
| **mean prior q̄** (0.203, 0.5) | **0.1209** | −0.247 | 0.487 | **9** / 20 / 1 | 0.85 |
| grid Bayes (0.2, 0.5) = "most plausible" | 0.1210 | −0.243 | 0.498 | 9 / 20 / 1 | 0.85 |
| **no-harm** (0.075, 0) | 0.6374 | **−0.0055** | 3.657 | **0** / 27 / 3 | 0.45 |
| **minimax regret** (0.15, 0.875) | 0.1471 | −0.286 | **0.316** | 10 / 18 / 2 | 0.76 |

**Secondary physics (q_w = 0.5):** naive K1 0.627, K3 3.513. q̄ K1 0.1251 (10 harmed / 19 improved), K2 −0.210. No-harm (0.02, 0.625): K1 0.4105, K2 −0.0078, 0 harmed / 27 improved, median retained 0.43. Minimax (0.2, 0.875): K1 0.163, K3 0.3225 (13 harmed / 16 improved, median retained 0.30).

- **Bayes optimality is realised:** the mean prior (and the grid candidate next to it) cuts the average regret by 87% (q_w=0.02) and 80% (q_w=0.5) relative to ignoring corruption, and no other prior in the grid does better on average.
- **But it harms 9–10 of the 30 truths.** The nine harmed truths at q_w=0.02 are the weak or diffuse channels: β_t = 0.02 at every α, β_t = 0.05 at α ≤ 0.25, β_t = 0.1 at α = 0, and β_t = 0.2 at α = 0 (B from −0.25 to −0.03; V₂,t from 0.001 to 0.09).
- **No-harm and Bayes are far apart.** The only candidates with zero harmful truths number 15 (q_w=0.02) and 19 (q_w=0.5); the best of them has 5–6× the Bayes risk (0.637 vs 0.121) and keeps a median 45% of the matched benefit.
- **Minimax regret** has the lowest worst-case regret (0.316 vs 0.487 for q̄ and 5.05 for naive) at a modest Bayes-risk cost (0.147 vs 0.121), but harms 10–13 truths.
- **The selection did not match the Opus expectation of α_a ≈ 0.5 for minimax:** with β also uncertain, the minimax candidate is concentrated (α_a = 0.875) with a moderate rate (0.15–0.2). The earlier 0.5 came from a β-fixed location-only screen.
- **A4 reproduces the exact picture** (saved checkpoints, no retraining; seed intervals conditional on the shared test set): at N=10⁵, K1 0.1218 ± 0.0003 for q̄ vs exact 0.1209; K3 0.486; 10 harmed / 20 improved. At N=10³: K1 0.154 ± 0.006, K3 0.517, 10 harmed / 20 improved. No-harm prior: 0 harmed, 27 improved, 3 other at both N. Minimax: 10 harmed, 18 improved.
- **Caveat on the retained-benefit column:** the worst-truth "retained" values (−190 …) are ratios to a tiny V₂,t (0.0013 for β_t=0.02, α_t=0) and should not be read as magnitudes; K2 and the harm counts are the primary measures.

## 4. Descriptive hypothesis check: does harm track overstatement of the lag-0 corruption rate?

6,540 (candidate, truth) pairs with β_a > 0 (both physics, selection set); 1,758 with B_exact significantly below 0.

| declared / true lag-0 mass | pairs | share harmed |
|---|---|---|
| ≤ 1 | 3,194 | 0.000 |
| (1, 1.5] | 544 | 0.000 |
| (1.5, 2] | 370 | 0.011 |
| (2, 3] | 498 | 0.189 |
| (3, 5] | 556 | 0.579 |
| (5, 10] | 606 | 0.934 |
| > 10 | 772 | 1.000 |

The smallest ratio with significant harm is 1.67, and no pair with ratio ≤ 1.5 is harmed. By contrast, the overall β ratio has harm at ratio 0.29 and 4.9% harmed among pairs with β_a ≤ β_t. The AUC for separating harmed from non-harmed pairs is **0.990 (lag-0 ratio)** vs **0.913 (β ratio)**.
This is descriptive: it uses all pairs from the structured grid, and the threshold depends on the true lag-0 mass and noise level. It supports the hypothesis (harm follows overstatement of the recent-lag rate more closely than the overall rate) but does not establish a causal rule.

## 5. Limits and flags for the Opus review

1. **No free robust prior** in this set: the Bayes prior harms 30–33% of truths, the no-harm prior keeps under half the benefit. Which criterion a user should adopt is not a scientific question; the report gives the trade-off.
2. The uncertainty set is a designed grid with uniform weights over (β, α); other weightings or sets change q̄ and the harmed list. The Bayes identity holds for any weights.
3. Learned results cover A4 only (saved checkpoints; lead physics). Direct estimators were not evaluated under uncertainty; they have no analogue of "declare a prior" apart from conditioning.
4. E2b uses exact-prior selection on one set and confirmation on another; selection among 110 candidates (not a learned search).
5. Not addressed: estimating q from corrupted data, mode-dependent placement.

## 6. Opus evidence review (2026-10-04, `claude-opus-5-5`)

Checked:
- **Provenance.** Base `4b9380f`; the run's dirty-code list equals the files now awaiting commit, with no code edits after the corrected run.
- **The α bug and its handling.** It was caught before analysis, a regression test was added, and the whole stage was re-run. That is the right procedure.
- **Validation.**
- **The harm and gain magnitudes,** recomputed from `results/e2b/e2b.json`.

**The results are sound.**

**Supported** (exogenous single-flip channel, supplied physics, the declared uncertainty set U and its uniform weights):
1. **C18 is confirmed numerically.** Under prior uncertainty independent of the record, composition with the mean prior is the Bayes estimator. The grid optimum coincides with q̄ in both physics, on selection and on independent confirmation, and the saved A4 models reproduce it.
2. **No free robust prior (C19).**
   - The Bayes prior lowers mean regret by 80–87%, but loses to ignoring corruption on 9–10 of 30 channels. These are the weak or diffuse channels: β_t ≤ 0.05, or uniform location with β_t ≤ 0.2.
   - In absolute terms its total loss over those channels is −1.35 (q_w=0.02) and −1.28 (q_w=0.5), against total gains of +26.2 and +16.3.
   - Insisting on no harm costs about two thirds of the total gain (no-harm prior: +9.3 and +6.5, with losses −0.006 and −0.008).
   - Minimax regret lowers the worst case but harms more channels.
   - Which criterion to adopt is a deployment decision, not a scientific finding. The scientific content is the size of the trade-off.
3. **H-loc is supported descriptively, not causally.** Declared-to-true lag-0 corruption mass separates harmful from harmless declarations almost perfectly on this grid (AUC 0.990; no harm at ratio ≤ 1.5) and better than the overall rate (0.913). A practical reading: *do not declare more recent-lag corruption than you believe is present*. Underdeclaring recent-lag mass costs benefit, not safety.
   This agrees with the per-record theorem for rate-only error (safe up to twice the true odds). It remains an empirical regularity of one designed grid, one simulator and one channel family.
4. The minimax selection (concentrated, α_a = 0.875) contradicts my earlier location-only expectation (α_a ≈ 0.5). The earlier figure came from a set without rate uncertainty and is superseded.

**Decision.** The bounded misspecification question (E2 + E2b) is answered within its scope; **experimental work on this branch stops here.** What remains needs a user research decision:
- (a) **Empirical-Bayes estimation of q from many corrupted records.** This is the direct remedy for the trade-off, because repeated deployment identifies the mixture, and it is the recommended next step if work continues.
- (b) **Mode-dependent placement,** which needs a new reference and enumeration validation first.
- (c) **The original MTM question.** Does multi-view consistency detect harmful trajectories beyond residual, rarity or dynamics signals? It remains unaddressed.

Unknown physics, closed-loop control and poisoning defences also remain out of scope.
