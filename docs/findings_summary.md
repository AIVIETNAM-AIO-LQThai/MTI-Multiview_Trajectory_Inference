# MTI findings summary (phase 1, 2026-10-04)

Consolidated by the Opus evidence review. Every number is reproducible from the committed code, configs and seeds; details in the linked reports.
**Scope of every statement:** S2 switching-mode scalar simulator; exogenous single-flip action-sign channel with a declared prior; known physics (ρ, c, q_a, q_w, λ);
physics-informed learners; one-step quadratic decision endpoint; L=8 for all learned results.

## Thesis supported by the evidence

> Under a declared exogenous corruption channel, **exact inference-time composition of a clean-law model with the declared channel** is a prior-flexible estimator of the decision-relevant aware belief.
> With matched supervision it is never worse than training directly on simulated channel data. It is much better when the test prior lies outside the training family, and better in-family at small N in the low-noise regime.
> The specific multi-view **route (folded vs candidate repair) does not matter**. What matters is analytic use of the channel, how the clean components are supervised, and exact Bayesian weighting.
> How large the payoff is depends strongly on the corruption prior.

This replaces the provisional direction "equivalence of exact routes, finite-sample estimation differences, and the cost of inference-time flexibility". The first part holds; the second holds in its composition-vs-direct form, not its route form; the third is quantified.

## Claim ledger

| # | claim | status | evidence |
|---|---|---|---|
| C1 | Candidate and folded exact routes compute the same aware posterior | **verified** (≤1.5e-14 in every cell; enumeration; mutation audit) | [oracle report](reports/oracle_v1_report.md) §2 |
| C2 | Naive excess = V_2 (recoverable) + I_loss (irrecoverable) | **verified** | oracle §2, T10 |
| C3/C4 | The value of awareness is prior-specific: 1.8% of the clean-history value (uniform prior, primary cell) up to 159% (corruption always on the last action) | **measured** | oracle §3–4, §11–13 |
| C10 | The aware belief is costlier than naive on clean records and after natural mode switches; it wins on the mixture | **measured** (oracle and every learned composition arm) | oracle §3; [pilot](reports/learned_pilot_report.md) §3.3 |
| C7 | Learned folded and candidate routes differ (H2, route form) | **not supported** (no material difference in 18/20 cells; the exceptions are predeclared-structural or both arms poor) | pilot §5 |
| C8 | Composition buys prior flexibility (H3) | **supported**; costs 9–17 queries per decision | pilot §5, [E1](reports/e1_report.md) §6 |
| C12 | Under matched supervision, composition vs direct estimation in-family | **better at small N (q_w=0.02); equivalent within the practical threshold at large N; undetectable at q_w=0.5** | E1 §3, §6 |
| C5 | Recalibration from (s_L, μ̃) alone | **lower bound only**: ≥6–11% of V_2 (uniform priors), 16–52% (lag-0 priors) | oracle §5, §11 |
| C6 | Contaminated queries are harder for learned components | **descriptive**: error grows 4–6× with contamination count; not isolated from target difficulty | pilot §3.3 |
| H5 | Compatibility/distillation would help | **not motivated** (route gap negligible, no cancellation) | pilot §5 |
| C11 | Learned results hold for physics-informed learners | **scope qualification** | pilot §5 |
| C9 | Poisoning defence / trained-policy robustness | **not addressed** | — |

## What is not established

- Any advantage of a masked-trajectory *route* as such.
- Results with generic sequence features, unknown physics, other channels (per-label, interval, mode-dependent, state corruption) or other simulators.
- Results at the L=32 history length for learned arms.
- Closed-loop or trained-policy robustness, and defences against poisoned training data.
- That explicit localisation is necessary.

## If the project continues (each needs separate authorization)

1. **External validity of C12:** remove the physics-informed features and the true-structure model families (generic GRU/transformer, unknown q_w); repeat E1 at N=10³.
2. **Channel misspecification:** composition relies on the *declared* channel. Measure how fast its advantage disappears when the true channel differs (wrong β, mode-dependent placement). This is the main practical risk of the approach.
3. **Policy relevance:** plug the composed belief into a multi-step controller and measure closed-loop cost.

## Provenance

Branch `exp/oracle-evidence`:

| stage | code commit | results commit |
|---|---|---|
| oracle v1 | `0f1b6d1` | `5fc4383` |
| shifted priors | `2168bee` / `f89ff2c` | — |
| learned pilot | `09621bb` | `6ad73fa` |
| E1 | `0ac6b34` | `abd83b7` |

Status file: [research_status.md](research_status.md).
