# MTI research charter

MTI = Multi-View Trajectory Inference. Repository name settled; scope provisional and narrowed by evidence.
Recorded 2026-10-04 (Opus planning pass, branch `exp/oracle-evidence`, base `c250658`).

## Question

> **Completion note (2026-10-05).** The original question below was narrowed by evidence. The completed work is summarised in [research_synthesis.md](research_synthesis.md): *Identifiability and decision costs of action-sign corruption in switching trajectories*. The claims ledger below is historical; the current claim-to-evidence map is [consolidation_review.md](consolidation_review.md) §4 and [findings_summary.md](findings_summary.md).

When a model learned from clean trajectories meets possibly contaminated trajectory evidence, how do
different query structures (full record, sign-repaired candidates, sign-folded magnitude-retaining
observers) estimate a decision-relevant belief under finite data and compute — and what accuracy or
compute does composing inference at test time cost relative to training directly for a declared
corruption channel?

Working direction (not a claimed contribution): *decision-relevant trajectory inference under corrupted
evidence: equivalence of exact routes, finite-sample estimation differences, and the cost of
inference-time flexibility.*

## Three separate questions

| Question | Establishes | Does not establish |
|---|---|---|
| Information | Decision value awareness can recover (V_2) and clean-prefix information irreversibly lost (I_loss) under the declared channel | More views of one record create no new population information |
| Estimation | Whether learned query routes differ in accuracy/efficiency/adaptability at matched data, access, compute | Exact-route equality ⇏ finite-learner equality; oracle results cannot show a learned advantage |
| Policy relevance | Whether belief error matters for a specified physical decision | Reconstruction, AUROC or one-step synthetic cost ⇏ robust trained-policy performance |

## Hypotheses (phase-1 status corrected 2026-10-04 after external review: H1 verified; H2 no general route advantage established; H3 prior flexibility supported under a correct declared prior, finite-data benefit regime-dependent (6/4/4/0 vs matched density-direct); H4 measured, prior-specific; H5 deprioritised, not disproved. See [findings_summary.md](findings_summary.md) v2)

- H1 Oracle equivalence: candidate and folded routes compute the same aware posterior under the declared law.
  Analytic + numerical verification only; says nothing about learning.
- H2 Finite-estimation difference: repair / fold / retain-corruption queries induce different learned
  errors — could favour either route, neither, a fitted HMM, or a direct estimator.
- H3 Flexibility trade-off: test-time composition of clean-law components handles declared prior changes
  without retraining, at head-estimation and compute cost. Compared against a direct estimator given the
  same full prior (β and π, equivalently q), not only against a fixed-prior one.
- H4 Decision opportunity: V_2 is materially positive in some regimes and negligible in others. Small V_2
  is a scope constraint, not a failed experiment.
- H5 Compatibility mechanism (secondary): route disagreement may diagnose/regularise estimation error;
  consistency neither proves accuracy nor requires attack localisation.

## Scope of the first assignment

In: S2 switching-mode scalar simulator; exogenous single-flip action-sign channel; exact inference;
one-step quadratic decision; information/opportunity map; learned-pilot **protocol** (not run).
Out: learned-model runs; mask diversity; separate encoders; compatibility/distillation; other simulator
families; per-label/interval/hidden-mode-dependent/state corruption; unknown physics; closed loop;
Decision Transformer/Walker2d transfer; poisoning-of-training defences.

## Standing assumptions

A1 Data are synthetic draws from the declared S2 law ([spec §1](mathematical_specification.md)); they are
genuine samples of that model, not recorded real-world trajectories.
A2 Channel is exogenous: θ ⟂ (modes, probe, decision noise) | H, with a fixed declared prior.
A3 Physics (ρ,c,q_a,q_w,λ) known to all methods; η known to the exact oracle only.
A4 Hidden modes and corruption labels are evaluation/stratification fields only, never learned inputs or
primary training labels.
A5 Stop-gradient on a shared teacher's outputs does not freeze the teacher: shared-encoder updates via
student queries still change teacher predictions. Any "protected teacher" claim needs frozen parameters,
a justified update scheme, or measured drift/accuracy.
A6 A zero average route gap does not dismiss a mechanism (opposite conditional effects can cancel);
small compatibility residuals do not prove accurate beliefs.

## Evidence / claim ledger

Status legend: **open** · **analytic** (derived under the declared law) · **verified** (numerically
checked, tests saved) · **measured** (finite-sample/MC result with uncertainty) · **refuted**.

| # | Claim | Status | Evidence | Next needed |
|---|---|---|---|---|
| C1 | Candidate and folded exact routes coincide (G=M, C_j=0, ℓ_j=r_j/(1−r_j)) | **verified** | T6–T8; every oracle cell ≤ 1.5e-14 ([report](reports/oracle_v1_report.md) §2) | none (says nothing about learning) |
| C2 | Naive excess = V_2 + I_loss | **verified** (exact given S; in expectation \|z\| ≤ 1.5) | T10 | — |
| C3 | V_2, I_loss, VOI_clean in the primary cell | **measured**: 0.0712 ± 0.0031, 0.2614 ± 0.0069, 4.007 ± 0.039 (V_2 = 1.8% of VOI_clean) | report §3 | — |
| C4 | Where V_2 is material | **measured**: 13% of VOI_clean at L=8,q_w=.02,β=.5; ≤ 1.6% at L=32 (L contrast confounded with per-position rate β/L); **prior-specific**: with corruption on lag 0, V_2 = 27–159% of VOI_clean (report §11–12) | report §4 | other channels/simulators for external validity |
| C5 | Share of V_2 recoverable from (s_L, μ̃) | **measured, bounded**: ≥ 6–11% (residual class); R only upper-bounds the context term | report §5 | learned recalibrator (A1r) |
| C6 | Candidate repairs carry more contaminated weight than folded queries | **measured at oracle weights** (0.97 vs 0.80 attacked; 0.17 vs 0 clean); links to learning errors untested | report §6 | learned pilot diagnostics |
| C7 | Learned routes differ at finite N (H2) | **no general advantage established**: 14 equivalent / 4 inconclusive / 2 material; compatibility training deprioritised, not disproved | [pilot §6](reports/learned_pilot_report.md) | — |
| C8 | Composition buys prior flexibility (H3) | **supported, bounded by the declared prior's accuracy** (E2): understatement and flattening safe; overstating a weak channel or concentrating where corruption is diffuse is worse than ignoring the channel | [E2 §6](reports/e2_report.md) | E2b robust priors |
| C9 | Poisoning defence / trained-policy robustness | out of scope | — | separate programme |
| C12 | Composition (A4) vs density-direct (A6pd), matched backbone/loss family | **6 better / 4 equivalent / 4 inconclusive / 0 worse**; 8/14 meet precision; concentrated-prior advantage confounded with sparse training coverage | [E1 §7](reports/e1_report.md) | E2 coverage control |
| C15–C19, H-loc | Misspecification results (see findings_summary v4) | measured / analytic+verified / descriptive | [E2](reports/e2_report.md), [E2b](reports/e2b_report.md) reports | user decision: empirical Bayes, mode-dependent channel, or MTM question |
| C20–C24 | Identification and adaptive composition (see findings_summary v6; C18/C20/C21/C23/C24 are instances of known theory) | verified / measured | [E3 report](reports/e3_report.md) | — |
| C25–C26 | Joint (η, q) identification (L ≤ 3 unidentified, L ≥ 4 separated within family); per-record harm oracle; route gap is finite-learner only | analytic + verified (pattern level) | [literature_review.md](literature_review.md) | — (E4 done) |
| C27–C28 | Joint identification removes within-family η-error absorption for L ≥ 4 (measured cost); one mild family violation absorbed persistently over the measured n; extra clean prefixes: CV-30/100/1,000 equivalent to J-MLE in 4/10/15 of 16 cells (not an equal budget) | measured | [E4 report](reports/e4_report.md); [consolidation_review](consolidation_review.md) §4 | completed; see [research_synthesis](research_synthesis.md) |
| C11 | Conclusions about learned estimation hold for physics-informed learners (LLR-type features; A4/A5 families contain the truth) | **scope qualification** | learned report §5 | generic-feature / unknown-physics study (out of current scope) |
| C10 | Aware belief costs realised cost in natural-switch states and on clean records | **measured**: D = −0.56 ± 0.10 (switch edge), −0.047 ± 0.008 (clean) in primary | report §3 | check in learned arms |

No numerical result from earlier conversations is inherited; every number must be reproduced here with
recorded code, configuration and seeds.
