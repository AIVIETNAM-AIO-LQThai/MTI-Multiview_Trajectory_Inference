# MTI findings summary — v4 (phase 1 corrected after external review + E2/E2b channel misspecification, 2026-10-04)

v1 overstated two claims ("never worse", "the route does not matter") and attributed a pipeline difference to supervision alone. This version replaces it;
the corrections are documented in [e1_report.md](reports/e1_report.md) §7 and [learned_pilot_report.md](reports/learned_pilot_report.md) §6.

**Scope of every statement:**
- S2 switching-mode scalar simulator, with physics (ρ, c, q_a, q_w, λ) supplied.
- Exogenous single-flip action-sign channel, with a declared and correct prior.
- One-step quadratic decision regret.
- Learned results only: L=8, physics-informed features, 5 seeds, the implemented 8-config grids and epoch caps.

## Working claim (after E2)

> With supplied physics and an exogenous action-sign channel, composing a clean-law estimator with a declared channel prior gives inference-time prior flexibility and can improve finite-data one-step decisions. **Its value is bounded by the accuracy of the declared prior.**
>
> - Under the true law, understating the corruption rate never hurt (proved per record when the location prior is right), and declaring a flatter location prior than the truth never hurt on the screened grid.
> - Overstating a weak channel, or concentrating location mass where corruption is actually diffuse, made channel-aware inference **worse than ignoring corruption**.
> - Better estimators (density composition, fitted HMM) implement a wrong prior more faithfully; direct estimators looked robust only through attenuated responsiveness to the prior.
> - Against a coverage-trained direct estimator, composition kept a finite-data advantage at N=10³ but not at N=10⁵.
> - A correctly structured HMM is near-oracle, so nothing here shows a need for neural inference.

**Under prior uncertainty (E2b):**
- Composition with the mean prior is the Bayes estimator, verified numerically. It cut mean regret by 80–87% over a 30-channel uncertainty set, but lost to ignoring corruption on 9–10 weak or diffuse channels.
- A prior that harms no channel keeps only about one third of the total gain. There is no free robust prior.
- Harm tracks over-declared recent-lag (lag-0) corruption mass (AUC 0.99, descriptive).

**Most important unresolved questions:** whether q can be estimated from many corrupted records (empirical Bayes), which would remove the trade-off in repeated deployment; mode-dependent placement; and the original MTM question (multi-view consistency detecting harmful trajectories), which remains unaddressed.

## Claim ledger

| # | claim | status | evidence |
|---|---|---|---|
| C1 | Exact candidate and folded routes compute the same aware posterior | **verified** (≤1.5e-14; enumeration; mutation audit) | oracle §2 |
| C2 | Naive excess = V_2 + I_loss | **verified** | oracle T10 |
| C3/C4 | Value of awareness is prior-specific (1.8% … 159% of VOI_clean) | **measured** | oracle §3–4, §11–13 |
| C10 | The aware belief costs more than naive on clean records and after natural switches, and wins on the mixture | **measured** | oracle §3; pilot §3.3 |
| C7 | Learned folded vs candidate routes differ | **no general advantage established** (14 eq / 4 inconcl. / 2 material); compatibility training deprioritised, not disproved | pilot §5–6 |
| C8 | Composition provides inference-time prior flexibility | **supported, structurally and empirically, under a correct declared prior**; costs L+1 to 2L+1 queries per decision | pilot; E1 |
| C12 | Composition (A4) vs density-direct (A6pd) at matched backbone and loss family | **6 better / 4 equivalent / 4 inconclusive / 0 worse** across 14 cells; only 8/14 meet the precision target; advantages at small N (q_w=0.02) and at concentrated priors | E1 §3, §7 |
| C12b | Direct estimators fail at concentrated priors | **withdrawn**; E2 coverage control: at N=10⁵ a coverage-trained direct estimator matches composition at concentrated priors (β_a ≥ 0.2); at N=10³ composition stays materially better (+0.26 … +0.53) | E1 §7; [E2 §6](reports/e2_report.md) |
| C13 | Pipeline (density vs regression) matters more than route | **supported as a pipeline statement**; supervision alone is **not** isolated (architecture/representation also differ) | pilot §6; E1 §7 |
| C14 | A fitted, correctly structured HMM is near-oracle | **measured**; limits the contribution of the learned results | pilot; E1 |
| C15 | A wrong declared prior can make channel-aware inference worse than ignoring corruption | **measured + partly proved**: rate understatement never hurts (per-record proof, 0 violations); overstatement harms for a weak true channel beyond ≈2–3×; location harm only for concentrated-declared/diffuse-true (−0.03 … −0.30); flattened declarations never harmed on the grid | E2 §3, §6 |
| C16 | Better estimators implement a wrong prior more faithfully; direct-arm robustness is attenuation | **measured** (A4/A5 reproduce exact harm; negative cross terms −0.54 … −0.71 for direct arms; better-trained conditioning loses more) | E2 §4, §6 |
| C17 | Aggregate regret hides transfers (composition costs on clean records and natural switches, gains on corrupted records; overstatement enlarges the costs) | **measured** | E2 §3 |
| C18 | Under prior uncertainty independent of the record, composition with the mean prior is Bayes-optimal | **analytic + verified** (grid optimum = q̄ in both physics; compound identity to 1e-10) | [E2b](reports/e2b_report.md) |
| C19 | No free robust prior: the Bayes prior harms weak/diffuse channels (9–10 of 30; total −1.3 vs +16…+26 gain); a no-harm prior keeps ≈⅓ of the gain; minimax lowers the worst case but harms more channels | **measured** (exact + A4 checkpoints) | E2b §3, §6 |
| H-loc | Harm tracks overstatement of the lag-0 (recent) corruption rate | **supported descriptively** (AUC 0.990 vs 0.913 for the overall rate; no harm at lag-0 ratio ≤ 1.5); not causal; one grid/simulator | E2b §4, §6 |
| C5 | (s_L, μ̃) recalibration | lower bound only | oracle §5, §11 |
| C6 | Contaminated queries are harder | descriptive (4–6×); not isolated from target difficulty | pilot §3.3 |
| C11 | Learned results hold for physics-informed learners only | scope qualification | pilot §5 |
| — | Multi-view consistency detects harmful trajectories beyond residual/rarity/dynamics signals (the original MTM question) | **not addressed** | — |
| C9 | MTM/DT poisoning defence; closed-loop control benefit; thesis-level novelty | **not established** | — |

## Next experiment

None scheduled. Candidates needing a user decision: empirical-Bayes estimation of q (recommended), mode-dependent placement, the original MTM consistency question.
