# E1 report: supervision vs composition (Sonnet 5.5; interpretation pending Opus review)

Design, arms and the three interpretation outcomes were fixed before running in [learned_pilot_protocol.md](../learned_pilot_protocol.md) §E1.
Full tables: [e1_tables.md](e1_tables.md). Pilot context: [learned_pilot_report.md](learned_pilot_report.md).

## 1. What was run

- **Code commit `0ac6b34` (clean tree).** Commands: `run_e1.py --stage all --out results/e1` (tuning 64 trainings + 20 units, 466 s total), `analyze_e1.py`. Raw: `results/e1/{tuning.json,units/*.json,analysis.json}`.
- **New arms** (CausalDensity backbone and token features of A4; trained on channel-simulated records of the same training prefixes as the pilot, original clean probe transition kept; belief read directly as tanh(logit_L/2), one forward pass):
  **A6d** fixed prior (P1) and **A6pd-broad** prior-conditioned (q appended to each token; broad family). Objective: NLL of all L recorded transitions of the corrupted record plus the clean probe transition. Unit tests confirm
  the probe-transition NLL on corrupted records is minimised at the aware belief (z > 4) and that inputs carry no probe or hidden information (85 tests pass).
- **Pairing:** same training prefixes, validation prefixes, test sets, priors, grid and seeds (1–5) as the pilot; pilot arms (A4, A2-G, A6p-broad, A6) are read from `results/learned_pilot/`, not retrained.
  Δ_min and verdict rules as in the pilot; ratios E_x/E_y are geometric means over seeds with t-intervals on the log ratio.

## 2. Endpoint E, lead physics (q_w=0.02), mean ± 95% CI over 5 seeds

| N | arm | P1 (in-family) | P2 (in) | P5 (in) | P3 (out) | P4 (out) |
|---|---|---|---|---|---|---|
| 10³ | A4 (clean density + composition) | 0.0259 ± 0.0070 | 0.0275 ± 0.0072 | 0.0311 ± 0.0086 | 0.0355 ± 0.0092 | 0.0335 ± 0.0098 |
| 10³ | **A6pd-broad** | 0.0419 ± 0.0105 | 0.1174 ± 0.0132 | 0.1662 ± 0.0212 | 0.5768 ± 0.0537 | 1.2523 ± 0.3712 |
| 10³ | A6p-broad (probe regression) | 0.0957 ± 0.0199 | 0.2073 ± 0.0518 | 0.2777 ± 0.0815 | 0.7368 ± 0.1791 | 2.3114 ± 0.5275 |
| 10³ | **A6d** (fixed P1) | 0.0436 ± 0.0068 | 0.1319 ± 0.0140 | 0.1976 ± 0.0245 | 0.6473 ± 0.0610 | 2.1662 ± 0.1772 |
| 10⁵ | A4 | 0.0008 ± 0.0001 | 0.0008 ± 0.0001 | 0.0010 ± 0.0001 | 0.0011 ± 0.0001 | 0.0010 ± 0.0001 |
| 10⁵ | A2-G | 0.0061 ± 0.0017 | 0.0073 ± 0.0021 | 0.0084 ± 0.0026 | 0.0080 ± 0.0019 | 0.0109 ± 0.0020 |
| 10⁵ | **A6pd-broad** | 0.0047 ± 0.0009 | 0.0110 ± 0.0023 | 0.0214 ± 0.0074 | 0.2348 ± 0.0306 | 1.8005 ± 0.3577 |
| 10⁵ | A6p-broad | 0.0097 ± 0.0019 | 0.0170 ± 0.0044 | 0.0122 ± 0.0037 | 0.0984 ± 0.0139 | 0.5345 ± 0.0652 |
| 10⁵ | **A6d** (fixed P1) | 0.0034 ± 0.0002 | 0.0829 ± 0.0038 | 0.1666 ± 0.0072 | 0.6339 ± 0.0163 | 2.1861 ± 0.0435 |
| | *Δ_min* | 0.0092 | 0.0411 | 0.0617 | 0.0617 | 0.0617 |

Secondary physics (q_w=0.5; P1/P2 only for channel-trained arms): N=10³: A4 0.128 / 0.115, A6pd-broad 0.123 / 0.147, A6d 0.121 / 0.162, A6p-broad 0.327 / 0.321. N=10⁵: A4 0.0047 / 0.0044, A6pd-broad 0.0098 / 0.0355, A6d 0.0120 / 0.0714, A6p-broad 0.0202 / 0.0400.

## 3. The predeclared comparisons

**C-b (supervision, direct held fixed): A6pd-broad vs A6p-broad.** Likelihood supervision lowers E in-family at N=10³: A6pd better by ≥Δ_min in 4 of 5 cells (ratio E_x/E_y 0.38–0.57), inconclusive in 1; at N=10⁵ equivalent in 4, inconclusive in 1 (ratios 0.48–1.73, CI of the P5 ratio 1.03–2.93).
**Out-of-family at N=10⁵ A6p-broad is better** (P3: 0.098 vs 0.235; P4: 0.535 vs 1.801; ratios 2.4 and 3.3), i.e. the density-direct prior-conditioned model extrapolates *worse* than probe regression to the simplex vertices.

**C-a (composition vs direct, supervision matched): A4 vs A6pd-broad.**
- *In-family, N=10⁵:* equivalent within Δ_min in 4 of 5 cells, inconclusive in 1 (secondary P2, d = −0.0311 ± 0.0040 vs Δ_min 0.0312). The ratio E_x/E_y is nevertheless 0.05–0.48 (A4 lower in every cell).
- *In-family, N=10³:* **lead physics: A4 better by ≥Δ_min at P2 (0.0275 vs 0.1174, ratio 0.23) and P5 (0.0311 vs 0.1662, ratio 0.18); P1 inconclusive (ratio 0.61, CI 0.39–0.97).** Secondary physics: inconclusive at both priors (ratios 1.03, 0.77).
- *Out-of-family P3/P4:* A4 better by ≥Δ_min in all four (N=10³ and 10⁵), ratios 0.00–0.06.

**C-d (flexibility under matched supervision): A4 vs A6d (fixed P1).** At P1 the two are equivalent at N=10⁵ (0.0008 vs 0.0034) and A4 is better at N=10³ (ratio 0.58). At the other priors A4 is better by ≥Δ_min in **all 8** lead-physics cells (P2–P5, both N); on the secondary physics (P2 only) the verdict is inconclusive at N=10³ and A4 better at N=10⁵. A6d degrades off its training prior exactly as the pilot's A6 did.

**C-c (supervision held, composition fixed): A4 vs A2-G.** Unchanged from the pilot: A4 better at N=10³ in all 10 cells; equivalent at N=10⁵ (ratios 0.1–0.3).

**Extra: A2-G vs A6pd-broad.** In-family N=10³: A6pd-broad better in 3, inconclusive in 2; N=10⁵ equivalent in 4 (1 inconclusive); out-of-family: A2-G better in all four. So *a matched-supervision direct estimator overtakes the pilot's probe-trained composition model (A2) at N=10³ in-family*, consistent with supervision being a large part of the pilot's A4-vs-A2 gap.

## 4. Mapping to the predeclared outcomes (for the Opus review; not decided here)

| outcome fixed in advance | observed |
|---|---|
| 1. A6pd equivalent to A4 in-family and worse at P3/P4 | **Matches** at N=10⁵ in Δ_min terms (4 of 5 equivalent) and at P3/P4 (A4 better in all). Does **not** match at N=10³ lead physics (A4 better at P2, P5) or in ratio terms at N=10⁵ (A4 2–20× lower). |
| 2. A6pd worse than A4 by ≥Δ_min in-family → composition carries an estimation advantage; report only if C-b shows supervision alone does not close the gap | **Matches at N=10³ on the lead physics** (P2, P5). C-b shows supervision closes part (A6pd better than A6p-broad by ratio 0.38–0.57) but not all (A4 still 0.18–0.23× A6pd). Not met on the secondary physics (inconclusive). |
| 3. A6pd better than A4 in-family | Not observed in any cell. |

## 5. Limitations and flags

1. **N=10⁵ equivalence is again Δ_min-saturation:** all in-family differences are < 0.02 in absolute terms; ratios say A4 is consistently lower, but not by a margin that matters for the decision.
2. **Confounds that E1 does not remove:** A4 is trained on *clean* prefixes with one view per step, A6pd/A6d on corrupted records (two views per step); a clean-law component may be an easier estimation target than a corrupted-record posterior independent of composition. The architecture and features are identical, but the training distributions differ by design.
3. Grid-edge selection at N=10⁵ again (A6pd all 12 epochs; A6d 12 epochs lead, 6 epochs secondary) — see [tuning.json](../../results/e1/tuning.json).
4. The secondary physics has high variance at N=10³ (several inconclusive cells); 5 seeds.
5. A6pd's poor vertex extrapolation (P3/P4, N=10⁵) shows that an in-family direct estimator is brittle at the prior's boundary independent of the supervision type.

## 6. Opus evidence review (2026-10-04, `claude-opus-5-5`)

Checked: provenance (all 20 units at clean commit `0ac6b34`), the C-a and C-d verdicts the conclusion rests on (re-read from `analysis.json`), and the claims in §2–4. **The measurements and the outcome mapping are correct.**

**On the "residual confound" (§5.2): it is the mechanism, not a nuisance.** The two arm families have the same declared channel, the same physics, the same prefixes, the same backbone, and matched (density) supervision. They differ in *how* the channel is used:
- composition (A4) applies the declared channel analytically at inference and learns only the clean law;
- the direct arms learn the channel's effect from simulated corrupted records.

"The clean law is easier to learn" is exactly the advantage composition can offer, so this cannot be removed without changing the question. The view-count difference (two corrupted views per step for the direct arms vs one clean view for A4) favours the direct arms, so it cannot explain A4's advantage.
The fixed-prior comparison C-d at P1 has no prior-family burden and is the cleanest in-family test. It gives the same answer: on the lead physics at N=10³, A4 is better by ≥Δ_min (−0.0177 ± 0.0071, ratio 0.58). At N=10⁵ the difference is within Δ_min (ratio 0.22).

**Verdict (within S2, the exogenous single-flip channel, physics-informed features, the one-step endpoint, 5 seeds):**
1. **Composition is never worse than direct training on simulated channel data under matched supervision.** Outcome 3 occurred in no cell.
2. **It is materially better off-family** (P3/P4: all cells, both N, ratios ≤ 0.06). Prior-conditioned *direct* models are brittle at the prior's boundary: the density-direct model was worse there than the probe-trained one.
3. **In-family, it is materially better at small N in the low-noise physics** (q_w=0.02, N=10³: P2, P5 vs A6pd; P1 vs A6d; P1 vs A6pd inconclusive with ratio 0.61). At N=10⁵ it is equivalent within the practical threshold, though 2–20× lower in error ratio.
   On the high-noise physics (q_w=0.5) no in-family advantage is detectable at N=10³ (ratios ≈ 1.0, wide CIs). There, all learned arms are poor: A4 is worse than the exact naive filter at P1.
4. Together with the pilot: the **multi-view route (folded vs candidate) is immaterial**. The levers are (a) using the declared channel analytically rather than learning it, (b) how the clean components are supervised, and (c) exact Bayesian weighting.

**Decision: stop the experimental phase.** The predeclared question is answered within its scope. The remaining open items are outside the current scope, not unresolved pilot details: external validity (generic features, unknown physics, other channels and simulators), closed-loop policy relevance, and poisoned training data. More seeds for the high-noise N=10³ cells would only tighten an already-practical "no detectable advantage". A consolidated statement is in [findings_summary.md](../findings_summary.md).

## 7. Correction after external review (2026-10-04, Opus 5.5) — supersedes the §6 wording where they conflict

An independent review (ChatGPT) reproduced the numbers and disputed the interpretation. Re-checked against `results/e1/analysis.json`:
- **"Never worse" is withdrawn.** C-a (A4 vs A6pd-broad) has 6 material A4 advantages, 4 practical equivalences and 4 inconclusive cells. No cell shows material inferiority, but an inconclusive interval also allows a material disadvantage.
  Example: q_w=0.5, N=10³, P1 gives d = +0.0045 ± 0.0384 against a margin of 0.0080. The supported statement is the count, with this uncertainty, not dominance.
- **Precision:** only 8 of 14 C-a cells (and 8/14 C-d, 5/14 C-b, 10/20 C-c) meet the half-width ≤ Δ_min/2 target. Results describe the implemented budgets (5 seeds, 8-config grid, epoch caps reached at N=10⁵), not converged or intrinsic sample efficiency.
- **Attribution:** A4 vs A6d/A6pd matches backbone, features and loss family, and differs by the intended pipeline choice (clean-law learning + analytic channel vs learning from simulated corrupted records). That comparison stands.
  C-b (A6pd vs A6p) changes architecture (unidirectional vs bidirectional GRU, 19k vs 47k parameters), token representation *and* loss. It compares two pipelines and **does not isolate supervision**. Claims are narrowed to "density pipeline vs regression pipeline".
- **Off-family results:** the training family Dirichlet(1,…,1) gives P(π_last > 0.9) = 10⁻⁷ and P(π_last > 0.5) ≈ 0.008. P3/P4 therefore test **sparse boundary coverage**, not merely "out-of-family" inputs.
  The composition arm's prior flexibility is structural and stands. "Direct estimators are brittle at the boundary" is withdrawn as a general claim: it is a statement about this training distribution, and E2 adds a coverage-aware direct control.
- **The fitted HMM (A5)** is near-oracle in every cell. Under a correctly specified simulator, a one-parameter structured estimator solves the problem, so these experiments do not show that neural inference is needed.
- **Analysis bug:** `analyze_e1.py` gave two supplementary comparisons the same id `extra`, so the JSON kept only A2-G vs A6pd (14 keys of A6d vs A6 overwritten). The markdown tables and the main comparisons C-a–C-d were unaffected. Fix and regeneration are scheduled (status file).
