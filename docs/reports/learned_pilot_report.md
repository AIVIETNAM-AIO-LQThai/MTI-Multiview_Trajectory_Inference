# Learned pilot report (v1; Sonnet 5.5; interpretation pending Opus review)

Everything is **measured** on synthetic S2 data against the exact oracle ([math spec](../mathematical_specification.md); oracle baselines
[oracle_v1_report.md](oracle_v1_report.md)). Full tables (every arm × prior × physics × N, all paired comparisons, diagnostics, costs):
[learned_pilot_tables.md](learned_pilot_tables.md). Design: [learned_pilot_protocol.md](../learned_pilot_protocol.md) v2.

## 1. What was run

- **Provenance:** code commit `09621bb` (clean tree), config = `experiments/run_learned.py` defaults, torch 2.14.1+cpu, Python 3.14.7, Windows 11, 18 single-thread worker processes.
  Commands: `run_learned.py --stage all --out results/learned_pilot` (tuning 1,177 s + 20 units, total 2,839 s), `analyze_learned.py`. Raw: `results/learned_pilot/{tuning.json,units/*.json,analysis.json}`.
- **Physics:** lead (ρ=0.9, c=1, q_a=1, λ=0.1, η=0.05, L=8, **q_w=0.02**) and secondary (**q_w=0.5**). **Priors:** P1 (β=.2 uniform), P2 (β=.5 uniform), P3 (β=.2, lag 0), P4 (β=.5, lag 0), P5 (β=.2, π∝2^{−lag}).
  Clean-trained arms were evaluated at every prior; channel-trained arms at all five on the lead physics and at P1/P2 on the secondary (protocol v2).
- **Arms:** A0 exact oracle; exact naive; A1 learned naive filter (¾N); A1r restricted-summary recalibrator (calibration ¼N, refit per prior); **A2** shared fixed-view BiGRU queried through folded (**G**), candidate (**M**) and
  averaged composition; **A3** unweighted masked ensemble; **A4** causal mixture-density scorer with candidate composition; **A5** two-state HMM with learned η (EM, known physics); **A6** channel-trained direct regression (P1); **A6p** prior-conditioned
  direct regression (narrow: β≤.3 uniform; broad: β≤.5, π~Dirichlet(1)). All neural arms: GRU hidden 64 (44–47k parameters; A4 17.7k).
- **Data/seeds:** N ∈ {10³, 10⁵} independent clean prefixes (+N/4 validation); tuning on seed 0 (8-config grid lr∈{1e-3,3e-3} × wd∈{1e-5,1e-3} × epoch cap {6,12} (N=10⁵) or {60,150} (N=10³), per arm, per physics and N, selected on the arm's deployable validation loss);
  main seeds 1–5 (fresh prefixes each). Test set: 20,000 independent prefixes per physics, Rao–Blackwellised over all L+1 corruption views, exact μ_2 used for evaluation only.
- **Endpoint:** E = E[κ(μ̂−μ_2)²] (0 = exact aware, V_2 = exact naive filter). Uncertainty: 95% t-interval over 5 training seeds (paired by seed for differences); test-set Monte Carlo error is separate and small by comparison. Δ_min = clip(0.10 V_2, 0.002 VOI_clean, 0.02 VOI_clean).
  Verdict rule: "x better" if the paired CI lies below −Δ_min, "equivalent" if inside ±Δ_min, else "inconclusive".

## 2. Limitations to read results through

1. **Grid-edge selection at N=10⁵.** All 12 N=10⁵ selections (6 arms × 2 physics) chose the epoch cap of 12, and 20 of the 24 selections overall chose the larger learning rate 3e-3 (`tuning.json`); the held-out validation loss was still the lowest at the largest budget. These models are probably not converged, so absolute errors at N=10⁵ are upper bounds on what each architecture can do, and arms with slower convergence (A2: three view-passes per step) are disadvantaged. The protocol grid was not changed after seeing this.
2. **A1 is a weak baseline:** it is trained on ¾N and, at N=10⁵, is worse than the *exact* naive filter (E 0.106 vs V_2 0.092 at P1, lead). "Captured opportunity" is therefore reported against the exact naive filter, not against A1.
3. **Structural privilege of A5 and A4.** A5's family (two symmetric states, known emission) contains the truth and only η is learned; A4's mixture family contains the true predictive mixture. Neither is a generic learner.
4. **A2 compares G, M, and average from one network.** At β=½ with a point-mass prior (P4) G collapses to one masked-belief query (spec T4, protocol note), so P4 G-vs-M numbers are partly structural.
5. Five seeds: several paired intervals do not meet the "half-width ≤ Δ_min/2" power rule (column in the tables); those cells are labelled inconclusive rather than equivalent. The pilot has not been extended to 10 seeds.
6. Probe-loss regression is high-variance by construction (the target contains the physical noise); this penalises A1, A2, A6, A6p relative to A4/A5, which use likelihoods of the observed transitions.

## 3. Results (selected; see tables for every cell)

### 3.1 Endpoint E, lead physics (q_w=0.02), N=10⁵, mean ± 95% CI over 5 seeds [captured opportunity]

| arm | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| exact naive (= V_2) | 0.0916 | 0.4109 | 1.6807 | 4.9071 | 0.6852 |
| A1 | 0.1061 ± 0.0076 [−16%] | 0.4379 ± 0.0173 [−7%] | 1.7221 ± 0.0392 [−2%] | 4.9999 ± 0.0973 [−2%] | 0.7160 ± 0.0228 [−4%] |
| A1r (refit per prior) | 0.0965 ± 0.0056 [−5%] | 0.3384 ± 0.0551 [18%] | 0.8986 ± 0.2193 [47%] | 1.7812 ± 0.3634 [64%] | 0.4616 ± 0.0939 [33%] |
| A5 HMM (learned η̂=0.0500) | 0.0000 [100%] | 0.0000 [100%] | 0.0000 [100%] | 0.0000 [100%] | 0.0000 [100%] |
| A4 density + candidate | 0.0008 ± 0.0001 [99%] | 0.0008 ± 0.0001 [100%] | 0.0011 ± 0.0001 [100%] | 0.0010 ± 0.0001 [100%] | 0.0010 ± 0.0001 [100%] |
| **A2-G** (folded) | 0.0061 ± 0.0017 [93%] | 0.0073 ± 0.0021 [98%] | 0.0080 ± 0.0019 [100%] | 0.0109 ± 0.0020 [100%] | 0.0084 ± 0.0026 [99%] |
| **A2-M** (candidate) | 0.0063 ± 0.0018 [93%] | 0.0076 ± 0.0024 [98%] | 0.0085 ± 0.0025 [99%] | 0.0075 ± 0.0022 [100%] | 0.0090 ± 0.0030 [99%] |
| A2-avg | 0.0062 ± 0.0017 | 0.0073 ± 0.0023 | 0.0079 ± 0.0022 | 0.0074 ± 0.0017 | 0.0086 ± 0.0028 |
| A3 unweighted ensemble | 0.0561 ± 0.0031 [39%] | 0.3049 ± 0.0087 [26%] | 1.3040 ± 0.0213 [22%] | 3.9486 ± 0.0540 [20%] | 0.5028 ± 0.0119 [27%] |
| A6 direct (trained P1) | 0.0079 ± 0.0022 [91%] | 0.1057 ± 0.0123 [74%] | 0.6873 ± 0.0474 [59%] | 2.3311 ± 0.1319 [52%] | 0.1934 ± 0.0202 [72%] |
| A6p-narrow | 0.0089 ± 0.0005 [90%] | 0.0596 ± 0.0130 [85%] | 0.6992 ± 0.0432 [58%] | 0.9915 ± 0.1367 [80%] | 0.2033 ± 0.0159 [70%] |
| A6p-broad | 0.0097 ± 0.0019 [89%] | 0.0170 ± 0.0044 [96%] | 0.0984 ± 0.0139 [94%] | 0.5345 ± 0.0652 [89%] | 0.0122 ± 0.0037 [98%] |
| *Δ_min* | 0.0092 | 0.0411 | 0.0617 | 0.0617 | 0.0617 |

Other cells: N=10³ lead, and the secondary physics at N=10³/10⁵, are in the tables. Same ordering at q_w=0.5/N=10⁵ (A5 ≈ 0, A4 0.005–0.007, A2-G 0.015–0.025, A6p-broad 0.020 at P1 and 0.040 at P2, A3 and A1 far from the oracle).
At N=10³ the probe-regression arms are poor: lead P1: A1 0.136, A2-G 0.185, A2-M 0.186, A6 0.110, A6p-broad 0.096 versus exact naive 0.092 (all worse than the exact naive filter at P1, i.e. negative captured opportunity), A4 0.026, A5 0.0001.

### 3.2 Paired comparisons against Δ_min (5 seeds; "—" = not evaluated)

- **A2-G vs A2-M** (folded vs candidate composition from the same network): *equivalent* in 14 of 20 (physics, N, prior) cells, *inconclusive* in 4, **A2-M better** in 1 (lead, N=10³, P4: d=+0.202 ± 0.129, Δ=0.062; the structural P4 collapse), **A2-G better** in 1 (secondary, N=10³, P1: −0.034 ± 0.023, Δ=0.008, where both arms are far worse than exact naive).
  Mean signed gap G−M at N=10⁵ is within ±0.0022 in every cell (clean and corrupted records have the same sign), E[κ(G−M)²] ≤ 0.013, mean |Ĉ_j| 0.013–0.021 (0.04–0.09 at N=10³), identity error ≤ 2.2e-15 (`learned_pilot_tables.md`, route-gap table).
- **A2-G vs A2-avg:** equivalent in 15, inconclusive in 3, A2-avg better in 2. **A3 (unweighted ensemble) is worse than A2-G by ≥Δ_min in 16 of 20 cells**; the rest are inconclusive.
- **Composition vs channel-trained direct regression** (lead physics N=10⁵; 5 priors): A2-G vs **A6p-broad**: equivalent at P1, P2, P5; **A2-G better at P3 (−0.0905 ± 0.0152) and P4 (−0.5236 ± 0.0654)**. At N=10³ A6p-broad is better at P1 (+0.089 ± 0.042) and A2-G is better at P3/P4 (−0.516, −1.913); P2/P5 inconclusive.
  Secondary physics (P1/P2 only): inconclusive at both N. A2-G vs **A6** (fixed prior P1): A2-G better in 7 of 14 cells where both exist, A6 better in 1, equivalent in 1, inconclusive in 5. A6 and A6p-narrow degrade at the shifted priors (A6 at P4: 2.33; A6p-narrow at P3: 0.70) while A2 and A6p-broad do not at P1/P2/P5.
- **A4 vs A2:** A4 better than A2-G by ≥Δ_min in all 10 N=10³ cells (A2-G − A4 = +0.16 … +1.01); at N=10⁵ the two are *equivalent* in 9 of 10 cells (one inconclusive), with A4 lower by 0.005–0.018 in every cell.
- **A5 vs A4:** A5 better in 6 cells (all N=10³), equivalent in 14. **A5 vs A2-G:** A5 better in 11, equivalent in 9.
- **A1r vs A1:** per-prior refit helps at the shifted priors (e.g. lead N=10⁵ P4: 4.9999 → 1.7812) but captures 5% (P1: negative) to 64% of V_2 at best, far below composition arms.

### 3.3 Mechanism diagnostics

- **Contamination count and query error (A2, κ-weighted MSE against exact conditionals, lead, N=10⁵):** repaired-flip queries T_jS with 0 / 1 / 2 contaminated positions: 0.0057 / 0.0225 / 0.0358;
  folded queries O_j(S) with 0 / 1 retained flips: 0.0069 / 0.0252; sign-head KL to the exact opposite-sign probability: 0.0017 / 0.0051. At N=10³: 0.176 / 0.550 / 0.804. The same ordering holds at q_w=0.5 (0.0174 / 0.0446 / 0.0679). Query error grows with contamination count by a factor of 4–6.
  (Descriptive; unweighted over the L+1 views; contaminated queries occur only on corrupted records.)
- **Clean-record harm and natural-switch cost** (realised-cost difference vs exact naive, lead, N=10⁵, P1; oracle: D_none +0.065, D_switch +0.916): A4 +0.067/+0.943; A2-G +0.074/+0.975; A2-M +0.074/+0.969; A6 +0.063/+0.913; A6p-broad +0.053/+0.694; A3 +0.012/+0.093; A1 +0.006/+0.054 (A1r −0.015 in the switch stratum).
  The composition arms reproduce the oracle's behaviour (small cost on clean records, large gain in natural-switch states). A6p-broad under-reproduces the switch-stratum gain (+0.694 ± 0.027). Per-stratum values for all arms, both physics and N=10⁵ are in the tables.
- **Inference cost per test record:** A1/A6/A6p 1 forward pass; A2-G 9, A2-M and average 17, A4 9, A5 analytic O(L). Training (N=10⁵, one thread, lead): A1 111 s, A2 568 s, A4 51 s, A6 157 s, A6p 174–183 s; A5 EM <1 s (η̂ = 0.0500 ± 0.0002 at N=10⁵, 0.0505 ± 0.0020 at N=10³). Prior-conditioned or channel-trained arms need training data/compute per training prior family; composition arms reuse one clean-trained model for all priors.

## 4. Open items and flags for the Opus review

1. **Hypotheses H2/H3 interpretation** (route differences; flexibility trade-off) — numbers above; the decision on what is supported is Opus's.
2. **Extend the grid or training budget at N=10⁵?** The edge selections (limitation 1) may matter most for the A2-vs-A4 gap and for the "captured" values of A1/A6; an extended-epoch rerun of the N=10⁵ units is a ≈1.3 CPU-hour job.
3. **Seeds:** which paired comparisons need 10 seeds to meet the Δ_min/2 power rule (see the "CI half-width ≤ Δ_min/2" column; mostly the N=10³ and secondary-physics rows).
4. **Mask diversity, separate encoders, compatibility/distillation** remain deferred; the route-gap and query-level diagnostics above are the data the protocol said should motivate them.
5. **P4 reduction:** report G and M at P4 separately from the other priors in any summary (limitation 4).
