# MTI mathematical specification — exact reference (v1)

Status: **approved by Opus planning pass, 2026-10-04**, branch `exp/oracle-evidence`, base commit `c250658`.
Scope: S2 simulator, single-flip action-sign channel, exact (oracle) inference, one-step decision
quantities, first information/opportunity experiment. Learned arms are in
[learned_pilot_protocol.md](learned_pilot_protocol.md). Changes to anything marked **[target]** require an
Opus analysis pass; implementation details marked **[impl]** are Sonnet's call.

Notation: `N(x; m, v)` is the normal density with mean `m`, **variance** `v`. `σ(x)=1/(1+e^{-x})`.
All indices are 0-based.

## 1. Joint law (S2) [target]

Parameters `θ_phys = (ρ, c, q_a, q_w, λ, η, L)` with |ρ|<1, c≠0, q_a>0, q_w>0, λ≥0, η∈[0,½], L≥1.
`D = c²+λ > 0`.

- Modes `m_0..m_L`: `m_0 ~ Unif{±1}`, `Pr(m_{k+1} ≠ m_k) = η`.
- Actions `a_0..a_{L-1} ~ iid N(0,q_a)`; noises `w_0..w_{L-1} ~ iid N(0,q_w)`; `s_0 ~ N(0, (c²q_a+q_w)/(1-ρ²))`.
  All mutually independent and independent of the mode path.
- `s_{k+1} = ρ s_k + c m_k a_k + w_k`, k = 0..L-1. **Transition k uses m_k**; the last recorded
  transition uses `m_{L-1}`; the decision uses `m_L`.
- Prefix `H = (s_0, a_0, s_1, …, a_{L-1}, s_L)`, stored as arrays `s[0..L]`, `a[0..L-1]`.
- Probe branch (stored separately, never an inference input): `a_L^pr ~ N(0,q_a)`,
  `s_{L+1}^pr = ρ s_L + c m_L a_L^pr + w_L^pr`.
- Decision branch (evaluation only): `s_{L+1}^dec = ρ s_L + c m_L a + w^dec` for the evaluated action `a`,
  with `w^dec` independent of the probe noise. Use common random numbers `w^dec` across compared actions.

Transition residual `δ_k = s_{k+1} − ρ s_k`; emission `e_k(m) = N(δ_k; c m a_k, q_w)`.
Clean density factorises as
`p_0(H) = N(s_0;0,v_0) · Π_k N(a_k;0,q_a) · Σ_{m_{0:L-1}} ½ Π P(m_k→m_{k+1}) Π_k e_k(m_k)`.
The first two factors are invariant under any action-sign flip, so all likelihood **ratios** below depend
only on the HMM transition likelihood `ℓℓ(H) = log Σ_m … Π e_k(m_k)` ("transition log-likelihood").
"Full-prefix" log density adds the sign-invariant term `log N(s_0) + Σ log N(a_k)`.

## 2. Channel [target]

`T_j` (j=0..L-1) negates the recorded `a_j` only; `T_none = id`. `S = T_θ H` with
`Pr(θ=none)=1-β`, `Pr(θ=j)=q_j=βπ_j`, `Σπ_j=1`, β∈[0,½] in this study, π fixed and independent of
everything. θ is independent of (modes, probe, decision noise) given H (**exogenous channel**). Primary:
uniform π. Zero masses `q_j=0` allowed and handled exactly (§6).

Sign convention for exact zeros [impl, fixed here]: recorded sign `sgn(a)=+1` if `a ≥ 0`, else −1. With
`a_j=0`, `T_jS=S`, `ℓ_j=1`, `r_j=½`; all formulas below remain valid without special-casing.

## 3. Clean-law HMM computations

Log-domain forward filter [impl: vectorised over prefixes]:
`log α_0(m) = log ½`; `log α̃_k(m) = log α_k(m) + log e_k(m)`;
`log α_{k+1}(m') = logsumexp_m(log α̃_k(m) + log P(m,m'))` (η=0 gives −∞ entries; must work).
`ℓℓ(H) = logsumexp_m log α̃_{L-1}(m)`.
`Λ(H) = log α̃_{L-1}(+) − log α̃_{L-1}(−)` (log-odds of m_{L-1}).
**Clean belief** `μ(H) = E_0[m_L|H] = (1−2η)·tanh(Λ/2)` — the final mode transition is explicit.
Naive belief `μ̃(S) = μ(S)`.

## 4. Candidate route

For each j: `log ℓ_j = ℓℓ(T_jS) − ℓℓ(S)`;
`log Z = logsumexp(log(1−β), {log q_j + log ℓ_j : q_j>0})`;
`p_none = (1−β)/Z`, `p_j = q_j ℓ_j / Z` (exactly 0 when q_j=0).
`M(S) := μ_2(S) = p_none μ(S) + Σ_j p_j μ(T_jS)`.
`Var(μ_H|S) = p_none μ(S)² + Σ p_j μ(T_jS)² − μ_2²` (variance of the clean conditional belief, not of m_L).
Derivation: θ exogenous and `T_θ` an involution with unit Jacobian ⇒
`Pr(θ=j|S) ∝ q_j p_0(T_jS)`, `Pr(none|S) ∝ (1−β)p_0(S)`; and `E[m_L|S,θ] = μ(T_θS)` ⇒
`μ_2 = E[m_L|S] = E[μ_H|S]`.

Reference implementation [impl]: one forward pass per query (L+1 records per S). An O(L)
forward–backward fast path is allowed only if it matches the reference to the §13 tolerances.

## 5. Folded route (computed independently) [target]

Observer `O_j(S) = (S with a_j's sign removed, |a_j|)`. Marginalising the uniform, mode-independent
sign gives the folded emission `ē_j = ½[N(δ_j; c|a_j|, q_w) + N(δ_j; −c|a_j|, q_w)]`, **which is the same for
m=±1**. Hence (audit note A3):

- `ψ_j = E_0[m_L|O_j]` = forward filter with `log e_j(·)` replaced by the constant `log ē_j`.
- `P(m_j=m|O_j) = P(m_j=m | all transitions except j)` from forward–backward with transition j neutral.
- Given `m_j=m`, `Pr(sign=+|m, δ_j, |a_j|) = σ(2c m |a_j| δ_j / q_w)`. So, in log form,
  `log p_j^+ = logsumexp_m[log P(m_j=m|O_j) + log σ( 2cm|a_j|δ_j/q_w)]`,
  `log p_j^- = logsumexp_m[log P(m_j=m|O_j) + log σ(−2cm|a_j|δ_j/q_w)]` (use `log σ(x) = −softplus(−x)`).
- `r_j` = probability of the sign **opposite** to the recorded one: if recorded sign is +,
  `log r_j = log p_j^-`, `log(1−r_j) = log p_j^+`; else swapped. The recorded sign is applied **after**
  querying O_j; it is never an O_j input.

Folded composition, using only folded quantities (audit note A1):
`Z_G = (1−2β) + Σ_j q_j/(1−r_j)`, `G(S) = [(1−2β)μ(S) + Σ_j q_j ψ_j/(1−r_j)] / Z_G`.
Identities to verify (not to assume): `ℓ_j = r_j/(1−r_j)`, `Z_G = Z`, `G = M`,
`C_j = ψ_j − (1−r_j)μ(S) − r_jμ(T_jS) = 0`.
Audit note A2: the identity G=M holds for any β<1; β≤½ is what makes G a **convex** combination of query
beliefs (weights `(1−2β)/Z` and `q_j/((1−r_j)Z)`, summing to 1).

Learned-route identity (for the protocol, not this run): with heads μ̂, ψ̂, r̂, **candidate ℓ̂ = r̂/(1−r̂)
from the same sign head** and the same Ẑ, `Ĝ − M̂ = Ẑ^{-1} Σ_j q_j Ĉ_j/(1−r̂_j)` (checked algebraically).
If the candidate route uses a separately learned density ratio, an extra term appears (audit note A7).

## 6. Numerical rules [target for the oracle]

Log-domain throughout; keep both `log r` and `log(1−r)`; never compute `1−r` by subtraction; never floor
probabilities or clip beliefs; zero masses by explicit masks (no `log 0 + finite` NaNs);
`1/(1−r_j) = exp(−log(1−r_j))`. q_w=0 is a separate discrete reference (not implemented in v1); never
evaluate a zero-variance Gaussian. float64 everywhere.

## 7. Decision quantities [target]

Cost `s_{L+1}² + λa²`. `κ(s_L) = ρ²c²s_L²/D`, `a*(s_L,u) = −ρc s_L u/D`.
`C(s,u,a) = ρ²s² + 2ρc s u a + D a² + q_w` (expected cost when `E[m_L|info]=u`).
For an info-measurable belief û and true conditional mean u: `C(s,u,a*(s,û)) = ρ²s²+q_w − κu² + κ(û−u)²`.
With the hidden mode itself: `cost(s, m_L, a*(û)) averaged over w = ρ²s²+q_w − κ + κ(û−m_L)²`.

Report-level analytic costs (expectations over the declared law):
zero action `E[ρ²s_L²+q_w]`; mode oracle `E[ρ²s_L²+q_w−κ]`; clean oracle `E[ρ²s_L²+q_w−κμ_H²]`;
naive = clean + `E[κ(μ̃−μ_H)²]`; aware = clean + `E[κ(μ_2−μ_H)²]`.
`VOI_clean = E[κμ_H²]` (decision value of clean history) is the reporting scale.

## 8. Estimands of the first report

- `V_2 = E[κ(μ̃−μ_2)²]` (opportunity of awareness); `I_loss = E[κ Var(μ_H|S)]` (irrecoverable);
  `X_naive = E[κ(μ̃−μ_H)²]` (naive excess over clean oracle). Population identity `X_naive = V_2 + I_loss`.
  It also holds **exactly conditionally on S** (audit note A5):
  `E[κ(μ̃−μ_H)²|S] = κ Σ_θ p_θ (μ(S)−μ(T_θS))² = κ(μ̃−μ_2)² + κVar(μ_H|S)`; with the realized H it holds
  only in expectation.
- Strata rule (audit note A4): strata defined by θ (exogenous) may use `κ(û−μ_H)²` because
  `E[m_L|H,θ]=μ_H`. **Strata defined by hidden modes must use `κ(û−m_L)²`** (the mode-conditional expected
  cost up to a stratum-common term); μ_H is not the conditional mean within such strata.
- Lag of location j: `lag = L−1−j` (lag 0 = last recorded action).
- Natural-switch diagnostic: primary edge `E_last = 1[m_{L−1} ≠ m_{L−2}]` (the edge into the mode of the
  last recorded transition — the one confusable with a lag-0 flip; L≥2). Also report the decision edge
  `E_dec = 1[m_L ≠ m_{L−1}]` (unpredictable from any record) and "any prefix switch" by edge lag. A natural
  switch is a property of the hidden path, not an apparent sign change in the record.
- Corruption-posterior diagnostic: categorical log score and Brier for θ over {none, 0..L−1}; binary
  "any corruption" (`1−p_none`) reliability (15 equal-mass bins), Brier, log loss, ECE; location top-1 accuracy
  given θ≠none; each compared against the prior-only predictor. Exact posteriors should be calibrated up
  to MC error — this checks the pipeline; it is not a detection result.

## 9. Restricted-summary recalibration

`B = (s_L, μ̃)`, `μ_B = E[μ_2|B]`. Since κ is B-measurable,
`V_2 = E[κ(μ̃−μ_B)²] + E[κ(μ_B−μ_2)²]`. Symmetries: μ_B is even in s_L and odd in μ̃ (§14 T13), so
use features `(|s_L|, μ̃)` and odd-symmetrise.
Approximation g [impl details free]: κ-weighted binned regression of the oracle target μ_2 on
(|s_L|-quantile × μ̃-quantile) bins, fitted on independent "fit" prefixes, resolution selected on independent
"validation" prefixes, evaluated on the main prefixes. Report all three held-out terms
`A=E[κ(μ̃−g)²]`, `R=E[κ(g−μ_2)²]`, `X=2E[κ(μ̃−g)(g−μ_2)]` with CIs, at ≥2 resolutions and ≥2 fit sizes.
Rigorous one-sided reading (audit note A6), for any g fixed before evaluation:
`R ≥ E[κ(μ_B−μ_2)²]` (remaining error upper-bounds the exact additional-context term) and
`V_2 − R ≤ E[κ(μ̃−μ_B)²]` (attained improvement lower-bounds the exact recalibration term), up to
evaluation MC error. This is a privileged oracle diagnostic, not a deployable recalibrator.

## 10. Input-contamination profile

For each (H,θ) and its record S, query set: `S`, `T_kS` (k=0..L−1), `O_k(S)` (k=0..L−1).
Contamination count = number of action positions where the query's **signed** action differs from H
(for O_k the sign at k is absent and not counted): S: 1[θ≠none]; `T_kS`: 0 if k=θ, else
1[θ≠none]+1; `O_k(S)`: 1[θ∉{none,k}].
Per query record: (i) transition log-likelihood ℓℓ — for folded queries the sign-marginalised
`log ½[p(·|+)+p(·|−)]` (a proper density of s_{1:L} given the folded conditioning set); (ii) paired
difference `ℓℓ(query) − ℓℓ(H)`; (iii) percentile of ℓℓ(query) in the clean distribution of ℓℓ(H);
(iv) one-step predictive log density at the flipped transition(s). Full-prefix scores differ by a
sign-invariant per-prefix term — report which is used (primary: transition). Route weights: candidate
`p_none, p_k`; folded `(1−2β)/Z, q_k/((1−r_k)Z)`. Summaries: weight-mass on contaminated queries by
route, κ-weighted, and by θ stratum. This profile is a proposed explanation for later finite-sample errors,
not evidence of a masked-route advantage or OOD robustness.

## 11. Monte Carlo design [target for the estimand; impl for engineering]

- Unit: independent clean prefix H. Primary estimator is Rao–Blackwellised over θ: for each H evaluate all
  L+1 views and `Y(H) = (1−β) f(H) + Σ_j q_j f(T_jH)`. Y is iid across prefixes; 95% normal interval
  `Ȳ ± 1.96 sd(Y)/√N`. θ strata use the paired per-view values (strata share prefixes; differences
  between strata use paired per-prefix differences).
- Check estimator: independent seed, θ sampled once per prefix by the channel stream, plain mean; must
  agree with the main estimate (|z| ≤ 4).
- Sample size: pilot N_p = 2×10⁴ (own seed); `N_main = max(10⁵, ⌈1.25(1.96·sd_Y/(0.05·V̂_2))²⌉)`, rounded up
  to whole chunks, cap 2×10⁶ per cell (if cap binds, report achieved precision). If `V̂_2 < 0.01·VOI_clean`,
  use the absolute target half-width ≤ `5×10⁻⁴·VOI_clean`. No adaptive re-inspection after the main run.
- Controls β=0 and η=½: V_2 = I_loss = 0 **identically** (η=½ ⇒ all beliefs 0); verify deterministically
  (|·| ≤ 1e−15 per sample) and report, rather than spend MC precision on them.
- Seeds: `numpy.random.SeedSequence(master, spawn_key=(cell_id, stream_id, chunk_id))`, streams
  0 s_0, 1 modes, 2 actions, 3 transition noise, 4 corruption, 5 probe, 6 decision noise.
  Masters: pilot 1001, main 2001, check 3001, recal-fit 4001, recal-validation 5001, tests 0–999.
  Chunk size is part of the config.

## 12. Experiment cells

Fixed: ρ=0.9, c=1, q_a=1, λ=0.1, π uniform.
Primary: L=8, q_w=0.5, β=0.2, η=0.05. Controls (primary otherwise): β=0; η=½.
Map (η=0.05): L∈{8,32} × q_w∈{0.02,0.5} × β∈{0.2,0.5} (8 cells incl. primary). Every cell is reported.

## 13. Tolerances

- Deterministic identities on beliefs/probabilities (values in [−1,1] or [0,1]): abs ≤ 1e−10.
- Log-likelihoods/log-ratios: `|Δ| ≤ 1e−9·max(1,|value|)`.
- Cost identities: rel ≤ 1e−10.
- Statistical checks: |z| ≤ 4 two-sided, all z reported (≤50 checks ⇒ family-wise false-alarm < 0.5%).
Tolerances may not be loosened to pass a test; a failure after focused debugging escalates to Opus with a
minimal failing case.

## 14. Validation tests (acceptance = all pass, outputs saved)

- T1 Enumeration over all mode paths m_{0:L} (L∈{1,2,3,5,8}; η∈{0,0.05,0.5}; q_w∈{0.02,0.5}; random
  ρ,c,λ in domain): full-prefix and transition log-likelihoods, μ(H) **including the m_{L−1}→m_L step**.
- T2 Enumeration over (θ, mode path) with full joint densities, random non-uniform π incl. zero masses:
  p_θ, μ_2, Var(μ_H|S).
- T3 Normalisation `p_none+Σp_j=1`; β=0 ⇒ p_none=1, μ_2=μ̃; q_j=0 ⇒ p_j=0 exactly.
- T4 Known location π=δ_k: `p_k = βℓ_k/(1−β+βℓ_k)`; at β=½, `μ_2 = ψ_k` for each k (incl. k=L−1). Not
  applied to uniform π.
- T5 Final prediction: `μ_H = (1−2η)E[m_{L−1}|H]` (enumeration); on simulated data `E[m_L μ_H] = E[μ_H²]`,
  `E[m_L μ_2] = E[μ_2²]` and binned calibration (|z|≤4). Mutation check: the same test with m_{L−1} as target
  must fail for η=0.05, demonstrating the test's power.
- T6 `log ℓ_j = log r_j − log(1−r_j)` (candidate vs independent folded smoother); r_j and ψ_j vs enumeration
  with sign-marginalised emission.
- T7 `C_j=0`, `Z_G=Z`, `G=M` for all j on clean and corrupted records, both q_w, L∈{8,32}.
- T8 `O_j(T_jH) = O_j(H)`; O_j carries |a_j| and no sign.
- T9 Decision: C(s,u,a) vs direct expectation over m_L; a* = numerical argmin; excess = κ(û−u)²;
  simulated decision-branch mean costs (CRN) vs analytic (|z|≤4).
- T10 Conditional decomposition (A5) per S to 1e−10; realized-H version in expectation (|z|≤4).
- T11 Controls η=½ and β=0 exact zeros.
- T12 Access: inference functions accept only a `Prefix(s, a)` object; perturbing probe/hidden/decision
  fields leaves outputs bit-identical.
- T13 Symmetries: negating all actions ⇒ ℓℓ invariant, μ, μ_2 negate; (s,a)→(−s,−a) ⇒ μ, μ_2 invariant.
- T14 Simulator moments: Var(s_0), Var(s_L) (stationary), mode-flip rate η, Var(a), Var(δ_k − c m_k a_k)=q_w,
  corruption frequencies β, π (|z|≤4).
- T15 Main (RB) vs check (single-draw) estimator agreement (in the experiment run).
- T16 Extreme regime q_w=0.02, L=32: all outputs finite; both log r and log(1−r) finite where the exact
  value is; probabilities sum to 1.

## Audit notes (Opus, 2026-10-04)

The briefing's formulas were checked algebraically and are consistent. Additions/clarifications, none of
which changes a target: A1 folded normaliser `Z_G=(1−2β)+Σq_j/(1−r_j)`; A2 β≤½ ⇔ convexity of the folded
route; A3 folded emission is mode-symmetric (ψ_j = filter with transition j deleted; r_j from the smoother
marginal at m_j) — this supplies a genuinely independent folded computation; A4 hidden-mode strata must
use κ(û−m_L)²; A5 the X=V_2+I_loss identity is exact conditionally on S; A6 one-sided bounds from any fixed
recalibrator; A7 the Ĝ−M̂ identity requires ℓ̂ from the same sign head and shared Ẑ; A8 β=0 and η=½
controls are exact zeros, so they are pipeline checks rather than measurements.
