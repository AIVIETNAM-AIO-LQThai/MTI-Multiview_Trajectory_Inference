# Decision-identification closeout: L ≤ 3 geometry, certificates and corrected claims

Branch `exp/joint-identification`, base `08dd58e`.
- **Part I** (Opus 5.5, 2026-10-12): verdicts on Codex's reply, the mathematics, and the implementation specification.
- **Part II**: Sonnet 5.5 implementation results.
- **Part III** (Opus 5.5, 2026-10-12): final review, the answers to Codex's §7 deliverable, and the compact thesis claim table.

Inputs:
- the Opus discussion note of 2026-10-11 (`decision_identification_discussion.md`, written in a separate clone; re-supplied by the user after Part II and now imported as [decision_identification_discussion.md](decision_identification_discussion.md), body unchanged, with a provenance header);
- Codex's reply ("bounded closeout of the decision-identification chapter");
- the verification archive `mti_decision_identification_verification.zip`, ported under `experiments/decision_id/` in Part II.

---

## Part I. Verdicts and mathematics (Opus)

### I.1 Verdicts on Codex's reply

**Overall.** I accept every substantive point, with one strengthening (I.2: a Jung-type upper bound) and one addition (I.5: L = 2 with η known has an exact answer). Nothing is refuted. The recommendation is to finish this chapter and write up the thesis; I see no indispensable gap that the bounded work leaves open (I.7).

| item | verdict | basis |
|---|---|---|
| §2 L = 3 materiality from the two-law lower bound (5 of 6 corrupted cells) | **Accept** | Matches the discussion's Table B (the "V*" column there *is* the endpoint-pair quantity). Only the label changes. |
| §3 Lower bound ≠ minimax; endpoint-ball criterion; b_pw is the wrong falsifier | **Accept** | Proof in I.2. In the discussion, the falsifier "b_pw above V*" was wrong: at lead T0, b_pw is 0.4% above V* while the endpoint midpoint can still be minimax. |
| §3 Frank–Wolfe reporting (gap not saved, support from rounding) and the "Vstar" label in `diam_se.py` | **Accept** | Confirmed in the archive code: no gap, iteration or termination output; the support is read off weights > 1e-4. |
| §4 L = 2 with η unknown is a **triangle**, and the least-favourable prior can have three support points | **Accept: verified independently** | Analytic re-derivation in I.3. Numerical check: beliefs match the HMM to 2.4e-15, and the three vertex worlds give identical record laws (4.9e-15), on 4 × 10⁵ records, secondary physics, h = 0.54. Monte Carlo G = [[4.015, −1.326], [−1.326, 3.918]], close to Codex's quadrature [[3.992, −1.316], [−1.316, 3.913]]. All three angle inner products are positive (0.26, 1.71, 0.53). Circumcentre weights 0.194 / 0.454 / 0.352 (Codex 0.196 / 0.454 / 0.350). Raw minimax 0.585 vs ¼·diam² 0.560 (Codex 0.583 / 0.557). Farthest-pair midpoint 0.817 (Codex 0.817). The ρ → 1 limit argument is re-derived and correct (I.3). |
| §5 Proposition A, uniform case | **Accept** | If F is uniform on {0,1}², then F̂(A) = 0 for A ≠ ∅, so every η′ < ½ with uniform F′ is compatible, and at η′ = ½ every channel is. The identified set is then not a compact arc. The 0.4999 in `eta_hi` is a numerical cutoff. |
| §5 Proposition B, wording | **Accept** | Exact statement: the sample profile is non-increasing in η, and the lower end of the domain is a maximiser. What the implementation returns depends on EM tolerance and tie rules, so describe the saved runs as "near the bottom", not as a theorem. |
| §5 Weighted identification (equality on {κ > 0}) | **Accept** | Definition in I.2. With ρ = 0.9 and c ≠ 0, κ > 0 a.s., so no current result changes. |
| §5 Clean calibration is not a universal collapse at L ≤ 3 | **Accept** | Clean prefixes must contain at least one mode edge (L_clean ≥ 2) to identify η. With η known: L = 3 collapses (Prop. (i)); L = 2 keeps the allocation segment (I.5); L = 1 keeps β. |
| §5 Q2: "bounds the decision cost below the margin", not "removes the ambiguity"; no intrinsic ordering of side information | **Accept** | Q2 is restricted to one explicit β-cap family on the L = 3 truths (spec II.4). |
| §5 Crossed SE double-counts the interaction | **Accept, and fix** | For a two-way crossed layout D_ri = μ + a_r + b_i + e_ri, the unbiased variance of the grand mean is (MS_R + MS_C − MS_E)/(RN). The formula used, s²(rep means)/R + s²(prefix means)/N, equals (MS_R + MS_C)/(RN), adding σ_e²/(RN) in expectation. Implement the ANOVA estimator, labelled, with its assumptions (spec II.5). |
| §6 The L = 3 four-function representation and the degree-8 stationary-point certificate | **Accept: derivation checked** | I.4. |

### I.2 Decision identification and minimax: definitions and certified bounds

**Setting.**
- Let Ξ(P) be the set of parameters compatible with the record law P.
- The **decision target set** is T(P) = {μ_ξ : ξ ∈ Ξ(P)} ⊂ L²(P, κ), with ‖f‖² = E_P[κ f²].
- **Decision identification:** diam T(P) = 0, i.e. μ_ξ = μ_ξ′ on {κ > 0}, P-a.s. If the sign of c is not supplied, use γ = cμ.

**Minimax.**

  V*(P) = inf_b sup_{ξ ∈ Ξ(P)} ‖b − μ_ξ‖² = r²(conv T(P)),

- This is the squared Chebyshev radius: the radius of the smallest ball enclosing T, which is also the smallest enclosing the closed convex hull, since the risk is convex in μ.
- The minimiser b* is unique, because sup_ξ ‖b − μ_ξ‖² is strictly convex in b.
- Priors on Ξ are not updated by S. So V* = max_π E_P[κ Var_π μ_ξ] (Γ-minimax; Berger 1985), and b* = E_{π*} μ_ξ.

**Certified bracket (universal).**

  ¼ diam²(P) ≤ V*(P) ≤ (n / (2(n + 1))) · diam²(P),

where n is the dimension of the affine hull of T(P).
- Lower bound: two points.
- Upper bound: Jung's theorem. In any Hilbert space r ≤ diam/√2 (the n → ∞ limit); this is a standard result, not re-read here.
- n = 1 (a segment): V* = ¼ diam² exactly.
- L = 2, η unknown: n ≤ 2, so V* ≤ ⅓ diam².
- L = 3: T lies in a 4-function span, so n ≤ 4 and V* ≤ 0.4 diam².

**Endpoint-ball criterion.** For a, c ∈ T with m = (a + c)/2 and d = ‖a − c‖, the following are equivalent:
- (1) the prior ½δ_a + ½δ_c is least favourable;
- (2) m is minimax and V* = d²/4;
- (3) T ⊂ B̄(m, d/2);
- (4) ⟨μ − a, μ − c⟩ ≤ 0 for all μ ∈ T.

*Proof.*
- (3) ⇔ (4) by expanding ‖μ − m‖² = ⟨μ − a, μ − c⟩ + d²/4.
- (3) ⇒ (2): sup_ξ R_ξ(m) ≤ d²/4, which equals the two-point lower bound.
- (2) ⇒ (1): the Bayes value of the ½/½ prior is d²/4 = V*.
- (1) ⇒ (3): if the ½/½ prior is least favourable, its Bayes value d²/4 equals V*. The unique minimax rule must then be Bayes for that prior, which makes it m. So sup R(m) = d²/4. ∎

**What the criterion tests.** It tests the endpoint midpoint m itself. The pointwise-envelope midpoint b_pw minimises a different, per-record criterion, and can be suboptimal globally even when m is minimax.

### I.3 L = 2: exact target-set geometry

**Assumptions.** Lemma 1; η ∈ [0, ½); the full single-flip simplex; h ∈ (0, 1). Let x_j = c ã_j δ_j / q_w and z_j = tanh x_j.

**Derivation (re-done).**
- The record density relative to an independent-uniform-modes base is (1 + h z₀z₁), with h = r(1 − 2β) and r = 1 − 2η.
- With σ₁ = −1 iff θ = 1:

  E[m̃₁σ₁ | S] = [(1 − 2q₁)z₁ + r(1 − 2q₀)z₀] / (1 + h z₀z₁).

- Hence μ = φ·(u, v), with φ = (z₁, z₀)/(1 + h z₀z₁), u = r(1 − 2q₁) and v = r²(1 − 2q₀).
- Feasibility:
  - β = (1 − h/r)/2 ≥ 0 requires r ∈ [h, 1].
  - u = h + 2r q₀ ∈ [h, r].
  - v = r² − r(u − h), which is increasing in r for r ≥ u.
- The target set is therefore the triangle 𝒯_h = {h ≤ u ≤ 1, hu ≤ v ≤ 1 + h − u}, with vertices A = (h, h²) (η = (1 − h)/2, β = 0), B = (h, 1) (η = 0, q₁ = (1 − h)/2) and C = (1, h) (η = 0, q₀ = (1 − h)/2).

**Minimax on the triangle.**
- Let G = E_P[κ φφᵀ]. V* is the squared circumradius of the triangle in the G metric if the triangle is acute; otherwise it is ¼ of the squared longest side.
- In the acute case the least-favourable prior has the circumcentre's barycentric weights, all positive: **three support points**.

**Limit ρ → 1.**
- Normalise by E_P[κ]. The s₀ term dominates κ, and s₀ is independent of (ã, δ).
- So G/E_P[κ] → E_P[φφᵀ] = [[g, −ht], [−ht, g]]. This uses the sign symmetry of the base law, with g = E_base[z₁²/(1 − h²z₀²z₁²)] and t = E_base[z₀²z₁²/(1 − h²z₀²z₁²)] < g.
- The vertex angle inner products are h(g − t), g + ht and (1 − h)(g + ht) at A, B and C: all positive.
- By continuity the triangle is acute for ρ close to 1. **Checked: correct.**

**Codex's two qualifications, kept.**
- Restrict to h > 0.
- With **η known** the set is the allocation segment {v = r² − r(u − h), u ∈ [h, r]}. Its affine dimension is 1, so V* = ¼ d² **exactly**, and the endpoint midpoint (between q₀ = 0 and q₁ = 0) is minimax. This is my addition.
- L = 1 for completeness:
  - with η known, μ = r(1 − 2β)z₀ with β ∈ [0, 1] (full simplex) gives the segment [−r z₀, r z₀], with V* = r² E_P[κ z₀²] and minimax rule b ≡ 0;
  - with η unknown, the segment is [−z₀, z₀].
  - Under a declared domain β ≤ ½ the segments halve. Admissibility depends on the channel domain, as Codex notes.

### I.4 L = 3: the four-function representation (checked) and the certificate

**Representation.**
- Let r = 1 − 2η′ and let g(f) = −1 only at f = (0, 1), the final-action footprint, with ĝ(C) = ¼ Σ_f g(f)χ_C(f).
- Let Q = B_{η′} ∗ (g F′) be the signed apparent law weighted by σ₂. Then Q̂(A) = r^{|A|} Σ_B ĝ(A △ B) F̂′(B), with F̂′(B) = P̂(B) r^{−|B|}.
- With the final step factor r:

  μ_r = [J_r(∅)z₂ + J_r({0})z₀z₁z₂ + J_r({1})z₁ + J_r({0,1})z₀] / [1 + P̂({0})z₀z₁ + P̂({1})z₁z₂ + P̂({0,1})z₀z₂],

  where J_r(A) = Σ_B ĝ(A △ B) P̂(B) r^{1+|A|−|B|}. This matches Codex.
- The denominator is fixed on Ξ(P). The coefficients are Laurent polynomials in r with exponents −1 to 3.

**Certificate on the evaluation measure.** Let ψ be the four functions over the denominator and M = E_P[κ ψψᵀ] (4 × 4). For a candidate centre with coefficient vector c, the risk is ρ_c(r) = (J(r) − c)ᵀ M (J(r) − c), a Laurent polynomial of degree −2 to 6. Then r³ ρ_c′(r) is a polynomial of degree ≤ 8. sup over r ∈ [r_hi, 1] of ρ_c is found at its real roots in the interval or at the endpoints.

Applications:
- **Endpoint midpoint** c = (J(r_hi) + J(1))/2. It is certified minimax on the evaluation measure iff sup_r ρ_c ≤ d²/4·(1 + tol).
- **Otherwise, a bracket.**
  - Lower bound: the Bayes value of any prior on the arc. Use the Frank–Wolfe prior on a fine grid; Jaggi 2013, §2 gives the duality gap.
  - Upper bound: the continuous sup risk of that prior's mixture rule, from the same polynomial.
  - Both are always valid; report the gap.
- **Separate error sources.** These are statements about the finite evaluation measure: 20,000 prefixes with all views. Population claims additionally need integration error (a prefix-level SE), parameter-discretisation error (removed by the continuous check), and optimisation error (the gap).

### I.5 Which conclusions change

1. **Q1 is rewritten around target-set geometry.**
   - The question is no longer "does the least-favourable prior sit on the two arc endpoints at L ≤ 3?"
   - It is now: "is the target set inside the diametral ball of its farthest pair?"
   - L = 2, η unknown: **no in general**. The triangle has a three-point least-favourable prior when it is acute in the G metric, as in secondary physics with h = 0.54.
   - L = 2, η known, and L = 1: **yes, exactly** (segments).
   - L = 3: certify per cell (spec II.3).
2. **The labels "V*" and "minimax floor" for the endpoint-pair quantity become "endpoint-pair lower bound ¼ d²".** The materiality conclusion survives: the lower bound exceeds the common reference margin in 5 of 6 corrupted cells and in both β = 0 cells.
3. **Prop. A gains its uniform-F exception and the η′ = ½ boundary stratum.** Prop. B is restated as an exact nesting statement, with "near-bottom in the saved runs".
4. **Clean calibration** collapses the ambiguity only at L ≥ 3. The β-cap result is reported as "bounds the decision cost", with the truth retained and the residual set stated.
5. **Crossed SE.** Replace with the ANOVA estimator, or label the old figure an approximation. The shared-prefix SE stays. The E4 conclusions are unchanged.
6. **The abstract and C25 wording.** "Quantified minimax floor" becomes "certified lower bound on the worst-case decision cost (two-law), with the minimax value certified numerically where stated".

### I.6 Specification for Sonnet (Part II)

**Ground rules.**
- No fitting, training or new data.
- Do not overwrite historical results; write new outputs under `results/decision_id/`.
- Keep pytest outputs out of committed artifacts.

**II.1 Portability.**
- Import the discussion unchanged as `docs/decision_identification_discussion.md`, with a one-line header noting its provenance (separate clone, 2026-10-11) and that it is superseded where this closeout differs. *(Done in Part III, after the user re-supplied the file.)*
- Port the archive to `experiments/decision_id/`:
  - `common.py` with repository-relative paths and no `/home/claude`;
  - keep seeds;
  - a test asserting that the numpy TestSet replica equals `mti.learned.evalset.TestSet` arrays (s_rec, a_rec, kap) for L = 3.
- Each script writes JSON with provenance: commit, dirty flag and script hash.

**II.2 L = 2 geometry** — `l2_geometry.py` → `results/decision_id/l2_geometry.json`.
- For both physics and the E4 truths at L = 2 with h > 0 (T0: h = 0.9; T3 and T6: 0.54; T4: 0.27), plus Codex's worlds A/B/C at h = 0.54:
  - G on the evaluation measure (L = 2 TestSet, master 8601, cell 0, views weighted by the truth);
  - the triangle vertices and vertex angle inner products;
  - V* (circumradius² or ¼ longest side²), the least-favourable weights, ¼ diam², the Jung bound ⅓ diam², and the worst risk of the farthest-pair midpoint;
  - the η-known segment V* = ¼ d²;
  - HMM agreement of φ·(u, v) (≤ 1e-12) and record-law equality across the three vertex worlds (≤ 1e-12);
  - normalisers: raw, E_P[κ] and reference VOI_clean.
- An optional cross-check of G by a 10⁶-record Monte Carlo.

**II.3 L = 3 certificate** — `l3_certify.py` → `results/decision_id/l3_certify.json`.
- For 2 physics × T0/T3/T4/T6:
  - the arc (with uniform-F detection: if F is uniform, report "non-compact; boundary stratum" and skip);
  - r_hi, and the Walsh coefficients P̂, ĝ;
  - an HMM agreement check of the four-function formula at ≥ 9 arc points (≤ 1e-10 max abs), plus record-law equality at the endpoints;
  - M (4 × 4) and the endpoint-pair lower bound d²/4 with its prefix SE;
  - the continuous worst case of the endpoint midpoint (roots of the degree-≤ 8 polynomial in [r_hi, 1], root residuals, endpoint values);
  - the certification flag;
  - Frank–Wolfe on a 401-point grid: iterations, final gap, termination reason, support with an explicit tolerance;
  - lower bound (FW Bayes value), upper bound (continuous sup of the FW rule), and the gap;
  - the Jung bound 0.4 diam².
- Report raw, E_P[κ] and reference VOI_clean, with the common reference margin labelled as such.
- Also report the worst risk of b_pw (descriptive only) and of the bottom-endpoint selection.

**II.4 One β-cap family** — `l3_beta_cap.py` → `results/decision_id/l3_beta_cap.json`.
- For L = 3 truths T3, T4 and T6 in both physics, and β̄ on a grid from β_true to the arc maximum:
  - the restricted set {η′ : β′(η′) ≤ β̄}, computed exactly (not assuming monotonicity: report the interval or intervals);
  - whether the truth is retained;
  - the lower bound ¼ d² of the restricted set, and its certified minimax bracket via II.3 on the sub-interval;
  - the smallest β̄ at which the upper bound falls below the reference margin, if any.

**II.5 SE reporting.**
- `experiments/decision_id/se_e4.py`. From the saved P1 units, compute:
  - the shared-prefix SE (fits fixed);
  - the delta-method SE (VOI estimated on the same prefixes);
  - the crossed ANOVA estimator (MS_R + MS_C − MS_E)/(RN), with its assumptions (independently sampled fits and prefixes; additive two-way model) printed in the JSON;
  - the cross-replicate correlation summary.
- Relabel the key in `check_e4_eval_se.py` as "independent-evaluation-set approximation (not a bound)". Leave the historical `eval_se_check.json` unchanged.

**II.6 Appendix scripts.** Port `kflip.py` and `nuisance.py` portably and rerun them. Outputs go to `results/decision_id/`.

**II.7 Tests** — `tests/test_decision_id.py`.
- the L = 2 belief formula against the HMM;
- three-world record-law equality;
- the triangle feasibility round trip (random (u, v) in 𝒯_h ↦ (r, q₀, q₁) valid);
- the L = 3 four-function formula against the HMM;
- the endpoint-ball criterion on a toy segment and a toy acute triangle;
- the crossed estimator's unbiasedness on a synthetic two-way design;
- replica = TestSet.

**II.8 conftest.** Write the test record to `results/oracle_v1/test_max_errors.json` only when the environment variable `MTI_WRITE_TEST_RECORD=1` is set; otherwise write to `.pytest_cache/`. Keep the existing ≥ 60-tests guard.

**II.9 Documentation corrections.** Apply the discussion's §9 corrections 1–13, as amended by I.1 and I.5, to:
- findings_summary (C25, C27, C28, scope line);
- research_synthesis (abstract, §2, §4.4 (iii)/(iv), §5.4, §6);
- consolidation_review (§3, §4, G1);
- literature_review §4.3;
- e4_report (§5 search domain, §9 item 2);
- research_status (HEAD `08dd58e`, Part II status).

Do **not** write Part III.

**Stop conditions (return to Opus).**
- Any HMM agreement or record-law check fails its tolerance.
- A polynomial root residual > 1e-8 (relative).
- A lower bound exceeds its upper bound.
- The replica differs from TestSet.
- A test fails.
- An L = 3 cell is non-certifiable with a gap > 1% of the bound (report it; do not tune).

### I.7 Is anything indispensable left after this?

No. Arguing against the recommendation, the only candidate gap is that the abstract's claim of decision-relevant ambiguity also covers L ≤ 2. That claim is settled exactly by I.3: the segments, and the triangle with an explicit V*. II.2 computes the numbers.

Out of scope for this thesis, as Codex says:
- a population theorem for the L = 3 midpoint;
- finite-sample confidence sets (non-regular at the boundary);
- plug-in physics;
- detectors;
- real data.

---

## Part II. Implementation results (Sonnet 5.5, 2026-10-12)

Base `08dd58e`; all outputs are in `results/decision_id/` (each JSON records commit, dirty-tree flag and script hash; the dirty flag is true because the closeout code is uncommitted — rerun after committing to refresh provenance). No model fitting, training or new data.

### II.A State found, deviations and items for Opus

1. **The "ported" scripts were the raw archive.** `experiments/decision_id/` contained the received files unchanged: `/home/claude/mti` in `e4common.py` and `se_shared.py`; scripts that compute at import time; JSON written to the working directory (which would have overwritten the received outputs). Ported per II.1: `did_common.py` (repository-relative paths, numpy-only TestSet replica, provenance); `ridgelib.py` (library only); received JSON preserved in `results/decision_id/archive_received_2026-10-11/`; the superseded received scripts `diam_se.py`, `ridge_l3_extra.py`, `se_shared.py` were moved unchanged to `archive_received_2026-10-11/scripts/` (not runnable there). I deleted `e4common.py` and `ridge_l3.py` from the working directory before noticing that the zip had also disappeared from `Downloads`; the user later re-extracted the original zip, and **all nine received scripts are now in the archive exactly as received** (see the archive `NOTE.txt`; an earlier in-session transcription of `e4common.py` was byte-identical to the recovered file). **The re-extraction briefly overwrote three of my ported files** (`ridgelib.py`, `kflip.py`, `nuisance.py` in `experiments/decision_id/`); I moved the raw files to the archive and restored the ported ones from the text I had written. Verification: `kflip.py` and `nuisance.py` are hash-identical to the runs recorded in `results/decision_id/*.json`; `ridgelib.py` has no recorded hash, so it was verified behaviourally: rerunning `l3_certify`, `l3_beta_cap`, `kflip` and `nuisance` reproduces the saved JSON exactly (max relative difference 0.0), and `tests/test_decision_id.py` passes.
2. **`decision_identification_discussion.md` and the zip are no longer in `Downloads`**, so the discussion note could **not** be imported unchanged (spec II.1) and was **not** reconstructed from memory. Documents cite it as "Opus discussion note of 2026-10-11, not in the repository". **Proposition A's proof, Table A and Proposition B's proof exist only there**; the numerical content is reproduced here (arc, η_hi by exact roots vs bisection, HMM agreement) but the written proofs are not in the repository. Needs the user to re-supply the file, or Opus to re-derive in Part III. *(Resolved: the user re-supplied the file; imported in Part III.)*
3. **Tolerance form corrected (not loosened).** My first bracket check used a purely relative tolerance on `lb ≤ ub`; in β-cap cells where the bound is ~8e-5 the Frank–Wolfe value (a difference of O(1) terms) exceeded the upper bound by 4.5e-16 absolute, so two T4 cells were flagged. The check now uses an absolute tolerance scaled by max diag(M_g) (1e-12). Re-run: all pass. The root-residual, HMM-agreement and log-likelihood tolerances are as specified (1e-8, 1e-10, 1e-12).
4. **Spec II.4 statistic is degenerate.** Restricted sets are nested, so the certified upper bound is nondecreasing in the cap and the "smallest cap with UB < margin" is trivially β_true (reported). The informative statistic is the **largest** cap with UB < margin (and an interpolated crossing), also reported.
5. **Frank–Wolfe.** Pairwise FW with exact line search replaces the received plain FW (20,000 iterations, final gap not saved, support read from rounded weights). It converges to a duality gap ≈ 1e-14 in ≈ 400 iterations; gap, iterations, termination reason and support tolerance are saved. The received archive's "gap 1.4e-6–5.0e-6" was non-convergence of the plain method.
6. **`check_e4_eval_se.py`**: the independent-set key was relabelled and its output redirected to `results/decision_id/eval_se_check_rerun.json`, so the committed `results/e4/stage1/eval_se_check.json` is never overwritten (not rerun; the historical file keeps its original key).
7. **conftest**: the test record is written to `.pytest_cache/test_max_errors.json` unless `MTI_WRITE_TEST_RECORD=1`. The committed `results/oracle_v1/test_max_errors.json` is therefore unchanged by this and the full-suite run (and does not contain the new tests' maxima).

### II.B L = 3: arc, lower bound, continuous certificate, Frank–Wolfe bracket (`l3_certify.py`)

Evaluation measure: the E4 evaluation prefixes (master 8601, cell 0, 20,000; all 4 views weighted by the reference truth; one prefix = one unit). Margin: the **common reference margin** of the reference truth, clip(0.1 V₂, 0.002 VOI, 0.02 VOI); not recomputed per compatible world. VOI_clean is the reference truth's; E_P[κ] (identified) is 3.972 (lead) and 5.857 (secondary).

| cell | arc η′ ∈ [0, η_hi] | ¼d²/VOI (SE) | ¼d² raw | ¼d²/E_P[κ] | margin/VOI | ¼d² / margin | farthest-from-truth/VOI | midpoint sup excess | FW gap (iters) | UB−LB (rel) | HMM agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lead_T0 | 0 … 0.05000 | 0.05028 (0.00263) | 0.1575 | 0.0397 | 0.0020 | 25.14 | 0.2011 | 0.0e+00 | 9.8e-15 (403) | 1.3e-13 | 6.4e-14 |
| lead_T3 | 0 … 0.08483 | 0.04647 (0.00041) | 0.1456 | 0.0366 | 0.0200 | 2.32 | 0.0717 | 2.2e-16 | 8.4e-15 (406) | 5.3e-14 | 3.9e-14 |
| lead_T4 | 0 … 0.05000 | 0.02436 (0.00078) | 0.0763 | 0.0192 | 0.0200 | 1.22 | 0.0975 | 8.9e-16 | 9.8e-15 (406) | 1.3e-13 | 4.6e-14 |
| lead_T6 | 0 … 0.07916 | 0.04362 (0.00045) | 0.1366 | 0.0344 | 0.0200 | 2.18 | 0.0755 | 4.4e-16 | 8.4e-15 (404) | 6.3e-14 | 3.8e-14 |
| sec_T0 | 0 … 0.05000 | 0.02663 (0.00174) | 0.1030 | 0.0176 | 0.0020 | 13.32 | 0.1065 | 2.2e-16 | 1.3e-14 (403) | 1.4e-13 | 5.3e-14 |
| sec_T3 | 0 … 0.08483 | 0.03234 (0.00039) | 0.1251 | 0.0214 | 0.0140 | 2.30 | 0.0491 | 4.4e-16 | 9.8e-15 (405) | 8.1e-14 | 3.6e-15 |
| sec_T4 | 0 … 0.05000 | 0.01365 (0.00041) | 0.0528 | 0.0090 | 0.0200 | 0.68 | 0.0546 | 1.1e-15 | 1.1e-14 (402) | 2.1e-13 | 7.5e-15 |
| sec_T6 | 0 … 0.07916 | 0.02946 (0.00040) | 0.1140 | 0.0195 | 0.0116 | 2.53 | 0.0505 | 4.4e-16 | 9.8e-15 (402) | 8.4e-14 | 2.6e-15 |

- **Checks passed in all 8 cells:** four-function belief vs HMM at 9 arc points (max ≤ 6.4e-14); record log-likelihood equality at both arc endpoints (≤ 5.3e-15); exact η_hi vs bisection (≤ 3e-17); polynomial root residuals ≤ 1e-8 (max 1.8e-15); dense-grid maximum never above the root-based supremum.
- **Reproduction against the received archive:** ¼d² raw differs by ≤ 6e-15 relative in every cell.
- **The endpoint midpoint is certified minimax on the evaluation measure in all 8 cells** (continuous worst case equals ¼d² to ≤ 1.1e-15 relative), and the Frank–Wolfe prior is the ½/½ endpoint prior (support 2, endpoint mass 1.0). The two-law lower bound therefore equals the minimax value on this measure. This is a numerical certificate on a 20,000-prefix measure, not a population theorem; no integration-error bound beyond the prefix-level SE is claimed.
- **Jung bound** on the minimax value: ≤ 0.4·diam² (affine dimension 4), i.e. at most 1.6 × the lower bound without using the certificate.
- **Descriptive, grid-based (not certificates):** the pointwise-envelope midpoint's worst risk is within 0.4% of ¼d² (lead T0) and ≤ 0.15% elsewhere; the bottom-endpoint selection's continuous worst case is the full diameter.
- **Margins:** ¼d² exceeds the common reference margin in 5 of 6 corrupted cells (sec T4: 0.68 of the margin) and in both β = 0 cells.

### II.C L = 2: triangle geometry (`l2_geometry.py`)

Same evaluation measure at L = 2 (identified h = r(1 − 2β): T0 0.90, T3 and T6 0.54, T4 0.27). V* is the squared circumradius of the triangle in the G metric when it is acute, else ¼ of the longest side squared.

| cell | h | kind | LFP weights (A, B, C) | angle cosines (A, B, C) | V* raw | ¼d² raw | V*/¼d² | farthest-pair midpoint worst | Jung ⅓d² | η-known segment V* | V* on fresh MC G | G rel. diff (MC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lead_T0 | 0.90 | circumcentre | 0.092, 0.495, 0.412 | 0.045, 0.969, 0.204 | 0.1564 | 0.1561 | 1.002 | 0.1629 | 0.2082 | 0.0000 | 0.1572 | 5.0e-03 |
| lead_T3 | 0.54 | circumcentre | 0.048, 0.489, 0.463 | 0.043, 0.866, 0.462 | 0.7781 | 0.7767 | 1.002 | 0.8354 | 1.0356 | 0.4298 | 0.7759 | 3.1e-03 |
| lead_T4 | 0.27 | circumcentre | 0.024, 0.491, 0.485 | 0.023, 0.789, 0.596 | 1.2741 | 1.2734 | 1.001 | 1.3318 | 1.6979 | 0.8576 | 1.2694 | 3.6e-03 |
| lead_T6 | 0.54 | circumcentre | 0.048, 0.489, 0.463 | 0.043, 0.866, 0.462 | 0.7781 | 0.7767 | 1.002 | 0.8354 | 1.0356 | 0.4298 | 0.7759 | 3.1e-03 |
| sec_T0 | 0.90 | circumcentre | 0.322, 0.483, 0.195 | 0.233, 0.931, 0.138 | 0.0824 | 0.0808 | 1.020 | 0.0971 | 0.1078 | 0.0000 | 0.0760 | 7.7e-02 |
| sec_T3 | 0.54 | circumcentre | 0.190, 0.455, 0.354 | 0.201, 0.817, 0.400 | 0.5916 | 0.5677 | 1.042 | 0.8187 | 0.7569 | 0.3145 | 0.5825 | 1.3e-02 |
| sec_T4 | 0.27 | circumcentre | 0.101, 0.463, 0.437 | 0.108, 0.758, 0.566 | 1.0824 | 1.0697 | 1.012 | 1.3210 | 1.4263 | 0.7212 | 1.0725 | 6.1e-03 |
| sec_T6 | 0.54 | circumcentre | 0.190, 0.455, 0.354 | 0.201, 0.817, 0.400 | 0.5916 | 0.5677 | 1.042 | 0.8187 | 0.7569 | 0.3145 | 0.5825 | 1.3e-02 |

- **Checks passed in all 8 cells:** belief formula φ·(u, v) vs the HMM at the truth and the three vertex worlds (max ≤ 4.7e-15); record-law equality of the three vertex worlds with the truth (≤ 3.6e-15); the truth lies in the triangle.
- **A three-point least-favourable prior holds in all 8 cells on both the evaluation measure and a fresh 10⁶-record sample** (master 1234). The secondary T3/T6 values (weights 0.196 / 0.454 / 0.350 on the fresh sample; V* 0.5826) agree with Codex's population quadrature (0.1964 / 0.4537 / 0.3499; 0.58288).
- **How close to the boundary:** in the lead cells the angle at A is nearly a right angle (cosine 0.02–0.05), so the A-weight is small (0.02–0.09) and V* exceeds ¼d² by only 0.05–0.2%; in the secondary cells the excess is 1.2–4.2%. Integration error in G (20,000 prefixes vs 10⁶ fresh) is up to 1.3% in all cells but 7.7% at secondary T0 (heavy-tailed κ), where V* changes from 0.0824 to 0.0760. The qualitative conclusion (three-point prior, no universal endpoint-midpoint formula) is unchanged under the fresh sample.
- **With η known** the compatible set is a segment: V* = ¼d² exactly, the endpoint midpoint is minimax. At β = 0 the segment collapses to a point (V* = 0).
- The farthest-pair midpoint can be far from minimax (0.82 vs 0.59 raw at secondary T3).

### II.D β-cap family on the L = 3 truths (`l3_beta_cap.py`)

25 caps from β_true to the largest β′ on the arc; restricted sets computed exactly as unions of intervals (monotone decreasing in η′ in all 6 cells); bracket via pairwise FW on the restricted grid and the continuous worst case of the FW mixture rule over the restricted set. Truth retained in every row.

| cell | β_true | β_max (arc) | UB/margin at cap = β_true | UB/margin at full arc | largest grid cap with UB < margin | interpolated crossing |
|---|---|---|---|---|---|---|
| lead_T3 | 0.200 | 0.270 | 0.337 | 2.32 | 0.2321 | 0.2347 |
| lead_T4 | 0.350 | 0.397 | 0.000 | 1.22 | 0.3929 | 0.3932 |
| lead_T6 | 0.200 | 0.271 | 0.257 | 2.18 | 0.2385 | 0.2396 |
| sec_T3 | 0.200 | 0.270 | 0.341 | 2.30 | 0.2321 | 0.2345 |
| sec_T4 | 0.350 | 0.397 | 0.000 | 0.68 | 0.3967 | none (full arc below margin) |
| sec_T6 | 0.200 | 0.271 | 0.303 | 2.53 | 0.2326 | 0.2341 |

- A cap bounds the worst-case decision cost (certified upper bound) but leaves a non-singleton residual set except at the single point cap = β_true on the boundary truth T4, where the residual set is the truth.
- For T3 and T6 (true β = 0.2) caps up to ≈ 0.23–0.24 keep the certified cost below the margin; secondary T4's full arc is already below the margin.

### II.E Standard errors (`se_e4.py`; P1 cells, L = 8, T0, n = 10⁴, R = 20, N = 20,000)

Saved regrets reproduce exactly (max relative error 0.0). Units: VOI_clean.

| contrast | physics | mean/VOI | replicate SE | shared-prefix SE (fits, VOI fixed) | delta-method | crossed ANOVA | crossed ANOVA (delta) | independent-set approx. | old crossed formula | min / mean cross-replicate corr |
|---|---|---|---|---|---|---|---|---|---|---|
| J − EX | lead | 0.00007 | 3.94e-05 | 3.95e-06 | 3.89e-06 | 3.95e-05 | 3.95e-05 | 2.45e-06 | 3.96e-05 | -1.000 / 0.232 |
| ETA⁻ − J | lead | 0.02831 | 7.38e-04 | 1.58e-03 | 1.55e-03 | 1.74e-03 | 1.72e-03 | 3.56e-04 | 1.74e-03 | 0.973 / 0.997 |
| J − EX | sec | 0.00011 | 5.15e-05 | 7.71e-06 | 7.66e-06 | 5.20e-05 | 5.20e-05 | 3.98e-06 | 5.21e-05 | -0.995 / 0.082 |
| ETA⁻ − J | sec | 0.02829 | 7.54e-04 | 1.39e-03 | 1.38e-03 | 1.58e-03 | 1.57e-03 | 3.16e-04 | 1.58e-03 | 0.918 / 0.987 |

- Shared-prefix SE and the delta-method SE reproduce Codex's and Opus's figures. The crossed ANOVA estimator (MS_R + MS_C − MS_E)/(RN) is unbiased under the stated two-way random-effects assumptions (checked on a synthetic design in `tests/test_decision_id.py`); the old (MS_R + MS_C)/(RN) figure double-counts σ_e²/(RN), which is negligible here (agrees to the 3rd digit).
- The independent-set approximation is 1.6–4.4 times smaller than the shared-prefix SE in these four contrasts; in general the sign of the difference is set by the cross-replicate covariances (individual J − EX correlations reach −0.9998), so it is no bound.
- Conclusions unchanged: ETA⁻ − J = 0.0283 is about 15 crossed SEs above Δ; J − EX remains within ±Δ.

### II.F Appendix scripts

- `kflip.py`: 0 mismatches over k ∈ {1, 2, 3}, L ∈ {2, …, 9} (injectivity iff L ≥ 2k + 1; joint identification iff L ≥ 2k + 2; LP agreement).
- `nuisance.py`: channel-invariant moments on 2×10⁵ records with β = 0.35: ρ̂ 0.9004 / 0.9002, q̂_a 1.0003, ĉ² 0.9998 / 0.9993, q̂_w 0.0201 (0.02) / 0.5002 (0.5); trusted-outcome slope of z on candidate beliefs at L = 3 (lead T3): 0.812, 1.000, 1.108 at η′ = 0, 0.05 (truth), η_hi.

### II.G Tests, conftest and documentation

- `tests/test_decision_id.py` (9 tests): replica = `TestSet`; L = 2 formula and vertex-world record laws; triangle round trip; L = 3 four-function vs HMM and η_hi roots vs bisection (incl. the uniform footprint case → non-compact); `sup_risk` vs dense grid and pairwise FW on a segment; endpoint-ball criterion and triangle minimax on toy cases; crossed-ANOVA unbiasedness; conftest guard.
- Full suite: 123 passed (114 + 9); `results/oracle_v1/test_max_errors.json` unchanged (record written to `.pytest_cache/`).
- Documentation corrections applied (spec II.9, as amended by I.1/I.5): findings_summary (scope line, L = 3 statement, C25 row), research_synthesis (abstract, §2 item 2, §4.4 (iii)/(iv) and the decision-identification definition, §5.4 L = 3 bullet, search domain, SE text, §5.5, §6), consolidation_review (header, §3 (iii)/(iv), C25 row, G1), literature_review (§4.3 correction block and excess-excursion sentence), e4_report (§3 note, §5 search domain, §9 item 2), e4_protocol (superseded note), research_status. Historical results and numbers are unchanged; corrections are marked as such.

### II.H Commands

```
python experiments/decision_id/l3_certify.py      # results/decision_id/l3_certify.json
python experiments/decision_id/l2_geometry.py     # results/decision_id/l2_geometry.json
python experiments/decision_id/l3_beta_cap.py     # results/decision_id/l3_beta_cap.json
python experiments/decision_id/se_e4.py           # results/decision_id/se_e4.json
python experiments/decision_id/kflip.py           # results/decision_id/kflip.json
python experiments/decision_id/nuisance.py        # results/decision_id/nuisance.json
python -m pytest -q                               # 123 passed; MTI_WRITE_TEST_RECORD=1 to refresh the committed record
```

### II.I Remaining risks

- Numerical certificates are on the 20,000-prefix evaluation measure; population statements need integration-error control (G differs by 0.3–7.7% between measures at L = 2).
- *(Resolved in Part III: the discussion note, with the Proposition A/B proofs, is now in `docs/`.)*
- Scripts' JSON provenance shows a dirty tree until the closeout is committed.
- The β-cap family covers only T3/T4/T6 at L = 3 and a cap on β; other side information remains analytical.

---

## Part III. Opus final review and thesis claim table (Opus 5.5, 2026-10-12)

### III.1 What I checked

**Inputs and provenance.**
- [decision_identification_discussion.md](decision_identification_discussion.md) is the original note: 39,477 bytes, 424 lines, body unchanged; a provenance header was added. References that said "not in the repository" now link to it.
- Sonnet's port and its restoration after the re-extraction:
  - `kflip.py` and `nuisance.py` are hash-identical to the runs recorded in the JSON;
  - `ridgelib.py` reproduces every dependent output exactly;
  - the archive holds all nine received scripts.

**Independent numerical spot checks** (methods that do not share Sonnet's code path):
1. **L = 3 endpoint-midpoint certificate from HMM beliefs** (`ExactRecords` on 201 arc points, not the four-function formula). In lead T0, secondary T3 and lead T6:
   - the midpoint's worst risk / (¼d²) − 1 is 0, 0 and 2e-16;
   - ¼d² matches Sonnet to 6 digits.
2. **L = 2 minimax by direct convex minimisation** of the maximum vertex risk (Nelder–Mead), not the circumcentre formula. Lead T3, secondary T3 and secondary T0: V* and the centre agree to 6 digits.
3. **One β-cap bracket from HMM beliefs** (lead T3, cap 0.2321, sub-arc [0.0287, 0.0848]): the midpoint's worst risk = ¼d² = 0.058422 = Sonnet's lower and upper bound, against Δ = 0.062648.
4. **Re-derived independently:** the L = 2 triangle (I.3) and the L = 3 four-function representation (I.4). The four-function representation agrees with the HMM to ≤ 6.4e-14.

**Mathematics re-checked.**
- *Proposition A (b):* the identity dF′/dt|₁ = −½ΔF follows from (ΔF)^(A) = −2|A|F̂(A), and the antipodal case gives F′ = (1 − t²)/4 < 0 for t > 1.
- *Connectedness of the arc* (not stated in the discussion; added here):
  - The maps T_t (multiply F̂(A) by t^{|A|}) form a semigroup, T_s T_t = T_{st}, and T_s preserves nonnegativity for s ∈ [0, 1] because it is the B-noise convolution.
  - So if F′ at t₂ > 1 is admissible, then F′ at any t ∈ [r, t₂] equals T_{t/t₂} applied to the admissible law at t₂, and is admissible too.
  - The admissible set is therefore exactly [r, t_hi]: Ξ(P) is an arc, and `eta_hi` (first sign change) is its endpoint.
  - In the uniform case t_hi = ∞, plus the η′ = ½ stratum.
- *Proposition B:* also holds at L ≤ 2, because every distribution on {0,1} or {∅} is a footprint law there.
- The endpoint-ball criterion and the Jung bound are as proved in I.2.

**Sonnet's deviations, accepted.**
- The absolute, scale-aware tolerance for "lower ≤ upper" is the correct form for a quantity computed as a difference of O(max d) terms.
- The β-cap "smallest cap" statistic was degenerate (nested sets); the largest cap is the informative one.
- Pairwise Frank–Wolfe replaces the plain version.
- Outputs are redirected so the committed artifacts are never overwritten.

**Documentation.** Sonnet's corrections are accurate. I changed three things:
- the ρ = 0 sentence in the definition of decision identification, which now says action identification needs equality only on {κ > 0};
- the links to the discussion note;
- a new closing section, research_synthesis §4.6.

### III.2 What was proved, computed, approximated, and changed

**Proved** (analytic; assumptions: S2, supplied physics or physics identified within S2, full single-flip simplex):
- the identification Proposition (i)–(v) (research_synthesis §4.4), with the qualifications to (iii) and (iv);
- Proposition A (the L = 3 identified set is an arc, with the uniform exception) and Proposition B (exact nesting of the profile at L ≤ 3);
- the L = 2 triangle and segment geometry, and the analytic ρ → 1 acuteness of the triangle;
- the universal bracket ¼diam² ≤ V* ≤ n/(2(n+1))·diam²;
- the endpoint-ball criterion;
- the trusted-outcome identity under its stated assumptions;
- the channel-invariant physics moments;
- the k-flip thresholds (structural sketch; LP-consistent for k ≤ 3, L ≤ 9).

**Computed on the evaluation measure** (20,000 E4 evaluation prefixes, master 8601, all views weighted by the reference truth):
- the L = 3 lower bounds, the continuous midpoint certificate and the Frank–Wolfe brackets (all 8 cells);
- the L = 2 metric G, the minimax value, the least-favourable weights and the segment values;
- the β-cap brackets.

These are exact computations or certificates for that finite measure.

**Approximated by population integration.**
- At L = 2, G was re-estimated on an independent 10⁶-record sample. It differs by 0.3–1.3% in seven cells and by 7.7% at secondary T0, where heavy-tailed κ dominates. The qualitative conclusion (an acute triangle, three-point prior) is unchanged.
- No population claim is made for the L = 3 minimax value beyond the evaluation measure. Prefix-level SEs (≤ 0.0026 VOI for ¼d²) quantify integration uncertainty of the lower bounds.

**Error sources kept separate.**
- Integration: prefix-level SE; for L = 2, the fresh-sample comparison.
- Parameter discretisation:
  - removed at L = 3 by the continuous degree-≤ 8 stationary-point check;
  - absent at L = 2, where the vertices are exact.
- Optimisation:
  - Frank–Wolfe duality gap ≈ 1e-14;
  - bracket gap ≤ 2.1e-13 relative;
  - polynomial root residuals ≤ 3e-15 relative.

**Claims that changed.**
1. "L = 3 ambiguity material only at β = 0": withdrawn. That held for the single comparison point η′ = 0.03. The two-law lower bound is material in 5 of 6 corrupted cells and in both β = 0 cells.
2. "V\*" / "minimax floor" for the endpoint-pair quantity is now "lower bound ¼d²". It equals the minimax value on the evaluation measure in the 8 tested L = 3 cells (certificate). It is not a universal formula: at L = 2 with η unknown the least-favourable prior has three points.
3. Q1 is restated around target-set geometry, and answered:
   - segments (L = 1; L = 2 with η known): exactly two-point;
   - L = 2 triangle: three-point;
   - L = 3 arcs: two-point, certified numerically.
4. Proposition (iii) gains the upper branch, the uniform exception and connectedness. Proposition (iv) holds only with η known or L ≥ 4.
5. The J-MLE reading at L = 3: an exact nesting result and a near-bottom selection in the saved runs, not "drift" or a "fitted gain".
6. Clean calibration collapses the ambiguity only at L ≥ 3. A β cap bounds the cost below the margin; it does not remove the ambiguity.
7. SE reporting: the shared-prefix SE is retained; the crossed SE is now the unbiased two-way ANOVA estimator; the independent-set figure is an approximation, not a bound. The search domain is stated as effective [0.001, 0.155]. None of these change any E4 conclusion.

**Evidence paths and commands.**
- Code: `experiments/decision_id/`. Outputs: `results/decision_id/*.json`.
- Commands: Part II.H.
- Received archive: `results/decision_id/archive_received_2026-10-11/`.
- Tests: `tests/test_decision_id.py`, and the full suite (123 passing; the committed test record is written only with `MTI_WRITE_TEST_RECORD=1`).

### III.3 Compact final thesis claim table

| # | claim (strongest justified wording) | status | evidence |
|---|---|---|---|
| 1 | Candidate and folded exact routes give the same aware belief; agreement does not certify the declared law; a learned route gap vanishes for a converged learner | proved + verified (≤ 1.5e-14) | synthesis §1, §3; C1, C26 |
| 2 | Naive excess = V₂ + I_loss; V₂ is prior-specific (from under 2% to 159% of VOI_clean); the recoverable per-record target is κ(μ − μ₂)² | proved / measured | synthesis §3, §5.1 |
| 3 | A wrong declared prior can be worse than ignoring corruption; understating the rate never hurts; no fixed prior is both Bayes-good and harmless on the tested uncertainty set | measured + partly proved | synthesis §5.2 (E2, E2b) |
| 4 | From unlabeled records with known physics: channel identified iff L ≥ 3 (η known); (η, q) jointly iff L ≥ 4; at η = ½ the decision is identified only with η known or L ≥ 4 | **proved** (structural corollary of known-support noisy population recovery) | synthesis §4.4 |
| 5 | Where identification fails, the compatible decision targets are a segment (L = 1; L = 2 with η known), a triangle (L = 2, η unknown) or an arc (L = 3, η unknown) | **proved** | synthesis §4.6; closeout I.3; discussion §3 |
| 6 | The worst-case decision cost has an exact two-law lower bound; at L = 3 it is 1.4–4.6% of VOI_clean at β = 0.2–0.35 (above the common reference margin in 5 of 6 cells) and 2.7–5.0% at β = 0; at L = 2 the minimax cost is 2.3–41% of VOI_clean (11.6–25 × the margin) | lower bound proved; values computed on the evaluation measure | closeout II.B–C; `l3_certify.json`, `l2_geometry.json` |
| 7 | No universal two-endpoint minimax formula: exactly two-point on segments; three-point least-favourable prior on the L = 2 triangle (8 of 8 cells; acute for ρ → 1 analytically); the L = 3 endpoint midpoint is minimax on the evaluation measure (certified, 8 of 8 cells) | proved (segments, ρ → 1 limit) / certified numerical | closeout I.2–I.4, II.B–C, III.1 |
| 8 | Likelihood adaptation is near-oracle with about 10⁴ records given a correct clean law; clean-law error at fixed persistence is absorbed as spurious corruption | measured | synthesis §5.3 (E3) |
| 9 | Joint persistence/channel estimation removes the within-family absorption at L ≥ 4 (equivalent to known η at L = 8, n = 10⁴); at L ≤ 3 its sample profile is exactly non-increasing in η and it selects near the bottom of the arc | measured (E4) + proved (Prop. B) | synthesis §5.4 |
| 10 | Extra clean prefixes (30 / 100 / 1,000) match joint estimation in 4 / 10 / 15 of 16 cells (more information, not an equal budget); clean calibration resolves the ambiguity only at L ≥ 3 | measured / proved | synthesis §5.5, §4.6 |
| 11 | A family violation (record-heterogeneous persistence) is absorbed persistently over the measured n; attribution is identified only relative to a clean family that cannot reproduce the corrupted law; lack of fit may still be detectable | measured (one mild instance) / proved (conditional) | synthesis §5.6, §6 |
| 12 | Side information as stated assumptions: a β cap bounds the L = 3 cost (caps ≈ 0.23–0.24 for true β = 0.2); trusted outcomes identify μ₂ under exogeneity and transfer; physics is identified within S2 by channel-invariant moments; k-flip thresholds L ≥ 2k + 1 / 2k + 2 | computed / proved corollaries | synthesis §4.6; closeout II.D, II.F |
| — | Harmful-record detector, poisoning defence, closed-loop or real-data benefit, unknown physics outside S2, population L = 3 minimax theorem, finite-sample identified-set estimation, family dose–response | **not established** | synthesis §7 |

### III.4 Verdict and stop

- **The stop condition is met.**
  - The identification claims are corrected.
  - The decision lower bounds are reproducible from committed-to-be scripts with saved outputs.
  - Every proposed decision rule carries an analytic or a numerical qualification.
- **No new theorem failed.** The universal two-endpoint statement was refuted at L = 2, as Codex showed, and the write-up uses the target-set geometry instead.
- **This closes the identification chapter.**
- **Next:** the thesis write-up from research_synthesis.md. None of the following is required for the bounded claim:
  - a population L = 3 minimax theorem;
  - an unknown-physics plug-in experiment;
  - a finite-sample identified-set estimator;
  - a family dose–response;
  - a learned detector;
  - real data.
- **Novelty stays proportional to the literature read.** The thresholds are a structural specialisation of known-support noisy population recovery. The decision-level characterisation and its measured costs are the contribution.
