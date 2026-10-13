"""Generate every numerical table of docs/decision_identification_chapter.md from saved JSON (no computation of new results, no hand-typed numbers).

    python experiments/decision_id/tables_chapter.py               # writes results/decision_id/chapter_tables.md and refreshes the chapter's table blocks
    python experiments/decision_id/tables_chapter.py --no-chapter  # only writes results/decision_id/chapter_tables.md

Each table states its source paths, units, normaliser and uncertainty. In the chapter, tables live between markers
<!-- TABLE:<name>:BEGIN --> and <!-- TABLE:<name>:END -->; only those blocks are rewritten.
Normalisers: raw one-step cost; E_P[kappa] (identified from the record law, since kappa depends only on the observed s_L); VOI_clean and the margin Delta
of the REFERENCE truth (reference-law scales, not identified from the corrupted record law).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from analyze_e4 import ci90, classify  # noqa: E402

RD = os.path.join(ROOT, "results", "decision_id")
CHAPTER = os.path.join(ROOT, "docs", "decision_identification_chapter.md")


def J(rel):
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


def last_commit(rel):
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out or "uncommitted"


def prov(rel, obj=None):
    p = (obj or {}).get("_provenance") or (obj or {}).get("provenance") or {}
    run = p.get("git_commit", "")[:7]
    dirty = p.get("working_tree_dirty_code")
    s = f"`{rel}` (file last committed in `{last_commit(rel)}`"
    if run:
        s += f"; produced at `{run}`" + ("" if dirty is None else (", dirty tree" if dirty else ", clean tree"))
    return s + ")"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def fmt(x, d=4):
    return f"{x:.{d}f}"


def sci(x, d=1):
    return f"{x:.{d}e}"


CELL = lambda k: k.replace("_L3_", " ").replace("_L2_", " ").replace("sec", "secondary")


def table_identification_checks():
    ji = J("results/derivation_checks/joint_identification.json")
    st = [J(os.path.relpath(f, ROOT)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "e4", "stage0", "units", "*.json")))]
    l3 = J("results/decision_id/l3_certify.json")
    l2 = J("results/decision_id/l2_geometry.json")
    i03 = ji["eta_grid"].index(0.03)
    rows = ["| check | result | what it supports | source |", "|---|---|---|---|"]
    r = ji["L3_ridge"]
    rows.append(f"| L = 3 ridge, record level (η = 0.05 → η′ = 0.03; β = 0.2 → β′ = {r['beta_ridge']:.3f}; q_w = 0.5) | max \\|Δ log p(S)\\| = {sci(r['max_abs_record_loglik_diff'])}; max \\|Δμ₂\\| = {r['max_abs_belief_diff']:.2f} | Prop. 3 (lower branch) and Prop. 6 (decision relevance) | {prov('results/derivation_checks/joint_identification.json')} |")
    for L in (3, 4, 5, 8):
        tvs = [ji["D"][f"L{L}_{t}"][i03] for t in ("clean", "uniform_b0.2", "lag0_b0.2")]
        rows.append(f"| pattern-level min TV over channels at η′ = 0.03 (η = 0.05), L = {L}, three truths | {', '.join(sci(v, 1) for v in tvs)} | {'non-identification' if L == 3 else 'separation'} at the pattern level (Lemma 1 makes pattern equality sufficient for record equality) | same |")
    for L in (3, 4, 8):
        ratios = [u["I_prof"] / u["I_eta"] for u in st if u["L"] == L]
        rows.append(f"| stage-0 profile Fisher information I_prof/I_η, L = {L} (2 physics × 4 truths; 2×10⁵ records) | {sci(min(ratios))} – {sci(max(ratios))} | {'flat ridge (local)' if L == 3 else 'local information about η'}; local, not part of any proof | `results/e4/stage0/units/*.json` (last committed in `{last_commit('results/e4/stage0/units')}`) |")
    cells = l3["cells"]
    eh = {k.split("_")[-1]: c["arc"]["eta_hi"] for k, c in cells.items() if k.startswith("lead")}
    rows.append(f"| L = 3 arc endpoints η_hi (exact roots), truths T0/T3/T4/T6 | {', '.join(f'{t} {v:.4f}' for t, v in eh.items())}; max \\|exact − bisection\\| = {sci(max(c['arc']['abs_diff'] for c in cells.values()), 0)} | Prop. 3 (upper branch iff full support) | {prov('results/decision_id/l3_certify.json', l3)} |")
    rows.append(f"| L = 3 record-law equality at both arc endpoints (8 cells) | max \\|Δ log p(S)\\| = {sci(max(max(c['checks']['record_loglik_diff_endpoints'].values()) for c in cells.values()))} | Prop. 3 | same |")
    rows.append(f"| L = 3 four-function belief vs HMM at 9 arc points (8 cells) | max \\|Δμ\\| = {sci(max(c['checks']['max_abs_hmm_vs_four_function'] for c in cells.values()))} | Prop. 10 (representation) | same |")
    c2 = l2["cells"]
    rows.append(f"| L = 2 belief formula φ·(u, v) vs HMM, truth and three vertex worlds (8 cells) | max \\|Δμ\\| = {sci(max(max(w['max_abs_formula_vs_hmm'] for w in c['checks'].values()) for c in c2.values()))}; max \\|Δ log p(S)\\| across worlds = {sci(max(max(w['max_abs_loglik_diff_vs_truth'] for w in c['checks'].values()) for c in c2.values()))} | Prop. 8 (triangle) | {prov('results/decision_id/l2_geometry.json', l2)} |")
    return "**Table 1. Numerical consistency checks of the identification results.** Evaluation records are the E4 evaluation prefixes (master 8601, cell 0, 20,000 prefixes, all views) unless stated. These checks are consistent with the propositions; none of them is part of a proof.\n\n" + "\n".join(rows) + "\n"


def table_l3():
    l3 = J("results/decision_id/l3_certify.json")
    rows = ["| cell | arc η′ ∈ [0, η_hi] | ¼d² raw | ¼d² / E_P[κ] | ¼d² / VOI_ref (prefix SE) | Δ / VOI_ref | ¼d² / Δ | farthest-from-truth / VOI_ref |", "|---|---|---|---|---|---|---|---|"]
    for k, c in l3["cells"].items():
        lb = c["lower_bound"]
        rows.append(f"| {CELL(k)} | 0 – {c['arc']['eta_hi']:.4f} | {fmt(lb['d2_over_4_raw'])} | {fmt(lb['d2_over_4_over_Ekappa'])} | {pct(lb['d2_over_4_over_voi_ref'])} ({pct(lb['se_prefix_over_voi_delta_method'], 3)}) | {pct(c['margin_ref_over_voi'])} | {lb['ratio_to_margin']:.2f} | {pct(c['farthest_from_truth']['over_voi_ref'])} |")
    rows2 = ["| cell | midpoint worst risk / (¼d²) − 1 | FW lower bound (raw) | FW upper bound (raw) | (UB − LB)/LB | FW duality gap (iterations) | support (tol 1e-6) | max root residual (rel.) | certified on this measure |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in l3["cells"].items():
        m, f = c["endpoint_midpoint"], c["frank_wolfe"]
        res = max(m["max_root_residual_rel"], f["upper_root_residual_rel"], c["farthest_from_truth"]["max_root_residual_rel"])
        rows2.append(f"| {CELL(k)} | {sci(m['excess_over_lower_bound_rel'])} | {f['lower_bound_raw']:.6f} | {f['upper_bound_raw']:.6f} | {sci(f['bracket_gap_rel'])} | {sci(f['fw_gap_raw'])} ({f['iterations']}) | {f['n_support']} points, endpoint mass {f['endpoint_mass']:.4f} | {sci(res)} | {'yes' if m['certified_minimax_on_evaluation_measure'] else 'no'} |")
    head = ("**Table 2a. L = 3, η unknown: decision cost of the identified arc** (lead and secondary physics; reference truths T0 β = 0, T3 β = 0.2, T4 β = 0.35, T6 β = 0.2). "
            "Units: one-step cost. Normalisers: E_P[κ] (identified) and the reference truth's VOI_clean (VOI_ref, a reference-law scale). Δ is the common reference margin of the reference truth, clip(0.1 V₂, 0.002 VOI_ref, 0.02 VOI_ref), "
            "not recomputed per compatible world. ¼d² is the endpoint-pair LOWER bound on the worst-case cost of any belief rule. Uncertainty: prefix-level SE over the 20,000 evaluation prefixes (delta method with VOI_ref estimated on the same prefixes). "
            f"Source: {prov('results/decision_id/l3_certify.json', l3)}, script `experiments/decision_id/l3_certify.py`.\n\n")
    head2 = ("\n**Table 2b. L = 3: tolerance-qualified numerical certificates on the fixed evaluation measure.** The midpoint column is the continuous worst case of the endpoint midpoint over the arc "
             "(floating-point roots of a degree-≤ 8 polynomial, plus endpoints and a 2,001-point grid cross-check), relative to ¼d²; 'certified' means it is ≤ 1e-9. "
             "FW: pairwise Frank–Wolfe on a 401-point arc grid; lower bound = Bayes value of the FW prior (weak duality), upper bound = continuous worst case of the FW mixture rule. "
             "These are statements about this finite measure in floating-point arithmetic, not exact-arithmetic or population certificates. Same source.\n\n")
    return head + "\n".join(rows) + "\n" + head2 + "\n".join(rows2) + "\n"


def table_l2():
    l2 = J("results/decision_id/l2_geometry.json")
    rows = ["| cell | h | minimax structure | least-favourable weights A / B / C | min angle cosine | V* raw | V* / E_P[κ] | V* / VOI_ref | V* / (¼d²) − 1 | Jung ⅓diam² raw | V* / Δ | η known: segment V* / VOI_ref | V* on fresh 10⁶ sample (raw) | G rel. diff. (fresh vs eval.) |", "|" + "---|" * 14]
    for k, c in l2["cells"].items():
        mm, n = c["minimax"], c["normalised"]
        w = mm["least_favourable_weights"]
        rows.append(f"| {CELL(k)} | {c['h']:.2f} | {mm['kind']} | {w['A']:.3f} / {w['B']:.3f} / {w['C']:.3f} | {min(c['angle_cosines'].values()):.3f} | {mm['V_star_raw']:.4f} | {n['V_star_over_Ekappa']:.4f} | {pct(n['V_star_over_voi_ref'])} | {pct(c['endpoint_pair_lower_bound']['V_over_lower'] - 1)} | "
                    f"{c['jung_upper_bound_raw']:.4f} | {n['V_star_over_margin']:.1f} | {pct(n['segment_V_over_voi_ref'])} | {c['minimax_on_fresh_mc_G']['V_star_raw']:.4f} | {pct(c['G_rel_diff_mc'], 1)} |")
    head = ("**Table 3. L = 2: the compatible target triangle (η unknown, h ∈ (0, 1)) and segment (η known).** Same normalisers and evaluation measure as Table 2a, at L = 2 (h = 0.90 for T0, 0.54 for T3/T6, 0.27 for T4). "
            "V* = minimax value of the triangle in the G = E_P[κφφᵀ] metric (circumradius² when acute, else ¼ longest side²), computed exactly for this measure; 'circumcentre' means a three-point least-favourable prior. "
            "The fresh-sample columns re-estimate G on 10⁶ independent records (master 1234) as an integration-error check. "
            f"Source: {prov('results/decision_id/l2_geometry.json', l2)}, script `experiments/decision_id/l2_geometry.py`.\n\n")
    return head + "\n".join(rows) + "\n"


def table_cap():
    bc = J("results/decision_id/l3_beta_cap.json")
    rows = ["| cell | β_true | largest β′ on the arc | certified UB / Δ, full arc | largest grid cap with UB < Δ | interpolated crossing | residual set at cap = β_true |", "|---|---|---|---|---|---|---|"]
    for k, c in bc["cells"].items():
        r0, r1 = c["rows"][0], c["rows"][-1]
        cr = c["interpolated_cap_where_upper_bound_crosses_margin"]
        rows.append(f"| {CELL(k)} | {c['beta_true']:.3f} | {c['beta_max_on_arc']:.3f} | {r1['upper_over_margin']:.2f} | {c['largest_grid_cap_with_certified_upper_below_margin']:.4f} | {('%.4f' % cr) if cr else 'none: full arc already below Δ'} | "
                    f"{'single point (the truth)' if r0['residual_set_is_point'] else 'interval η′ ∈ [' + ', '.join(f'{x:.4f}' for x in r0['intervals_eta'][0]) + ']'} |")
    head = ("**Table 4. L = 3: one declared side-information family, a cap β′ ≤ β̄.** 25 caps from β_true to the largest β′ on the arc; restricted sets computed exactly; certified upper bound (UB) on the restricted minimax value from the continuous worst case of a pairwise Frank–Wolfe mixture rule. "
            "A cap bounds the worst-case cost; it identifies the decision only where the residual set is a point. Same evaluation measure and margin as Table 2a. "
            f"Source: {prov('results/decision_id/l3_beta_cap.json', bc)}, script `experiments/decision_id/l3_beta_cap.py`.\n\n")
    return head + "\n".join(rows) + "\n"


def e4_units():
    U = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "e4", "stage1", "units", "*.json")))]
    cells = defaultdict(list)
    for u in U:
        if u["kind"] == "main":
            cells[(u["pname"], u["L"], u["truth"])].append(u)
    return cells, U[0]["provenance"]


def table_e4():
    cells, pv = e4_units()
    se = J("results/decision_id/se_e4.json")
    n = 10000
    rows = ["| physics | L | truth | R | Δ / VOI_ref | EX-MLE | J-MLE | η fixed at 0.03 | J − EX [90% CI] | (η = 0.03) − J [90% CI] | J vs EX (margin Δ) |", "|" + "---|" * 11]
    for pn in ("lead", "sec"):
        for L in (8, 4):
            for tn in ("T0", "T3"):
                us = cells[(pn, L, tn)]
                voi, dl = us[0]["voi"], us[0]["delta"]
                g = lambda a: np.array([u["arms"][a][str(n)]["regret"] for u in us])
                d1, d2 = g("J-MLE") - g("EX-MLE"), g("ETA--MLE") - g("J-MLE")
                f = lambda d: (lambda m, lo, hi: f"{m / voi:+.5f} [{lo / voi:+.5f}, {hi / voi:+.5f}]")(*ci90(d))
                c1 = {"a worse": "J worse", "a better": "J better"}.get(classify(d1, dl), classify(d1, dl))
                rows.append(f"| {'secondary' if pn == 'sec' else pn} | {L} | {tn} | {len(us)} | {dl / voi:.4f} | {g('EX-MLE').mean() / voi:.5f} | {g('J-MLE').mean() / voi:.5f} | {g('ETA--MLE').mean() / voi:.4f} | {f(d1)} | {f(d2)} | {c1} |")
    srow = ["| physics | contrast | shared-prefix SE (fits, VOI fixed) | crossed ANOVA SE (replicates + prefixes) | replicate SE |", "|---|---|---|---|---|"]
    for pn in ("lead", "sec"):
        for c, nm in (("J_minus_EX", "J − EX"), ("ETAm_minus_J", "(η = 0.03) − J")):
            v = se[pn][c]
            srow.append(f"| {'secondary' if pn == 'sec' else pn} | {nm} | {sci(v['se_shared_prefix_fixed_fits_fixed_voi'], 2)} | {sci(v['se_crossed_anova_fixed_voi'], 2)} | {sci(v['se_rep_over_voi'], 2)} |")
    head = ("**Table 5. Supporting empirical context (E4): where the parameters are identified, joint estimation is feasible at the tested sample size.** n = 10⁴ corrupted prefixes, mean regret relative to the true-law oracle (regret 0) in units of VOI_ref; "
            "90% paired t-intervals over R replicates, conditional on the fixed evaluation set (fit uncertainty only). This is one design point per cell, not a finite-sample guarantee. "
            f"Source: `results/e4/stage1/units/*.json` (produced at `{pv['git_commit'][:7]}` + the A1 patch; last committed in `{last_commit('results/e4/stage1/units')}`), script `experiments/run_e4.py`; intervals recomputed here with `experiments/analyze_e4.py` helpers.\n\n")
    head2 = ("\n**Table 5b. Uncertainty accounting for the L = 8, T0 contrasts** (units of VOI_ref). Crossed ANOVA = unbiased (MS_R + MS_C − MS_E)/(RN) under independently sampled fits and prefixes. "
             f"Source: {prov('results/decision_id/se_e4.json', se)}, script `experiments/decision_id/se_e4.py`.\n\n")
    return head + "\n".join(rows) + "\n" + head2 + "\n".join(srow) + "\n"


def table_propb():
    cells, pv = e4_units()
    n = 10000
    rows = ["| physics | truth | R | median η̂ | range of η̂ | share at the 0.001 search bound | mean regret / VOI_ref |", "|---|---|---|---|---|---|---|"]
    for pn in ("lead", "sec"):
        for tn in ("T0", "T3", "T4", "T6"):
            us = cells[(pn, 3, tn)]
            e = np.array([u["arms"]["J-MLE"][str(n)]["eta"] for u in us])
            reg = np.array([u["arms"]["J-MLE"][str(n)]["regret"] for u in us]) / us[0]["voi"]
            rows.append(f"| {'secondary' if pn == 'sec' else pn} | {tn} | {len(us)} | {np.median(e):.4f} | {e.min():.4f} – {e.max():.4f} | {np.mean(e <= 0.0011):.2f} | {reg.mean():.3f} |")
    head = ("**Table 6. Observed selections of the E4 J-MLE implementation at L = 3** (n = 10⁴; true η = 0.05). These describe one implementation (grid 0.005–0.15, refinement ±0.005 clipped to [0.001, 0.499], EM tolerance and tie rules); "
            "they are not consequences of Proposition 11 alone, which only guarantees that the lower end of the search domain is a maximiser of the sample profile. "
            f"Source: `results/e4/stage1/units/*_L3_*.json` (produced at `{pv['git_commit'][:7]}` + the A1 patch).\n\n")
    return head + "\n".join(rows) + "\n"


def table_corollaries():
    nu = J("results/decision_id/nuisance.json")
    kf = J("results/decision_id/kflip.json")
    pm = nu["physics_moments"]
    rows = ["| corollary | quantity | value (truth) | source |", "|---|---|---|---|"]
    for pn in ("lead", "sec"):
        p = pm[pn]
        rows.append(f"| physics identified within S2 by channel-invariant moments ({'secondary' if pn == 'sec' else pn}; 2×10⁵ records, β = 0.35 on the last action) | ρ̂, q̂_a, ĉ², q̂_w | {p['rho']:.4f} ({p['rho_true']}), {p['q_a']:.4f} ({p['q_a_true']}), {p['c2']:.4f} ({p['c2_true']}), {p['q_w']:.4f} ({p['q_w_true']}) | {prov('results/decision_id/nuisance.json', nu)} |")
    to = nu["trusted_outcome"]["arc_points"]
    rows.append(f"| trusted outcomes select the decision-equivalence class (L = 3, lead T3, 4×10⁵ records) | slope of u(y − ρs_L)/q_u on candidate μ₂ at η′ = 0, 0.05 (truth), η_hi | {', '.join(f'{r['slope_z_on_mu2']:.3f}' for r in to)} (1 at the truth) | same |")
    rows.append(f"| at most k flips: thresholds L ≥ 2k + 1 (η known), L ≥ 2k + 2 (joint) | LP consistency, k ≤ 3, L = 2–9 | {kf['mismatches']} mismatches in {len(kf['rows'])} cases | {prov('results/decision_id/kflip.json', kf)} |")
    return "**Table 7. Structural corollaries: numerical consistency.** Moment estimates are single-sample values (no fitting of models). Scripts `experiments/decision_id/nuisance.py`, `experiments/decision_id/kflip.py`.\n\n" + "\n".join(rows) + "\n"


def headline():
    l3 = J("results/decision_id/l3_certify.json")["cells"]
    l2 = J("results/decision_id/l2_geometry.json")["cells"]
    bc = J("results/decision_id/l3_beta_cap.json")["cells"]
    corr = [c for c in l3.values() if c["w_truth"][0] < 1]
    clean = [c for c in l3.values() if c["w_truth"][0] == 1]
    lbv = lambda cs: [c["lower_bound"]["d2_over_4_over_voi_ref"] for c in cs]
    above = sum(c["lower_bound"]["exceeds_common_reference_margin"] for c in corr)
    l2c = [c for c in l2.values() if c["beta"] > 0]
    crosses = [c["interpolated_cap_where_upper_bound_crosses_margin"] for c in bc.values() if c["truth"] in ("T3", "T6")]
    rows = ["| headline quantity | value | source |", "|---|---|---|",
            f"| L = 3 endpoint-pair lower bound / VOI_ref, corrupted truths (β = 0.2–0.35; 6 cells) | {pct(min(lbv(corr)), 1)} – {pct(max(lbv(corr)), 1)}; above Δ in {above} of {len(corr)} | Table 2a |",
            f"| L = 3 endpoint-pair lower bound / VOI_ref, β = 0 (2 cells) | {pct(min(lbv(clean)), 1)} – {pct(max(lbv(clean)), 1)}; above Δ in {sum(c['lower_bound']['exceeds_common_reference_margin'] for c in clean)} of {len(clean)} | Table 2a |",
            f"| L = 3 endpoint midpoint: max relative excess of its continuous worst case over ¼d² (8 cells) | {sci(max(c['endpoint_midpoint']['excess_over_lower_bound_rel'] for c in l3.values()))} | Table 2b |",
            f"| L = 3 FW bracket (UB − LB)/LB, max over 8 cells | {sci(max(c['frank_wolfe']['bracket_gap_rel'] for c in l3.values()))} | Table 2b |",
            f"| L = 2 (η unknown) cells with a three-point least-favourable prior | {sum(c['minimax']['kind'] == 'circumcentre' for c in l2.values())} of {len(l2)} (fresh sample: {sum(c['minimax_on_fresh_mc_G']['kind'] == 'circumcentre' for c in l2.values())} of {len(l2)}) | Table 3 |",
            f"| L = 2 (η unknown) V* / VOI_ref, β > 0 / β = 0 | {pct(min(c['normalised']['V_star_over_voi_ref'] for c in l2c), 1)} – {pct(max(c['normalised']['V_star_over_voi_ref'] for c in l2c), 1)} / {', '.join(pct(c['normalised']['V_star_over_voi_ref'], 1) for c in l2.values() if c['beta'] == 0)} | Table 3 |",
            f"| L = 2 (η unknown) V* / Δ | {min(c['normalised']['V_star_over_margin'] for c in l2.values()):.1f} – {max(c['normalised']['V_star_over_margin'] for c in l2.values()):.1f} | Table 3 |",
            f"| L = 2 (η unknown) V* / (¼d²) − 1 | {pct(min(c['endpoint_pair_lower_bound']['V_over_lower'] for c in l2.values()) - 1)} – {pct(max(c['endpoint_pair_lower_bound']['V_over_lower'] for c in l2.values()) - 1)} | Table 3 |",
            f"| L = 2 (η known) segment V* / VOI_ref, β > 0 | {pct(min(c['normalised']['segment_V_over_voi_ref'] for c in l2c), 1)} – {pct(max(c['normalised']['segment_V_over_voi_ref'] for c in l2c), 1)} | Table 3 |",
            f"| L = 2 integration check: G relative difference, evaluation vs fresh 10⁶ sample | {pct(min(c['G_rel_diff_mc'] for c in l2.values()), 1)} – {pct(max(c['G_rel_diff_mc'] for c in l2.values()), 1)} | Table 3 |",
            f"| β cap at which the certified UB crosses Δ, T3/T6 (true β = 0.2) | {min(crosses):.3f} – {max(crosses):.3f} | Table 4 |"]
    return "**Table 0. Headline numbers quoted in the text** (each is a min–max or count over the cells of the referenced table).\n\n" + "\n".join(rows) + "\n"


BLOCKS = [("headline", headline), ("identification_checks", table_identification_checks), ("l3", table_l3), ("l2", table_l2), ("cap", table_cap),
          ("e4", table_e4), ("propb", table_propb), ("corollaries", table_corollaries)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chapter", action="store_true")
    args = ap.parse_args()
    out = {name: fn() for name, fn in BLOCKS}
    md = "# Tables for docs/decision_identification_chapter.md\n\nGenerated by `experiments/decision_id/tables_chapter.py` from saved JSON only.\n\n" + "\n".join(out[n] + "\n" for n, _ in BLOCKS)
    open(os.path.join(RD, "chapter_tables.md"), "w", encoding="utf-8").write(md)
    print("wrote", os.path.join(RD, "chapter_tables.md"))
    if not args.no_chapter and os.path.exists(CHAPTER):
        t = open(CHAPTER, encoding="utf-8").read()
        for name, _ in BLOCKS:
            pat = re.compile(rf"(<!-- TABLE:{name}:BEGIN -->\n)(?:.*?\n)?(<!-- TABLE:{name}:END -->)", re.S)
            if not pat.search(t):
                raise SystemExit(f"marker for table '{name}' not found in the chapter")
            t = pat.sub(lambda m: m.group(1) + out[name] + m.group(2), t)
        open(CHAPTER, "w", encoding="utf-8").write(t)
        print("refreshed table blocks in", CHAPTER)


if __name__ == "__main__":
    main()
