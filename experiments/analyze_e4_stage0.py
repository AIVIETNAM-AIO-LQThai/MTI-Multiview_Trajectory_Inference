"""Summarise E4 stage 0 and evaluate its stop rules (protocol section 3). python experiments/analyze_e4_stage0.py"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "results", "e4", "stage0", "units", "*.json"))):
    r = json.load(open(f))
    c = {x["eta"]: x for x in r["curve"]}
    rows.append(r | dict(c03=c[0.03], c08=c[0.08]))
out = ["| physics | L | truth | I_prof/I_eta | SE_pred(eta) n=1e4 | TV(0.03) | KL(0.03) | n_LR(0.03) | beta*(0.03) | regret*(0.03)/VOI | regret*(0.08)/VOI | KL(0.08) |", "|" + "---|" * 12]
for r in rows:
    f = lambda x, p=3: "—" if x is None else f"{x:.{p}g}"
    out.append(f"| {r['physics']} | {r['L']} | {r['truth']} | {r['I_prof'] / r['I_eta']:.2e} | {f(r['se_pred_n1e4'])} | {r['c03']['pattern_tv']:.2e} | {r['c03']['kl']:.2e} | "
               f"{f(r['n_lr_reject_eta03'])} | {r['c03']['beta_star']:.3f} | {r['c03']['regret'] / r['voi']:.4f} | {r['c08']['regret'] / r['voi']:.4f} | {r['c08']['kl']:.2e} |")
print("\n".join(out))
stops = []
for r in rows:
    if r["L"] == 3 and r["truth"] == "T3":
        ridge_kl = max(abs(x["kl"]) for x in r["curve"] if x["eta"] < 0.05)
        if ridge_kl > 1e-6 or r["I_prof"] > 1e-3 * r["I_eta"]:
            stops.append(f"L=3 T3 {r['physics']}: ridge KL {ridge_kl:.2e}, I_prof/I_eta {r['I_prof'] / r['I_eta']:.2e}")
    if r["L"] == 8 and r["c03"]["kl"] < 1e-6:
        stops.append(f"L=8 {r['physics']} {r['truth']}: KL(0.03) {r['c03']['kl']:.2e}")
print("STOP RULES FIRED:" if stops else "no stop rule fired", *stops, sep="\n")
pred = [(r["physics"], r["L"], r["truth"], r["se_pred_n1e4"]) for r in rows if r["L"] == 8 and r["se_pred_n1e4"] and r["se_pred_n1e4"] > 0.01]
print("stage-0 prediction: H-E4 predicted to FAIL at n=1e4 (SE > 0.01) for:", pred if pred else "none (L=8 SE <= 0.01 in every cell)")
open(os.path.join(ROOT, "results", "e4", "stage0", "summary.md"), "w").write("\n".join(out) + "\n")
