"""Mutation audit: confirm that the test-suite fails when the implementation is deliberately wrong.

    python tests/mutation_audit.py            # runs every mutation in a subprocess, prints/writes a table

Each mutation edits the source of mti/inference.py (in memory, never on disk) before the tests import it.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTATIONS = {
    "M1_final_step_dropped (target m_{L-1} instead of m_L)": (
        "mu = (1.0 - 2.0 * eta) * np.tanh(lam / 2.0)", "mu = np.tanh(lam / 2.0)"),
    "M2_candidate_normaliser_drops_(1-beta)": (
        "log_none = np.log1p(-chan.beta)", "log_none = 0.0"),
    "M3_folded_recorded_sign_swapped": (
        "log_1mr = np.where(pos, log_pplus, log_pminus)\n    log_r = np.where(pos, log_pminus, log_pplus)",
        "log_1mr = np.where(pos, log_pminus, log_pplus)\n    log_r = np.where(pos, log_pplus, log_pminus)"),
    "M4_folded_normaliser_uses_(1-beta)_not_(1-2beta)": (
        "log_c0 = np.log(1.0 - 2.0 * chan.beta)", "log_c0 = np.log(1.0 - chan.beta)"),
    "M5_candidate_omits_one_location (L-1 candidates)": (
        "terms = np.concatenate([np.full((ll_S.shape[0], 1), log_none), _log_q(chan)[None, :] + log_ell], axis=1)",
        "_lq = _log_q(chan).copy(); _lq[-1] = -np.inf\n    terms = np.concatenate([np.full((ll_S.shape[0], 1), log_none), _lq[None, :] + log_ell], axis=1)"),
    "M6_folded_drops_magnitude (sign marginal uses |a|=0)": (
        "x = 2.0 * phys.c * np.abs(prefix.a) * delta / phys.q_w", "x = 0.0 * delta"),
}

CHILD = r"""
import importlib.util, os, sys, types
root, mut = sys.argv[1], int(sys.argv[2])
import mti  # package path
src = open(os.path.join(root, "src", "mti", "inference.py")).read()
old, new = __import__("json").loads(sys.argv[3])
assert old in src, "mutation target not found"
spec = importlib.util.spec_from_loader("mti.inference", loader=None)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "mti"
exec(compile(src.replace(old, new), "mutated_inference.py", "exec"), mod.__dict__)
sys.modules["mti.inference"] = mod
import pytest
os.environ["MTI_MUTATION"] = "1"
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", os.path.join(root, "tests"), "--ignore=" + os.path.join(root, "tests", "mutation_audit.py")]))
"""


def main():
    rows = []
    for i, (name, (old, new)) in enumerate(MUTATIONS.items()):
        p = subprocess.run([sys.executable, "-c", CHILD, ROOT, str(i), json.dumps([old, new])],
                           capture_output=True, text=True, cwd=ROOT)
        killed = p.returncode != 0
        fails = [l.split("::")[1].split(" ")[0] for l in p.stdout.splitlines() if l.startswith("FAILED")]
        groups = sorted({f.split("[")[0] for f in fails})
        rows.append(dict(mutation=name, killed=killed, n_failing_cases=len(fails), failing_tests=groups))
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}\n          {len(fails)} failing cases: {', '.join(groups)}")
    out = os.path.join(ROOT, "results", "oracle_v1", "mutation_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    return 0 if all(r["killed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
