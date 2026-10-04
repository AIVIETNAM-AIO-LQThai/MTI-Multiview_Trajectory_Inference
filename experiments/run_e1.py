"""E1 (protocol v2, section E1): channel-trained density-direct arms A6d (fixed prior P1) and A6pd-broad (prior-conditioned).

    python experiments/run_e1.py --stage all --out results/e1

Reuses the pilot's training prefixes (same masters, cell ids, seeds), test sets, priors and grid, so every comparison with the pilot's arms
(A4, A2-G, A6p-broad, A6, ...) is paired by seed. Existing arms are not retrained; analyze_e1.py reads them from results/learned_pilot/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import torch

from run_learned import PHYSICS, _init_worker, channel_priors, grid, jsonable, priors, provenance
from mti.learned.data import MASTER_TRAIN, MASTER_VAL, make_probe_set
from mti.learned.evalset import TestSet, query_density_belief
from mti.learned.train import TrainCfg, train_A6d, train_A6pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1_ARMS = ["A6d", "A6pd-broad"]


def train_arm(arm, tr, va, phys, cfg):
    if arm == "A6d":
        return train_A6d(tr, va, phys, priors(phys.L)["P1"], cfg)
    return train_A6pd(tr, va, phys, "broad", cfg)


def tune_task(args):
    phys_idx, N, arm, ci, cfg, smoke = args
    phys = PHYSICS[phys_idx]
    tr = make_probe_set(phys, N, MASTER_TRAIN, phys_idx, 0)
    va = make_probe_set(phys, N // 4, MASTER_VAL, phys_idx, 0)
    t = time.time()
    _, info = train_arm(arm, tr, va, phys, cfg)
    return dict(phys_idx=phys_idx, N=N, arm=arm, cfg_idx=ci, cfg=asdict(cfg), val=info.best_val, epochs_run=info.epochs_run, seconds=time.time() - t)


def run_unit(args):
    phys_idx, N, seed, cfgs, n_test = args
    phys = PHYSICS[phys_idx]
    L = phys.L
    t0 = time.time()
    pri = priors(L)
    ts = TestSet(phys, n_test, cell_id=phys_idx)
    for k, ch in pri.items():
        ts.add_prior(k, ch)
    n, V = ts.n, ts.V
    tr = make_probe_set(phys, N, MASTER_TRAIN, phys_idx, seed)
    va = make_probe_set(phys, N // 4, MASTER_VAL, phys_idx, seed)
    rows, train = [], {}
    for arm in E1_ARMS:
        model, info = train_arm(arm, tr, va, phys, TrainCfg(**cfgs[arm], seed=seed))
        train[arm] = dict(best_val=info.best_val, epochs_run=info.epochs_run, steps=info.steps, view_passes=info.view_passes,
                          seconds=info.seconds, params=info.n_params)
        for k in channel_priors(phys_idx):
            if arm == "A6d":
                mu = query_density_belief(model, ts.s_rec, ts.a_rec, phys).reshape(n, V)
            else:
                q = torch.as_tensor(np.tile(pri[k].q.astype(np.float32), (n * V, 1)))
                mu = query_density_belief(model, ts.s_rec, ts.a_rec, phys, cond=q).reshape(n, V)
            rows.append(dict(arm=arm, prior=k, **ts.summarize(k, mu)))
    return dict(phys_idx=phys_idx, q_w=phys.q_w, N=N, seed=seed, n_test=n_test, cfgs=cfgs, train=train, rows=rows, unit_seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e1"))
    ap.add_argument("--stage", choices=["tune", "main", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_test", type=int, default=20_000)
    args = ap.parse_args()
    Ns = [1_000, 100_000] if not args.smoke else [800, 2_000]
    n_test = args.n_test if not args.smoke else 1_500
    os.makedirs(os.path.join(args.out, "units"), exist_ok=True)
    prov = provenance()
    t0 = time.time()
    tune_path = os.path.join(args.out, "tuning.json")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        if args.stage in ("tune", "all"):
            tasks = [(p, N, arm, ci, cfg, args.smoke) for p in (0, 1) for N in Ns for arm in E1_ARMS for ci, cfg in enumerate(grid(N, args.smoke))]
            print(f"tuning: {len(tasks)} trainings", flush=True)
            res = list(pool.map(tune_task, tasks, chunksize=1))
            sel = {}
            for r in res:
                key = f"{r['phys_idx']}|{r['N']}|{r['arm']}"
                if key not in sel or r["val"] < sel[key]["val"]:
                    sel[key] = r
            json.dump(jsonable(dict(provenance=prov, results=res, selected=sel, seconds=time.time() - t0)), open(tune_path, "w"), indent=1)
            print(f"[{time.time() - t0:.0f}s] tuning done", flush=True)
        if args.stage in ("main", "all"):
            sel = json.load(open(tune_path))["selected"]
            tasks = []
            for p in (0, 1):
                for N in Ns:
                    cfgs = {arm: {k: v for k, v in sel[f"{p}|{N}|{arm}"]["cfg"].items() if k != "seed"} for arm in E1_ARMS}
                    tasks += [(p, N, s, cfgs, n_test) for s in args.seeds]
            tasks.sort(key=lambda t: -t[1])
            print(f"main: {len(tasks)} units", flush=True)
            futs = [pool.submit(run_unit, t) for t in tasks]
            for t, f in zip(tasks, futs):
                r = f.result()
                r["provenance"] = prov
                json.dump(jsonable(r), open(os.path.join(args.out, "units", f"phys{t[0]}_N{t[1]}_s{t[2]}.json"), "w"))
                print(f"[{time.time() - t0:.0f}s] unit phys{t[0]} N{t[1]} seed{t[2]} done ({r['unit_seconds']:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
