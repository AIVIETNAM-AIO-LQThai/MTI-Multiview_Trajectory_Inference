"""Feasibility gate (protocol v2 decision 9): time every neural arm at N=1e5 on the lead physics and project the pilot's cost.

    python experiments/run_learned_gate.py --out results/learned_gate

One CPU thread per training process (small GRUs do not scale with threads); the pilot runs many processes in parallel.
Timings use 1 epoch of N=1e5 prefixes (390 steps at batch 256) and a 20,000-prefix test set (RB over L+1 views).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from mti.learned.data import MASTER_TRAIN, MASTER_VAL, make_probe_set
from mti.learned.evalset import TestSet, candidate_from_density, compose_F, query_density, query_F, query_full
from mti.learned.hmm_fit import fit_eta
from mti.learned.train import TrainCfg, train_A1, train_A2, train_A4, train_A6, train_A6p
from mti.params import ChannelSpec, Physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "learned_gate"))
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n_test", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    phys = Physics(rho=0.9, c=1.0, q_a=1.0, q_w=0.02, lam=0.1, eta=0.05, L=8)
    chan = ChannelSpec.uniform(8, 0.2)
    res = dict(threads=args.threads, n_train=args.n, n_test=args.n_test, torch=torch.__version__)
    tr = make_probe_set(phys, args.n, MASTER_TRAIN, 0, 0)
    va = make_probe_set(phys, args.n // 4, MASTER_VAL, 0, 0)
    cfg = TrainCfg(epochs=1, patience=1)
    steps_per_epoch = args.n // cfg.batch
    arms = {"A1": lambda: train_A1(tr, va, phys, cfg), "A2": lambda: train_A2(tr, va, phys, cfg), "A4": lambda: train_A4(tr, va, phys, cfg),
            "A6": lambda: train_A6(tr, va, phys, chan, cfg), "A6p-broad": lambda: train_A6p(tr, va, phys, "broad", cfg)}
    models, train_s = {}, {}
    for k, f in arms.items():
        t = time.time()
        m, info = f()
        models[k] = m
        train_s[k] = dict(seconds_per_epoch=time.time() - t, steps=info.steps, view_passes=info.view_passes, params=info.n_params)
        print(f"{k}: {train_s[k]['seconds_per_epoch']:.1f}s/epoch (1 thread), {info.steps} steps", flush=True)
    res["train_1_epoch"] = train_s
    t = time.time()
    eta, it, _ = fit_eta(tr.s.numpy().astype(float), tr.a.numpy().astype(float),
                         (tr.d_pr.numpy() + phys.rho * tr.s[:, -1].numpy()).astype(float), tr.a_pr.numpy().astype(float), phys)
    res["A5_em_seconds"] = time.time() - t
    res["A5_eta_hat_N1e5"] = eta
    t = time.time()
    ts = TestSet(phys, args.n_test, cell_id=0)
    ts.add_prior("P1", chan)
    res["testset_build_seconds_one_prior"] = time.time() - t
    q = {}
    t = time.time(); query_F(models["A2"], ts.s_rec, ts.a_rec, phys); q["A2_query_F"] = time.time() - t
    t = time.time(); query_density(models["A4"], ts.s_rec, ts.a_rec, phys); q["A4_query_density"] = time.time() - t
    t = time.time(); query_full(models["A1"], ts.s_rec, ts.a_rec, phys); q["A1_query_full"] = time.time() - t
    res["test_query_seconds_1thread"] = q
    json.dump(res, open(os.path.join(args.out, "gate.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
