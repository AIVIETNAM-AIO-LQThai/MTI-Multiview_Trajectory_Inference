"""Training loops for the learned arms (A1, A1r, A2/A3, A4, A6, A6p). CPU, float32."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..params import ChannelSpec, Physics
from .data import ProbeSet, channel_probs, corrupt, probe_loss, sample_family
from .features import make_features, onehot_mask
from .models import CausalDensity, MaskedBelief, count_params


@dataclass
class TrainCfg:
    lr: float = 2e-3
    wd: float = 1e-4
    batch: int = 256
    epochs: int = 12
    patience: int = 4
    views: int = 2          # extra folded (A2) or corrupted (A6/A6p) views per prefix per step
    sign_weight: float = 1.0
    seed: int = 0
    hidden: int = 64


@dataclass
class FitInfo:
    best_val: float
    epochs_run: int
    steps: int
    view_passes: int
    seconds: float
    n_params: int
    history: list = field(default_factory=list)


def fit(model: nn.Module, step_loss, val_loss, n: int, cfg: TrainCfg, views_per_step: int) -> FitInfo:
    """Generic loop: AdamW + cosine decay, validation each epoch, keep best state (early stopping)."""
    gen = torch.Generator().manual_seed(10_000 + cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    B = min(cfg.batch, max(8, n // 8))
    steps_per_epoch = max(1, n // B)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, k / total))))
    best, best_state, bad, hist, steps = float("inf"), copy.deepcopy(model.state_dict()), 0, [], 0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(steps_per_epoch):
            idx = perm[i * B:(i + 1) * B]
            loss = step_loss(idx, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            steps += 1
        model.eval()
        with torch.no_grad():
            v = float(val_loss())
        hist.append(v)
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitInfo(best, len(hist), steps, steps * B * views_per_step, time.time() - t0, count_params(model), hist)


def _batched(fn, n: int, bs: int = 8192):
    tot = 0.0
    for lo in range(0, n, bs):
        hi = min(n, lo + bs)
        tot += float(fn(lo, hi)) * (hi - lo)
    return tot / n


# ---------------------------------------------------------------- A1: naive clean filter (full view only)
def train_A1(tr: ProbeSet, va: ProbeSet, phys: Physics, cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    L = phys.L
    model = MaskedBelief(cfg.hidden)
    nomask = torch.zeros(1, L, dtype=torch.bool)

    def step_loss(idx, gen):
        s, a = tr.s[idx], tr.a[idx]
        pred, _ = model(make_features(s, a, nomask.expand(len(idx), -1), phys))
        return probe_loss(pred, tr.d_pr[idx], tr.a_pr[idx], phys)

    def val_loss():
        def f(lo, hi):
            p, _ = model(make_features(va.s[lo:hi], va.a[lo:hi], nomask.expand(hi - lo, -1), phys))
            return probe_loss(p, va.d_pr[lo:hi], va.a_pr[lo:hi], phys)
        return _batched(f, va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=1)


# ---------------------------------------------------------------- A2/A3: shared fixed-view model (full + folded views)
def _a2_losses(model, s, a, d_pr, a_pr, j, phys, sign_weight):
    """Composite objective for one set of folded slots j: full-view probe loss + folded probe loss + sign BCE."""
    B, L = a.shape
    nomask = torch.zeros(B, L, dtype=torch.bool)
    full, _ = model(make_features(s, a, nomask, phys))
    pf, sl = model(make_features(s, a, onehot_mask(j, L), phys))
    rows = torch.arange(B)
    bce = nn.functional.binary_cross_entropy_with_logits(sl[rows, j], (a[rows, j] >= 0).to(sl.dtype))
    return probe_loss(full, d_pr, a_pr, phys), probe_loss(pf, d_pr, a_pr, phys), bce


def train_A2(tr: ProbeSet, va: ProbeSet, phys: Physics, cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    L = phys.L
    model = MaskedBelief(cfg.hidden)
    vj = (torch.arange(va.n) % L)                                    # deterministic validation slots

    def step_loss(idx, gen):
        s, a, d, ap = tr.s[idx], tr.a[idx], tr.d_pr[idx], tr.a_pr[idx]
        B = len(idx)
        full_l, fold_l, sign_l = 0.0, 0.0, 0.0
        for _ in range(cfg.views):
            j = torch.randint(0, L, (B,), generator=gen)
            fl, pl, bl = _a2_losses(model, s, a, d, ap, j, phys, cfg.sign_weight)
            full_l, fold_l, sign_l = full_l + fl / cfg.views, fold_l + pl / cfg.views, sign_l + bl / cfg.views
        return full_l + fold_l + cfg.sign_weight * sign_l

    def val_loss():
        def f(lo, hi):
            fl, pl, bl = _a2_losses(model, va.s[lo:hi], va.a[lo:hi], va.d_pr[lo:hi], va.a_pr[lo:hi], vj[lo:hi], phys, cfg.sign_weight)
            return fl + pl + cfg.sign_weight * bl
        return _batched(f, va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=1 + cfg.views)


# ---------------------------------------------------------------- A4: causal mixture density scorer
def _tokens(s, a, phys):
    x = make_features(s, a, torch.zeros(a.shape, dtype=torch.bool), phys)
    return x[:, :-1, :4]


def mixture_logpdf(delta, a, logit_w, phys):
    """log of w N(delta; c a, q_w) + (1-w) N(delta; -c a, q_w), w = sigmoid(logit_w)."""
    const = -0.5 * np.log(2 * np.pi * phys.q_w)
    lp = nn.functional.logsigmoid(logit_w) + const - (delta - phys.c * a) ** 2 / (2 * phys.q_w)
    lm = nn.functional.logsigmoid(-logit_w) + const - (delta + phys.c * a) ** 2 / (2 * phys.q_w)
    return torch.logaddexp(lp, lm)


def _a4_nll(model, s, a, d_pr, a_pr, phys):
    logits = model(_tokens(s, a, phys))                              # (B, L+1)
    delta = torch.cat([s[:, 1:] - phys.rho * s[:, :-1], d_pr[:, None]], dim=1)
    act = torch.cat([a, a_pr[:, None]], dim=1)
    return -mixture_logpdf(delta, act, logits, phys).mean()


def train_A4(tr: ProbeSet, va: ProbeSet, phys: Physics, cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    model = CausalDensity(cfg.hidden)

    def step_loss(idx, gen):
        return _a4_nll(model, tr.s[idx], tr.a[idx], tr.d_pr[idx], tr.a_pr[idx], phys)

    def val_loss():
        return _batched(lambda lo, hi: _a4_nll(model, va.s[lo:hi], va.a[lo:hi], va.d_pr[lo:hi], va.a_pr[lo:hi], phys), va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=1)


# ---------------------------------------------------------------- E1: channel-trained density-direct (A6d / A6pd)
def _a6d_nll(model, s, a_corrupt, d_pr, a_pr, phys, cond=None):
    """NLL of every recorded transition of the (possibly corrupted) record plus the original clean probe transition.

    Each conditional is a +-c*a mixture over the RECORDED action; a flip swaps the component, absorbed by the weight.
    The probe transition is clean, so w_L targets P(m_L=+1 | S) under the training channel and the belief 2 w_L - 1 targets mu_2.
    """
    logits = model(_tokens(s, a_corrupt, phys), cond)                # (B, L+1)
    delta = torch.cat([s[:, 1:] - phys.rho * s[:, :-1], d_pr[:, None]], dim=1)
    act = torch.cat([a_corrupt, a_pr[:, None]], dim=1)
    return -mixture_logpdf(delta, act, logits, phys).mean()


def train_A6d(tr: ProbeSet, va: ProbeSet, phys: Physics, chan: ChannelSpec, cfg: TrainCfg):
    """Fixed-prior density-direct arm (channel P1): density NLL on channel-simulated records, original probe kept."""
    torch.manual_seed(cfg.seed)
    probs = channel_probs(chan)
    model = CausalDensity(cfg.hidden)
    vg = torch.Generator().manual_seed(555 + cfg.seed)
    va_a = corrupt(va.a, probs, vg)

    def step_loss(idx, gen):
        tot = 0.0
        for _ in range(cfg.views):
            a = corrupt(tr.a[idx], probs, gen)
            tot = tot + _a6d_nll(model, tr.s[idx], a, tr.d_pr[idx], tr.a_pr[idx], phys) / cfg.views
        return tot

    def val_loss():
        return _batched(lambda lo, hi: _a6d_nll(model, va.s[lo:hi], va_a[lo:hi], va.d_pr[lo:hi], va.a_pr[lo:hi], phys), va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=cfg.views)


def train_A6pd(tr: ProbeSet, va: ProbeSet, phys: Physics, family: str, cfg: TrainCfg):
    """Prior-conditioned density-direct arm: q appended to every token, priors drawn per example from `family`."""
    torch.manual_seed(cfg.seed)
    L = phys.L
    model = CausalDensity(cfg.hidden, cond_dim=L)
    vg = torch.Generator().manual_seed(777 + cfg.seed)
    vprobs, vq = sample_family(family, va.n, L, vg)
    va_a = corrupt(va.a, vprobs, vg)

    def step_loss(idx, gen):
        tot = 0.0
        for _ in range(cfg.views):
            probs, q = sample_family(family, len(idx), L, gen)
            a = corrupt(tr.a[idx], probs, gen)
            tot = tot + _a6d_nll(model, tr.s[idx], a, tr.d_pr[idx], tr.a_pr[idx], phys, q) / cfg.views
        return tot

    def val_loss():
        return _batched(lambda lo, hi: _a6d_nll(model, va.s[lo:hi], va_a[lo:hi], va.d_pr[lo:hi], va.a_pr[lo:hi], phys, vq[lo:hi]), va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=cfg.views)


# ---------------------------------------------------------------- A6 / A6p: channel-trained direct regression
def train_A6(tr: ProbeSet, va: ProbeSet, phys: Physics, chan: ChannelSpec, cfg: TrainCfg):
    """Fixed-prior direct regression h(S) trained on channel-simulated records of the training prefixes (original probe kept)."""
    torch.manual_seed(cfg.seed)
    probs = channel_probs(chan)
    model = MaskedBelief(cfg.hidden)
    nomask = torch.zeros(1, phys.L, dtype=torch.bool)
    vg = torch.Generator().manual_seed(555 + cfg.seed)
    va_a = corrupt(va.a, probs, vg)                                  # fixed channel-simulated validation records

    def step_loss(idx, gen):
        s, ap, d = tr.s[idx], tr.a_pr[idx], tr.d_pr[idx]
        tot = 0.0
        for _ in range(cfg.views):
            a = corrupt(tr.a[idx], probs, gen)
            pred, _ = model(make_features(s, a, nomask.expand(len(idx), -1), phys))
            tot = tot + probe_loss(pred, d, ap, phys) / cfg.views
        return tot

    def val_loss():
        def f(lo, hi):
            p, _ = model(make_features(va.s[lo:hi], va_a[lo:hi], nomask.expand(hi - lo, -1), phys))
            return probe_loss(p, va.d_pr[lo:hi], va.a_pr[lo:hi], phys)
        return _batched(f, va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=cfg.views)


def train_A6p(tr: ProbeSet, va: ProbeSet, phys: Physics, family: str, cfg: TrainCfg):
    """Prior-conditioned direct regression h(S, q), priors drawn per example from `family` ('narrow' | 'broad')."""
    torch.manual_seed(cfg.seed)
    L = phys.L
    model = MaskedBelief(cfg.hidden, cond_dim=L)
    nomask = torch.zeros(1, L, dtype=torch.bool)
    vg = torch.Generator().manual_seed(777 + cfg.seed)
    vprobs, vq = sample_family(family, va.n, L, vg)
    va_a = corrupt(va.a, vprobs, vg)

    def step_loss(idx, gen):
        s, ap, d = tr.s[idx], tr.a_pr[idx], tr.d_pr[idx]
        tot = 0.0
        for _ in range(cfg.views):
            probs, q = sample_family(family, len(idx), L, gen)
            a = corrupt(tr.a[idx], probs, gen)
            pred, _ = model(make_features(s, a, nomask.expand(len(idx), -1), phys), q)
            tot = tot + probe_loss(pred, d, ap, phys) / cfg.views
        return tot

    def val_loss():
        def f(lo, hi):
            p, _ = model(make_features(va.s[lo:hi], va_a[lo:hi], nomask.expand(hi - lo, -1), phys), vq[lo:hi])
            return probe_loss(p, va.d_pr[lo:hi], va.a_pr[lo:hi], phys)
        return _batched(f, va.n)

    return model, fit(model, step_loss, val_loss, tr.n, cfg, views_per_step=cfg.views)


# ---------------------------------------------------------------- A1r: restricted-summary recalibrator on A1's outputs
class Recalibrator(nn.Module):
    """g(|s_L|, f) = f + sign(f) * u(|s_L|, |f|): odd in f, contains the identity (zero-initialised correction)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, abs_s: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        u = self.net(torch.stack([abs_s, f.abs()], dim=-1)).squeeze(-1)
        return f + torch.sign(f) * u        # sign(0) = 0 keeps g exactly odd


def train_A1r(a1: MaskedBelief, cal: ProbeSet, cal_val: ProbeSet, phys: Physics, chan: ChannelSpec, cfg: TrainCfg, draws: int = 8):
    """Fit g on channel-simulated calibration records of THIS A1 realisation; supervision = probe loss, no oracle lookup.

    `draws` independent corruption draws per calibration prefix are pooled (views of the same prefix, not new prefixes).
    """
    torch.manual_seed(cfg.seed)
    probs = channel_probs(chan)
    gen = torch.Generator().manual_seed(999 + cfg.seed)
    L = phys.L
    nomask = torch.zeros(1, L, dtype=torch.bool)
    sig_s = float(np.sqrt(phys.var_s0))

    def pool(ps: ProbeSet, k: int):
        fs, ab, ds, aps = [], [], [], []
        with torch.no_grad():
            for _ in range(k):
                a = corrupt(ps.a, probs, gen)
                f = torch.cat([a1(make_features(ps.s[lo:lo + 8192], a[lo:lo + 8192], nomask.expand(len(a[lo:lo + 8192]), -1), phys))[0]
                               for lo in range(0, ps.n, 8192)])
                fs.append(f)
                ab.append(ps.s[:, -1].abs() / sig_s)
                ds.append(ps.d_pr)
                aps.append(ps.a_pr)
        return torch.cat(fs), torch.cat(ab), torch.cat(ds), torch.cat(aps)

    f_tr, s_tr, d_tr, a_tr = pool(cal, draws)
    f_va, s_va, d_va, a_va = pool(cal_val, 2)
    g = Recalibrator()

    def step_loss(idx, _gen):
        return probe_loss(g(s_tr[idx], f_tr[idx]), d_tr[idx], a_tr[idx], phys)

    def val_loss():
        return probe_loss(g(s_va, f_va), d_va, a_va, phys)

    info = fit(g, step_loss, val_loss, len(f_tr), cfg, views_per_step=1)
    return g, info
