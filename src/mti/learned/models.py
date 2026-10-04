"""Networks. All neural arms share one backbone family (GRU, hidden 64); heads are small MLPs."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import FEAT_DIM


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MaskedBelief(nn.Module):
    """Bidirectional GRU over the full prefix; belief head (tanh) from both end states, sign head per position.

    Used for A1 (belief only), A2/A3 (belief + sign at the folded slot), A6 (direct regression on corrupted
    records) and A6p (additionally conditioned on the corruption prior q).
    """

    def __init__(self, hidden: int = 64, cond_dim: int = 0):
        super().__init__()
        self.hidden, self.cond_dim = hidden, cond_dim
        self.gru = nn.GRU(FEAT_DIM + cond_dim, hidden, batch_first=True, bidirectional=True)
        self.belief = _mlp(2 * hidden, hidden, 1)
        self.sign = _mlp(2 * hidden, hidden, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        if self.cond_dim:
            x = torch.cat([x, cond[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        out, _ = self.gru(x)                                         # (B, L+1, 2H)
        H = self.hidden
        end = torch.cat([out[:, -1, :H], out[:, 0, H:]], dim=-1)     # forward state after the last token, backward after the first
        belief_pre = self.belief(end).squeeze(-1)
        sign_logit = self.sign(out).squeeze(-1)                      # (B, L+1); logit that the removed sign is '+'
        return torch.tanh(belief_pre), sign_logit


class CausalDensity(nn.Module):
    """Causal GRU over transitions; head gives w_k = P(mode_k = +1 | transitions < k) for k = 0..L.

    Transition density: delta_k | past, a_k ~ w_k N(c a_k, q_w) + (1 - w_k) N(-c a_k, q_w) (mixture-capable).
    The decision belief is 2 w_L - 1.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(4, hidden, batch_first=True)
        self.head = _mlp(hidden, hidden, 1)
        self.h0_logit = nn.Parameter(torch.zeros(1))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        """tok (B, L, 4) transition tokens (s_k, a_k, delta_k, asinh z_k) -> logit of w_k, shape (B, L+1)."""
        out, _ = self.gru(tok)                                       # state after tokens 0..k
        logits = self.head(out).squeeze(-1)                          # logits for w_{k+1}, k = 0..L-1
        first = self.h0_logit.expand(tok.shape[0], 1)
        return torch.cat([first, logits], dim=1)


def count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))
