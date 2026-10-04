"""Learned-pilot arms (protocol v2): features, networks, training loops, composition, evaluation.

Access rules (docs/learned_pilot_protocol.md): every learned input is a deterministic function of a permitted
`Prefix` (or its folded / transformed view) and the declared physics (rho, c, q_a, q_w, lambda, L). Hidden modes,
corruption labels, oracle beliefs and the probe never enter an input; the probe only forms the training target
(delta_L - c a_L h(I))^2.
"""
