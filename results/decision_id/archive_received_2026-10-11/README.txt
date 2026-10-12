Verification scripts for docs/decision_identification_discussion.md (MTI, exp/joint-identification @ 08dd58e).
Evaluation-only: no fitting, no training. Run from this directory with the MTI repository at /home/claude/mti
(edit the sys.path line in e4common.py to point at your checkout's src/). numpy + scipy only (no torch).

  ridge_l3.py        complete L=3 arcs, record-likelihood checks, diameter, least-favourable prior (writes ridge_l3.json; also used as ridgelib.py)
  ridge_l3_extra.py  stable endpoint likelihood checks; regret of midpoint / bottom / naive rules (ridge_l3_extra.json)
  diam_se.py         minimax floor V* with prefix-level SEs (diam_se.json)
  se_shared.py       shared-prefix, delta-method and crossed SEs for the E4 P1 contrasts from saved units (se_shared.json)
  kflip.py           k-flip thresholds by brute force and LP (k<=3, L<=9)
  nuisance.py        channel-invariant moment identification of physics; trusted-outcome slope check at L=3
  e4common.py        numpy-only replicas of run_e4.PHYS / truths and TestSet.__init__ (identical simulate calls)
