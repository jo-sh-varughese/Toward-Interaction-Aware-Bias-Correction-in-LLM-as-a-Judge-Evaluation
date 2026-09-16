"""
sim_ssit_power.py
=================
Design guidance for running SSIT: how many pairs and samples/cell do you
need to detect a position x verbosity interaction of a given size?

Test: wild cluster bootstrap on the per-pair DiD
  d_k = [m_k(1,1) - m_k(0,1)] - [m_k(1,0) - m_k(0,0)]   (clusters = pairs),
rejecting H0: i_PV = 0 at 5%.

Power depends on the judge's *decisiveness* -- how close its cell
pick-target rates sit to 1.  A near-deterministic judge (rates ~0.9,
like the pilot's) has low within-cell variance, which helps, but the
probability scale is compressed near the ceiling, which hurts.  We
report power at three decisiveness levels; for each we solve for the
log-odds bias that produces the target probability-scale i_PV.

Judge model: logit Pr(pick target) = c0 + bP(2P-1) + bPV(2P-1)V,
delta absorbed into c0's spread (sd 1.0).  c0 is set for each
decisiveness level; bP is fixed at -0.9; bPV solves for the target i_PV.

numpy only.  seed = 2024.  Writes results/ssit_power.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEED = 2024
SIG = lambda x: 1.0 / (1.0 + np.exp(-x))
BP = -0.90
SD_DELTA = 1.0
N_PANELS = 500
B_WILD = 799
ALPHA = 0.05

# c0 chosen so the mean cell pick-target rate is ~ the decisiveness level
DECISIVE = {"stochastic (rate ~0.65)": 0.7,
            "moderate (rate ~0.80)": 1.6,
            "near-det. (rate ~0.90)": 2.7}


def _iPV(c0, bpv, sd, m=400_000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, sd, m)
    def cell(p, v):
        return SIG(c0 + d + BP * (2 * p - 1) + bpv * (2 * p - 1) * v).mean()
    return (cell(1, 1) - cell(0, 1)) - (cell(1, 0) - cell(0, 0))


def _bpv_for(c0, target, sd):
    grid = np.linspace(0.0, 4.0, 161)
    err = [abs(_iPV(c0, b, sd) - target) for b in grid]
    return float(grid[int(np.argmin(err))])


def _panels_did(rng, n_pairs, n_panels, samples, c0, bpv, sd):
    out = np.empty((n_panels, n_pairs))
    for j in range(n_panels):
        d = rng.normal(0.0, sd, n_pairs)
        m = {}
        for p in (0, 1):
            for v in (0, 1):
                pr = SIG(c0 + d + BP * (2 * p - 1) + bpv * (2 * p - 1) * v)
                m[(p, v)] = (rng.random((samples, n_pairs)) < pr).mean(axis=0)
        out[j] = (m[(1, 1)] - m[(0, 1)]) - (m[(1, 0)] - m[(0, 0)])
    return out


def _wild_power(did, rng):
    P, n = did.shape
    dbar = did.mean(axis=1)
    dev = did - dbar[:, None]
    rej = np.zeros(P)
    step = max(1, 4000 // n)
    for a in range(0, P, step):
        b = min(P, a + step)
        G = rng.integers(0, 2, (b - a, B_WILD, n)) * 2 - 1
        stat = np.abs((G * dev[a:b, None, :]).mean(axis=2))
        rej[a:b] = (stat >= np.abs(dbar[a:b])[:, None]).mean(axis=1) < ALPHA
    return rej.mean()


def main():
    rng = np.random.default_rng(SEED)
    targets = [0.05, 0.10, 0.15, 0.20]
    npairs = [20, 30, 50, 100]
    S = 2                       # samples/cell for the headline table

    res = {"targets": targets, "npairs": npairs, "samples": S, "power": {}}
    print(f"Power of the SSIT wild-cluster test (5% level), {N_PANELS} panels, "
          f"{S} samples/cell\n")
    for label, c0 in DECISIVE.items():
        print(f"  {label}")
        print(f"    {'true i_PV':>10} | " + " ".join(f"{n:>6}p" for n in npairs))
        for t in targets:
            b = _bpv_for(c0, t, SD_DELTA)
            row = []
            for n in npairs:
                did = _panels_did(rng, n, N_PANELS, S, c0, b, SD_DELTA)
                pw = round(_wild_power(did, rng), 3)
                row.append(pw)
                res["power"][f"{label}|{t}|{n}"] = pw
            print(f"    {t:>10.2f} |   " + "  ".join(f"{x:.2f}" for x in row))
        print()

    # samples/cell sweep at the pilot-like config
    swp = {}
    c0 = DECISIVE["near-det. (rate ~0.90)"]
    b = _bpv_for(c0, 0.15, SD_DELTA)
    for s in (1, 2, 3, 5):
        did = _panels_did(rng, 20, N_PANELS, s, c0, b, SD_DELTA)
        swp[s] = round(_wild_power(did, rng), 3)
    res["samples_sweep_neardet_iPV0.15_n20"] = swp
    print("Samples/cell at n=20, near-deterministic judge, true i_PV=0.15:")
    for s, pw in swp.items():
        print(f"    {s} samples/cell:  power {pw:.2f}")

    # ---- analytical power (Eq. in the paper) vs simulation ----------------
    from math import erf, sqrt
    Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    Z = 1.959963985                                     # z_{0.975}
    print("\nAnalytical power  Phi(|i|*sqrt(n)/sd_d - 1.96)  vs wild-bootstrap sim")
    print(f"  {'sd_d':>6}{'i_PV':>7}{'n':>6}{'  analytic':>11}{'  sim':>7}")
    ana = {}
    for c0 in (DECISIVE["moderate (rate ~0.80)"],):
        for t in (0.10, 0.15, 0.20):
            b = _bpv_for(c0, t, SD_DELTA)
            for n in (20, 30, 50, 100):
                did = _panels_did(rng, n, 400, S, c0, b, SD_DELTA)
                sd_d = float(np.mean([d.std(ddof=1) for d in did]))
                a = Phi(t * sqrt(n) / sd_d - Z) + Phi(-t * sqrt(n) / sd_d - Z)
                s_pw = _wild_power(did, rng)
                ana[f"{t}|{n}"] = {"sd_d": round(sd_d, 3),
                                   "analytic": round(a, 3), "sim": round(s_pw, 3)}
                print(f"  {sd_d:>6.3f}{t:>7.2f}{n:>6}{a:>11.2f}{s_pw:>7.2f}")
    res["analytic_vs_sim"] = ana
    n80 = lambda sd, i: 7.8 * sd * sd / (i * i)
    print(f"\n  n for 80% power (7.8 * sd_d^2 / i_PV^2):")
    for sd in (0.20, 0.25, 0.30, 0.40):
        print(f"    sd_d={sd}: i=0.10 -> {n80(sd,0.10):.0f},  "
              f"i=0.15 -> {n80(sd,0.15):.0f},  i=0.20 -> {n80(sd,0.20):.0f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/ssit_power.json").write_text(json.dumps(res, indent=2))

    print("\n--- LaTeX: tab:ssit_power (near-deterministic judge, 2 samples/cell) ---")
    print("\\begin{tabular}{lcccc}\n\\toprule")
    print("true $i_{PV}$ & " + " & ".join(f"{n} pairs" for n in npairs)
          + " \\\\\n\\midrule")
    lab = "near-det. (rate ~0.90)"
    for t in targets:
        print(f"${t:.2f}$ & " + " & ".join(
            f"${res['power'][f'{lab}|{t}|{n}']:.2f}$" for n in npairs) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
