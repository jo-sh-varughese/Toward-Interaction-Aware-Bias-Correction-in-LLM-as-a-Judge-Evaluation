"""
sim_prop2_coverage.py
=====================
Simulation check for Proposition 2 of the paper: a *marginal* PPI rectifier
under-covers the debiased win rate under bias-cell composition shift between
the calibration and evaluation sets, and Interaction-Aware PPI (factorial
rectifier) restores nominal coverage.

Also runs the structural-misspecification variants V6-V8 (probit link,
active three-way term, quality-dependent slope): IA-PPI's rectifier is
still the same factorial logistic-style cell model, i.e. mis-specified.

Outputs two LaTeX-ready tables to stdout and writes results/prop2.json.
numpy only.  seed = 2024.
"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
from ia_ppi import calibration_only, marginal_ppi, ia_ppi

SEED = 2024
sig = lambda x: 1.0 / (1.0 + np.exp(-x))
from math import erf
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / np.sqrt(2))))

BP, BV = 0.50, 0.55                       # judge main effects (fixed)
PI_EVAL = np.array([0.10, 0.15, 0.15, 0.60])   # cell (P,V): 00,01,10,11
PI_CAL = np.array([0.25, 0.25, 0.25, 0.25])    # balanced calibration
DELTA_MU, DELTA_SD = 0.10, 0.80           # latent quality-gap distribution


def theta_true(rng, m=4_000_000):
    d = rng.normal(DELTA_MU, DELTA_SD, m)
    return sig(d).mean()


def draw(rng, n, pi, bpv, variant="base"):
    """Return (Yhat, P, V, H) for n pairs. H = unbiased gold, Yhat = biased judge."""
    cell = rng.choice(4, size=n, p=pi)
    P = (cell >> 1).astype(float)
    V = (cell & 1).astype(float)
    d = rng.normal(DELTA_MU, DELTA_SD, n)
    H = rng.binomial(1, sig(d))
    if variant == "V6":                   # probit link for the judge
        bias = BP * P + BV * V + bpv * P * V
        Yhat = rng.binomial(1, Phi(d + bias))
    elif variant == "V7":                 # active three-way (S marginalised at 0.5)
        S = rng.integers(0, 2, n).astype(float)
        bias = BP * P + BV * V + bpv * P * V - 0.30 * V * S + 0.40 * P * V * S
        Yhat = rng.binomial(1, sig(d + bias))
    elif variant == "V8":                 # quality-dependent position slope
        bias = (BP + 0.6 * np.tanh(d)) * P + BV * V + bpv * P * V
        Yhat = rng.binomial(1, sig(d + bias))
    else:
        bias = BP * P + BV * V + bpv * P * V
        Yhat = rng.binomial(1, sig(d + bias))
    return Yhat, P, V, H


def run(bic_levels, n_reps=1500, n_cal=800, n_eval=20000, variant="base"):
    rng = np.random.default_rng(SEED)
    th = theta_true(rng)
    rows = []
    for b in bic_levels:
        bpv = b * np.sqrt(BP * BV)
        cover = {k: 0 for k in ("cal", "mppi", "iappi")}
        width = {k: 0.0 for k in ("cal", "mppi", "iappi")}
        for _ in range(n_reps):
            Yc, Pc, Vc, Hc = draw(rng, n_cal, PI_CAL, bpv, variant)
            Ye, Pe, Ve, He = draw(rng, n_eval, PI_EVAL, bpv, variant)
            for key, fn in (
                ("cal",   lambda: calibration_only(Hc)),
                ("mppi",  lambda: marginal_ppi(Ye, Yc, Hc)),
                ("iappi", lambda: ia_ppi(Ye, Pe, Ve, Yc, Pc, Vc, Hc)),
            ):
                est, (lo, hi), se = fn()
                cover[key] += (lo <= th <= hi)
                width[key] += hi - lo
        rows.append(dict(
            bic=round(b, 2), theta=round(float(th), 4),
            cov_cal=cover["cal"] / n_reps,
            cov_mppi=cover["mppi"] / n_reps,
            cov_iappi=cover["iappi"] / n_reps,
            w_cal=width["cal"] / n_reps,
            w_mppi=width["mppi"] / n_reps,
            w_iappi=width["iappi"] / n_reps,
        ))
        print(f"  [{variant}] BIC={b:>4}: cover  cal={rows[-1]['cov_cal']:.3f}  "
              f"marg-PPI={rows[-1]['cov_mppi']:.3f}  IA-PPI={rows[-1]['cov_iappi']:.3f}   "
              f"(width ratio cal/IA = {rows[-1]['w_cal']/rows[-1]['w_iappi']:.2f})")
    return rows


if __name__ == "__main__":
    out = {}
    print("== Proposition 2: coverage under cell-composition shift ==")
    out["base"] = run([0.0, 0.4, 0.8, 1.2, 2.0])
    print("\n== Structural-misspecification variants (BIC = 1.5) ==")
    for v in ("V6", "V7", "V8"):
        out[v] = run([1.5], variant=v)

    Path("results").mkdir(exist_ok=True)
    Path("results/prop2.json").write_text(json.dumps(out, indent=2))

    print("\n--- LaTeX: tab:coverage ---")
    for r in out["base"]:
        print(f"{r['bic']:.1f} & {r['cov_cal']:.3f} & {r['cov_mppi']:.3f} & {r['cov_iappi']:.3f} \\\\")
    print("\n--- LaTeX: tab:misspec (coverage @ BIC=1.5) ---")
    for v in ("V6", "V7", "V8"):
        r = out[v][0]
        print(f"{v} & {r['cov_iappi']:.3f} & marg-PPI {r['cov_mppi']:.3f} \\\\")
