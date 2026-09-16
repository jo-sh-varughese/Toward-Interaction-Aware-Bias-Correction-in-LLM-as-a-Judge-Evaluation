"""
sim_ssit_robustness.py
======================
What does SSIT actually assume?

The probability-scale readout i_PV is a *design-based* estimand: the
difference-in-differences of the four within-pair cell means
  E[pick-target | P,V] .
Because every pair is run through all four (P,V) swap cells, this is an
unbiased estimate of the population interaction for ANY judge decision
rule -- there is no link, no additivity, no homogeneity assumption. Only
the *secondary* log-odds readout b_PV uses the logistic swap model
(Eq. 2 of the paper) and can be biased by pooling over the unknown
per-pair quality gap delta.

This script checks that claim. It plants a judge, computes that judge's
TRUE i_PV by high-N Monte Carlo, and asks whether SSIT recovers it from
60-pair panels with no gold labels, across six decision rules:

  logit            additive logistic (SSIT's own model)          -- baseline
  het-delta        logistic, wide delta ~ N(0,1.6)               -- heavy pooling
  qdep-sym         position & interaction slopes vary with delta
                     symmetrically: bP,bPV += c*tanh(delta)      -- model violated
  qdep-asym        bias fades when the target is clearly better
                     (realistic: no position effect on easy pairs)-- model violated
  probit           probit link instead of logit                  -- link violated
  rule             OUT OF FAMILY: no link at all. Heavy-tailed
                     (Student-t) quality perception + a hard
                     confidence threshold; when unsure, a
                     probability-scale positional/verbosity
                     tie-break.                                   -- model absent

numpy only. seed = 2024. Writes results/ssit_robustness.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ssit import ssit, ssit_boot, CELLS

SEED = 2024
SIG = lambda x: 1.0 / (1.0 + np.exp(-x))
from math import erf
PHI = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / np.sqrt(2))))

B_P, B_V, B_PV = -0.55, 0.30, 0.60          # logit-family planted params
A_P, A_PV, TAU = -0.42, 0.52, 0.9           # rule-judge: heuristic strengths, conf. threshold
N_PAIRS, REPS, BOOT = 60, 300, 1500


def _pick_target(rng, d, p, v, dgp):
    """Return Pr(judge picks the target response) for one (pair, cell).
    d = quality gap (target minus other), on the model's own scale."""
    pp = 2 * p - 1
    if dgp == "rule":
        # out of family: no link. one Student-t quality read per call,
        # hard confidence gate, else a prob-scale positional tie-break.
        g = d + rng.standard_t(3) * 0.9
        if abs(g) > TAU:
            return 1.0 if g > 0 else 0.0
        ptar = 0.5 + 0.5 * (A_P * pp + A_PV * pp * v)
        return float(np.clip(ptar, 0.02, 0.98))
    bP, bPV = B_P, B_PV
    if dgp == "qdep-sym":
        bP = B_P + 0.6 * np.tanh(d)
        bPV = B_PV + 0.5 * np.tanh(d)
    elif dgp == "qdep-asym":
        fade = 1.0 - 0.7 * SIG(2.0 * d)          # -> ~0.3 when target clearly better
        bP, bPV = B_P * fade, B_PV * fade
    lin = d + bP * pp + B_V * v + bPV * pp * v
    return float(PHI(lin) if dgp == "probit" else SIG(lin))


def _delta(rng, n, dgp):
    return rng.normal(0.0, 1.6 if dgp == "het-delta" else 1.0, n)


def true_iPV(dgp, m=200_000):
    """That judge's actual probability-scale i_PV (population DiD)."""
    rng = np.random.default_rng(12345)
    d = _delta(rng, m, dgp)
    cell = {}
    for p in (0, 1):
        for v in (0, 1):
            # vectorised pick-target prob per pair for this cell
            if dgp == "rule":
                g = d + rng.standard_t(3, m) * 0.9
                pp = 2 * p - 1
                conf = np.abs(g) > TAU
                ptar = np.where(conf, (g > 0).astype(float),
                                np.clip(0.5 + 0.5 * (A_P * pp + A_PV * pp * v),
                                        0.02, 0.98))
                cell[(p, v)] = ptar.mean()
            else:
                pp = 2 * p - 1
                bP, bPV = B_P, B_PV
                if dgp == "qdep-sym":
                    bP, bPV = B_P + 0.6 * np.tanh(d), B_PV + 0.5 * np.tanh(d)
                elif dgp == "qdep-asym":
                    fade = 1.0 - 0.7 * SIG(2.0 * d)
                    bP, bPV = B_P * fade, B_PV * fade
                lin = d + bP * pp + B_V * v + bPV * pp * v
                cell[(p, v)] = (PHI(lin) if dgp == "probit" else SIG(lin)).mean()
    return ((cell[(1, 1)] - cell[(0, 1)]) - (cell[(1, 0)] - cell[(0, 0)]))


def _panel(rng, n, dgp):
    d = _delta(rng, n, dgp)
    recs = []
    for k in range(n):
        for p, v, s in CELLS:
            pr = _pick_target(rng, d[k], p, v, dgp)
            r1 = int(rng.random() < pr)
            recs.append(dict(P=p, V=v, S=s, pair_id=k,
                             Y_bin=(r1 if p == 1 else 1 - r1)))
    return recs


def run(dgp):
    rng = np.random.default_rng(SEED)
    iPV, bPV = [], []
    rep_ci = None
    for r in range(REPS):
        recs = _panel(rng, N_PAIRS, dgp)
        d = ssit(recs)
        iPV.append(d["i_PV_prob"])
        bPV.append(d["b_PV"])
        if r == 0:
            out, _ = ssit_boot(recs, B=BOOT, seed=SEED)
            rep_ci = out["i_PV_prob"]["ci"]
    iPV, bPV = np.array(iPV), np.array(bPV)
    ti = float(true_iPV(dgp))
    return dict(dgp=dgp, true_iPV=round(ti, 3),
               iPV_mean=round(float(iPV.mean()), 3), iPV_sd=round(float(iPV.std()), 3),
               iPV_bias=round(float(iPV.mean() - ti), 3),
               bPV_mean=round(float(bPV.mean()), 3), bPV_sd=round(float(bPV.std()), 3),
               frac_iPV_pos=round(float((iPV > 0).mean()), 3),
               rep_ci=[round(c, 3) for c in rep_ci] if rep_ci else None)


NOTES = {"logit": "additive logistic (SSIT's own model)",
         "het-delta": "logistic, wide quality-gap spread",
         "qdep-sym": "bias slopes vary with quality, symmetric",
         "qdep-asym": "bias fades when target clearly better (realistic)",
         "probit": "probit link (SSIT assumes logit)",
         "rule": "OUT OF FAMILY: t-noise + hard threshold, no link"}


def main():
    rows = [run(x) for x in
            ("logit", "het-delta", "qdep-sym", "qdep-asym", "probit", "rule")]
    Path("results").mkdir(exist_ok=True)
    Path("results/ssit_robustness.json").write_text(
        json.dumps({"n_pairs": N_PAIRS, "reps": REPS, "rows": rows}, indent=2))

    print(f"{'DGP':<11}{'true i_PV':<11}{'SSIT i_PV mean(SD)':<22}"
          f"{'bias':<8}{'P(>0)':<7}{'rep 95% CI':<18}{'note'}")
    print("-" * 104)
    for r in rows:
        ci = r["rep_ci"]
        cs = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "n/a"
        print(f"{r['dgp']:<11}{r['true_iPV']:<+11.3f}"
              f"{r['iPV_mean']:+.3f} ({r['iPV_sd']:.3f})      "
              f"{r['iPV_bias']:<+8.3f}{r['frac_iPV_pos']:<7.2f}{cs:<18}"
              f"{NOTES[r['dgp']]}")

    print("\n--- LaTeX: tab:ssit_robust ---")
    print("\\begin{tabular}{lcccc}\n\\toprule")
    print("Judge decision rule & true $i_{PV}$ & SSIT $\\hat\\imath_{PV}$ "
          "& bias & $\\Pr(\\hat\\imath_{PV}{>}0)$ \\\\\n\\midrule")
    for r in rows:
        print(f"{NOTES[r['dgp']]} & ${r['true_iPV']:+.2f}$ "
              f"& ${r['iPV_mean']:+.2f}\\,(\\pm{r['iPV_sd']:.2f})$ "
              f"& ${r['iPV_bias']:+.2f}$ & {r['frac_iPV_pos']:.2f} \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
