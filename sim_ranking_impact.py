"""
sim_ranking_impact.py
=====================
Does the position x verbosity interaction actually change a leaderboard?

Setup (AlpacaEval-style).  Six models are ranked by their win rate against
a fixed reference, which always sits in slot 1; each test model is in
slot 2.  Models alternate verbose / terse; the reference is terse, so a
verbose test model has a "length gap" (V=1) and a terse one does not.
True skill is a tight ladder (0.05-logit rungs; adjacent win rate ~51%).

Judge bias, calibrated to the pilot (S 5):

    logit P(pick the test model) = (skill gap) + b_P * s + b_PV * s * V
    s = -1  (test model is always in slot 2);  V in {0,1}
    b_P = -0.60 (a pull toward slot 1 / the reference),  b_PV = +0.60
      ->  i_PV ~ +0.15 win-rate points; position penalty on a verbose
          test model is CANCELLED by the interaction, on a terse one it is not

Rankings compared (corrections estimate coefficients from a calibration
set, not oracle values):

  none : raw win-rate-vs-reference
  SC   : subtract the marginal position effect (regress on s) + marginal V.
         With s constant across the leaderboard this is a constant shift and
         cannot fix a verbose-vs-terse differential (Prop. 1).
  JBC  : regress s + V + s*V, subtract all three  (= IA-PPI point estimate).
         Calibrated on an EQUAL-SKILL gold set (true win rate = 0.5).
  IASC : subtract b_P*s + b_PV*s*V, with b_P, b_PV read off the four swap
         cells of a calibration subset by difference-in-differences
         (Prop. 5).  GOLD-FREE and no equal-skill assumption: the
         calibration pairs have arbitrary unknown skill gaps.  1x eval cost.

seed = 2024, numpy only.
"""
from __future__ import annotations
import numpy as np

rng = np.random.default_rng(2024)
sig = lambda x: 1.0 / (1.0 + np.exp(-x))

MODELS  = ["m1", "m2", "m3", "m4", "m5", "m6"]
THETA   = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])   # true order m1<...<m6
VERBOSE = np.array([0, 1, 0, 1, 0, 1])                      # vs a terse reference
THETA_REF = 0.12
B_P, B_PV = -0.60, 0.60
N_EVAL, N_CAL = 20000, 12000
TRUTH = list(range(len(MODELS)))[::-1]                      # m6 best ... m1 worst


def _l(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calib():
    """Balanced equal-skill calibration: both slots, V ~ Bernoulli(0.5)."""
    P = rng.integers(0, 2, N_CAL); V = rng.integers(0, 2, N_CAL); s = 2 * P - 1
    Y = (rng.random(N_CAL) < sig(B_P * s + B_PV * s * V)).astype(float)
    bP_m = 0.5 * (_l(Y[s == 1].mean()) - _l(Y[s == -1].mean()))
    bV_m = _l(Y[V == 1].mean()) - _l(Y[V == 0].mean())
    cell = {(si, vi): _l(Y[(s == si) & (V == vi)].mean())
            for si in (-1, 1) for vi in (0, 1)}
    X = np.array([[1, si, vi, si * vi] for si in (-1, 1) for vi in (0, 1)], float)
    yv = np.array([cell[(si, vi)] for si in (-1, 1) for vi in (0, 1)])
    _, bP_j, bV_j, bPV_j = np.linalg.solve(X, yv)
    return dict(bP_m=bP_m, bV_m=bV_m, bP_j=bP_j, bV_j=bV_j, bPV_j=bPV_j)


def calib_iasc(n_pairs=8000, delta_sd=0.7, seed=7):
    """Gold-free swap calibration: pairs with ARBITRARY unknown skill gaps,
    each run through the four (P,V) swap cells.  b_P, b_PV by DiD on the
    cell log-odds (delta and b_V cancel -- no gold labels, no equal-skill
    assumption).  delta_sd = spread of the (unknown) calibration-pair
    quality gaps: smaller = pairs chosen to be roughly quality-matched
    (a cheap proxy filter), which shrinks the logit-pooling attenuation.
    Uses its own rng so it does not perturb the leaderboard draw."""
    r = np.random.default_rng(seed)
    delta = r.normal(0.0, delta_sd, n_pairs)          # unknown, heterogeneous
    Vg = r.integers(0, 2, n_pairs)                    # pair length-gap
    lab = {}
    for P in (0, 1):
        s = 2 * P - 1
        L = delta + B_P * s + 0.30 * Vg + B_PV * s * Vg   # 0.30 = a b_V main effect
        pick_t = (r.random(n_pairs) < sig(L)).astype(float)
        for v in (0, 1):
            lab[(P, v)] = _l(pick_t[Vg == v].mean())
    pe0 = lab[(1, 0)] - lab[(0, 0)]                   # = 2 b_P
    pe1 = lab[(1, 1)] - lab[(0, 1)]                   # = 2 b_P + 2 b_PV
    return dict(bP_ss=0.5 * pe0, bPV_ss=0.5 * (pe1 - pe0))


def leaderboard(method, C):
    s = -1                                    # test model always slot 2
    scores = np.empty(len(MODELS))
    for m in range(len(MODELS)):
        V = int(VERBOSE[m] == 1)              # reference is terse
        raw = (rng.random(N_EVAL) < sig((THETA[m] - THETA_REF) + B_P * s + B_PV * s * V)).mean()
        if method == "none":
            scores[m] = raw
        elif method == "sc":
            scores[m] = sig(_l(raw) - C["bP_m"] * s - C["bV_m"] * V)
        elif method == "jbc":
            scores[m] = sig(_l(raw) - C["bP_j"] * s - C["bV_j"] * V - C["bPV_j"] * s * V)
        elif method == "iasc":
            scores[m] = sig(_l(raw) - C["bP_ss"] * s - C["bPV_ss"] * s * V)
        elif method == "swap":
            bwd = (rng.random(N_EVAL) < sig((THETA[m] - THETA_REF) + B_P * (-s) + B_PV * (-s) * V)).mean()
            scores[m] = 0.5 * (raw + bwd)
    return list(np.argsort(-scores)), scores


def kendall(order):
    p = {m: k for k, m in enumerate(order)}; t = {m: k for k, m in enumerate(TRUTH)}
    import itertools
    c = d = 0
    for a, b in itertools.combinations(range(len(order)), 2):
        c += np.sign(p[a] - p[b]) == np.sign(t[a] - t[b])
        d += np.sign(p[a] - p[b]) != np.sign(t[a] - t[b])
    return (c - d) / (c + d)


if __name__ == "__main__":
    C = calib()
    C.update(calib_iasc())
    print("IASC gold-free coeffs vs calibration-pair quality spread "
          "(oracle bP=%.2f bPV=%.2f):" % (B_P, B_PV))
    for sd in (0.4, 0.7, 1.0, 1.5):
        c = calib_iasc(delta_sd=sd)
        print(f"  delta_sd={sd}: bP_ss={c['bP_ss']:+.2f}  bPV_ss={c['bPV_ss']:+.2f}")
    print("true order (best first):", [MODELS[m] for m in TRUTH])
    print(f"calib  bP_marg={C['bP_m']:+.2f} (oracle bP+0.5bPV={B_P+0.5*B_PV:+.2f})   "
          f"bP_joint={C['bP_j']:+.2f}  bPV_joint={C['bPV_j']:+.2f}")
    print(f"iasc   bP_ss={C['bP_ss']:+.2f}  bPV_ss={C['bPV_ss']:+.2f}  "
          f"(gold-free, DiD on swap cells; oracle {B_P:+.2f} / {B_PV:+.2f})\n")
    print(f"{'method':<7}{'ranking (best first)':<30}{'tau':<9}{'inversions vs truth'}")
    print("-" * 66)
    import json
    from pathlib import Path
    dump = {"true_order": [MODELS[m] for m in TRUTH],
            "coeffs": {k: round(float(C[k]), 3) for k in
                       ("bP_m", "bP_j", "bPV_j", "bP_ss", "bPV_ss")},
            "oracle": {"b_P": B_P, "b_PV": B_PV}, "methods": {}}
    for name, key in [("none", "none"), ("SC", "sc"), ("swap", "swap"),
                      ("JBC", "jbc"), ("IASC", "iasc")]:
        o, sc = leaderboard(key, C)
        names = [MODELS[m] for m in o]
        inv = [f"{names[k]}>{names[k+1]}" for k in range(len(names) - 1)
               if TRUTH.index(o[k]) > TRUTH.index(o[k + 1])]
        tau = kendall(o)
        print(f"{name:<7}{str(names):<30}{tau:+.3f}   {inv or 'none (correct)'}")
        dump["methods"][name] = {"order": names, "tau": round(float(tau), 3),
                                 "inversions": inv}
    # ---- robustness: is the method ordering stable across bias magnitudes? ----
    # The pilot shows near-cancellation on verbose pairs (b_P + b_PV ~ 0) with
    # a probability-scale i_PV ~ +0.15; the log-odds magnitude is attenuated by
    # delta-pooling and is a lower bound.  Sweep b_P over a wide range with
    # b_PV = -b_P (the cancellation regime) and confirm none/SC stay broken,
    # swap/JBC stay exact, and IASC stays close.
    _bp0, _bpv0 = B_P, B_PV
    sweep = []
    print(f"\n{'b_P':>6}{'b_PV':>7}   tau by method (none / SC / IASC / JBC / swap)")
    print("-" * 62)
    for bp in (-0.40, -0.60, -1.00, -1.50):
        B_P, B_PV = bp, -bp
        rng = np.random.default_rng(2024)          # same clean state per row
        Cs = calib()
        Cs.update(calib_iasc())
        taus = {}
        for nm, ky in (("none", "none"), ("SC", "sc"), ("swap", "swap"),
                       ("JBC", "jbc"), ("IASC", "iasc")):
            o, _ = leaderboard(ky, Cs)
            taus[nm] = round(float(kendall(o)), 3)
        sweep.append({"b_P": bp, "b_PV": -bp, "tau": taus})
        print(f"{bp:>6.2f}{-bp:>7.2f}   "
              + " / ".join(f"{taus[n]:+.2f}" for n in
                           ("none", "SC", "IASC", "JBC", "swap")))
    B_P, B_PV = _bp0, _bpv0
    dump["robustness_sweep"] = sweep

    # ---- downstream: RLHF preference-label flips from an uncorrected i_PV ----
    # On the length-mismatched cell the judge's pick-target probability is
    # shifted by i_PV relative to the quality-only preference.  Fraction of
    # comparisons on that cell whose LABELLED winner differs from the
    # quality-only winner = P(quality pref within i_PV of 0.5), estimated with
    # a uniform-ish quality-gap prior (Beta(2,2) on the pref prob).
    rr = np.random.default_rng(2024)
    label_flip = {}
    # two priors on the quality-only preference prob: "decisive" (U-shaped,
    # most pairs have a clear better answer) and "ambiguous" (mass near 0.5).
    priors = {"decisive": (0.8, 0.8), "ambiguous": (2.0, 2.0)}
    for iPV in (0.05, 0.10, 0.15, 0.20):
        row = {}
        for name, (a, b) in priors.items():
            q = rr.beta(a, b, 400_000)
            flipped = ((q < 0.5) & (q + iPV > 0.5)) | ((q > 0.5) & (q - iPV < 0.5))
            row[name] = round(float(flipped.mean()), 3)
        label_flip[iPV] = row
    dump["rlhf_label_flip_frac_on_affected_cell"] = label_flip
    print("\nRLHF: upper bound on the fraction of length-mismatched comparisons "
          "whose labelled winner is set by the interaction, not quality\n"
          "(assumes the judge shifts pick-prob by the full i_PV on the cell):")
    for iPV, row in label_flip.items():
        print(f"  i_PV={iPV}:  decisive prior {row['decisive']:.1%}, "
              f"ambiguous prior {row['ambiguous']:.1%}  (of the affected cell)")

    Path("results").mkdir(exist_ok=True)
    Path("results/ranking_impact.json").write_text(json.dumps(dump, indent=2))
