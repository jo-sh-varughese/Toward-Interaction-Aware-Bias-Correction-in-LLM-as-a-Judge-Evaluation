"""
iasc.py -- Interaction-Aware Swap Correction
===========================================
A GOLD-LABEL-FREE debiasing method for pairwise LLM-judge win rates.

Motivation.  Full swap-averaging (judge every pair in both orders, average
the pick rate) removes the position main effect b_P *and* every
position-coupled interaction -- including b_PV*(2P-1)*V -- because both
flip sign under a position swap.  But it costs 2x the judge budget on the
*entire* evaluation set, and the common AlpacaEval-style protocol judges
each pair once.

IASC targets the same removal at 1x + epsilon cost (in expectation;
pooling over heterogeneous quality gaps attenuates the estimated
coefficients, so in practice it UNDER-corrects by a data-dependent
amount -- see the "Guarantee" note below):

  1.  On a small calibration subset C, run the 4 (P,V) swap cells and
      read off the position-coupled coefficients by difference-in-
      differences (Prop. 5):

          PE(V) = logit pi1(1,V) - logit pi1(0,V) = 2 b_P + 2 b_PV V
          b_P   = PE(0) / 2
          b_PV  = ( PE(1) - PE(0) ) / 2                  # = Delta_SS / 2

      Each pair's unknown quality gap delta and the verbosity main
      effect b_V load only on the pair intercept and cancel in every
      swap contrast -- so C needs NO gold labels and NO equal-skill
      assumption (unlike the JBC calibration in the paper).

  2.  Apply  logit_corr = logit(yhat) - b_P (2P-1) - b_PV (2P-1) V
      to every single-order judgment in the bulk evaluation set E.

Guarantee (Prop. 5), BEFORE attenuation.  With the true b_P, b_PV,
E[ logit_corr ] = delta + b_V V + b_S S : the position main effect and
every position-coupled interaction are removed without labels; the
verbosity / self-preference *main* effects are NOT identified by swap
symmetry and remain (they need gold labels or a length-matched design).
In practice b_P, b_PV are fit by pooling over pairs with unknown,
heterogeneous quality gaps delta, which attenuates them toward 0 -- so
IASC removes only PART of the position-coupled bias, by an amount that
depends on the (unobservable) spread of delta.  It is a partial,
label-free correction, not a replacement for gold calibration.

numpy only.  Reuses the saturated-cell solver from ssit.py.
"""
from __future__ import annotations
import numpy as np
from ssit import _coef, _arrays, CELLS

_SIG = lambda x: 1.0 / (1.0 + np.exp(-x))
_CLIP = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, float), _CLIP, 1 - _CLIP)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- fit
def iasc_fit(recs, include_S=False):
    """Gold-free position-coupled coefficients from the swap cells of a
    calibration subset.  `recs` are collect_free.py records (need P,V,S,
    Y_bin, pair_id).  Returns {b_P, b_PV[, b_PS, b_PVS]} in log-odds."""
    P, V, S, r1, item = _arrays(recs)
    b = _coef(P, V, S, r1, item)                      # 8-vector, ssit.NAMES order
    out = {"b_P": float(b[1]), "b_PV": float(b[4])}
    if include_S:
        out["b_PS"] = float(b[5])
        out["b_PVS"] = float(b[7])
    return out


# ---------------------------------------------------------- correction
def iasc_correct(yhat, P, V, fit, S=None):
    """Return IASC-corrected pick-target probabilities.

    yhat : judge pick-target rate for each item (0/1 or sample mean)
    P,V  : slot / length-gap indicators in {0,1} for each item
    fit  : dict from iasc_fit
    """
    yhat, P, V = map(lambda a: np.asarray(a, float), (yhat, P, V))
    s = 2 * P - 1
    corr = _logit(yhat) - fit["b_P"] * s - fit["b_PV"] * s * V
    if S is not None and "b_PS" in fit:
        S = np.asarray(S, float)
        corr = corr - fit["b_PS"] * s * S - fit.get("b_PVS", 0.0) * s * V * S
    return _SIG(corr)


# ------------------------------------------------- reference: swap-avg
def swap_average(recs):
    """Full 2x swap-averaged pick-target rate per pair (needs both P
    cells).  Returns dict pair_id -> mean over cells of picked-target."""
    P, V, S, r1, item = _arrays(recs)
    out = {}
    for it in np.unique(item):
        m = item == it
        out[str(it)] = float(r1[m].mean())
    return out


# --------------------------------------------------- win-rate + CI
def iasc_winrate(recs_eval, fit, single_order_P=1, samples_key="sample",
                 B=3000, seed=2024, cal_recs=None):
    """Debiased win rate from the *single-order* (P == single_order_P)
    slice of recs_eval, corrected with `fit`.  If cal_recs is given the
    coefficients are re-fit on each bootstrap resample of cal_recs so the
    CI includes calibration uncertainty; otherwise `fit` is held fixed.

    Returns (est, (lo, hi), se_boot).
    """
    rng = np.random.default_rng(seed)
    ev = [r for r in recs_eval if r["P"] == single_order_P]
    if not ev:
        raise ValueError("no single-order records at P=%d" % single_order_P)

    def _wr(rs, ft):
        pid = sorted({r["pair_id"] for r in rs})
        y = np.array([np.mean([r["Y_bin"] for r in rs if r["pair_id"] == p]) for p in pid])
        # pick-target rate: Y_bin is pick-A; at P=1 target r1 is in A
        yt = y if single_order_P == 1 else 1 - y
        Pv = np.full(len(pid), single_order_P)
        Vv = np.array([next(r["V"] for r in rs if r["pair_id"] == p) for p in pid])
        return float(iasc_correct(yt, Pv, Vv, ft).mean())

    est = _wr(ev, fit)
    items = sorted({r["pair_id"] for r in ev})
    by_item = {it: [r for r in ev if r["pair_id"] == it] for it in items}
    cal_items = None
    if cal_recs is not None:
        cal_by = {}
        for r in cal_recs:
            cal_by.setdefault(r["pair_id"], []).append(r)
        cal_items = sorted(cal_by)

    draws = []
    for _ in range(B):
        pick = rng.choice(items, len(items), replace=True)
        rs = [r for it in pick for r in by_item[it]]
        ft = fit
        if cal_recs is not None:
            cpick = rng.choice(cal_items, len(cal_items), replace=True)
            crs = [r for it in cpick for r in cal_by[it]]
            ft = iasc_fit(crs, include_S=("b_PS" in fit))
        draws.append(_wr(rs, ft))
    draws = np.array(draws)
    return est, (float(np.percentile(draws, 2.5)),
                 float(np.percentile(draws, 97.5))), float(draws.std())


if __name__ == "__main__":
    # self-test: plant known bias, unknown per-pair quality gap, recover
    rng = np.random.default_rng(0)
    tb = dict(b_P=-0.55, b_V=0.60, b_S=0.0, b_PV=0.65)
    recs = []
    for k in range(300):
        delta = rng.normal(0.0, 1.3)                  # unknown, heterogeneous
        for p, v, s in CELLS:
            pp = 2 * p - 1
            L = delta + tb["b_P"] * pp + tb["b_V"] * v + tb["b_PV"] * pp * v
            r1 = int(rng.random() < _SIG(L))
            recs.append(dict(P=p, V=v, S=s, pair_id=k,
                             Y_bin=(r1 if p == 1 else 1 - r1)))
    fit = iasc_fit(recs)
    print("planted  b_P=%+.2f  b_PV=%+.2f" % (tb["b_P"], tb["b_PV"]))
    print("IASC fit b_P=%+.3f  b_PV=%+.3f  (gold-free, delta~N(0,1.3))"
          % (fit["b_P"], fit["b_PV"]))
    # corrected single-order slice vs full swap-average
    P, V, S, r1, item = _arrays(recs)
    m1 = P == 1
    sa = np.mean(list(swap_average(recs).values()))
    raw1 = r1[m1].mean()
    cor1 = iasc_correct(
        [r1[(item == it) & m1].mean() for it in np.unique(item)],
        np.ones(len(np.unique(item))),
        [V[(item == it) & (P == 1)][0] for it in np.unique(item)], fit).mean()
    print("\npick-target rate, single order (P=1):")
    print("  raw            %.3f" % raw1)
    print("  + IASC         %.3f   <- gold-free, 1x cost" % cor1)
    print("  full swap-avg  %.3f   <- 2x cost reference" % sa)
