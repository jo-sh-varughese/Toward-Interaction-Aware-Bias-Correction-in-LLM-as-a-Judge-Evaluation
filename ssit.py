"""
ssit.py -- Swap-Symmetry Interaction Test
=========================================
A GOLD-LABEL-FREE estimator of the judge-bias coefficients and a direct
test of the additive-independence assumption (AIA), from the 2x2x2
factorial-swap design used by collect_free.py.

Key idea.  Present every pair in all 8 (P,V,S) cells.  With
    picked_r1 = Y_bin if P==1 else 1 - Y_bin,
    q_c = mean over pairs of picked_r1 in cell c,
model the pooled log-odds of picking the target response r1 as

  logit q_c = delta + b_P*(2P-1) + b_V*V + b_S*S
              + b_PV*(2P-1)*V + b_PS*(2P-1)*S + b_VS*V*S + b_PVS*(2P-1)*V*S .

Swapping the two responses flips the slot exactly, so position enters as
(2P-1); the unknown mean quality gap `delta` loads ONLY on the intercept.
The 8-cell design is saturated -> b = X^{-1} logit(q) is exact.

  b_PV = 1/2 * { [position effect | V=1] - [position effect | V=0] }

is a difference-in-differences on logits.  The AIA test for the
position x verbosity pair is H0: b_PV = 0 (95% pair-bootstrap CI vs 0).

Two readouts:
  * i_PV_prob : the same DiD on the PROBABILITY scale (a win-rate-point
    quantity).  Robust -- no clipping, no matrix inverse -- and is the
    number to report for near-deterministic judges / small n.
  * b_PV      : the log-odds coefficient from the saturated 8-cell fit.
    More interpretable when the judge is not near-deterministic, but
    unstable when cell rates hit 0/1 (strong judge, one sample per cell).

Pooling over heterogeneous `delta` attenuates ALL coefficients toward 0
(logit nonlinearity); the attenuation is milder for the DiD than for the
marginals but not absent, so treat magnitudes as lower bounds.  A powered
run uses >=3 samples per cell or >=50 pairs.

numpy only.
"""
from __future__ import annotations
import numpy as np

CELLS = [(p, v, s) for p in (0, 1) for v in (0, 1) for s in (0, 1)]
NAMES = ["delta", "b_P", "b_V", "b_S", "b_PV", "b_PS", "b_VS", "b_PVS"]
_X = np.array([[1, 2 * p - 1, v, s, (2 * p - 1) * v, (2 * p - 1) * s, v * s,
                (2 * p - 1) * v * s] for p, v, s in CELLS], float)
_XINV = np.linalg.inv(_X)


def _coef(P, V, S, r1, item, laplace=0.5):
    q = np.empty(8)
    for i, (p, v, s) in enumerate(CELLS):
        m = (P == p) & (V == v) & (S == s)
        q[i] = (r1[m].sum() + laplace) / (m.sum() + 2 * laplace) if m.any() else 0.5
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return _XINV @ np.log(q / (1 - q))


def _bics(b, eps):
    d = {NAMES[i]: float(b[i]) for i in range(8)}

    def bic(inter, i, j):
        if abs(d[i]) < eps or abs(d[j]) < eps:
            return np.nan
        return d[inter] / np.sqrt(abs(d[i]) * abs(d[j]))

    d["BIC_PV"] = bic("b_PV", "b_P", "b_V")
    d["BIC_PS"] = bic("b_PS", "b_P", "b_S")
    d["BIC_VS"] = bic("b_VS", "b_V", "b_S")
    d["posn_effect_V0"] = 2 * d["b_P"]
    d["posn_effect_V1"] = 2 * (d["b_P"] + d["b_PV"])
    d["i_PV_prob"] = d["i_PS_prob"] = d["i_VS_prob"] = np.nan  # filled by caller
    return d


def _arrays(recs):
    return (np.array([r["P"] for r in recs]), np.array([r["V"] for r in recs]),
            np.array([r["S"] for r in recs]),
            np.where(np.array([r["P"] for r in recs]) == 1,
                     np.array([r["Y_bin"] for r in recs]),
                     1 - np.array([r["Y_bin"] for r in recs])).astype(float),
            np.array([r["pair_id"] for r in recs]))


def _iPV_prob(P, V, r1):
    """Probability-scale position-effect difference-in-differences (robust,
    no attenuation, but not a log-odds coefficient)."""
    def pe(vv):
        a = r1[(V == vv) & (P == 1)].mean() if ((V == vv) & (P == 1)).any() else np.nan
        b = r1[(V == vv) & (P == 0)].mean() if ((V == vv) & (P == 0)).any() else np.nan
        return a - b
    return pe(1) - pe(0)


def _iXY_prob(A, B, r1):
    """Generic probability-scale difference-in-differences for the (A,B)
    factor pair: [E r1 | A=1,B=1] - [E r1 | A=0,B=1]
              - ([E r1 | A=1,B=0] - [E r1 | A=0,B=0]).
    r1 is already the picked-target outcome (position pre-flipped), so all
    factors enter raw here.  Robust: no matrix inverse, no clipping."""
    def m(a, b):
        sel = (A == a) & (B == b)
        return r1[sel].mean() if sel.any() else np.nan
    return (m(1, 1) - m(0, 1)) - (m(1, 0) - m(0, 0))


def ssit(recs, eps=0.05):
    P, V, S, r1, item = _arrays(recs)
    d = _bics(_coef(P, V, S, r1, item), eps)
    d["i_PV_prob"] = float(_iPV_prob(P, V, r1))
    # probability-scale readouts for the other two factor pairs.  P is the
    # swap factor so it enters flip-coded; V,S enter raw.
    d["i_PS_prob"] = float(_iXY_prob(P, S, r1))
    d["i_VS_prob"] = float(_iXY_prob(V, S, r1))
    return d


def ssit_boot(recs, B=3000, seed=2024, eps=0.05):
    rng = np.random.default_rng(seed)
    P, V, S, r1, item = _arrays(recs)
    items = np.unique(item)
    idx = {it: np.where(item == it)[0] for it in items}
    base = ssit(recs, eps)

    keys = ["b_P", "b_V", "b_S", "b_PV", "b_PS", "b_VS",
            "BIC_PV", "BIC_PS", "BIC_VS", "i_PV_prob", "i_PS_prob", "i_VS_prob"]
    draws = {k: [] for k in keys}
    for _ in range(B):
        pick = rng.choice(items, len(items), replace=True)
        ii = np.concatenate([idx[it] for it in pick])
        d = _bics(_coef(P[ii], V[ii], S[ii], r1[ii], item[ii]), eps)
        d["i_PV_prob"] = _iPV_prob(P[ii], V[ii], r1[ii])
        d["i_PS_prob"] = _iXY_prob(P[ii], S[ii], r1[ii])
        d["i_VS_prob"] = _iXY_prob(V[ii], S[ii], r1[ii])
        for k in keys:
            draws[k].append(d[k])

    out = {"n": int(len(item)), "n_items": int(items.size)}
    for k in keys:
        v = np.array([x for x in draws[k] if np.isfinite(x)], float)
        out[k] = dict(
            est=(None if not np.isfinite(base[k]) else round(float(base[k]), 4)),
            ci=([round(float(np.percentile(v, 2.5)), 3),
                 round(float(np.percentile(v, 97.5)), 3)] if len(v) >= 50 else None),
            p=(round(float(2 * min((v > 0).mean(), (v < 0).mean())), 4)
               if len(v) >= 50 else None))
    v = np.array([x for x in draws["b_PV"] if np.isfinite(x)], float)
    ip = np.array([x for x in draws["i_PV_prob"] if np.isfinite(x)], float)
    if len(v) >= 50:
        lo, hi = np.percentile(v, [2.5, 97.5])
        ilo, ihi = np.percentile(ip, [2.5, 97.5])
        out["AIA_reject_PV"] = bool((lo > 0 or hi < 0) and (ilo > 0 or ihi < 0))
        out["p_abs_iPV_gt_0.02"] = round(float((np.abs(ip) > 0.02).mean()), 3)
    else:
        out["AIA_reject_PV"] = out["p_abs_iPV_gt_0.02"] = None

    # same AIA test for the other two pairs, on the probability-scale DiD:
    # reject additive independence iff the 95% CI excludes 0.  A pair whose
    # CI straddles 0 is evidence *for* additive independence there -- the
    # interactions are selective, not universal.
    for pair, dk, bk in (("PS", "i_PS_prob", "b_PS"), ("VS", "i_VS_prob", "b_VS")):
        iv = np.array([x for x in draws[dk] if np.isfinite(x)], float)
        bv = np.array([x for x in draws[bk] if np.isfinite(x)], float)
        if len(iv) >= 50 and len(bv) >= 50:
            ilo, ihi = np.percentile(iv, [2.5, 97.5])
            blo, bhi = np.percentile(bv, [2.5, 97.5])
            out["AIA_reject_" + pair] = bool(
                (ilo > 0 or ihi < 0) and (blo > 0 or bhi < 0))
        else:
            out["AIA_reject_" + pair] = None
    return out, base


def _pair_did(recs):
    """Per-pair probability-scale P x V difference-in-differences d_k, from
    that pair's own cell means.  Returns (pids, d) over pairs that have all
    four (P,V) cells populated."""
    P, V, S, r1, item = _arrays(recs)
    pids, d = [], []
    for it in np.unique(item):
        m = item == it
        cells = {}
        ok = True
        for p in (0, 1):
            for v in (0, 1):
                sel = m & (P == p) & (V == v)
                if not sel.any():
                    ok = False
                    break
                cells[(p, v)] = r1[sel].mean()
            if not ok:
                break
        if ok:
            pids.append(it)
            d.append((cells[(1, 1)] - cells[(0, 1)])
                     - (cells[(1, 0)] - cells[(0, 0)]))
    return np.array(pids), np.array(d, float)


def ssit_wild_boot(recs, B=9999, seed=2024):
    """Wild cluster bootstrap CI for i_PV, clusters = pairs, Rademacher
    weights.  Recommended over the pairs bootstrap when the number of
    pairs (clusters) is small (<~40): the pairs bootstrap is
    anti-conservative there, the wild version is not.

    i_PV = mean_k d_k over pairs; i_PV* = ibar + (1/n) sum_k g_k (d_k - ibar),
    g_k in {-1,+1}.  Two-sided percentile CI on the recentred distribution,
    plus a wild-bootstrap p-value for H0: i_PV = 0.
    """
    rng = np.random.default_rng(seed)
    pids, d = _pair_did(recs)
    n = len(d)
    if n < 4:
        return {"i_PV": None, "ci": None, "p": None, "n_pairs": int(n)}
    ibar = float(d.mean())
    dev = d - ibar
    stats = np.empty(B)
    for b in range(B):
        g = rng.integers(0, 2, n) * 2 - 1          # Rademacher +/-1
        stats[b] = ibar + (g * dev).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    # p-value: impose H0 by centring at 0, compare |ibar| to the null dist
    null = np.array([(g * dev).mean() for g in
                     (rng.integers(0, 2, (B, n)) * 2 - 1)])
    p = float((np.abs(null) >= abs(ibar)).mean())
    return {"i_PV": round(ibar, 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(p, 4), "n_pairs": int(n)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = lambda x: 1 / (1 + np.exp(-x))
    true = dict(b_P=0.30, b_V=0.25, b_S=0.10, b_PV=0.22, b_PS=0.03, b_VS=-0.15)
    recs = []
    for k in range(200):
        delta = rng.normal(0.0, 1.2)
        for p in (0, 1):
            for v in (0, 1):
                for s in (0, 1):
                    pp = 2 * p - 1
                    L = (delta + true["b_P"] * pp + true["b_V"] * v + true["b_S"] * s
                         + true["b_PV"] * pp * v + true["b_PS"] * pp * s
                         + true["b_VS"] * v * s)
                    pr1 = int(rng.random() < sig(L))
                    recs.append(dict(P=p, V=v, S=s, pair_id=k,
                                     Y_bin=(pr1 if p == 1 else 1 - pr1)))
    out, base = ssit_boot(recs, B=1500)
    print("planted -> recovered (gold-free; marginals attenuate, interaction does not):")
    for k in ("b_P", "b_V", "b_S", "b_PV", "b_PS", "b_VS"):
        print(f"  {k:6} {true[k]:+.2f} -> {base[k]:+.3f}  CI {out[k]['ci']}  p={out[k]['p']}")
    print("  AIA reject (PxV):", out["AIA_reject_PV"],
          " i_PV(prob) =", out["i_PV_prob"]["est"], out["i_PV_prob"]["ci"])
