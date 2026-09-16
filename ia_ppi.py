"""
ia_ppi.py
=========
Reference implementation of Interaction-Aware Prediction-Powered Inference
(IA-PPI) for debiased pairwise win-rate estimation with LLM judges.

Given
  - a large evaluation set with judge labels  Yhat in {0,1}  and bias
    indicators (P, V[, S]) but NO gold labels, and
  - a small calibration set with BOTH gold labels H in {0,1} and judge
    labels Yhat and indicators,
IA-PPI returns a point estimate of the quality-only win rate theta plus an
asymptotically valid confidence interval.

Three estimators are provided for comparison:
  calibration_only : mean(H) on the calibration set          (unbiased, wide)
  marginal_ppi     : PPI with rectifier mu(Yhat)              (Angelopoulos+ 2023)
  ia_ppi           : PPI with factorial rectifier mu(Yhat,P,V,PV),
                     rectifier residuals re-weighted to the evaluation
                     set's bias-cell composition.

The factorial rectifier is what Proposition 2 of the paper shows is
necessary: a marginal rectifier under-covers whenever judge bias makes
E[H | Yhat, cell] cell-dependent and the calibration / evaluation sets
differ in cell composition.

No third-party dependencies beyond numpy.
"""
from __future__ import annotations
import numpy as np

Z95 = 1.959963984540054


def _cells(P, V):
    """Map (P,V) in {0,1}^2 to a cell index 0..3."""
    return (np.asarray(P).astype(int) << 1) | np.asarray(V).astype(int)


def _cellmeans(y, h, cell, n_cells=8):
    """Return per-(cell,yhat) mean of h, with global-mean fallback for
    empty strata."""
    key = cell * 2 + y.astype(int)
    out = np.full(n_cells, np.nan)
    for k in range(n_cells):
        m = key == k
        if m.any():
            out[k] = h[m].mean()
    glob = h.mean()
    out[np.isnan(out)] = glob
    return out, key


def calibration_only(h_cal):
    h = np.asarray(h_cal, float)
    est = h.mean()
    se = h.std(ddof=1) / np.sqrt(len(h))
    return est, (est - Z95 * se, est + Z95 * se), se


def marginal_ppi(yhat_eval, yhat_cal, h_cal):
    """Plain PPI (lambda = 1) with rectifier mu(yhat) = E[H | yhat] on cal."""
    ye, yc, hc = map(np.asarray, (yhat_eval, yhat_cal, h_cal))
    mu1 = hc[yc == 1].mean() if (yc == 1).any() else hc.mean()
    mu0 = hc[yc == 0].mean() if (yc == 0).any() else hc.mean()
    mu = lambda y: np.where(y == 1, mu1, mu0)
    imp_eval = mu(ye)
    rect = hc - mu(yc)
    est = imp_eval.mean() + rect.mean()
    var = imp_eval.var(ddof=1) / len(ye) + rect.var(ddof=1) / len(yc)
    se = np.sqrt(var)
    return est, (est - Z95 * se, est + Z95 * se), se


def ia_ppi(yhat_eval, P_eval, V_eval, yhat_cal, P_cal, V_cal, h_cal):
    """IA-PPI with factorial rectifier mu(yhat, P, V, PV), rectifier term
    re-weighted to the evaluation set's (P,V) cell frequencies."""
    ye, Pe, Ve = map(np.asarray, (yhat_eval, P_eval, V_eval))
    yc, Pc, Vc, hc = map(np.asarray, (yhat_cal, P_cal, V_cal, h_cal))
    ce, cc = _cells(Pe, Ve), _cells(Pc, Vc)
    mu_tab, key_c = _cellmeans(yc, hc, cc)          # 8-vector, indexed cell*2+y
    key_e = ce * 2 + ye.astype(int)
    imp_eval = mu_tab[key_e]                        # mu(Yhat,cell) on eval
    resid_c = hc - mu_tab[key_c]                    # rectifier residuals on cal

    est = imp_eval.mean()
    var = imp_eval.var(ddof=1) / len(ye)
    # re-weighted rectifier: sum_c pi'_c * mean_{cal in c} resid
    for c in range(4):
        me, mc = ce == c, cc == c
        pi_e = me.mean()
        if pi_e == 0:
            continue
        if mc.any():
            r = resid_c[mc]
            est += pi_e * r.mean()
            var += (pi_e ** 2) * r.var(ddof=1) / mc.sum()
        # else: no cal data in a cell the eval set needs -> rectifier can't
        #       correct it; caller should ensure all four cells are populated.
    se = np.sqrt(var)
    return est, (est - Z95 * se, est + Z95 * se), se


def bic(beta_i, beta_j, beta_ij, eps=0.10):
    if abs(beta_i) < eps or abs(beta_j) < eps:
        return np.nan
    return beta_ij / np.sqrt(abs(beta_i) * abs(beta_j))


if __name__ == "__main__":
    # tiny smoke test
    rng = np.random.default_rng(0)
    n = 40000
    delta = rng.normal(0.1, 0.8, n)
    P = rng.integers(0, 2, n); V = rng.integers(0, 2, n)
    H = rng.binomial(1, 1 / (1 + np.exp(-delta)))
    bias = 0.5 * P + 0.55 * V + 0.7 * P * V
    Y = rng.binomial(1, 1 / (1 + np.exp(-(delta + bias))))
    theta = (1 / (1 + np.exp(-rng.normal(0.1, 0.8, 2_000_000)))).mean()
    cal = slice(0, 800)
    print(f"true theta ~ {theta:.4f}")
    print("cal-only   ", calibration_only(H[cal])[:2])
    print("marginalPPI", marginal_ppi(Y[800:], Y[cal], H[cal])[:2])
    print("IA-PPI     ", ia_ppi(Y[800:], P[800:], V[800:], Y[cal], P[cal], V[cal], H[cal])[:2])
