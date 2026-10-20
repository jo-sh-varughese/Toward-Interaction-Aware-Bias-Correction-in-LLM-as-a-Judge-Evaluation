"""
analyze_tmlr.py
===============
Pilot analysis for the TMLR manuscript (paper_tmlr.tex).  Reproduces every
pilot number in the paper from data/raw (paraphrase V) and data/raw_padded
(padding-only V).  numpy only, seed 2024, ~1 min.

1. Tie coding.  Three codings of a TIE verdict on the "picked the target"
   scale:
     legacy : tie -> slot B (collect_free.py's Y_bin; the workshop version).
              NOT swap-equivariant: a tie is invariant under a swap, but
              this maps it to whichever response sits in slot B.
     half   : tie -> 1/2 (Y_star).  Swap-equivariant.  PRIMARY.
     strict : tie -> 0 (target must be named).  Swap-equivariant.
   Identity (exact, checked numerically below):
     i_PV(legacy) = i_PV(strict) + [t(0,0) - t(0,1)].

2. Clusters.  The stimulus file cycles 12 templates (make_stimulus_pairs.py
   uses i % len(templates)), so the 20 judged "pairs" are 12 distinct
   stimuli.  All inference clusters on the TEMPLATE: duplicate pairs are
   pooled as extra samples of the same stimulus.  With 12 clusters the
   sign-flip randomization test is enumerated exactly (2^12 assignments).

3. Scale.  i_PV is a probability-scale (win-rate) DiD.  A judge whose
   log-odds are additive can still show i_PV != 0 through curvature near
   the ceiling.  For each judge we fit the ADDITIVE model
       logit pi = delta_k + b_P (2P-1) + b_V V        (template effects)
   and report the curvature-only baseline i_PV^0 it implies, plus a
   parametric-bootstrap test of the observed i_PV against it; and the
   log-odds interaction b_PV from the same model with a b_PV (2P-1) V term,
   with a template-cluster bootstrap CI.

Output: results/tmlr_pilot.json and a printed summary.
"""
from __future__ import annotations
import itertools, json, glob
from pathlib import Path
import numpy as np

SEED = 2024
CODINGS = ("legacy", "half", "strict")
NBOOT = 4000          # parametric bootstrap (vectorised)
NBOOT_CLUSTER = 400   # template-cluster bootstrap refits for b_PV
CELLS4 = [(0, 0), (0, 1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- data
def templates():
    rows = [json.loads(l) for l in open("data/stimulus_pairs.jsonl", encoding="utf-8")]
    tid = {}
    for r in rows:
        tid.setdefault(r["prompt"], len(tid))
    return {r["pair_id"]: tid[r["prompt"]] for r in rows}


TPL = templates()


def load(d):
    recs = []
    for fp in sorted(glob.glob(f"{d}/judgments_*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("verdict") in ("invalid", None):
                continue
            r["tpl"] = TPL[r["pair_id"]]
            recs.append(r)
    return recs


def target_pick(r, coding):
    v = r["verdict"]
    if v == "tie":
        if coding == "legacy":
            return 0.0 if r["P"] == 1 else 1.0
        return 0.5 if coding == "half" else 0.0
    named_A = v == "A"
    return float(named_A if r["P"] == 1 else not named_A)


# ------------------------------------------------- design-based readout
def cell_means(rs, coding, A="P", Bf="V", key="tpl"):
    by = {}
    for r in rs:
        by.setdefault(r[key], {}).setdefault((r[A], r[Bf]), []).append(
            target_pick(r, coding))
    return {k: {c: float(np.mean(v)) for c, v in cc.items()}
            for k, cc in by.items() if len(cc) == 4}


def tie_means(rs, key="tpl"):
    by = {}
    for r in rs:
        by.setdefault(r[key], {}).setdefault((r["P"], r["V"]), []).append(
            float(r["verdict"] == "tie"))
    return {k: {c: float(np.mean(v)) for c, v in cc.items()}
            for k, cc in by.items() if len(cc) == 4}


def did(cm):
    return np.array([(c[(1, 1)] - c[(0, 1)]) - (c[(1, 0)] - c[(0, 0)])
                     for c in cm.values()], float)


def signflip_p(d):
    """Exact two-sided sign-flip p-value (all 2^n assignments, n <= 16)."""
    n = len(d)
    if n < 4 or np.allclose(d, 0):
        return 1.0
    G = np.array(list(itertools.product((-1, 1), repeat=n)), float)
    null = np.abs(G @ d) / n
    return float((null >= abs(d.mean()) - 1e-12).mean())


def wild_ci(d, rng, B=9999):
    m = d.mean()
    G = rng.integers(0, 2, (B, len(d))) * 2 - 1
    st = m + (G * (d - m)).mean(axis=1)
    return [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def summarise(rs, coding, rng, A="P", Bf="V"):
    d = did(cell_means(rs, coding, A, Bf))
    return dict(i=round(float(d.mean()), 4), ci=[round(x, 4) for x in wild_ci(d, rng)],
                p=round(signflip_p(d), 4), n_clusters=int(len(d)),
                sd=round(float(np.std(d, ddof=1)), 3))


def pe_by_v(rs, coding):
    cm = cell_means(rs, coding)
    a = np.array([[c[(1, 0)] - c[(0, 0)], c[(1, 1)] - c[(0, 1)]] for c in cm.values()])
    return [round(float(x), 4) for x in a.mean(0)]


# ------------------------------------------------- model-based readouts
_sig = lambda x: 1.0 / (1.0 + np.exp(-x))


def aggregate(rs, coding):
    """Template x cell table: arrays (nK, 4) of judgment counts and outcome
    sums, cells in CELLS4 order; only templates with all four cells."""
    by = {}
    for r in rs:
        by.setdefault(r["tpl"], {}).setdefault((r["P"], r["V"]), []).append(
            target_pick(r, coding))
    ks = [k for k in sorted(by) if len(by[k]) == 4]
    n = np.array([[len(by[k][c]) for c in CELLS4] for k in ks], float)
    y = np.array([[sum(by[k][c]) for c in CELLS4] for k in ks], float)
    return n, y


_S = np.array([2 * p - 1 for p, _ in CELLS4], float)
_V = np.array([v for _, v in CELLS4], float)


def fit_logit(n, y, inter, lam=0.05, lam_b=1 / 9, iters=100):
    """Binomial logit on the template x cell table: template fixed effects
    delta_k + shared b_P (2P-1) + b_V V [+ b_PV (2P-1) V].  y may hold
    half-counts (ties).  A weak ridge on delta_k keeps templates with all
    picks on one side finite; an N(0, 3^2) ridge on the slopes keeps them
    finite under quasi-separation (near-deterministic judges)."""
    nK = n.shape[0]
    X = np.column_stack([_S, _V] + ([_S * _V] if inter else []))   # (4, pb)
    pb = X.shape[1]
    d, b = np.zeros(nK), np.zeros(pb)
    for _ in range(iters):
        mu = _sig(d[:, None] + (X @ b)[None, :])
        w = n * mu * (1 - mu) + 1e-9
        r = y - n * mu
        g = np.concatenate([r.sum(1) - lam * d, (r @ X).sum(0) - lam_b * b])
        H = np.zeros((nK + pb, nK + pb))
        H[np.arange(nK), np.arange(nK)] = w.sum(1) + lam
        H[:nK, nK:] = w @ X
        H[nK:, :nK] = H[:nK, nK:].T
        H[nK:, nK:] = X.T @ np.diag(w.sum(0)) @ X + lam_b * np.eye(pb)
        step = np.linalg.solve(H, g)
        d += step[:nK]
        b += step[nK:]
        if np.max(np.abs(step)) < 1e-9:
            break
    return d, b


def _did_rates(q):
    """q: (..., nK, 4) cell rates -> mean over templates of the DiD."""
    return ((q[..., 3] - q[..., 1]) - (q[..., 2] - q[..., 0])).mean(-1)


def scale_checks(rs, coding, rng):
    n, y = aggregate(rs, coding)
    obs = float(_did_rates(y / n))
    d0, b0 = fit_logit(n, y, inter=False)
    pi0 = _sig(d0[:, None] + b0[0] * _S[None, :] + b0[1] * _V[None, :])
    base = float(_did_rates(pi0))
    # parametric bootstrap of the design-based statistic under the additive
    # fit (binary draws: conservative relative to the tie-softened outcome)
    sims = _did_rates(rng.binomial(n.astype(int), pi0, size=(NBOOT,) + n.shape) / n)
    p_curv = float(min(1.0, 2 * min((sims >= obs).mean(), (sims <= obs).mean())))
    # log-odds interaction, template-cluster bootstrap
    _, b1 = fit_logit(n, y, inter=True)
    bs = []
    for _ in range(NBOOT_CLUSTER):
        idx = rng.integers(0, n.shape[0], n.shape[0])
        bs.append(fit_logit(n[idx], y[idx], inter=True)[1][2])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(i_obs=round(obs, 4), curv_baseline=round(base, 4),
                excess=round(obs - base, 4), p_vs_curv=round(p_curv, 4),
                bP_add=round(float(b0[0]), 3), bV_add=round(float(b0[1]), 3),
                bPV=round(float(b1[2]), 3), bPV_ci=[round(float(lo), 3), round(float(hi), 3)])


# ---------------------------------------------------------------- IASC
def iasc_pilot(rs, coding):
    """Single-order (P=1) target-pick rate: raw, IASC-corrected per (template,V)
    with gold-free coefficients from the cell log-odds DiD, and 2x swap-avg."""
    eps = 0.5
    def lo(q):
        return np.log(q / (1 - q))
    rate = {}
    for c in CELLS4:
        y = [target_pick(r, coding) for r in rs if (r["P"], r["V"]) == c]
        rate[c] = (sum(y) + eps) / (len(y) + 2 * eps)
    bP = 0.5 * (lo(rate[(1, 0)]) - lo(rate[(0, 0)]))
    bPV = 0.5 * ((lo(rate[(1, 1)]) - lo(rate[(0, 1)])) - (lo(rate[(1, 0)]) - lo(rate[(0, 0)])))
    cm = cell_means(rs, coding)
    raw, cor, swap = [], [], []
    for c in cm.values():
        for v in (0, 1):
            q = np.clip(c[(1, v)], 1e-3, 1 - 1e-3)
            raw.append(c[(1, v)])
            cor.append(_sig(lo(q) - bP - bPV * v))
            swap.append(0.5 * (c[(1, v)] + c[(0, v)]))
    return dict(bP=round(float(bP), 3), bPV=round(float(bPV), 3),
                raw=round(float(np.mean(raw)), 3), iasc=round(float(np.mean(cor)), 3),
                swap=round(float(np.mean(swap)), 3))


# --------------------------------------------------------------- driver
def block(recs, rng, model_checks=True):
    judges = sorted({r["judge"] for r in recs})
    groups = {j: [r for r in recs if r["judge"] == j] for j in judges}
    groups["Pooled"] = recs
    groups["Pooled (excl. command-r)"] = [r for r in recs if r["judge"] != "cohere-command-r"]
    out = {}
    for g, rs in groups.items():
        tm = tie_means(rs)
        tc = {c: float(np.mean([v[c] for v in tm.values()])) for c in CELLS4}
        row = {"n_judgments": len(rs),
               "n_pairs": len({r["pair_id"] for r in rs}),
               "n_templates": len({r["tpl"] for r in rs}),
               "tie_rate": round(float(np.mean([r["verdict"] == "tie" for r in rs])), 3),
               "tie_by_cell": {f"{c[0]}{c[1]}": round(v, 3) for c, v in tc.items()},
               "t00_minus_t01": round(tc[(0, 0)] - tc[(0, 1)], 4)}
        for cd in CODINGS:
            row[cd] = summarise(rs, cd, rng)
            row[cd]["pe_by_v"] = pe_by_v(rs, cd)
            row[cd]["win_rate"] = round(float(np.mean([target_pick(r, cd) for r in rs])), 3)
        if model_checks and row["half"]["sd"] > 0:
            for cd in ("legacy", "half"):
                row[cd]["scale"] = scale_checks(rs, cd, rng)
        row["iasc_half"] = iasc_pilot(rs, "half")
        out[g] = row
    for cd in CODINGS:
        adj = holm(np.array([out[j][cd]["p"] for j in judges]))
        for j, a in zip(judges, adj):
            out[j][cd]["p_holm"] = round(float(a), 4)
    return out


def main():
    rng = np.random.default_rng(SEED)
    raw, pad = load("data/raw"), load("data/raw_padded")
    res = {"paraphrase_V": block(raw, rng), "padding_V": block(pad, rng)}

    gaps = []
    for rs in (raw, pad):
        for j in sorted({r["judge"] for r in rs}):
            sub = [r for r in rs if r["judge"] == j]
            L, S, T = (cell_means(sub, "legacy", key="pair_id"),
                       cell_means(sub, "strict", key="pair_id"), tie_means(sub, key="pair_id"))
            for k in L:
                dl = (L[k][(1, 1)] - L[k][(0, 1)]) - (L[k][(1, 0)] - L[k][(0, 0)])
                ds = (S[k][(1, 1)] - S[k][(0, 1)]) - (S[k][(1, 0)] - S[k][(0, 0)])
                gaps.append(dl - ds - (T[k][(0, 0)] - T[k][(0, 1)]))
    res["identity_max_abs_gap"] = float(np.max(np.abs(gaps)))

    fp = {}
    for name, rs in [("Pooled", raw)] + [(j, [r for r in raw if r["judge"] == j])
                                        for j in sorted({r["judge"] for r in raw})]:
        fp[name] = {A + Bf: {cd: summarise(rs, cd, rng, A, Bf) for cd in ("legacy", "half")}
                    for A, Bf in (("P", "V"), ("P", "S"), ("V", "S"))}
    res["factor_pairs"] = fp
    res["by_task"] = {}
    for t in sorted({r["task"] for r in raw}):
        rs = [r for r in raw if r["task"] == t]
        res["by_task"][t] = {"templates": len({r["tpl"] for r in rs}),
                             **{cd: dict(i=round(float(did(cell_means(rs, cd)).mean()), 4))
                                for cd in ("legacy", "half")}}

    Path("results").mkdir(exist_ok=True)
    Path("results/tmlr_pilot.json").write_text(json.dumps(res, indent=2))

    for blk in ("paraphrase_V", "padding_V"):
        print(f"\n===== {blk}")
        for g, o in res[blk].items():
            print(f"{g:26} J={o['n_judgments']:4} pairs={o['n_pairs']} tpl={o['n_templates']} "
                  f"tie={o['tie_rate']:.3f} t00-t01={o['t00_minus_t01']:+.3f}")
            for cd in CODINGS:
                s = o[cd]
                line = (f"   {cd:7} i={s['i']:+.3f} [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}] "
                        f"p={s['p']:.4f} holm={s.get('p_holm', '-')} sd={s['sd']} "
                        f"PE|V={s['pe_by_v']} win={s['win_rate']}")
                if "scale" in s:
                    c = s["scale"]
                    line += (f"\n            curv0={c['curv_baseline']:+.3f} excess={c['excess']:+.3f} "
                             f"p_curv={c['p_vs_curv']:.3f} bP={c['bP_add']:+.2f} bV={c['bV_add']:+.2f} "
                             f"bPV={c['bPV']:+.2f} {c['bPV_ci']}")
                print(line)
            print("   IASC(half):", o["iasc_half"])
    print("\nidentity max |gap| =", res["identity_max_abs_gap"])
    for n, o in fp.items():
        print(n, {k: (v["half"]["i"], v["half"]["ci"], v["half"]["p"]) for k, v in o.items()})
    print(json.dumps(res["by_task"]))


if __name__ == "__main__":
    main()
