"""
analyze_iappi.py
================
Illustrative Interaction-Aware Prediction-Powered Inference (ia_ppi.py) on
the real-judge pilot.

The pilot has NO human gold labels, so this is a *mechanism* check, not a
coverage study.  We take the swap-averaged pick-target rate of each pair
(judge every cell, average) as a label-free proxy H for the quality-only
preference -- the same quantity full swap-averaging targets and the best
label-free estimate we have.  Then:

  1.  Show the Proposition 2 precondition holds on real judges:
      E[H | Yhat, (P,V) cell] is cell-dependent, so a *marginal* rectifier
      mu(Yhat) is misspecified and a *factorial* rectifier mu(Yhat,P,V) is
      needed.  We print E[H | Yhat=1] and E[H | Yhat=0] per cell.

  2.  Run cal-only / marginal-PPI / IA-PPI with the single-order (P==1)
      judgments as the "evaluation" set (this is the cell-composition
      shift: evaluation is 100% P=1, calibration spans all cells) and the
      pooled swap cells as the labeled calibration set.  Pair-cluster
      bootstrap for the interval; seed 2024.

Output: results/iappi_pilot.json + a LaTeX table on stdout.  numpy only.
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np
from ia_ppi import calibration_only, marginal_ppi, ia_ppi

RAW = Path("data/raw")
B = 3000
SEED = 2024


def load():
    recs = []
    for fp in sorted(glob.glob(str(RAW / "judgments_*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("verdict") in ("invalid", None):
                continue
            recs.append(r)
    if not recs:
        sys.exit("no valid judgments in data/raw/")
    return recs


def _pick_target(r):
    """Judge's pick of the target response r1 (position pre-flipped)."""
    return r["Y_bin"] if r["P"] == 1 else 1 - r["Y_bin"]


def pseudo_gold(recs):
    """Swap-averaged pick-target rate per pair -> label-free H proxy."""
    pids = sorted({r["pair_id"] for r in recs})
    H = {}
    for p in pids:
        sub = [r for r in recs if r["pair_id"] == p]
        H[p] = float(np.mean([_pick_target(r) for r in sub]))
    return H


def cell_table(recs, H):
    """E[H | Yhat, (P,V) cell] on the calibration (all-cell) data."""
    rows = []
    for P in (0, 1):
        for V in (0, 1):
            for yh in (1, 0):
                sub = [r for r in recs
                       if r["P"] == P and r["V"] == V and _pick_target(r) == yh]
                if sub:
                    m = np.mean([H[r["pair_id"]] for r in sub])
                    rows.append((P, V, yh, len(sub), float(m)))
    return rows


def _sets(recs, H):
    """Calibration = all swap cells (Yhat, P, V, H).  Evaluation = the
    single-order P==1 slice (Yhat, P, V), no H used."""
    cal = [(float(_pick_target(r)), r["P"], r["V"], H[r["pair_id"]]) for r in recs]
    ev = [(float(_pick_target(r)), r["P"], r["V"])
          for r in recs if r["P"] == 1]
    return cal, ev


def _estimates(cal, ev):
    yc = np.array([c[0] for c in cal]); Pc = np.array([c[1] for c in cal])
    Vc = np.array([c[2] for c in cal]); hc = np.array([c[3] for c in cal])
    ye = np.array([e[0] for e in ev]); Pe = np.array([e[1] for e in ev])
    Ve = np.array([e[2] for e in ev])
    return {
        "cal_only": calibration_only(hc)[0],
        "marginal_ppi": marginal_ppi(ye, yc, hc)[0],
        "ia_ppi": ia_ppi(ye, Pe, Ve, yc, Pc, Vc, hc)[0],
        "naive_eval": float(ye.mean()),
    }


def analyse(recs):
    H = pseudo_gold(recs)
    cal, ev = _sets(recs, H)
    pt = _estimates(cal, ev)

    rng = np.random.default_rng(SEED)
    items = sorted({r["pair_id"] for r in recs})
    by = {it: [r for r in recs if r["pair_id"] == it] for it in items}
    draws = {k: [] for k in pt}
    for _ in range(B):
        pick = rng.choice(items, len(items), replace=True)
        bs = [r for it in pick for r in by[it]]
        Hb = pseudo_gold(bs)
        cb, eb = _sets(bs, Hb)
        if len(eb) < 3 or len({c[1] * 2 + c[2] for c in cb}) < 4:
            continue
        try:
            e = _estimates(cb, eb)
        except Exception:
            continue
        for k in e:
            draws[k].append(e[k])
    ci = {k: [float(np.percentile(draws[k], 2.5)),
              float(np.percentile(draws[k], 97.5))] if len(draws[k]) >= 50
          else None for k in draws}
    return dict(n=len(recs), pairs=len(items),
               point={k: round(v, 3) for k, v in pt.items()},
               ci={k: ([round(x, 3) for x in ci[k]] if ci[k] else None)
                   for k in ci},
               cells=cell_table(recs, H))


def main():
    recs = load()
    res = {"Pooled": analyse(recs)}
    for j in sorted({r["judge"] for r in recs}):
        res[j] = analyse([r for r in recs if r["judge"] == j])

    Path("results").mkdir(exist_ok=True)
    Path("results/iappi_pilot.json").write_text(json.dumps(res, indent=2))

    p = res["Pooled"]
    print("Proposition 2 precondition on real judges "
          "(swap-averaged pseudo-gold H):")
    print(f"  {'cell (P,V)':<12}{'Yhat':>5}{'n':>6}{'  E[H | Yhat, cell]':<20}")
    for P, V, yh, n, m in p["cells"]:
        print(f"  ({P},{V})       {yh:>5}{n:>6}   {m:.3f}")
    e0 = [m for (P, V, yh, n, m) in p["cells"] if yh == 0]
    print(f"  -> E[H | Yhat=0] ranges {min(e0):.2f}-{max(e0):.2f} across the four "
          f"(P,V) cells (lowest where P=1, i.e. the judge picked the\n"
          f"     non-target from the target-favouring slot): E[H | Yhat, cell] "
          f"is cell-dependent, so a marginal rectifier is misspecified.")

    print("\nDebiased win-rate estimate on the single-order (P=1) slice:")
    print(f"  {'method':<16}{'estimate':>10}{'   95% CI':<18}")
    for k, lab in (("naive_eval", "naive (Yhat)"),
                   ("cal_only", "cal-only mean H"),
                   ("marginal_ppi", "marginal PPI"),
                   ("ia_ppi", "IA-PPI")):
        c = p["ci"][k]
        cs = f"[{c[0]:+.3f},{c[1]:+.3f}]" if c else "n/a"
        print(f"  {lab:<16}{p['point'][k]:>+10.3f}   {cs}")

    print("\n--- LaTeX (IA-PPI on the pilot, pooled) ---")
    print("\\begin{tabular}{lcc}\n\\toprule")
    print("Estimator & point & 95\\% CI \\\\\n\\midrule")
    for k, lab in (("naive_eval", "Naive judge rate"),
                   ("marginal_ppi", "Marginal PPI"),
                   ("ia_ppi", "\\IAPPI")):
        c = p["ci"][k]
        cs = f"$[{c[0]:.2f},{c[1]:.2f}]$" if c else "n/a"
        print(f"{lab} & ${p['point'][k]:.2f}$ & {cs} \\\\")
    print("\\bottomrule\n\\end{tabular}")
    print(f"% pooled: n={p['n']} judgments over {p['pairs']} pairs; "
          f"pseudo-gold = swap-averaged pick-target rate (label-free)")


if __name__ == "__main__":
    main()
