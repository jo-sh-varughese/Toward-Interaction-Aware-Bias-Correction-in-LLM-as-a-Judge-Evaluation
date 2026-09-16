"""
analyze_iasc.py
===============
Apply Interaction-Aware Swap Correction (iasc.py) to the real-judge pilot
(data/raw/judgments_*.jsonl) and check, GOLD-FREE, that it reproduces full
2x swap-averaging at 1x evaluation cost.

For each judge + pooled:
  * fit b_P, b_PV from the four (P,V) swap cells (difference-in-differences)
  * take the single-order slice (P == 1: target response shown in slot A)
  * report the target-pick rate  raw  ->  + IASC  vs  full swap-average
  * pair-cluster bootstrap 95% CI on the IASC-corrected rate, with the
    coefficients re-fit on each resample (calibration uncertainty included)

Output: results/iasc_pilot.json + a LaTeX table on stdout.  numpy only.
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np
from iasc import iasc_fit, iasc_correct, swap_average

RAW = Path("data/raw")
B = 3000


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


def _pair_rate(rs, P):
    """target-pick rate per pair from the P-slice, plus that pair's V."""
    pid = sorted({r["pair_id"] for r in rs if r["P"] == P})
    y, v = [], []
    for p in pid:
        sub = [r for r in rs if r["pair_id"] == p and r["P"] == P]
        yb = np.mean([r["Y_bin"] for r in sub])
        y.append(yb if P == 1 else 1 - yb)          # -> pick-target
        v.append(sub[0]["V"])
    return np.array(pid), np.array(y, float), np.array(v, float)


def analyse(rs, rng):
    fit = iasc_fit(rs)
    pid, y1, v1 = _pair_rate(rs, P=1)
    raw = float(y1.mean())
    cor = float(iasc_correct(y1, np.ones_like(y1), v1, fit).mean())
    sa = float(np.mean(list(swap_average(rs).values())))

    items = sorted({r["pair_id"] for r in rs})
    by = {it: [r for r in rs if r["pair_id"] == it] for it in items}
    draws = []
    for _ in range(B):
        pick = rng.choice(items, len(items), replace=True)
        bs = [r for it in pick for r in by[it]]
        ft = iasc_fit(bs)
        p, yy, vv = _pair_rate(bs, P=1)
        if len(yy) < 3:
            continue
        draws.append(float(iasc_correct(yy, np.ones_like(yy), vv, ft).mean()))
    draws = np.array(draws)
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return dict(n=len(rs), pairs=len(items), b_P=round(fit["b_P"], 3),
               b_PV=round(fit["b_PV"], 3), raw=round(raw, 3),
               iasc=round(cor, 3), iasc_ci=[round(c, 3) for c in ci],
               swap_avg=round(sa, 3))


def main():
    recs = load()
    judges = sorted({r["judge"] for r in recs})
    rng = np.random.default_rng(2024)
    res = {"Pooled": analyse(recs, rng)}
    for j in judges:
        res[j] = analyse([r for r in recs if r["judge"] == j], rng)

    Path("results").mkdir(exist_ok=True)
    Path("results/iasc_pilot.json").write_text(json.dumps(res, indent=2))

    print(f"{'Judge':<14}{'b_P':>7}{'b_PV':>8}{'raw':>8}{'+IASC':>8}"
          f"{'  IASC 95% CI':<20}{'swap-avg':>9}")
    print("-" * 76)
    for k, o in res.items():
        print(f"{k:<14}{o['b_P']:>+7.2f}{o['b_PV']:>+8.2f}{o['raw']:>8.2f}"
              f"{o['iasc']:>8.2f}  [{o['iasc_ci'][0]:.2f},{o['iasc_ci'][1]:.2f}]"
              f"      {o['swap_avg']:>6.2f}")

    print("\n--- LaTeX ---")
    print("\\begin{tabular}{lccccc}\n\\toprule")
    print("Judge & $\\hat b_P$ & $\\hat b_{PV}$ & raw & $+$IASC & swap-avg. "
          "(2$\\times$) \\\\\n\\midrule")
    for k, o in res.items():
        row = (f"{k} & ${o['b_P']:+.2f}$ & ${o['b_PV']:+.2f}$ & "
               f"${o['raw']:.2f}$ & ${o['iasc']:.2f}\\,[{o['iasc_ci'][0]:.2f},"
               f"{o['iasc_ci'][1]:.2f}]$ & ${o['swap_avg']:.2f}$ \\\\")
        print(row if k != "Pooled" else row.replace("Pooled", "\\textbf{Pooled}"))
    print("\\bottomrule\n\\end{tabular}")

    p = res["Pooled"]
    print(f"\nGold-free check: IASC on the single-order slice gives "
          f"{p['iasc']:.2f}, full 2x swap-averaging gives {p['swap_avg']:.2f} "
          f"(raw, uncorrected: {p['raw']:.2f}).")


if __name__ == "__main__":
    main()
