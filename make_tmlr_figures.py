"""
make_tmlr_figures.py -- figures for paper_tmlr.tex from results/tmlr_pilot.json
(run analyze_tmlr.py first).  matplotlib only.

fig_forest.pdf : per-judge i_PV under slot-coded ("legacy") vs swap-equivariant
                 (tie = 1/2) outcome coding, wild-cluster 95% CIs over the 12
                 distinct stimuli, with the curvature-only baseline implied by
                 an additive-log-odds fit of each judge.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOT_LEGACY = "#eb6834"     # categorical slot 2 (orange)
SLOT_HALF = "#2a78d6"       # categorical slot 1 (blue)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

NAMES = {"gpt-oss-20b": "gpt-oss-20b", "gpt-oss-120b": "gpt-oss-120b",
         "gemma-3-27b": "gemma", "groq-qwen": "qwen3-27b",
         "groq-allam": "allam-2-7b", "mistral-small": "mistral-small",
         "cohere-command-r": "command-r", "Pooled": "Pooled (7 judges)"}


def main():
    res = json.loads(Path("results/tmlr_pilot.json").read_text())["paraphrase_V"]
    rows = [k for k in NAMES if k in res]
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    for y, k in enumerate(rows[::-1]):
        o = res[k]
        for cd, col, mk, dy, face in (("legacy", SLOT_LEGACY, "s", 0.17, "white"),
                                      ("half", SLOT_HALF, "o", -0.17, SLOT_HALF)):
            s = o[cd]
            ax.plot([100 * v for v in s["ci"]], [y + dy] * 2, color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(100 * s["i"], y + dy, marker=mk, ms=6, mfc=face, mec=col, mew=1.6,
                    color=col, zorder=3)
        if "scale" in o["half"]:
            c = 100 * o["half"]["scale"]["curv_baseline"]
            ax.plot([c, c], [y - 0.36, y + 0.02], color=INK, lw=1.4, zorder=4)
        if k == "Pooled":
            ax.axhspan(y - 0.5, y + 0.5, color=GRID, alpha=0.6, zorder=-1, lw=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAMES[k] for k in rows[::-1]])
    ax.set_xlabel(r"position$\times$length interaction $\hat\imath_{PV}$ (percentage points of win rate)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(-10, 32)
    h = [plt.Line2D([], [], color=SLOT_LEGACY, marker="s", mfc="white", mew=1.6, lw=2,
                    label="tie coded as slot B (legacy)"),
         plt.Line2D([], [], color=SLOT_HALF, marker="o", lw=2,
                    label=r"tie coded as $\frac{1}{2}$ (swap-equivariant)"),
         plt.Line2D([], [], color=INK, lw=1.4, marker="|", ms=9, linestyle="none",
                    label="curvature-only baseline")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3,
              frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    Path("tmlr/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("tmlr/figures/fig_forest.pdf")
    fig.savefig("tmlr/figures/fig_forest.png", dpi=200)


if __name__ == "__main__":
    main()
