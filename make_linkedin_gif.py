"""
make_linkedin_gif.py
====================
Animated GIF for a LinkedIn post: how a position x verbosity interaction
(measured on real judges, i_PV ~ 0.15) scrambles an AlpacaEval-style
leaderboard, why sequential correction can't fix it, and how joint
correction restores the true order.

Outputs linkedin_demo.gif.  matplotlib + numpy + pillow.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from itertools import combinations

sig = lambda x: 1 / (1 + np.exp(-x))

# --- model set / judge bias (same mechanism as sim_ranking_impact.py) ---
NAMES   = ["A", "B", "C", "D", "E", "F"]
THETA   = np.array([0.00, 0.07, 0.14, 0.21, 0.28, 0.35])   # true: F best ... A worst
VERBOSE = np.array([0, 0, 1, 1, 1, 1])   # A,B answer at the reference length;
                                         # C-F have a length gap with it
THETA_REF = 0.17
B_P, B_PV = -0.60, 0.60          # judge: a 2nd-position pull that VANISHES when
                                # the two answers differ in length  ->  i_PV ~ 0.15
bP_marg = B_P + 0.5 * B_PV       # what SC recovers (Prop. 1)
TRUE_ORDER = list(np.argsort(-THETA))

def scores(alpha, mode):
    s = -1.0
    raw = (THETA - THETA_REF) + alpha * (B_P * s + B_PV * s * VERBOSE)
    if mode == "raw":
        L = raw
    elif mode == "sc":                                      # subtract marginal position (a constant)
        L = raw - alpha * bP_marg * s
    else:                                                   # jbc: subtract s + V + s*V
        L = raw - alpha * (B_P * s + B_PV * s * VERBOSE)
    return sig(L)

def kendall(order):
    p = {m: k for k, m in enumerate(order)}
    t = {m: k for k, m in enumerate(TRUE_ORDER)}
    c = d = 0
    for a, b in combinations(range(6), 2):
        c += np.sign(p[a] - p[b]) == np.sign(t[a] - t[b])
        d += np.sign(p[a] - p[b]) != np.sign(t[a] - t[b])
    return (c - d) / (c + d)

# ---- storyboard: (label, mode_from, mode_to, alpha_from, alpha_to, n, hold) ----
# the seam is loss-less: jbc @ alpha=1  ==  raw @ alpha=0  (same bars, tau=+1).
ease = lambda f: f * f * (3 - 2 * f)                        # smoothstep
SEG = [
    ("An unbiased judge ranks 6 models correctly",              "raw", "raw", 0.0, 0.0, 10, True),
    ("A position×length interaction sets in  (î$_{PV}\\!\\to\\!0.15$)",
                                                                "raw", "raw", 0.0, 1.0, 22, False),
    ("The two weakest models now top the leaderboard",          "raw", "raw", 1.0, 1.0, 12, True),
    ("Standard fix: correct position, then length, separately", "raw", "sc",  1.0, 1.0, 10, False),
    ("A main-effects correction just shifts every score\n— the ranking does not move",
                                                                "sc",  "sc",  1.0, 1.0, 14, True),
    ("Estimate the biases jointly (with the interaction term)", "sc",  "jbc", 1.0, 1.0, 12, False),
    ("True order recovered",                                    "jbc", "jbc", 1.0, 1.0, 12, True),
]
frames = []
for label, m0, m1, a0, a1, n, _hold in SEG:
    for k in range(n):
        f = ease(k / max(n - 1, 1))
        a = a0 * (1 - f) + a1 * f
        blend = scores(a, m0) * (1 - f) + scores(a, m1) * f
        frames.append((label, blend))

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=96)
fig.subplots_adjust(left=0.13, right=0.79, top=0.80, bottom=0.12)
ypos = np.arange(6)[::-1]                                   # F on top
cmap = plt.cm.RdYlGn
colors = {m: cmap(0.12 + 0.76 * (TRUE_ORDER[::-1].index(m) / 5)) for m in range(6)}
bars = ax.barh(ypos, np.zeros(6), height=0.62,
               color=[colors[m] for m in range(6)], edgecolor="white", linewidth=1.2)
ax.axvline(0.5, color="#888", lw=1, ls=(0, (4, 4)), zorder=0)
ax.set_xlim(0.30, 0.90)
ax.set_ylim(-0.7, 5.7)
ax.set_yticks(ypos)
ax.set_yticklabels([f"Model {NAMES[m]}" for m in range(6)], fontsize=12)
ax.set_xlabel("measured win-rate vs. reference   (each pair judged once, "
              "position fixed)", fontsize=10.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
title = ax.text(0.0, 1.14, "", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom")
tau_box = ax.text(1.04, 0.5, "", transform=ax.transAxes, fontsize=15, ha="left",
                  va="center", fontweight="bold")
val_txt = [ax.text(0, y, "", va="center", ha="left", fontsize=11) for y in ypos]
foot = fig.text(0.13, 0.012, "true skill order:  F > E > D > C > B > A"
                "        (î$_{PV}$ = the position×length interaction; "
                "when it is non-zero, the two biases no longer compose)",
                fontsize=8.5, color="#777")

N = len(frames)
FADE = 6                                    # frames of title fade at the seam

def update(i):
    label, sc = frames[i]
    for b, m in zip(bars, range(6)):
        b.set_width(sc[m])
    order = list(np.argsort(-sc))
    tau = kendall(order)
    for t, y, m in zip(val_txt, ypos, range(6)):
        t.set_position((sc[m] + 0.008, y)); t.set_text(f"{sc[m]:.2f}")
    ta = min(i / FADE, (N - 1 - i) / FADE, 1.0)             # fade title across loop seam
    title.set_text(label); title.set_alpha(ta)
    col = "#1a7f37" if tau > 0.9 else ("#b35900" if tau > 0.4 else "#c0392b")
    tau_box.set_text(f"Kendall $\\tau$\n{tau:+.2f}")
    tau_box.set_color(col); tau_box.set_alpha(0.35 + 0.65 * ta)
    return list(bars) + [title, tau_box] + val_txt

anim = FuncAnimation(fig, update, frames=N, blit=False)
anim.save("linkedin_demo.gif", writer=PillowWriter(fps=14))
print(f"wrote linkedin_demo.gif  ({N} frames, ~{N/14:.1f}s loop)")
