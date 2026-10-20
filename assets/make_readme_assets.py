"""
make_readme_assets.py -- the README's light/dark SVG panels.

Typeset in STIX Two Text (SIL OFL), subset per panel and embedded as woff2,
so the panels render identically on GitHub in either theme.  Numbers are
read from results/tmlr_pilot.json (run analyze_tmlr.py first).

    pip install fonttools brotli matplotlib   # fonts are fetched on first run
    python assets/make_readme_assets.py
"""
from __future__ import annotations
import base64, io, json, urllib.request
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools import subset
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
FONTS = OUT / "fonts"
GF = "https://github.com/google/fonts/raw/main/ofl/stixtwotext/"
W = 1000

THEMES = {
    "light": dict(ink="#23224A", acc="#7A3D12", mut="#66648C", rule="#CBC9DD",
                  blue="#2F5D9E", fill="#F3F2F8"),
    "dark": dict(ink="#E7E6F6", acc="#D9975E", mut="#A3A1C6", rule="#4A4870",
                 blue="#8FB0E8", fill="#2A2944"),
}


# ------------------------------------------------------------------ fonts
def fonts():
    FONTS.mkdir(parents=True, exist_ok=True)
    spec = {"r": ("STIXTwoText%5Bwght%5D.ttf", 400),
            "i": ("STIXTwoText-Italic%5Bwght%5D.ttf", 400),
            "b": ("STIXTwoText%5Bwght%5D.ttf", 600),
            "bi": ("STIXTwoText-Italic%5Bwght%5D.ttf", 700)}
    out = {}
    for k, (src, w) in spec.items():
        p = FONTS / f"stix-{k}.ttf"
        if not p.exists():
            var = FONTS / src.replace("%5B", "[").replace("%5D", "]")
            if not var.exists():
                urllib.request.urlretrieve(GF + src, var)
            instancer.instantiateVariableFont(TTFont(var), {"wght": w}).save(p)
        out[k] = p
    return out


FONT_PATHS = fonts()
_TT = {k: TTFont(p) for k, p in FONT_PATHS.items()}


def width(text, cls, size):
    f = _TT[cls]
    cmap, hmtx, upm = f.getBestCmap(), f["hmtx"], f["head"].unitsPerEm
    text = plain(text)
    return sum(hmtx[cmap.get(ord(ch), cmap[ord("?")])][0] for ch in text) * size / upm


def wrap(text, cls, size, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if width(t, cls, size) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


import re as _re
_MK = _re.compile(r"([_^])\{([^}]*)\}")


def plain(s):
    """Text with sub/superscript markup removed (for measuring and subsetting)."""
    return _MK.sub(lambda m: m.group(2), s)


def spans(s, size):
    """Render _{x} / ^{x} markup as tspans with a baseline shift."""
    out, pos, small = [], 0, round(size * 0.68, 1)
    for m in _MK.finditer(s):
        if m.start() > pos:
            out.append(f"<tspan>{esc(s[pos:m.start()])}</tspan>")
        dy = size * (0.28 if m.group(1) == "_" else -0.38)
        out.append(f'<tspan dy="{dy:.1f}" font-size="{small}">{esc(m.group(2))}</tspan>'
                   f'<tspan dy="{-dy:.1f}"> </tspan>')
        pos = m.end()
    out.append(f"<tspan>{esc(s[pos:])}</tspan>")
    return "".join(out)


class Panel:
    def __init__(self, h, title):
        self.h, self.title, self.items, self.used = h, title, [], {k: set() for k in _TT}

    def text(self, x, y, s, cls="r", size=16, col="ink", anchor="start", extra=""):
        self.used[cls].update(plain(s) + " ")
        self.items.append(("t", x, y, s, cls, size, col, anchor, extra))

    def raw(self, svg):
        self.items.append(("raw", svg))

    def line(self, x1, y1, x2, y2, col="rule", w=1):
        self.items.append(("l", x1, y1, x2, y2, col, w))

    def rect(self, x, y, w, h, col="fill", stroke=None, rx=0):
        self.items.append(("rect", x, y, w, h, col, stroke, rx))

    def para(self, x, y, s, maxw, cls="r", size=17, lh=27, col="ink"):
        for i, ln in enumerate(wrap(s, cls, size, maxw)):
            self.text(x, y + i * lh, ln, cls, size, col)
        return y + len(wrap(s, cls, size, maxw)) * lh

    def render(self, theme):
        c = THEMES[theme]
        css = []
        for k, chars in self.used.items():
            if not chars:
                continue
            opts = subset.Options()
            opts.flavor = "woff2"
            opts.layout_features = ["kern", "liga", "lnum", "pnum"]
            f = TTFont(FONT_PATHS[k])
            sub = subset.Subsetter(opts)
            sub.populate(text="".join(sorted(chars)) + "?")
            sub.subset(f)
            buf = io.BytesIO()
            f.flavor = "woff2"
            f.save(buf)
            b64 = base64.b64encode(buf.getvalue()).decode()
            css.append(f"@font-face{{font-family:'X{k}';src:url(data:font/woff2;base64,{b64}) "
                       f"format('woff2');}}\n.{k}{{font-family:'X{k}','STIX Two Text',Georgia,serif;"
                       f"font-feature-settings:'kern','liga','lnum','pnum';}}")
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {self.h}" '
               f'width="{W}" height="{self.h}" role="img"><title>{esc(self.title)}</title>'
               f'<defs><style>{chr(10).join(css)}</style></defs>']
        for it in self.items:
            if it[0] == "t":
                _, x, y, s, cls, size, col, anchor, extra = it
                out.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" '
                           f'fill="{c[col]}" text-anchor="{anchor}" {extra} xml:space="preserve">{spans(s, size)}</text>')
            elif it[0] == "l":
                _, x1, y1, x2, y2, col, w = it
                out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                           f'stroke="{c[col]}" stroke-width="{w}"/>')
            elif it[0] == "rect":
                _, x, y, w, h, col, stroke, rx = it
                st = f' stroke="{c[stroke]}" stroke-width="1"' if stroke else ""
                out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                           f'fill="{c[col]}"{st}/>')
            else:
                s = it[1]
                for k, v in c.items():
                    s = s.replace("{" + k + "}", v)
                out.append(s)
        out.append("</svg>")
        return "\n".join(out)

    def save(self, name):
        for th in THEMES:
            (OUT / f"{name}-{th}.svg").write_text(self.render(th), encoding="utf-8")


def head(p, label, right, y=34):
    p.line(6, 8, 994, 8, "ink", 1.6)
    p.text(6, y, label, "r", 13, "acc", extra='letter-spacing="1.1"')
    p.text(994, y, right, "i", 14.5, "mut", "end")


# ----------------------------------------------------------------- panels
R = json.loads((ROOT / "results/tmlr_pilot.json").read_text())
PV = R["paraphrase_V"]
POOL = PV["Pooled"]["half"]


def plate():
    p = Panel(470, "Do LLM-Judge Biases Compose? A label-free, scale-aware test for "
                   "position-length interaction. Johan S. Varughese.")
    head(p, "PLATE  ·  LLM-AS-A-JUDGE EVALUATION", "Manuscript prepared for TMLR, 2026")
    p.text(6, 104, "Do LLM-Judge Biases Compose?", "r", 50, "ink")
    p.text(6, 146, "A Label-Free, Scale-Aware Test for Position–Length Interaction",
           "i", 25, "mut")
    p.text(6, 186, "Johan S. Varughese", "b", 17, "ink")
    p.line(6, 214, 994, 214)
    # the 2x2 swap design with pooled cell rates (half coding)
    rates = {(1, 0): 0.936, (0, 0): 0.937, (1, 1): 0.977, (0, 1): 0.931}
    x0, y0, cw, ch = 352, 272, 112, 60
    p.text(6, 250, "The swap design", "bi", 16, "acc")
    p.para(6, 276, "Each stimulus is judged in all four cells. Shown: pooled rate of "
           "choosing the target, 7 judges, ties scored ½.", 186, "i", 15, 21, "mut")
    p.text(x0 + cw / 2, y0 - 12, "target in slot A", "i", 14, "mut", "middle")
    p.text(x0 + cw * 1.5 + 8, y0 - 12, "target in slot B", "i", 14, "mut", "middle")
    for j, v in enumerate((0, 1)):
        yy = y0 + j * (ch + 8)
        p.text(x0 - 12, yy + ch / 2 + 5, ("lengths matched", "target lengthened")[j],
               "i", 14, "mut", "end")
        for i, pp in enumerate((1, 0)):
            xx = x0 + i * (cw + 8)
            p.rect(xx, yy, cw, ch, "fill", "rule", 3)
            p.text(xx + cw / 2, yy + ch / 2 + 11, f"{rates[(pp, v)]:.3f}", "r", 27,
                   "acc" if (pp, v) == (1, 1) else "ink", "middle")
    # equation
    ex = 640
    p.text(ex, 270, "Position effect when lengths match", "i", 15, "mut")
    p.text(ex, 298, "0.936 − 0.937 = −0.001", "r", 22, "ink")
    p.text(ex, 334, "Position effect when the target is longer", "i", 15, "mut")
    p.text(ex, 362, "0.977 − 0.931 = +0.046", "r", 22, "ink")
    p.line(ex, 380, 994, 380)
    p.text(ex, 412, "î_{PV} = " + f"+{POOL['i']:.3f}", "r", 24, "acc")
    p.text(ex + 170, 412, f"95% CI [{POOL['ci'][0]:.3f}, {POOL['ci'][1]:.3f}]", "i", 17, "mut")
    p.text(ex, 444, "No labels, no model of the judge.", "i", 15, "mut")
    p.save("plate")


def highlights():
    p = Panel(470, "Highlights of the paper.")
    head(p, "HIGHLIGHTS", "What the evidence in the paper shows")
    cells = [
        ("½ · i_{PV}", "the exact error any single-offset position correction leaves on "
         "the verbose-versus-terse contrast, for every judge and every link.", "Prop. 1"),
        ("+4.7 pts", f"pooled position×length interaction across seven judges, 95% CI "
         f"[{100 * POOL['ci'][0]:.1f}, {100 * POOL['ci'][1]:.1f}], using no human labels.",
         "Table 3"),
        ("−0.1 → +4.6", "position bias in win-rate points when lengths match, then "
         "when the target is lengthened.", "§6"),
        ("10 pts", "inflation of one judge's interaction from counting a tie as a "
         "vote for slot B. The identity is exact.", "Prop. 3"),
        ("12", "distinct stimuli behind the pilot's 20 pairs. All inference "
         "clusters on them, which removes every per-judge detection.", "App. A"),
        ("0 labels", "needed by SSIT. The quality gap of each stimulus cancels "
         "inside its own four swap cells.", "Prop. 2"),
    ]
    colw = (988 - 2 * 36) / 3
    for n, (big, txt, ref) in enumerate(cells):
        r, cidx = divmod(n, 3)
        x = 6 + cidx * (colw + 36)
        y = 92 + r * 200
        p.text(x, y, big, "r", 44, "acc", extra='letter-spacing="-0.5"')
        yy = p.para(x, y + 32, txt, colw - 8, "r", 16, 22)
        p.text(x, yy + 4, ref, "i", 13.5, "mut")
        if cidx:
            p.line(x - 18, y - 40, x - 18, y + 112)
    p.save("highlights")


def abstract():
    txt = ("Pipelines that use large language models as pairwise judges remove "
           "position bias and length bias in separate steps, assuming the two add. "
           "I show that this assumption matters, can be tested without human labels, "
           "and is easy to test wrongly. On the win-rate scale, any correction that "
           "applies one position offset across length conditions misstates the "
           "verbose-versus-terse contrast by exactly half the position×length "
           "interaction; swap-averaging and stratified correction are exact. The "
           "Swap-Symmetry Interaction Test estimates that interaction from the four "
           "swap cells of each stimulus as a difference of means. Two artefacts can "
           "manufacture what it measures: coding a tie as a vote for slot B, which "
           "breaks swap symmetry, and link curvature near the ceiling. Corrected for "
           "both, seven judges from six families show no individually significant "
           "interaction, but a pooled one survives: +4.7 win-rate points, with "
           "position bias absent at matched length and present only when the target "
           "is longer.")
    p = Panel(400, "Abstract.")
    lines = wrap("Abstract. " + txt, "r", 19, 680)
    y = 26
    for i, ln in enumerate(lines):
        if i == 0:
            p.text(6, y, "Abstract.", "bi", 19, "ink")
            p.text(6 + width("Abstract. ", "bi", 19), y, ln[len("Abstract. "):], "r", 19, "ink")
        else:
            p.text(6, y, ln, "r", 19, "ink")
        y += 30
    notes = [(116, "Proposition 1. Exact because it concerns cell means only; the "
                   "log-odds version fails by non-collapsibility."),
             (266, "The tie artefact appeared in this project's own first analysis "
                   "and is corrected here."),
             (356, "95% CI [1.6, 7.8]; sign-flip p = 0.012 over 12 distinct stimuli.")]
    for (yy, s), k in zip(notes, "123"):
        p.text(712, yy - 5, k, "r", 9.3, "acc")
        p.para(722, yy, s, 270, "i", 15, 21, "mut")
    p.h = max(y, 380) + 10
    p.save("abstract")


def theory():
    p = Panel(560, "Four results.")
    head(p, "§3–§5  ·  FOUR RESULTS", "Stated on the scale that pipelines report")
    rows = [
        ("Proposition 1", "Additive corrections",
         "e(1) − e(0) = ½ i_{PV}",
         "Subtracting one position offset c from a single-order win rate leaves error "
         "e(v) = ½Δ(v) − c. Whatever c is, the verbose-versus-terse contrast is off by "
         "exactly half the interaction. Swap-averaging and length-stratified "
         "correction are exact."),
        ("Proposition 2", "Design-based identification",
         "E[d̂_{k}] = i_{k}",
         "Judging stimulus k in all four (slot, length) cells makes its "
         "difference-in-differences unbiased for its interaction. The stimulus's "
         "unknown quality gap enters every cell and cancels."),
        ("Proposition 3", "Tie coding",
         "î^{slot} = î^{strict} + t_{00} − t_{01}",
         "Counting a tie as a vote for slot B is not swap-equivariant. It adds the "
         "fall in tie rate under lengthening to the estimate, and it makes a "
         "position-invariant judge look biased toward slot B."),
        ("Proposition 4", "Curvature",
         "i^{0} ≈ 2β_{P} E[σ′(δ+β_{V}) − σ′(δ)]",
         "An additive log-odds judge still shows win-rate interaction near the "
         "ceiling, but only through a position main effect. We estimate this "
         "baseline per judge and test the excess against it."),
    ]
    y = 78
    for lab, name, eq, txt in rows:
        p.text(6, y, lab, "bi", 16, "acc")
        p.text(6, y + 22, name, "i", 15, "mut")
        p.text(200, y, eq, "r", 19, "ink")
        yy = p.para(470, y, txt, 524, "r", 15.5, 21)
        y = max(yy, y + 50) + 22
        p.line(6, y - 26, 994, y - 26)
    p.h = y - 10
    p.save("theory")


def table():
    names = [("gpt-oss-20b", "gpt-oss-20b"), ("gpt-oss-120b", "gpt-oss-120b"),
             ("gemma-3-27b", "gemma"), ("groq-qwen", "qwen3-27b"),
             ("groq-allam", "allam-2-7b"), ("mistral-small", "mistral-small"),
             ("cohere-command-r", "command-r"), ("Pooled", "Pooled, 7 judges")]
    p = Panel(470, "Table: per-judge interaction under slot-coded and swap-equivariant ties.")
    head(p, "TABLE 3  ·  SEVEN JUDGES, SIX FAMILIES",
         "Paraphrase manipulation, 12 distinct stimuli")
    cols = [(6, "Judge", "start"), (300, "ties", "end"), (420, "slot-coded î_{PV}", "end"),
            (640, "ties scored ½: î_{PV}  [95% CI]", "end"), (740, "p", "end"),
            (860, "curvature i^{0}", "end"), (994, "log-odds β̂_{PV}", "end")]
    for x, h, a in cols:
        p.text(x, 72, h, "i", 14.5, "mut", a)
    p.line(6, 84, 994, 84)
    y = 112
    for key, lab in names:
        o = PV[key]
        h, s = o["half"], o["legacy"]
        sc = h.get("scale")
        pooled = key == "Pooled"
        if pooled:
            p.line(6, y - 20, 994, y - 20)
            y += 6
        col = "acc" if pooled else "ink"
        vals = [lab, f"{o['tie_rate']:.2f}", f"{s['i']:+.3f}",
                (f"{h['i']:+.3f}  [{h['ci'][0]:+.2f}, {h['ci'][1]:+.2f}]"
                 if h["sd"] else "0  (always picks target)"),
                f"{h['p']:.3f}" if h["sd"] else "—",
                f"{sc['curv_baseline']:+.3f}" if sc else "—",
                f"{sc['bPV']:+.2f}" if sc else "—"]
        vals = [vals[0]] + [v.replace("-", "−") for v in vals[1:]]
        for (x, _, a), v in zip(cols, vals):
            p.text(x, y, v, "r", 16, col if x > 6 else ("acc" if pooled else "ink"), a)
        y += 32
    p.line(6, y - 18, 994, y - 18, "ink", 1.2)
    y = p.para(6, y + 8, "Slot coding inflates every judge that ties. Once ties are scored "
               "½ and inference clusters on stimuli, no judge is individually significant; "
               "the pooled interaction excludes zero (sign-flip p = 0.012) and clears its "
               "curvature baseline (p = 0.009).", 988, "i", 15, 21, "mut")
    p.h = y + 6
    p.save("table")


def links():
    rows = [("paper", "Paper", "tmlr/paper_tmlr.pdf", "TMLR submission build, 13 pages"),
            ("code", "Code", "analyze_tmlr.py", "reproduces every pilot number"),
            ("web", "Website", "johansv.netlify.app", ""),
            ("mail", "Email", "johan.varughesee@gmail.com", "")]
    for key, lab, val, note in rows:
        p = Panel(38, f"{lab}: {val}")
        p.text(6, 26, lab, "i", 18.5, "acc")
        p.text(170, 26, val, "r", 19, "ink", extra='text-decoration="underline"')
        if note:
            p.text(994, 26, note, "i", 15, "mut", "end")
        p.save(f"link-{key}")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for k in ("r", "i"):
        font_manager.fontManager.addfont(str(FONT_PATHS[k]))
    fam = TTFont(FONT_PATHS["r"])["name"].getDebugName(1)
    rows = ["gpt-oss-20b", "gpt-oss-120b", "gemma-3-27b", "groq-qwen", "groq-allam",
            "mistral-small", "cohere-command-r", "Pooled"]
    lab = dict(zip(rows, ["gpt-oss-20b", "gpt-oss-120b", "gemma", "qwen3-27b",
                          "allam-2-7b", "mistral-small", "command-r", "Pooled, 7 judges"]))
    for th, c in THEMES.items():
        plt.rcParams.update({"font.family": fam, "font.size": 13, "svg.fonttype": "path"})
        fig, ax = plt.subplots(figsize=(10, 4.6))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.axvline(0, color=c["mut"], lw=0.9)
        for y, k in enumerate(rows[::-1]):
            o = PV[k]
            for cd, col, mk, dy, face in (("legacy", c["acc"], "s", 0.17, "none"),
                                          ("half", c["blue"], "o", -0.17, c["blue"])):
                s = o[cd]
                ax.plot([100 * v for v in s["ci"]], [y + dy] * 2, color=col, lw=2.2,
                        solid_capstyle="round")
                ax.plot(100 * s["i"], y + dy, marker=mk, ms=7, mfc=face, mec=col, mew=1.8)
            if "scale" in o["half"]:
                cb = 100 * o["half"]["scale"]["curv_baseline"]
                ax.plot([cb, cb], [y - 0.38, y + 0.02], color=c["ink"], lw=1.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab[k] for k in rows[::-1]], color=c["ink"])
        ax.set_xlim(-10, 32)
        ax.tick_params(colors=c["mut"], length=0)
        ax.xaxis.grid(True, color=c["rule"], lw=0.7)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("position × length interaction, percentage points of win rate",
                      color=c["mut"])
        h = [plt.Line2D([], [], color=c["acc"], marker="s", mfc="none", mew=1.8, lw=2.2,
                        label="tie counted for slot B"),
             plt.Line2D([], [], color=c["blue"], marker="o", lw=2.2, label="tie scored ½"),
             plt.Line2D([], [], color=c["ink"], marker="|", ms=11, lw=0, mew=1.6,
                        label="curvature-only baseline")]
        lg = ax.legend(handles=h, loc="lower right", frameon=False, ncol=1)
        for t in lg.get_texts():
            t.set_color(c["ink"])
        fig.tight_layout()
        buf = io.StringIO()
        fig.savefig(buf, format="svg", transparent=True)
        plt.close(fig)
        body = buf.getvalue()
        body = body[body.index("<svg"):]
        (OUT / f"_fig-{th}.svg").write_text(body, encoding="utf-8")
    # wrap with a caption panel
    import re
    for th in THEMES:
        body = (OUT / f"_fig-{th}.svg").read_text(encoding="utf-8")
        inner = re.sub(r"^<svg[^>]*>", "", body).rsplit("</svg>", 1)[0]
        inner = re.sub(r"<metadata>.*?</metadata>", "", inner, flags=re.S)
        vb = re.search(r'viewBox="([^"]+)"', body).group(1).split()
        sc = 988 / float(vb[2])
        fh = float(vb[3]) * sc
        p = Panel(int(fh + 120), "Figure 1. Per-judge position-length interaction "
                                 "under two tie codings, with curvature baselines.")
        head(p, "FIGURE 1", "Ties, curvature, and what survives")
        p.raw(f'<g transform="translate(6,46) scale({sc:.4f})">{inner}</g>')
        y = p.para(6, fh + 70, "Figure 1. Each judge's interaction with 95% intervals "
                   "over 12 stimuli. Squares count a tie for slot B; circles score it ½. "
                   "Ticks mark the interaction an additive log-odds judge would show "
                   "through curvature alone.", 988, "i", 15, 21, "mut")
        p.h = int(y + 4)
        (OUT / f"figure-{th}.svg").write_text(p.render(th), encoding="utf-8")
        (OUT / f"_fig-{th}.svg").unlink()


if __name__ == "__main__":
    plate(); highlights(); abstract(); theory(); table(); figure(); links()
    print(sorted(x.name for x in OUT.glob("*.svg")))
