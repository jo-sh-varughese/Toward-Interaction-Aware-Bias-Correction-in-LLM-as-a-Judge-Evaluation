<div align="center">

# ⚖️ Do LLM-Judge Biases Compose?

**A label-free test for the position × verbosity interaction (SSIT / IASC)**

[![Venue](https://img.shields.io/badge/NeurIPS_2026-Workshop_on_Evaluation_of_LLMs_as_Judges_(JUDGe)-181717?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/core_result-label--free_test-C5F135?style=for-the-badge)](#)
[![Pre-registered](https://img.shields.io/badge/full_study-pre--registered-C5F135?style=for-the-badge)](#-real-judge-validation-path)
[![License](https://img.shields.io/badge/license-MIT-C5F135?style=for-the-badge)](#-license)

</div>

---

> [!TIP]
> **TL;DR** — a *systems* methods paper: judge validity is about how a judge's error composes with the debiasing pipeline around it and the decisions its scores feed. LLM-judge pipelines correct position then verbosity bias *sequentially*, assuming the two add. This gives a label-free check of that assumption plus everything needed to use it.
> 1. **Which pipelines are exposed (Prop. 1 + Cor. 1):** a pipeline that judges once and *subtracts* a fitted position term leaves $\tfrac{1}{2}\beta_{PV}$; one that *log-odds swap-averages* removes it (up to a Jensen term); joint estimation is exact. Direct answer to "does swap-averaging fully debias?" — yes for P×V on the log-odds scale, no for the surface-form and self-preference pairs.
> 1b. **Downstream cost (§4):** an uncorrected `i_PV` = 0.15 inverts a constructed leaderboard (Kendall τ +0.20 → +1.0 with a working correction) and — as a loose upper bound — mislabels 26–44% of length-mismatched RLHF comparisons, in a direction that rewards verbosity.
> 2. **The check (SSIT):** a difference-in-differences over the four position×verbosity swap cells. Its win-rate-scale readout `i_PV` is a difference of four cell means, so it is *design-based* — no human labels, no model of the judge (`sim_ssit_robustness.py`: recovered to 0.004 across six decision rules, one with no link function). It is label-free but **not free** — a dedicated collection at ~8× the judgment cost of one evaluation plus a length manipulation.
> 3. **Sizing the check (`sim_ssit_power.py`):** power depends on the judge's per-pair effect-to-noise ratio, which you *observe* once you run the check. Generic rule: ~50–100 pairs at 2 samples/cell for a 0.15 interaction on a mid-range judge; fewer for a decisive one. `analyze_real.py` prints a design-based realized power per judge.
> 4. **A 1×-budget correction (IASC, Prop. 3):** a log-odds shrinkage — removes a fraction of each position-coupled term, sign preserved, `β_V` untouched. On the probability scale it can overshoot swap-averaging (Jensen). A heuristic floor, not a guaranteed win-rate improvement.
> 5. **Five-judge pilot + decision procedure + reviewer checklist.** SSIT on 5 open judges / 4 families. `i_PV` > 0 on **4 of 5** (+0.07 to +0.17; wild `p` ≤ 0.032), significant on 3; pooled +0.12 [+0.06, +0.18]. The 5th (command-r) picks the stronger response on every pair → SSIT correctly returns 0. **Padding-only re-test** on 2 judges (length-only `V`, `r1` verbatim): the interaction survives / grows (mistral +0.075 → +0.10, `p` 0.032 → 0.003) — a real position×length effect, not style. Remaining: frontier judge, n ≥ 50, all judges on padding-only = the pre-registered study.
>
> Simulation magnitudes elsewhere in this repo are illustrative, not measurements.

---

## ⚠️ Important: this is a simulation study

This repo's headline numbers ($\mathrm{BIC}_{PV}=+1.16$, the $4.3\times$ residual, etc.) come from a **controlled simulation calibrated to published empirical ranges** — they are **not** measurements of real deployed judges (GPT-4o, Gemini, Claude, etc.). The paper is explicit about this, and so is this README. What *is* established analytically (not just simulated) is **Proposition 1**: if a position×verbosity interaction exists at all, sequential correction cannot fully remove it. Whether real judges have such an interaction — and how large — is the open question this repo is built to answer. See [Real-Judge Validation Path](#-real-judge-validation-path).

---

## 📚 Table of Contents

- [Why this exists](#-why-this-exists)
- [Key findings (simulation)](#-key-findings-simulation)
- [The theory, in one box](#-the-theory-in-one-box)
- [BIC: the diagnostic](#-bic-the-diagnostic)
- [JBC: the correction algorithm](#️-jbc-the-correction-algorithm)
- [Results at a glance](#-results-at-a-glance)
- [Quick start](#-quick-start)
- [Repository structure](#-repository-structure)
- [Real-judge validation path](#-real-judge-validation-path)
- [Limitations](#-limitations)
- [Citation](#-citation)
- [License](#-license)

---

## 🤔 Why this exists

Every major LLM evaluation pipeline — MT-Bench, Chatbot Arena, AlpacaEval, RLHF reward modeling — uses an LLM to judge pairs of responses, and every one of them corrects for **position bias** (order effects), **verbosity bias** (longer = "better"), and **self-preference bias** (a model favoring its own outputs), each with its own well-published fix.

These three fixes are almost always applied **one after another**, implicitly assuming the biases are *additively independent* — that correcting for position doesn't change how verbosity behaves, and vice versa.

**Nobody has checked that assumption.** This paper does, in three parts:

1. An **analytical proof** that if position and verbosity biases interact ($\beta_{PV} \neq 0$), sequential correction structurally leaves a residual — no amount of better estimation fixes a *methodology* problem.
2. A **calibrated simulation** (8,000 judgments, 5 agent profiles, 4 task types) quantifying how big that residual could plausibly be.
3. A **ready-to-run, free, pre-registered pipeline** to find out if real judges actually behave this way.

---

## ✨ Key findings (simulation)

| | |
|---|---|
| 🧮 **Proposition 1 (analytical, not simulated)** | Sequential correction leaves residual log-odds $\Delta_{SC} = \tfrac{1}{2}\beta_{PV}$ whenever $\beta_{PV} \neq 0$ — proven via omitted-variable bias |
| 📈 **$\mathrm{BIC}_{PV} = +1.16$** | 95% CI $[+0.55, +2.15]$, $p<10^{-5}$ — position×verbosity interaction is positive (*amplifying*) across all 5 simulated agents and 4 task types |
| 📉 **$\mathrm{BIC}_{VS} = -1.26$** | 95% CI $[-1.62, -0.94]$, $p<10^{-5}$ — verbosity×self-preference interaction is negative (*suppressive*) |
| ➖ **$\mathrm{BIC}_{PS} = +0.27$** | $p=0.465$, n.s. — no evidence against treating position and self-preference corrections independently |
| 💰 **~3,900 / 100K** | Estimated mislabeled reward pairs in an RLHF corpus at $\mathrm{BIC}_{PV}\approx1.26$ ($4.3\times$ the null-condition residual) |
| 🎯 **Action threshold: $\lvert\mathrm{BIC}\rvert = 0.43$** | Below this, sequential correction appears adequate in simulation; above it, joint correction (JBC) may help |
| ⚠️ **JBC calibration cliff** | Below $N_{\text{cal}}=500$ calibration pairs, JBC *increases* residual bias relative to sequential correction. Above 500, it cuts residual bias by **27–49%** at $\mathrm{BIC}\geq1.0$ |

---

## 🧮 The theory, in one box

<details open>
<summary><b>Proposition 1 — Sequential Correction (SC) Residual Under Interaction</b></summary>

The judge's decision is modeled as a logistic regression over position ($P$), verbosity ($V$), self-preference ($S$), and their interactions:

$$\logit(P(Y{=}A)) = \gamma(Q) + \beta_P P + \beta_V V + \beta_S S + \beta_{PV}PV + \beta_{PS}PS + \beta_{VS}VS + \beta_{PVS}PVS + \epsilon$$

Sequential correction estimates $\hat\beta_P$ via marginal regression on $P$ alone, then $\hat\beta_V$ from position-adjusted data. Under a balanced $2{\times}2$ design with $\beta_{PV} \neq 0$:

$$\hat\beta_P^{\text{marg}} \xrightarrow{p} \beta_P + \tfrac{1}{2}\beta_{PV} \qquad \hat\beta_V^{(2)} \xrightarrow{p} \beta_V$$

The marginal estimate of $\beta_P$ silently absorbs half the interaction term. Subtracting these corrected effects leaves a residual log-odds in the $(P{=}1,V{=}1)$ cell:

$$\Delta_{SC} = \tfrac{1}{2}\beta_{PV} \;\neq\; 0$$

**In plain terms:** if position bias and verbosity bias *interact at all*, correcting for each one separately — in any order, however precisely — cannot fully remove the joint effect. Only **joint estimation** of $[\beta_P, \beta_V, \beta_{PV}]$ can. This is a proof, not a simulation result.
</details>

<details>
<summary><b>What the simulation adds</b></summary>

The proposition tells you *that* a residual exists if $\beta_{PV}\neq0$ — it doesn't tell you how big $\beta_{PV}$ is in practice, or whether the resulting residual is large enough to care about. The simulation (calibrated to published empirical ranges for position, verbosity, and self-preference bias across GPT-4o, Gemini, Claude, Llama-3-70B, and Mistral-style profiles) exists purely to put plausible numbers on that question — and to validate that the BIC/JBC pipeline correctly recovers known ground-truth parameters before pointing it at real judges.
</details>

---

## 📐 BIC: the diagnostic

$$\mathrm{BIC}_{ij} = \frac{\beta_{ij}}{\sqrt{\lvert\beta_i\rvert \cdot \lvert\beta_j\rvert}}$$

A dimensionless, cross-agent-comparable ratio:

- $\mathrm{BIC}_{ij} \approx 0$ → consistent with additive independence (sequential correction is fine)
- $\mathrm{BIC}_{ij} > 0$ → **amplifying** interaction
- $\mathrm{BIC}_{ij} < 0$ → **suppressive** interaction
- $\lvert\mathrm{BIC}_{ij}\rvert > 1$ → interaction exceeds the geometric mean of the two individual biases
- **n/a** when $\lvert\hat\beta_i\rvert < 0.10$ (too small to normalize meaningfully)

> [!NOTE]
> BIC has higher sampling variance than the raw $\beta_{ij}$ (delta method) — it's an **interpretive, ordinal indicator**, not a precision instrument. Bootstrap CIs in this study are typically $\pm1.0$ at $N{=}200$ pairs per cell.

---

## 🛠️ JBC: the correction algorithm

```text
1. Collect a calibration set 𝒞 of N_cal pairs spanning all 4 (P,V) ∈ {0,1}² conditions
2. Fit logit(P(Y=A)) ~ P + V + P·V on 𝒞  →  get β̂_P, β̂_V, β̂_PV
3. Compute BIC_PV from these estimates

If |BIC_PV| < 0.43:
    → sequential correction (SC) appears adequate — no change needed

Else:
    → for each evaluation pair i with (P_i, V_i):
          L_i* = logit(ŵ_i_raw) − β̂_P·P_i − β̂_V·V_i − β̂_PV·P_i·V_i
          ŵ_i* = σ(L_i*)        # jointly-corrected win probability
```

**Cost:** at the minimum recommended $N_{\text{cal}}=500$, that's $4 \times 500 = 2{,}000$ extra judge calls — a one-time cost (≤25% overhead on a 200-pair eval campaign) for any *recurring* evaluation setup.

---

## 📊 Results at a glance

**Pooled BIC estimates** (8,000 judgments, 5 agents × 4 task types, bootstrap $B{=}300$):

| Interaction | $\widehat{\mathrm{BIC}}$ | 95% CI | $p$ | Interpretation |
|---|---|---|---|---|
| Position × Verbosity | **+1.16** | $[+0.55, +2.15]$ | $<10^{-5}$ | Amplifying, consistent across all profiles |
| Verbosity × Self-Pref. | **−1.26** | $[-1.62, -0.94]$ | $<10^{-5}$ | Suppressive, narrowest CI of the three |
| Position × Self-Pref. | +0.27 | $[-0.36, +1.36]$ | 0.465 | No evidence of interaction |

**Regression fixed effects** (mixed-effects logistic, $N{=}7{,}376$, cluster-robust SEs):

| Predictor | $\hat\beta$ | $z$ | $p$ |
|---|---|---|---|
| $P$ (position) | +0.458 | +10.56 | $<0.001$ |
| $V$ (verbosity) | +0.707 | +8.25 | $<0.001$ |
| $S$ (self-pref.) | +0.304 | +3.84 | $<0.001$ |
| $P \times V$ | **+0.663** | +4.80 | $<0.001$ |
| $V \times S$ | **−0.582** | −5.40 | $<0.001$ |
| $P \times S$ | +0.099 | +0.95 | 0.342 |
| $P \times V \times S$ | +0.385 | +1.79 | 0.074 |

Both significant interaction terms recover their data-generating-process values ($\beta_{PV}^*=0.55$, $\beta_{VS}^*=-0.30$) closely — a pipeline-correctness check, not a real-world claim.

**JBC vs. sequential correction**, by calibration size (relative reduction in residual bias; 200 reps/cell):

| BIC level | $N_{\text{cal}}{=}100$ | $N_{\text{cal}}{=}500$ | $N_{\text{cal}}{=}2{,}000$ |
|---|---|---|---|
| 0.0 | −8% | −2% | +1% |
| 0.5 | −6% | −0% | +7%\* |
| 1.0 | −10% | **+10%**\*\* | **+25%**\*\*\* |
| 1.5 | −11% | **+27%**\*\*\* | **+47%**\*\*\* |
| 2.0 | +1% | **+44%**\*\*\* | **+49%**\*\*\* |

*Negative = JBC worse than SC. The flip from negative to strongly positive between $N_{\text{cal}}{=}100$ and $N_{\text{cal}}{=}500$ is the single most actionable finding for practitioners.*

---

## 🚀 Quick Start

### Reproduce the paper's simulations (numpy only, no keys, ~3 min)

```bash
pip install numpy scipy pandas
python sim_prop2_coverage.py      # regenerates the IA-PPI coverage table exactly
python ia_ppi.py                  # smoke-test the estimator
```

### Run the real-judge pilot (free-tier keys, ~2 h)

```bash
pip install numpy scipy groq

python make_stimulus_pairs.py --n 50 --out data/stimulus_pairs.jsonl   # no key

# set any subset (all free tier, no card except Anthropic):
export GROQ_API_KEY=...        # console.groq.com          -> gpt-oss
export GOOGLE_API_KEY=...      # aistudio.google.com       -> gemini, gemma
export CEREBRAS_API_KEY=...    # cloud.cerebras.ai         -> llama, qwen
export MISTRAL_API_KEY=...     # console.mistral.ai        -> mistral
export OPENROUTER_API_KEY=...  # openrouter.ai             -> many :free
export COHERE_API_KEY=...      # dashboard.cohere.com      -> command-r
export SAMBANOVA_API_KEY=...   # cloud.sambanova.ai        -> deepseek, qwen
export NVIDIA_API_KEY=...      # build.nvidia.com          -> nemotron, phi
export GITHUB_TOKEN=...        # github.com/marketplace/models -> phi, deepseek, jamba
export HF_TOKEN=...            # huggingface.co            -> granite, gemma
export TOGETHER_API_KEY=...    # api.together.xyz          -> llama (free endpoint)
# export ANTHROPIC_API_KEY=... # console.anthropic.com     (PAID, ~cents on Haiku)

python collect_free.py --list-models   # Groq's catalogue changes often
# pick judges across FAMILIES (20 configs available, 13 families):
python collect_free.py --judges gpt-oss-20b,sambanova-deepseek,nvidia-nemotron,github-phi,cohere-command-r --n 50
python analyze_real.py                  # -> results/real_bic.json + LaTeX table
```

Judge configs are in `FREE_JUDGES` in `collect_free.py`; adding a provider is
one config line + one 3-line REST caller (all OpenAI-compatible except Cohere,
Google, Anthropic).

`analyze_real.py` runs the **gold-free Swap-Symmetry Interaction Test**
(`ssit.py`): per-judge and pooled `i_PV` (probability-scale DiD) and `b_PV`
(log-odds) with pair-cluster bootstrap CIs, plus the pre-registered AIA
decision. No human labels required.

---

## 📁 Repository Structure

```
.
├── ssit.py                   # Swap-Symmetry Interaction Test (Prop. 2) + self-test
├── iasc.py                   # Interaction-Aware Swap Correction (Prop. 3), partial/label-free
├── ia_ppi.py                 # valid-CI variant: post-stratified PPI rectifier
├── sim_ssit_robustness.py    # SSIT i_PV vs true i_PV across 6 judge decision rules
├── sim_ssit_power.py         # power of the SSIT wild-cluster test vs n, samples/cell
├── sim_prop2_coverage.py     # PPI coverage under cell-composition shift (Prop. 4)
├── sim_ranking_impact.py     # constructed leaderboard demo + bias-magnitude sweep
├── analyze_real.py           # SSIT on the pilot: 3 factor pairs + by-task breakdown
├── analyze_iasc.py           # IASC on the pilot
├── analyze_iappi.py          # IA-PPI on the pilot (code check only)
├── collect_free.py           # multi-provider free-tier judgment collector (13 families)
├── make_stimulus_pairs.py    # build data/stimulus_pairs.jsonl
├── PANEL_EXPANSION.md        # runbook for the pre-registered full study
├── data/   raw/ (pilot judgments), stimulus_pairs.jsonl, raw_pilot_v1/ (archived)
├── results/  *.json outputs from the sims and analyses
└── paper_neurips_workshop.tex / paper_zenodo.tex / *.pdf
```

---

## 🔬 Real-judge validation path

A **pre-registered, zero-financial-cost** replication is ready to run against real judge APIs (Groq for Llama/Mixtral, Google AI Studio for Gemini — both free-tier):

- **50-pair pilot**: ~1,600 API calls, ~2 hours
- **Primary question**: does real-judge $\lvert\mathrm{BIC}_{PV}\rvert$ exceed the simulation-derived action threshold of $0.43$?
- **Either answer is informative** — a positive result would suggest AlpacaEval/MT-Bench-style sequential correction pipelines may warrant revision; a negative result would be the first empirical validation of sequential correction's adequacy.

> [!IMPORTANT]
> This is the part of the project most in need of contributors. If you have free-tier API access and ~2 hours, see `real_judge_pipeline/preregistration.md`.

---

## ⚠️ Limitations

- **All quantitative estimates are simulation-based.** Real-judge BIC magnitudes are unknown until the validation pipeline runs.
- **Wide confidence intervals** (~±1.0) at $N{=}200$ pairs/cell — BIC is an *ordinal* indicator at this scale, not a precise estimate.
- The $N_{\text{cal}}\geq500$ JBC threshold is itself simulation-derived and may shift for real judges with different noise profiles.
- Coverage limited to English text, 4 task categories, pairwise comparisons only.
- Mean inter-agent $\kappa = 0.060$ reflects deliberate profile heterogeneity by design — see `appendix/kappa_recovery.md` for what this implies about estimate reliability.

---

## 📖 Citation

```bibtex
@inproceedings{varughese2026compose,
  title     = {Do {LLM}-Judge Biases Compose? A Label-Free Test for the
               Position x Verbosity Interaction},
  author    = {Varughese, Johan S},
  booktitle = {NeurIPS 2026 Workshop on the Evaluation of LLMs as Judges (JUDGe)},
  year      = {2026},
  note      = {Non-archival. Code + data: see repository}
}
```

### Manuscript

- `paper_neurips_workshop.tex` → `paper_neurips_workshop.pdf` — **anonymous**
  submission build. Official `neurips_2026.sty`, `[dblblindworkshop]` option
  (double-blind, line numbers, "Submitted to NeurIPS 2026" footer).
- `paper_zenodo.tex` → `paper_zenodo.pdf` — identical content, `zenodobuild=1`
  selects `[dblblindworkshop,preprint]`: **attributed** (name + email), no line
  numbers, "Preprint" footer. For the Zenodo deposit.
- `neurips_2026.sty` — official style file. `neurips_2025.sty` (old drafting
  reimplementation) is unused and kept only for history.
- `paper_icmla_v4_archived.tex` — earlier IEEE-format version (BIC / JBC framing),
  kept for reference only.

**Structure.** A systems methods paper: judge-bias corrections are chained
assuming additivity; SSIT checks that label-free.
§3 which pipelines are exposed (Prop. 1 + Cor. 1) ·
§4 downstream cost (leaderboard τ, RLHF label-flip bound) ·
§5 SSIT — design-based `i_PV` (`sim_ssit_robustness.py`), §5.2 power analysis
(`sim_ssit_power.py`) ·
§6 IASC as a shrinkage correction ·
§7 worked example on 3 judges, read with a realized-power column ·
§8 decision procedure + deployment-disclosure checklist + pre-registered study.
PPI/coverage and the `BIC` diagnostic are in appendices.

### What the pilot found (gold-free)

`analyze_real.py` runs the **Swap-Symmetry Interaction Test** (`ssit.py`) — a
difference-in-differences over the four P×V swap cells that identifies the
interaction with **no human labels** (the per-pair quality gap is the
conditioned-out term).

| Judge | family | `i_PV` | wild 95% CI | wild `p` | rlz. power | AIA | `b_V` |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b | openai | +0.12 | [+0.01, +0.21] | 0.026 | 0.64 | yes | +0.5 |
| gpt-oss-120b | openai | +0.17 | [+0.04, +0.30] | 0.009 | 0.69 | yes | +1.5 |
| gemma | google | +0.16 | [+0.07, +0.26] | 0.0001 | 0.91 | yes | **−1.4** |
| mistral-small | mistral | +0.07 | [+0.01, +0.14] | 0.032 | 0.48 | no | +1.4 |
| command-r | cohere | +0.00 | [0, 0] | 1.0 | — | no | +0.0 |
| **Pooled (5)** | | **+0.09** | **[+0.04, +0.13]** | **≈0** | 0.97 | yes | |
| **Pooled (excl. command-r)** | | **+0.12** | **[+0.06, +0.18]** | **≈0** | — | yes | |

**4 of 5 judges show `i_PV` > 0** (+0.07 to +0.17), significant on 3, marginal
on mistral (`p` = 0.032). By Eq. (3) the detecting judges' `SD(d_k)` ≈ 0.2–0.3
gives 60–90% power at n ≈ 20 — the detections match the a-priori formula.
**command-r** picks the stronger response on all 20 pairs → SSIT correctly
returns 0 (the "bias fades on easy pairs" regime; the pre-registration fixes
this with quality-matched stimuli).

**Not a paraphrase artifact.** For the 2 judges re-collected with a
**padding-only `V`** (`r1` kept verbatim + redundant restatement — length only,
no rephrasing; `make_padded_v.py`, `data/raw_padded/`): mistral `i_PV` = +0.10
(`p` = 0.003, *up* from +0.075), command-r still 0. So `P×V` is a genuine
position×length interaction, not position×style.

**Selective across pairs, but weakly** (`analyze_real.py` tests all three).
Pooled `P×S` points the same way (+0.12 [+0.00, +0.24]) but its CI touches
zero, no single judge is significant, and the `S` manipulation swaps in the
judge's own generation — so part of it may be `P×(length)`. `V×S` (outside the
swap subspace) is null and sign-inconsistent. The *direction* matches IASC's
prediction; on 3 judges / ~20 pairs, with no multiplicity correction across the
three pairs, only `P×V` survives a conservative reading. By task, `i_PV` is
strongest on Math Reasoning, near-zero on open-ended QA, never sign-reversing.

`analyze_iappi.py` runs IA-PPI on the pilot as a **code check only** — no gold
labels, near-deterministic judges, so it cannot separate the estimators. The
coverage result is simulation-only.

### Does it move a leaderboard? (`sim_ranking_impact.py`)

An AlpacaEval-style leaderboard (6 models vs. a fixed reference, tight skill
ladder, alternating verbose/terse), judge calibrated to the pilot's
`i_PV = +0.15`:

| method | Kendall τ | |
|---|---|---|
| no correction | +0.20 | worst model ranks above best |
| **sequential correction** | **+0.20** | **unchanged — SC is a constant shift** |
| **IASC** (label-free, 1× cost) | **+0.87** | one adjacent pair (DiD attenuation) |
| swap every pair (2× cost) | +1.00 | correct |
| JBC / IA-PPI (equal-skill gold cal.) | +1.00 | correct |

The demo is **constructed** to expose the mechanism (verbosity aligned
adversarially with rank; real leaderboards randomize position) — absolute τ
values are artefacts, the *ordering of methods* is the point. **SC does nothing**
(the differential lives entirely in β_PV, which SC can't touch). JBC (with an
equal-skill gold set) and 2× swap-averaging recover the order. **IASC removes
the gross inversion label-free at 1× cost, but not every adjacent swap** — it
under-corrects because β̂_PV attenuates, and the residual grows with the bias
(τ ∈ [+0.60, +0.87] across a 4× magnitude sweep). It's a partial fix, not a
substitute for gold calibration.

**Caveats:** 18–20 pairs/judge, no frontier deployed judge (Gemini
rate-limited; Groq hit its 200K-tokens/day cap mid-run). Only Gemma has ≥3
samples/cell, so only its log-odds `b_PV` is precise. Data in `data/raw/`
(v1 gpt-oss + v2 gemma, the analysis set) and `data/raw_pilot_v1/`.

**Full test:** ≥8 judges / ≥5 families — the collector ships 21 configs across
13 families over 12 providers (Groq, Google, Cerebras, Mistral, OpenRouter,
Cohere, SambaNova, NVIDIA, GitHub Models, Hugging Face, Together; Anthropic
paid) + `--samples N`. See the pre-registered protocol in the paper.

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

if you run the real-judge pipeline, **please open a PR with your results** — that's the whole point ⭐

</div>
