<div align="center">

# ⚖️ Toward Interaction-Aware Bias Correction in LLM-as-a-Judge Evaluation

### Toward Interaction-Aware Bias Correction in LLM-as-a-Judge Evaluation
**The Bias Interaction Coefficient (BIC) & Joint Bias Correction (JBC)**

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=for-the-badge)](#)
[![Conference](https://img.shields.io/badge/ICMLA-2025-181717?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/results-simulation--based-C5F135?style=for-the-badge)](#-important-this-is-a-simulation-study)
[![Pre-registered](https://img.shields.io/badge/real--judge_study-pre--registered-C5F135?style=for-the-badge)](#-real-judge-validation-path)
[![License](https://img.shields.io/badge/license-MIT-C5F135?style=for-the-badge)](#-license)

</div>

---

> [!TIP]
> **TL;DR** — Pairwise LLM-judge pipelines correct for position bias, then verbosity bias, then self-preference bias — *sequentially*, as if each acts independently. We prove that if position and verbosity interact, sequential correction provably leaves a residual of $\tfrac{1}{2}\beta_{PV}$. In a calibrated simulation, that residual grows to **~4.3×** the no-interaction baseline — roughly **3,900 mislabeled pairs per 100K RLHF reward pairs**. We introduce **BIC**, a one-number diagnostic to detect this, and **JBC**, a joint-correction algorithm — plus a pre-registered, zero-cost pipeline to test all of this on real judges.

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

```bash
git clone https://github.com/<your-username>/bias-interaction-coefficient.git
cd bias-interaction-coefficient
pip install -r requirements.txt

# Reproduce the full simulation study (seed=2024)
python run_simulation.py

# Compute BIC + run JBC on your own judge data
python compute_bic.py --calibration data/your_calibration_set.csv
python jbc_correct.py --eval data/your_eval_set.csv --bic-estimates output/bic.json
```

---

## 📁 Repository Structure

```
.
├── run_simulation.py        # full 8,000-judgment factorial simulation (seed=2024)
├── compute_bic.py            # fits P + V + P·V logit, computes BIC_ij with bootstrap CIs
├── jbc_correct.py             # Joint Bias Correction (Algorithm 1)
├── src/
│   ├── dgp.py                 # data-generating process / agent profile parameters
│   ├── theory.py               # Proposition 1 residual computation
│   └── stats.py                # cumulative logit + mixed-effects regression helpers
├── figures/                    # heatmap, forest plot, interaction profiles, power curves
├── real_judge_pipeline/        # pre-registered, free-tier replication study
│   ├── pilot_50pairs.py        # 50-pair pilot, ~1,600 API calls, Groq + AI Studio
│   └── preregistration.md
└── appendix/
    ├── kappa_recovery.md        # BIC recovery accuracy under low inter-agent agreement
    └── power_analysis.md
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
@inproceedings{bias_interaction_coefficient_2025,
  title     = {Toward Interaction-Aware Bias Correction in LLM-as-a-Judge
               Evaluation: The Bias Interaction Coefficient and
               Joint Bias Correction},
  author    = {TODO: Add author(s)},
  booktitle = {Proceedings of ICMLA 2025},
  year      = {2025},
  note      = {Code: https://github.com/<your-username>/bias-interaction-coefficient}
}
```

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

if you run the real-judge pipeline, **please open a PR with your results** — that's the whole point ⭐

</div>
