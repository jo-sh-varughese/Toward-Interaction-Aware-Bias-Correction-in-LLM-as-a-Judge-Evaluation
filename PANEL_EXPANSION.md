# Panel expansion runbook (weakness #1)

Everything else in the 5-point review is already done in code (see the
"What changed" section at the bottom). This is the one item that needs
**your** free-tier API keys — a few sign-ups, then ~1 evening of
wall-clock. Goal: get from 3 judges / 2 families to **8-10 judges /
>=5 families**, which is what turns the pilot from "suggestive" into
"robust" and removes a reviewer's main line of attack.

## 0. Keep the design frozen

Do **not** change the stimulus set, the 8 `(P,V,S)` cells, or the judge
prompt. New judges must be a clean panel extension of the
pre-registration (Appendix A of the paper), not a protocol change. Use
the **same** `data/stimulus_pairs.jsonl` and `--n 20 --samples 2`.

## 1. Get free keys (no card needed for any of these)

| env var | sign-up | gives you (family) |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com | gpt-oss (openai) |
| `GOOGLE_API_KEY` | aistudio.google.com | gemini, gemma (google, google-gemma) |
| `CEREBRAS_API_KEY` | cloud.cerebras.ai | llama-3.3-70b (meta), qwen-3-32b (qwen) |
| `MISTRAL_API_KEY` | console.mistral.ai | mistral-small (mistral) |
| `SAMBANOVA_API_KEY` | cloud.sambanova.ai | DeepSeek-V3 (deepseek), Qwen3-32B (qwen) |
| `NVIDIA_API_KEY` | build.nvidia.com | nemotron-49b (nvidia), phi-4 (microsoft) |
| `COHERE_API_KEY` | dashboard.cohere.com | command-r (cohere) |
| `GITHUB_TOKEN` | github.com/marketplace/models (PAT w/ "Models" scope) | Phi-4, DeepSeek-V3, Jamba-1.5 (microsoft, deepseek, ai21) |
| `HF_TOKEN` | huggingface.co (read token) | granite-3.3 (ibm), gemma-3-27b (google-gemma) |
| `TOGETHER_API_KEY` | api.together.xyz | Llama-3.3-70B free endpoint (meta) |

Optional paid, ~cents: `ANTHROPIC_API_KEY` -> `claude-haiku` (anthropic) —
a clean cross-family anchor for trivial cost.

On Windows use `py` instead of `python`.

## 2. Check live model IDs first

Provider catalogues drift (the file already has stale-ish IDs like
`gemma-4-31b-it`, `mixtral-*`). Before a long run:

```
py collect_free.py --list-models          # Groq catalogue
```

For Google/others, if a judge 404s, open `collect_free.py`, find its
entry in `FREE_JUDGES`, and fix `model_id`. The `family` string only has
to be *consistent* (it drives the self-preference contrast S).

## 3. Collect (resumable — safe to Ctrl-C and rerun)

Pick judges to **maximize family coverage**. A good 7-judge target on top
of the existing 3:

```
py collect_free.py --judges cerebras-llama70b,cerebras-qwen32b,sambanova-deepseek,mistral-small,cohere-command-r,nvidia-nemotron,hf-granite --n 20 --samples 2
```

That plus the current `gpt-oss-20b/120b` + `gemma-3-27b` = **10 judges
across openai, meta, qwen, deepseek, mistral, cohere, nvidia, ibm,
google-gemma** (>=8 families). Output appends to `data/raw/`.

Rate limits mean this runs over a few hours; the loop skips already-done
`(pair, cell, judge, sample)` on restart. If a provider hard-fails, drop
it from `--judges` and continue with the rest.

Better still (more power, ~2.5x the calls): `--n 50` after
`py make_stimulus_pairs.py --n 50 --out data/stimulus_pairs.jsonl`
(regenerate the stimulus file only once, before any collection).

## 4. Re-run the analysis (no code changes needed)

```
py analyze_real.py     # -> results/real_bic.json + all LaTeX tables
py analyze_iasc.py     # -> results/iasc_pilot.json
py analyze_iappi.py    # -> results/iappi_pilot.json
```

`analyze_real.py` already prints, for the expanded panel: the per-judge
`i_PV` / `b_PV` / `b_V` table (paper Table 1), the all-three-pairs AIA
table, and `i_PV` by task category (both in the pilot appendix). Paste
the regenerated LaTeX blocks into `paper_neurips_workshop.tex`, update
the "3 judges / 2 families / ~20 pairs" counts, and — per the
pre-registered design (paper Appendix A) — switch to the padding-only
verbosity manipulation, held-out `S` generator, and wild-cluster
bootstrap for the full run. `P x V` should only strengthen with more
judges; if it doesn't, report it.

---

## Paper state (methods paper)

*Judge-bias corrections are chained assuming additivity; SSIT checks
that for free, and here's everything you need to use it.*

- **Framing** — a systems paper for JUDGe: judge validity = how the
  judge's error composes with the pipeline + downstream decisions.
- **§3 + Cor. 1** — exposure taxonomy, framed as the answer to the
  workshop's open question "does swap-averaging fully debias?":
  subtract-a-term pipelines exposed (½β_PV); log-odds swap-averaging
  removes P×V up to a Jensen term; joint estimation exact; non-position
  pairs untouched by swap-averaging.
- **§4 Downstream** (new) — `sim_ranking_impact.py` now also computes an
  RLHF preference-label-flip upper bound. i_PV=0.15 → leaderboard τ
  +0.20→+1.0, 26–44% of the length-mismatched cell mislabelled.
- **§8** — checklist reframed as a "deployment-disclosure" component
  (model card for the eval pipeline).
- **§7** — paraphrase confound reframed as a surface-form-sensitivity
  finding (position bias modulated by rephrasing, not only length).
- **§4.2 + `sim_ssit_power.py`** — power depends on the judge's per-pair
  effect-to-noise ratio, which is *observed* once you run the check.
  Generic: ~50–100 pairs at 2 samples/cell for a 0.15 effect; fewer for
  a decisive judge. `analyze_real.py` prints a design-based realized
  power per judge (pilot: 0.64 / 0.69 / 0.91) — this reconciles the 3/3
  detections with the generic table (which assumes a noisier judge).
- **§6** — "worked example," read with a realized-power column. Framed as
  a tool demo; magnitude may be paraphrase-inflated, full study settles.
- **§7** — decision procedure (numbered) + a checklist for reviewers of
  judge-based evaluations.
- **IASC (Prop. 3)** — restated as a shrinkage correction (fraction λ
  removed, can't over-correct), proved via the standard pooled-slope
  attenuation argument.
- Leaderboard demo cut from the body.

- **§3** Prop. 1 — marginal correction leaves ½·β_PV (framed as textbook
  omitted-variable bias, not a "structural" mystery).
- **§4 + `sim_ssit_robustness.py`** — the `i_PV` readout is a difference
  of four cell means, so it's *design-based*: the sim shows it recovers
  each judge's TRUE population interaction to within 0.004 across six
  decision rules, **including an out-of-family linkless threshold
  judge**. Only the secondary log-odds `b_PV` uses the model. This is
  the answer to "the identifying assumption is unvalidated."
- **Wild cluster bootstrap** (`ssit.ssit_wild_boot`, Rademacher weights)
  — added because ~20 clusters makes the pairs bootstrap suspect. On
  the pilot all three judges land at `p` ≤ 0.03, pooled `p` ≈ 0; the
  two bootstraps' intervals nearly coincide, so the pairs bootstrap
  wasn't badly off here. `analyze_real.py` prints both.
- **§5** IASC — honestly scoped: partial, under-corrects by a
  data-dependent amount, leaves the verbosity main effect.
- **§6** pilot + an explicit "threats to validity" paragraph.
- **§7** practitioner guidance + the pre-registered full study.
- IA-PPI / PPI-coverage / BIC moved to appendices.

Body is ~5 pages; 0 overfull boxes; 0 undefined refs. Both PDFs rebuilt
(`paper_neurips_workshop.pdf` anon, `paper_zenodo.pdf` attributed, new
title "Do LLM-Judge Biases Compose?").
