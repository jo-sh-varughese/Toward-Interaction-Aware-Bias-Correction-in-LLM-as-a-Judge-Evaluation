"""
collect_free.py
===============
Collects real LLM judge data using ONLY free API tiers.
Zero cost. Sign up takes 5 minutes per service.

FREE APIS USED:
  Groq  — https://console.groq.com  (free account, no card needed)
          Models: llama-3.1-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
          Limits: 30 req/min, 14,400 req/day FREE

  Google AI Studio — https://aistudio.google.com (free, no card needed)
          Models: gemini-1.5-flash (fast, free), gemini-1.5-pro (slower, free)
          Limits: 15 req/min, 1M tokens/day FREE

SETUP (5 minutes):
  1. pip install groq google-generativeai
  2. Go to https://console.groq.com  -> API Keys -> Create key
     Export GROQ_API_KEY=your_key
  3. Go to https://aistudio.google.com -> Get API key
     Export GOOGLE_API_KEY=your_key
  4. python collect_free.py

WHAT IT COLLECTS:
  N_PAIRS pairs x 8 cells x N_JUDGES judges = total judgments
  Default: 50 pairs x 8 cells x 4 judges = 1,600 calls (free, ~2 hrs)
  Full:   200 pairs x 8 cells x 4 judges = 6,400 calls (free, ~8 hrs overnight)
"""

import os, json, time, re, hashlib, argparse, logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Literal, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Free judge configurations ──────────────────────────────────
# NOTE: Groq's model catalogue changes frequently (llama-3.x-versatile and
# mixtral-8x7b were both decommissioned during 2025). Before running, list
# the live models and override `model_id` if needed:
#     curl -s https://api.groq.com/openai/v1/models \
#          -H "Authorization: Bearer $GROQ_API_KEY" | python -m json.tool
# or run:  python collect_free.py --list-models
# The `family` field only needs to be *consistent* for the self-preference
# contrast (S): pair judges so that some share a family with a response
# generator and some do not.
FREE_JUDGES = {
    "gpt-oss-20b": {
        "api":      "groq",
        "model_id": "openai/gpt-oss-20b",
        "family":   "openai",
        "rpm":      30,
        "delay":    2.2,
    },
    "gpt-oss-120b": {
        "api":      "groq",
        "model_id": "openai/gpt-oss-120b",
        "family":   "openai",
        "rpm":      30,
        "delay":    2.2,
    },
    "groq-qwen": {                        # Qwen family — new to the panel
        "api":      "groq",
        "model_id": "qwen/qwen3.8-27b",
        "family":   "qwen",
        "rpm":      30,
        "delay":    2.2,
    },
    "groq-allam": {                       # ALLaM (SDAIA) — 7B, Arabic-focused
        "api":      "groq",
        "model_id": "allam-2-7b",
        "family":   "allam",
        "rpm":      30,
        "delay":    2.2,
    },
    # Google AI Studio — free, no card required
    "gemini-2.5-flash": {
        "api":      "google",
        "model_id": "gemini-2.5-flash",
        "family":   "google",
        "rpm":      10,
        "delay":    6.5,
    },
    "gemma-3-27b": {                      # Gemma is a distinct family from Gemini
        "api":      "google",
        "model_id": "gemma-4-31b-it",
        "family":   "google-gemma",
        "rpm":      30,
        "delay":    2.0,
    },
    # Anthropic — PAID (not free tier). Needs ANTHROPIC_API_KEY (sk-ant-...).
    # Tiny cost for the pilot: ~150-300 calls/judge; cents on Haiku, ~$1-2 on
    # Sonnet. Gives the cross-family contrast the free judges lack.
    "claude-haiku": {
        "api":      "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "family":   "anthropic",
        "rpm":      50,
        "delay":    1.3,
    },
    "claude-sonnet": {
        "api":      "anthropic",
        "model_id": "claude-sonnet-5",
        "family":   "anthropic",
        "rpm":      50,
        "delay":    1.3,
    },
    # ---- More free tiers (all no card required) --------------------
    # Cerebras — https://cloud.cerebras.ai  (free key, very fast).
    # Model catalogue varies by key; list with --list-models-cerebras.
    "cerebras-llama70b": {
        "api": "cerebras", "model_id": "llama-3.3-70b",
        "family": "meta", "rpm": 30, "delay": 2.1,
    },
    "cerebras-qwen32b": {
        "api": "cerebras", "model_id": "qwen-3-32b",
        "family": "qwen", "rpm": 30, "delay": 2.1,
    },
    # provider-swap replicas (same model as the Groq / Google judges,
    # different inference backend) for a robustness check
    "cerebras-gptoss120b": {
        "api": "cerebras", "model_id": "gpt-oss-120b",
        "family": "openai", "rpm": 30, "delay": 2.1,
    },
    "cerebras-gemma": {
        "api": "cerebras", "model_id": "gemma-4-31b",
        "family": "google", "rpm": 30, "delay": 2.1,
    },
    # Mistral — https://console.mistral.ai  (free tier)
    "mistral-small": {
        "api": "mistral", "model_id": "mistral-small-latest",
        "family": "mistral", "rpm": 30, "delay": 2.1,
    },
    # OpenRouter — https://openrouter.ai  (many ':free' models)
    "openrouter-llama-free": {
        "api": "openrouter", "model_id": "meta-llama/llama-3.3-70b-instruct:free",
        "family": "meta", "rpm": 20, "delay": 3.2,
    },
    # Cohere — https://dashboard.cohere.com  (trial key)
    "cohere-command-r": {
        "api": "cohere", "model_id": "command-r-08-2024",
        "family": "cohere", "rpm": 20, "delay": 3.2,
    },
    # SambaNova Cloud — https://cloud.sambanova.ai  (free, no card, fast)
    "sambanova-deepseek": {
        "api": "sambanova", "model_id": "DeepSeek-V3-0324",
        "family": "deepseek", "rpm": 30, "delay": 2.1,
    },
    "sambanova-qwen": {
        "api": "sambanova", "model_id": "Qwen3-32B",
        "family": "qwen", "rpm": 30, "delay": 2.1,
    },
    # NVIDIA NIM — https://build.nvidia.com  (free credits)
    "nvidia-nemotron": {
        "api": "nvidia", "model_id": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "family": "nvidia", "rpm": 40, "delay": 1.6,
    },
    "nvidia-phi": {
        "api": "nvidia", "model_id": "microsoft/phi-4",
        "family": "microsoft", "rpm": 40, "delay": 1.6,
    },
    # GitHub Models — https://github.com/marketplace/models  (free w/ GH account;
    # GITHUB_TOKEN = a fine-grained PAT with the "Models" permission)
    "github-phi": {
        "api": "github", "model_id": "microsoft/Phi-4",
        "family": "microsoft", "rpm": 15, "delay": 4.2,
    },
    "github-deepseek": {
        "api": "github", "model_id": "deepseek/DeepSeek-V3-0324",
        "family": "deepseek", "rpm": 15, "delay": 4.2,
    },
    "github-jamba": {
        "api": "github", "model_id": "ai21-labs/AI21-Jamba-1.5-Large",
        "family": "ai21", "rpm": 15, "delay": 4.2,
    },
    # Hugging Face Inference Providers — https://huggingface.co  (free tier;
    # HF_TOKEN = a read token)
    "hf-granite": {
        "api": "huggingface", "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "family": "ibm", "rpm": 20, "delay": 3.2,
    },
    "hf-gemma": {
        "api": "huggingface", "model_id": "google/gemma-3-27b-it",
        "family": "google-gemma", "rpm": 20, "delay": 3.2,
    },
    # Together AI — https://api.together.xyz  (has a free Llama endpoint)
    "together-llama-free": {
        "api": "together",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "family": "meta", "rpm": 20, "delay": 3.2,
    },
}


def list_groq_models():
    import urllib.request
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
    data = json.load(urllib.request.urlopen(req))
    ids = sorted(m["id"] for m in data.get("data", []))
    print("Active Groq models:")
    for i in ids:
        print("  ", i)
    return ids

# ── Prompt template ────────────────────────────────────────────
PROMPT = """\
You are evaluating two responses to the following task. \
Determine which response is of higher quality based solely \
on accuracy, relevance, and completeness. \
Do NOT factor in length, formatting style, or response order.

[TASK]
{prompt}

[Response A]
{response_A}

[Response B]
{response_B}

Output your verdict on the final line in exactly this format:
VERDICT: A    or    VERDICT: B    or    VERDICT: TIE

Evaluation:"""

# ── Experimental cells ──────────────────────────────────────────
CELLS = [
    {"position": "A_first", "verbosity": "equal",    "identity": "neutral"},
    {"position": "A_first", "verbosity": "expanded",  "identity": "neutral"},
    {"position": "A_first", "verbosity": "equal",    "identity": "match"},
    {"position": "A_first", "verbosity": "expanded",  "identity": "match"},
    {"position": "B_first", "verbosity": "equal",    "identity": "neutral"},
    {"position": "B_first", "verbosity": "expanded",  "identity": "neutral"},
    {"position": "B_first", "verbosity": "equal",    "identity": "match"},
    {"position": "B_first", "verbosity": "expanded",  "identity": "match"},
]

# ── Verdict extraction ──────────────────────────────────────────
def extract_verdict(text: str) -> str:
    matches = re.findall(r"VERDICT:\s*([ABab]|TIE|tie)", text, re.IGNORECASE)
    if matches:
        v = matches[-1].upper()
        return "tie" if v == "TIE" else v
    text_lower = text.lower()
    if "response a is better" in text_lower: return "A"
    if "response b is better" in text_lower: return "B"
    if "both" in text_lower and "equal" in text_lower: return "tie"
    log.warning("Could not parse verdict from: %s", text[-100:])
    return "invalid"

def jid(pair_id, judge, pos, verb, iden, samp=0):
    key = f"{pair_id}|{judge}|{pos}|{verb}|{iden}|{samp}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]

# ── API callers ─────────────────────────────────────────────────
def call_groq(model_id, prompt, temp=0.0):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role":"user","content":prompt}],
        temperature=temp, max_tokens=512)
    return {
        "text":    resp.choices[0].message.content or "",
        "latency": int((time.monotonic()-t0)*1000),
        "in_tok":  resp.usage.prompt_tokens,
        "out_tok": resp.usage.completion_tokens,
    }

def call_google(model_id, prompt, temp=0.0):
    """Direct REST call to the Generative Language API (the deprecated
    google.generativeai SDK hangs with some key types)."""
    import urllib.request
    key = os.environ["GOOGLE_API_KEY"]
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model_id}:generateContent?key={key}")
    gcfg = {"temperature": temp, "maxOutputTokens": 800}
    # Gemini 2.5/3.x "think" by default and can burn the whole budget before
    # the verdict line; disable. Gemma / lite models reject the param -> omit.
    if model_id.startswith("gemini-2.5") or model_id.startswith("gemini-3"):
        gcfg["thinkingConfig"] = {"thinkingBudget": 0}
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": gcfg}).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    cand = (d.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    um = d.get("usageMetadata", {})
    return {
        "text":    text,
        "latency": int((time.monotonic() - t0) * 1000),
        "in_tok":  um.get("promptTokenCount", 0),
        "out_tok": um.get("candidatesTokenCount", 0),
    }

def call_anthropic(model_id, prompt, temp=0.0):
    """Anthropic Messages API via REST (no SDK dependency)."""
    import urllib.request
    body = json.dumps({
        "model": model_id,
        "max_tokens": 512,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    text = "".join(b.get("text", "") for b in d.get("content", [])
                   if b.get("type") == "text")
    u = d.get("usage", {})
    return {
        "text":    text,
        "latency": int((time.monotonic() - t0) * 1000),
        "in_tok":  u.get("input_tokens", 0),
        "out_tok": u.get("output_tokens", 0),
    }

def _openai_compat(url, key_env, model_id, prompt, temp, extra_headers=None):
    import urllib.request
    body = json.dumps({
        "model": model_id, "temperature": temp, "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    hdr = {"content-type": "application/json",
           "authorization": f"Bearer {os.environ[key_env]}",
           # some providers (Cerebras) sit behind Cloudflare and 403 the
           # default python-urllib User-Agent
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36"}
    if extra_headers:
        hdr.update(extra_headers)
    req = urllib.request.Request(url, data=body, headers=hdr)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    u = d.get("usage", {})
    return {"text": msg.get("content") or "",
            "latency": int((time.monotonic() - t0) * 1000),
            "in_tok": u.get("prompt_tokens", 0),
            "out_tok": u.get("completion_tokens", 0)}


def call_cerebras(model_id, prompt, temp=0.0):
    return _openai_compat("https://api.cerebras.ai/v1/chat/completions",
                          "CEREBRAS_API_KEY", model_id, prompt, temp)


def call_mistral(model_id, prompt, temp=0.0):
    return _openai_compat("https://api.mistral.ai/v1/chat/completions",
                          "MISTRAL_API_KEY", model_id, prompt, temp)


def call_openrouter(model_id, prompt, temp=0.0):
    return _openai_compat("https://openrouter.ai/api/v1/chat/completions",
                          "OPENROUTER_API_KEY", model_id, prompt, temp)


def call_sambanova(model_id, prompt, temp=0.0):
    return _openai_compat("https://api.sambanova.ai/v1/chat/completions",
                          "SAMBANOVA_API_KEY", model_id, prompt, temp)


def call_nvidia(model_id, prompt, temp=0.0):
    return _openai_compat("https://integrate.api.nvidia.com/v1/chat/completions",
                          "NVIDIA_API_KEY", model_id, prompt, temp)


def call_github(model_id, prompt, temp=0.0):
    return _openai_compat("https://models.github.ai/inference/chat/completions",
                          "GITHUB_TOKEN", model_id, prompt, temp)


def call_huggingface(model_id, prompt, temp=0.0):
    return _openai_compat("https://router.huggingface.co/v1/chat/completions",
                          "HF_TOKEN", model_id, prompt, temp)


def call_together(model_id, prompt, temp=0.0):
    return _openai_compat("https://api.together.xyz/v1/chat/completions",
                          "TOGETHER_API_KEY", model_id, prompt, temp)


def call_cohere(model_id, prompt, temp=0.0):
    import urllib.request
    body = json.dumps({
        "model": model_id, "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.cohere.com/v2/chat", data=body,
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {os.environ['COHERE_API_KEY']}"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    parts = d.get("message", {}).get("content", [])
    text = "".join(p.get("text", "") for p in parts)
    u = d.get("usage", {}).get("tokens", {})
    return {"text": text, "latency": int((time.monotonic() - t0) * 1000),
            "in_tok": u.get("input_tokens", 0), "out_tok": u.get("output_tokens", 0)}


API_FN = {"groq": call_groq, "google": call_google, "anthropic": call_anthropic,
          "cerebras": call_cerebras, "mistral": call_mistral,
          "openrouter": call_openrouter, "cohere": call_cohere,
          "sambanova": call_sambanova, "nvidia": call_nvidia,
          "github": call_github, "huggingface": call_huggingface,
          "together": call_together}

# ── Self-generation (for Factor S) ─────────────────────────────
def collect_self_gen(pairs, judge_name, judge_cfg, out_dir):
    """Collect judge's own responses for Factor S (identity match)."""
    outf = out_dir / f"selfgen_{judge_name}.jsonl"
    existing = {}
    if outf.exists():
        with open(outf) as f:
            for line in f:
                r = json.loads(line)
                existing[r["pair_id"]] = r["text"]

    api_fn = API_FN[judge_cfg["api"]]
    results = dict(existing)

    for pair in pairs:
        if pair["pair_id"] in existing:
            continue
        prompt = f"Please respond to the following task clearly and helpfully.\n\n{pair['prompt']}\n\nResponse:"
        for attempt in range(3):
            try:
                res = api_fn(judge_cfg["model_id"], prompt, temp=0.7)
                results[pair["pair_id"]] = res["text"]
                with open(outf,"a") as f:
                    f.write(json.dumps({"pair_id":pair["pair_id"],
                                        "judge":judge_name,
                                        "text":res["text"]})+"\n")
                time.sleep(judge_cfg["delay"])
                break
            except Exception as e:
                log.warning("Self-gen attempt %d failed: %s", attempt+1, e)
                time.sleep(5)
    return results

# ── Main collection loop ────────────────────────────────────────
def run(pairs_path, out_dir, judge_names, n_pairs=None, samples=1):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samp_temp = 0.0 if samples == 1 else 0.7

    with open(pairs_path) as f:
        all_pairs = [json.loads(l) for l in f]
    if n_pairs:
        all_pairs = all_pairs[:n_pairs]
    log.info("Loaded %d pairs", len(all_pairs))

    # Load completed judgment IDs
    done = set()
    for jf in out_dir.glob("judgments_*.jsonl"):
        with open(jf) as f:
            for line in f:
                try: done.add(json.loads(line)["jid"])
                except: pass

    judges = {n: FREE_JUDGES[n] for n in judge_names if n in FREE_JUDGES}
    total_needed = len(all_pairs) * 8 * len(judges)
    log.info("Judges: %s", list(judges.keys()))
    log.info("Total judgments needed: %d  (already done: %d)",
             total_needed, len(done))

    for jname, jcfg in judges.items():
        log.info("=== %s (%s) ===", jname, jcfg["api"].upper())
        api_fn = API_FN[jcfg["api"]]
        outf   = out_dir / f"judgments_{jname}.jsonl"

        # Collect self-generations for Factor S
        log.info("  Collecting self-generations …")
        selfgen = collect_self_gen(all_pairs, jname, jcfg, out_dir)

        n_done = 0
        with open(outf, "a", buffering=1) as fout:
            for pair in all_pairs:
                pid = pair["pair_id"]

                for cell in CELLS:
                  for si in range(samples):
                    jid_ = jid(pid, jname,
                               cell["position"],
                               cell["verbosity"],
                               cell["identity"], si)
                    if jid_ in done:
                        n_done += 1
                        continue

                    # Resolve which text goes in A and B
                    r1 = pair["response_r1"]
                    r2 = pair["response_r2"]
                    if cell["verbosity"] == "expanded":
                        # SSIT_V_FIELD lets a run use a padding-only manipulation
                        # (response_r1_padded) instead of the paraphrase one.
                        vf = os.environ.get("SSIT_V_FIELD", "response_r1_expanded")
                        r1 = pair.get(vf, pair.get("response_r1_expanded", r1))
                    if cell["identity"] == "match":
                        r1 = selfgen.get(pid, r1)

                    rA, rB = (r1,r2) if cell["position"]=="A_first" else (r2,r1)

                    prompt_text = PROMPT.format(
                        prompt=pair["prompt"],
                        response_A=rA, response_B=rB)

                    # Call with retry
                    result_text = "invalid"
                    latency = 0
                    for attempt in range(4):
                        try:
                            res = api_fn(jcfg["model_id"], prompt_text,
                                         temp=samp_temp)
                            result_text = res["text"]
                            latency     = res["latency"]
                            break
                        except Exception as e:
                            wait = (attempt+1) * 10
                            log.warning("Attempt %d failed (%s). Retry in %ds",
                                        attempt+1, e, wait)
                            time.sleep(wait)

                    verdict = extract_verdict(result_text)
                    record = {
                        "jid":      jid_,
                        "pair_id":  pid,
                        "sample":   si,
                        "task":     pair.get("task","unknown"),
                        "judge":    jname,
                        "position": cell["position"],
                        "verbosity":cell["verbosity"],
                        "identity": cell["identity"],
                        "P": 1 if cell["position"]=="A_first" else 0,
                        "V": 1 if cell["verbosity"]=="expanded" else 0,
                        "S": 1 if cell["identity"]=="match" else 0,
                        "verdict":  verdict,
                        "Y_bin":    1 if verdict=="A" else 0,
                        "Y_star":   (1. if verdict=="A" else
                                     .5 if verdict=="tie" else 0.),
                        "latency_ms": latency,
                        "ts": datetime.utcnow().isoformat(),
                    }
                    fout.write(json.dumps(record)+"\n")
                    done.add(jid_)
                    n_done += 1

                    if n_done % 100 == 0:
                        log.info("  %s: %d / %d complete",
                                 jname, n_done,
                                 len(all_pairs)*8*samples)

                    time.sleep(jcfg["delay"])

        log.info("  %s: DONE (%d judgments)", jname, n_done)

    log.info("Collection complete. Output: %s", out_dir)

# ── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",  default="data/stimulus_pairs.jsonl",
                        help="Path to stimulus pairs JSONL")
    parser.add_argument("--out",    default="data/raw/",
                        help="Output directory")
    parser.add_argument("--judges", default="groq-a,groq-c,gemini-flash",
                        help="Comma-separated judge names from FREE_JUDGES")
    parser.add_argument("--n",      type=int, default=50,
                        help="Number of pairs (50=free in 2hrs, 200=free overnight)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Draws per (pair,cell); >1 uses temperature 0.7 "
                             "and stabilises SSIT log-odds fits")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the live Groq model list and exit")
    args = parser.parse_args()
    if args.list_models:
        list_groq_models()
        raise SystemExit(0)
    run(args.pairs, args.out,
        [j.strip() for j in args.judges.split(",")],
        n_pairs=args.n, samples=args.samples)
