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
FREE_JUDGES = {
    # Groq — completely free, fast
    "llama-3.1-70b": {
        "api":      "groq",
        "model_id": "llama-3.1-70b-versatile",
        "family":   "meta",
        "rpm":      30,
        "delay":    2.1,   # seconds between calls (stay under 30 RPM)
    },
    "mixtral-8x7b": {
        "api":      "groq",
        "model_id": "mixtral-8x7b-32768",
        "family":   "mistral",
        "rpm":      30,
        "delay":    2.1,
    },
    "gemma2-9b": {
        "api":      "groq",
        "model_id": "gemma2-9b-it",
        "family":   "google",
        "rpm":      30,
        "delay":    2.1,
    },
    # Google AI Studio — free, no card required
    "gemini-1.5-flash": {
        "api":      "google",
        "model_id": "gemini-1.5-flash",
        "family":   "google",
        "rpm":      15,
        "delay":    4.1,   # 15 RPM -> 4s between calls
    },
}

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

def jid(pair_id, judge, pos, verb, iden):
    key = f"{pair_id}|{judge}|{pos}|{verb}|{iden}"
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
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_id)
    t0 = time.monotonic()
    resp = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temp, max_output_tokens=512))
    return {
        "text":    resp.text,
        "latency": int((time.monotonic()-t0)*1000),
        "in_tok":  resp.usage_metadata.prompt_token_count,
        "out_tok": resp.usage_metadata.candidates_token_count,
    }

API_FN = {"groq": call_groq, "google": call_google}

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
def run(pairs_path, out_dir, judge_names, n_pairs=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

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
                    jid_ = jid(pid, jname,
                               cell["position"],
                               cell["verbosity"],
                               cell["identity"])
                    if jid_ in done:
                        n_done += 1
                        continue

                    # Resolve which text goes in A and B
                    r1 = pair["response_r1"]
                    r2 = pair["response_r2"]
                    if cell["verbosity"] == "expanded":
                        r1 = pair.get("response_r1_expanded", r1)
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
                            res = api_fn(jcfg["model_id"], prompt_text, temp=0.0)
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
                                 len(all_pairs)*8)

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
    parser.add_argument("--judges", default="llama-3.1-70b,gemini-1.5-flash",
                        help="Comma-separated judge names from FREE_JUDGES")
    parser.add_argument("--n",      type=int, default=50,
                        help="Number of pairs (50=free in 2hrs, 200=free overnight)")
    args = parser.parse_args()
    run(args.pairs, args.out,
        [j.strip() for j in args.judges.split(",")],
        n_pairs=args.n)
