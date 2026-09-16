"""
make_padded_v.py
================
Build a PADDING-ONLY verbosity manipulation to sit alongside the
paraphrase one (response_r1_expanded).

response_r1_padded = response_r1 (verbatim) + 3 sentences that restate
points ALREADY MADE, no new information, same plain style.  This
isolates length from rephrasing: r1 is preserved word-for-word, only
extended.

Reads  data/stimulus_pairs.jsonl
Writes data/stimulus_pairs_padded.jsonl  (first --n pairs, adds the field)

Needs MISTRAL_API_KEY (used only to generate the redundant sentences).
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from pathlib import Path

SRC = "data/stimulus_pairs.jsonl"
OUT = "data/stimulus_pairs_padded.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

PROMPT = """Here is a response to a task. Write exactly 3 additional sentences \
that restate and lightly rephrase points ALREADY made in the response. \
Add NO new facts, examples, caveats, or structure. Keep the same plain, \
unformatted style (no markdown, no headers, no lists). Output only the 3 \
sentences, nothing else.

RESPONSE:
{r1}"""


def mistral(text):
    key = os.environ["MISTRAL_API_KEY"]
    body = json.dumps({
        "model": "mistral-small-latest",
        "temperature": 0.3,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions", data=body,
        headers={"content-type": "application/json",
                 "authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")][:N]
    done = {}
    if Path(OUT).exists():
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            done[r["pair_id"]] = r.get("response_r1_padded")

    with open(OUT, "w", encoding="utf-8") as f:
        for i, p in enumerate(rows):
            pid = p["pair_id"]
            if done.get(pid):
                p["response_r1_padded"] = done[pid]
            else:
                for attempt in range(4):
                    try:
                        extra = mistral(PROMPT.format(r1=p["response_r1"]))
                        # strip stray markdown / numbering
                        extra = extra.replace("*", "").replace("#", "")
                        extra = " ".join(ln.lstrip("0123456789.- ").strip()
                                         for ln in extra.splitlines() if ln.strip())
                        p["response_r1_padded"] = p["response_r1"].rstrip() + " " + extra
                        break
                    except Exception as e:
                        print(f"  {pid} attempt {attempt+1}: {e}")
                        time.sleep(4)
                time.sleep(1.5)
            r1t = len(p["response_r1"].split())
            padt = len(p["response_r1_padded"].split())
            print(f"{pid:22} r1={r1t}w  padded={padt}w  (x{padt/max(r1t,1):.1f})")
            f.write(json.dumps(p) + "\n")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
