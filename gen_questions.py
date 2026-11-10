#!/usr/bin/env python3
"""Generate synthetic training questions for each passage (Claude Haiku).

The 100 evaluation questions are never read here: training data comes only
from the passages themselves.

    python3 gen_questions.py --data data --out data/train_questions.json

Standard library only. Reads ANTHROPIC_API_KEY from the environment or from
~/.config/calinalabs/keys.env.
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request


def load_keys():
    f = pathlib.Path.home() / ".config" / "calinalabs" / "keys.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Claude:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def ask(self, model, prompt, max_tokens=1500, tries=10):
        body = json.dumps({"model": model, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"}
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.URL, data=body, headers=headers), timeout=120) as r:
                    return "".join(b.get("text", "") for b in json.load(r)["content"])
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < tries - 1:
                    time.sleep(min(60, 2 ** attempt) + random.random())
                    continue
                raise RuntimeError(f"Claude API -> HTTP {e.code}: {e.read()[:300]!r}") from None
            except urllib.error.URLError:
                if attempt < tries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

MODEL = "claude-haiku-5-5"
PER_PASSAGE = 3
PROMPT = """Here is a passage from a software product's developer documentation.

<heading>{heading}</heading>
<passage>
{text}
</passage>

Write {n} different questions that a developer might ask, which this passage answers.
- Vary them: one short and direct, one about how to do something, one that needs a detail from the passage.
- Phrase them the way a user would ask, not by copying the heading or the passage's wording.
- Each question must be answerable from this passage alone.

Return only a JSON list of {n} strings."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    load_keys()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY")
    docs = json.loads((a.data / "anthropic_summary_indexed_docs.json").read_text())
    llm = Claude()

    def one(d):
        raw = llm.ask(MODEL, PROMPT.format(heading=d["chunk_heading"], text=d["text"][:6000], n=PER_PASSAGE), 600)
        m = re.search(r"\[.*\]", raw, re.S)
        qs = json.loads(m.group(0)) if m else []
        return [{"chunk_link": d["chunk_link"], "question": q.strip()} for q in qs if isinstance(q, str) and q.strip()][:PER_PASSAGE]

    rows = []
    with cf.ThreadPoolExecutor(8) as pool:
        for n, r in enumerate(pool.map(one, docs), 1):
            rows += r
            if n % 50 == 0 or n == len(docs):
                print(f"  {n}/{len(docs)} passages")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"Saved {len(rows)} questions for {len({r['chunk_link'] for r in rows})} passages to {a.out}")


if __name__ == "__main__":
    main()
