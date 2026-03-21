"""
benchmark_loop2.py — Dubweave autoresearch Loop 2
--------------------------------------------------
Evaluates translation_prompt.md against eval_corpus.json.
Uses OpenRouter (Gemini 2.0 Flash) for translation,
Claude Haiku as judge.

Usage:
    pixi run python benchmark_loop2.py                  # score current prompt
    pixi run python benchmark_loop2.py --baseline       # record as BASELINE
    pixi run python benchmark_loop2.py --status         # print last 5 results

Cost: ~$0.02 per run (30 sentences × translate + judge)
"""

import argparse, csv, datetime, json, os, sys
from pathlib import Path
import urllib.request

CORPUS_PATH   = Path("eval_corpus.json")
PROMPT_PATH   = Path("translation_prompt.md")
RESULTS_PATH  = Path("results_loop2.tsv")

OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "").strip()
TRANSLATE_MODEL  = "google/gemini-2.0-flash-001"
JUDGE_MODEL      = "anthropic/claude-haiku-4-5"  # fast, cheap, reliable judge
OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_corpus(system_prompt: str, sentences: list[str]) -> list[str]:
    """Translate all sentences using the current system prompt."""
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    user_msg = (
        f"Translate these {len(sentences)} numbered sentences to Brazilian Portuguese.\n"
        "Output ONLY numbered translations, same count, same order.\n\n"
        f"{numbered}"
    )
    payload = json.dumps({
        "model": TRANSLATE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    raw = data["choices"][0]["message"]["content"].strip()
    import re
    lines = [re.sub(r"^\d+[.)]\s*", "", l).strip()
             for l in raw.splitlines() if l.strip()]
    return lines[:len(sentences)]

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_translations(pairs: list[dict]) -> list[float]:
    """
    Score each (english, reference, translation) triple 1-5.
    Returns list of float scores.
    """
    items = "\n\n".join(
        f"[{i+1}]\n"
        f"English:     {p['english']}\n"
        f"Reference:   {p['reference']}\n"
        f"Translation: {p['translation']}"
        for i, p in enumerate(pairs)
    )
    user_msg = (
        "Score each translation 1-5 for Brazilian Portuguese quality.\n"
        "Criteria: correct PT-BR (not PT-PT), accurate meaning, natural register.\n"
        "5=perfect PT-BR, 4=minor issues, 3=acceptable, 2=PT-PT markers or errors, 1=wrong.\n"
        "Output ONLY a JSON array of numbers, e.g. [4,5,3,5,4,...]\n\n"
        f"{items}"
    )
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a Brazilian Portuguese linguist. Output only JSON."},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    raw = data["choices"][0]["message"]["content"].strip()
    import re
    raw = re.sub(r"```json|```", "", raw).strip()
    scores = json.loads(raw)
    return [float(s) for s in scores]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prompt() -> dict:
    corpus   = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    prompt   = PROMPT_PATH.read_text(encoding="utf-8").strip()
    english  = [c["english"]   for c in corpus]
    refs     = [c["reference"] for c in corpus]

    print(f"  Translating {len(english)} sentences...")
    translations = translate_corpus(prompt, english)

    pairs = [{"english": e, "reference": r, "translation": t}
             for e, r, t in zip(english, refs, translations)]

    print(f"  Judging {len(pairs)} translations...")
    scores = judge_translations(pairs)

    avg   = sum(scores) / len(scores)
    pct_5 = sum(1 for s in scores if s == 5) / len(scores)
    pct_4 = sum(1 for s in scores if s >= 4) / len(scores)
    pct_2 = sum(1 for s in scores if s <= 2) / len(scores)

    return {
        "S":      round(avg,   4),   # primary metric: mean judge score
        "pct_5":  round(pct_5, 4),
        "pct_4+": round(pct_4, 4),
        "pct_2-": round(pct_2, 4),
        "n":      len(scores),
        "scores": scores,
    }

# ---------------------------------------------------------------------------
# Logging / CLI  (same pattern as benchmark.py)
# ---------------------------------------------------------------------------

_TSV_FIELDS = ["timestamp","status","S","pct_5","pct_4+","pct_2-","n","description"]

def log_result(scores: dict, status: str, description: str = "") -> None:
    row = {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "status":      status,
        "S":           scores["S"],
        "pct_5":       scores["pct_5"],
        "pct_4+":      scores["pct_4+"],
        "pct_2-":      scores["pct_2-"],
        "n":           scores["n"],
        "description": description,
    }
    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        if write_header: writer.writeheader()
        writer.writerow(row)

def read_best_S() -> float | None:
    if not RESULTS_PATH.exists(): return None
    best = None
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") in ("KEEP","BASELINE"):
                try:
                    s = float(row["S"])
                    if best is None or s > best: best = s
                except: pass
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--status",   action="store_true")
    parser.add_argument("--description", "-d", default="")
    args = parser.parse_args()

    if args.status:
        if not RESULTS_PATH.exists():
            print("No results_loop2.tsv yet.")
            return
        with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows[-10:]:
            print(f"[{row['timestamp']}] {row['status']:8s} "
                  f"S={row['S']}  5s={row['pct_5']}  "
                  f"4+={row['pct_4+']}  2-={row['pct_2-']}  "
                  f"{row['description']}")
        return

    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print("Scoring translation_prompt.md...")
    scores = score_prompt()

    if args.baseline:
        status = "BASELINE"
    else:
        best = read_best_S()
        if best is None:
            print("WARNING: no BASELINE yet. Run with --baseline first.")
            status = "?"
        else:
            status = "KEEP" if scores["S"] > best else "DISCARD"

    log_result(scores, status, args.description)

    print(f"\n  S      = {scores['S']:.4f}  {'← NEW BEST' if status=='KEEP' else ''}")
    print(f"  5/5    = {scores['pct_5']:.3f}")
    print(f"  4+/5   = {scores['pct_4+']:.3f}")
    print(f"  2-/5   = {scores['pct_2-']:.3f}")
    print(f"  n      = {scores['n']}")
    print(f"\n  status = {status}")
    print(f"  Logged → {RESULTS_PATH}")

if __name__ == "__main__":
    main()