"""
benchmark_loop3.py — Dubweave autoresearch Loop 3
--------------------------------------------------
Evaluates normalizer_rules.json against two corpora:

  1. ptpt_injection_corpus.json   — sentences with injected PT-PT constructions.
                                    Measures detection_rate: how many are corrected.

  2. First N lines of corpus/ptbr_clean_corpus.txt — native PT-BR sentences
     (the 1000 phonetically balanced sentences).
     Measures false_positive_rate: rules should NOT change these.

Composite score:
  S = detection_rate - FP_WEIGHT * false_positive_rate
  Higher is better. Max theoretical S = 1.0.

Usage:
    pixi run python benchmark_loop3.py                  # score current rules
    pixi run python benchmark_loop3.py --baseline       # record as BASELINE
    pixi run python benchmark_loop3.py --status         # print last 10 results

Cost: zero — pure Python regex, no API calls, ~1 second per run.
"""

import argparse
import csv
import datetime
import json
import re
import sys
from pathlib import Path

ROOT         = Path(__file__).parent.parent
AR           = Path(__file__).parent
RULES_PATH   = ROOT / "normalizer_rules.json"
INJECT_PATH  = AR / "ptpt_injection_corpus.json"
CLEAN_PATH   = ROOT / "corpus" / "ptbr_clean_corpus.txt"
RESULTS_PATH = AR / "results_loop3.tsv"

FP_WEIGHT      = 0.50   # false positive penalty weight
CLEAN_LINES    = 200    # how many clean PT-BR sentences to test for false positives

# ---------------------------------------------------------------------------
# Apply rules
# ---------------------------------------------------------------------------

def _gerund_replacement(verb: str) -> str:
    """Convert PT-PT 'a + infinitive' verb to PT-BR gerund."""
    if verb.endswith("ar"):
        return verb[:-2] + "ando"
    elif verb.endswith("er"):
        return verb[:-2] + "endo"
    elif verb.endswith("ir"):
        return verb[:-2] + "indo"
    return verb + "ndo"


def apply_rules(text: str, rules: list) -> str:
    """Apply all normalizer rules to text. Returns normalized text."""
    for rule in rules:
        if rule.get("type") == "gerund":
            verbs = rule.get("verbs", [])
            if verbs:
                verb_pattern = "|".join(re.escape(v) for v in verbs)
                pattern = rf"\ba ({verb_pattern})\b"
                def _replace(m):
                    verb = m.group(1)
                    return _gerund_replacement(verb)
                text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
        else:
            pattern     = rule.get("pattern", "")
            replacement = rule.get("replacement", "")
            if not pattern:
                continue
            try:
                def _replace_preserve_case(m, repl=replacement):
                    if m.group(0) and m.group(0)[0].isupper():
                        return repl[0].upper() + repl[1:]
                    return repl
                text = re.sub(pattern, _replace_preserve_case, text, flags=re.IGNORECASE)
            except re.error as e:
                print(f"  WARNING: invalid pattern '{pattern}': {e}", file=sys.stderr)
    return text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_detection(rules: list, corpus: list) -> dict:
    """
    For each sentence in injection corpus, check if applying rules produces
    a text that differs from ptpt (i.e. at least one correction was made).
    We measure:
      - corrected: rules changed the ptpt text (something was caught)
      - perfect:   normalized text matches reference ptbr exactly
    """
    corrected = 0
    perfect   = 0
    n         = len(corpus)

    details = []
    for item in corpus:
        ptpt = item["ptpt"]
        ref  = item["ptbr"]
        normalized = apply_rules(ptpt, rules)

        was_corrected = (normalized != ptpt)
        is_perfect    = (normalized.strip() == ref.strip())

        corrected += int(was_corrected)
        perfect   += int(is_perfect)
        details.append({
            "id":           item["id"],
            "corrected":    was_corrected,
            "perfect":      is_perfect,
            "input":        ptpt,
            "output":       normalized,
            "reference":    ref,
        })

    return {
        "detection_rate": corrected / n,
        "perfect_rate":   perfect   / n,
        "n":              n,
        "details":        details,
    }


def score_false_positives(rules: list, clean_sentences: list) -> dict:
    """
    For each clean PT-BR sentence, check if rules incorrectly modify it.
    Any change is a false positive.
    """
    changed = 0
    n       = len(clean_sentences)

    for sent in clean_sentences:
        normalized = apply_rules(sent, rules)
        if normalized.strip() != sent.strip():
            changed += 1

    return {
        "false_positive_rate": changed / n if n > 0 else 0.0,
        "n_changed":           changed,
        "n":                   n,
    }


def score_rules() -> dict:
    config = json.loads(RULES_PATH.read_text(encoding="utf-8-sig"))
    rules  = config.get("rules", [])

    inject_corpus = json.loads(INJECT_PATH.read_text(encoding="utf-8"))

    # Load clean PT-BR corpus
    clean_sentences = []
    if CLEAN_PATH.exists():
        raw = CLEAN_PATH.read_text(encoding="utf-8").splitlines()
        # Lines are formatted: "0001. Sentence here." — strip the number prefix
        for line in raw:
            line = line.strip()
            if not line:
                continue
            # Remove leading number prefix like "0001. "
            clean = re.sub(r"^\d+\.\s*", "", line)
            if clean:
                clean_sentences.append(clean)
            if len(clean_sentences) >= CLEAN_LINES:
                break
    else:
        print(f"  WARNING: {CLEAN_PATH} not found — false positive test skipped", file=sys.stderr)

    det = score_detection(rules, inject_corpus)
    fp  = score_false_positives(rules, clean_sentences) if clean_sentences else {"false_positive_rate": 0.0, "n": 0, "n_changed": 0}

    S = det["detection_rate"] - FP_WEIGHT * fp["false_positive_rate"]

    return {
        "S":              round(S,                          5),
        "detection_rate": round(det["detection_rate"],     4),
        "perfect_rate":   round(det["perfect_rate"],       4),
        "fp_rate":        round(fp["false_positive_rate"], 4),
        "n_inject":       det["n"],
        "n_clean":        fp["n"],
        "n_rules":        len([r for r in rules if r.get("type") != "gerund"]) + (1 if any(r.get("type") == "gerund" for r in rules) else 0),
        "details":        det["details"],
    }


# ---------------------------------------------------------------------------
# TSV logging
# ---------------------------------------------------------------------------

_TSV_FIELDS = [
    "timestamp", "status", "S",
    "detection_rate", "perfect_rate", "fp_rate",
    "n_inject", "n_clean", "n_rules", "description",
]


def log_result(scores: dict, status: str, description: str = "") -> None:
    row = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "status":         status,
        "S":              scores["S"],
        "detection_rate": scores["detection_rate"],
        "perfect_rate":   scores["perfect_rate"],
        "fp_rate":        scores["fp_rate"],
        "n_inject":       scores["n_inject"],
        "n_clean":        scores["n_clean"],
        "n_rules":        scores["n_rules"],
        "description":    description or "",
    }
    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_best_S() -> float | None:
    if not RESULTS_PATH.exists():
        return None
    best = None
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") in ("KEEP", "BASELINE"):
                try:
                    s = float(row["S"])
                    if best is None or s > best:
                        best = s
                except (ValueError, KeyError):
                    pass
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubweave Loop 3 benchmark — score normalizer_rules.json"
    )
    parser.add_argument("--baseline",    action="store_true")
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print missed detections and false positives")
    parser.add_argument("--description", "-d", default="")
    args = parser.parse_args()

    if args.status:
        if not RESULTS_PATH.exists():
            print("No results_loop3.tsv yet.")
            return
        with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows[-10:]:
            print(
                f"[{row['timestamp']}] {row['status']:8s} "
                f"S={row['S']:7s}  det={row['detection_rate']}  "
                f"perf={row['perfect_rate']}  fp={row['fp_rate']}  "
                f"rules={row['n_rules']}  {row['description']}"
            )
        return

    print("Scoring normalizer_rules.json...")
    scores = score_rules()

    if args.baseline:
        status = "BASELINE"
    else:
        best = read_best_S()
        if best is None:
            print("WARNING: no BASELINE yet. Run with --baseline first.", file=sys.stderr)
            status = "?"
        else:
            status = "KEEP" if scores["S"] > best else "DISCARD"

    log_result(scores, status, args.description)

    print(f"\n  S              = {scores['S']:.5f}  {'← NEW BEST' if status == 'KEEP' else ''}")
    print(f"  detection_rate = {scores['detection_rate']:.4f}   (how many PT-PT caught)")
    print(f"  perfect_rate   = {scores['perfect_rate']:.4f}   (exact match to reference)")
    print(f"  fp_rate        = {scores['fp_rate']:.4f}   (false positives on clean PT-BR)")
    print(f"  n_inject       = {scores['n_inject']}   n_clean = {scores['n_clean']}")
    print(f"  n_rules        = {scores['n_rules']}")
    print(f"\n  status         = {status}")

    if args.verbose:
        print("\n  Missed detections (PT-PT not caught):")
        for d in scores["details"]:
            if not d["corrected"]:
                print(f"    [{d['id']}] {d['input']}")
        print("\n  Imperfect corrections (caught but not perfect):")
        for d in scores["details"]:
            if d["corrected"] and not d["perfect"]:
                print(f"    [{d['id']}] {d['output']}")
                print(f"           ref: {d['reference']}")

    if status == "DISCARD":
        best = read_best_S()
        print(f"  best so far    = {best:.5f}")

    print(f"\n  Logged → {RESULTS_PATH}")


if __name__ == "__main__":
    main()