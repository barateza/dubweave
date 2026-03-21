"""
benchmark.py — Dubweave autoresearch Loop 1
-------------------------------------------
Evaluates a merge_config.json against the fixed Whisper corpus and logs
a composite score to results.tsv.

Usage:
    pixi run python benchmark.py                  # score current config
    pixi run python benchmark.py --baseline       # record as BASELINE row
    pixi run python benchmark.py --status         # print last 5 results

The agent edits merge_config.json, then calls:
    pixi run python benchmark.py
and reads the printed score + tsv to decide KEEP or DISCARD.
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import _merge_segments from the pipeline without loading any heavy deps.
# The stubs from test_core.py are reused so we don't pay the Whisper/TTS
# import cost just to run a merge benchmark.
# ---------------------------------------------------------------------------

import unittest.mock as _mock

for _mod in (
    "gradio", "dotenv", "torch", "whisper", "yt_dlp",
    "TTS", "TTS.api", "transformers", "kokoro", "soundfile",
    "numpy",
):
    _sys_mod = __import__("sys").modules
    if _mod not in _sys_mod:
        _sys_mod[_mod] = _mock.MagicMock()

# dotenv stub
__import__("sys").modules["dotenv"].load_dotenv = lambda *a, **kw: None  # type: ignore

from app import _merge_segments  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR   = Path("corpus")
CONFIG_PATH  = Path("merge_config.json")
RESULTS_PATH = Path("results.tsv")

# ---------------------------------------------------------------------------
# Composite metric weights
# ---------------------------------------------------------------------------

W_FIT       = 0.40   # α — % segments whose predicted synth fits in slot
W_BOUNDARY  = 0.25   # β — % segments ending on sentence boundary
W_SWEET     = 0.25   # γ — % segments in the 2.5–8 s "sweet spot"
W_OVER      = 0.10   # δ — % segments over 12 s (penalty, subtracted)

SWEET_MIN   = 2.5    # seconds
SWEET_MAX   = 8.0    # seconds
OVER_THRESH = 12.0   # seconds — pathological over-merge threshold

SENTENCE_ENDINGS = frozenset(".?!…—\"'")

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_corpus(config: dict) -> dict:
    """
    Run merge + score over every JSON in corpus/ and return metric dict.

    Metric components
    -----------------
    fit       : fraction of merged segments whose *predicted* synthesis
                duration (len(text) / chars_per_sec) fits within the
                original slot duration.  High = few timing overflows.

    boundary  : fraction of merged segments whose text ends with
                sentence-terminal punctuation.  High = clean utterances
                that translate well.

    sweet     : fraction of merged segments with duration in [2.5, 8.0] s.
                Penalises both tiny fragments (< 2.5 s) and over-long
                utterances (> 8 s) that stress the translation context window.

    over      : fraction of segments longer than 12 s (subtracted).
                Safety valve against pathological over-merging that sweet
                doesn't catch at the extreme tail.

    S         : composite score, bounded [0, 1] (approximately).
                  S = α·fit + β·boundary + γ·sweet − δ·over
    """
    min_words   = int(config.get("min_words",   4))
    max_words   = int(config.get("max_words",   40))
    max_dur     = config.get("max_duration")        # None or float
    chars_sec   = float(config.get("chars_per_sec", 14.5))

    if max_dur is not None:
        max_dur = float(max_dur)

    fit_n = boundary_n = sweet_n = over_n = total_n = 0

    corpus_files = sorted(CORPUS_DIR.glob("*_whisper.json"))
    if not corpus_files:
        raise FileNotFoundError(
            f"No *_whisper.json files found in {CORPUS_DIR}/. "
            "Run prepare_corpus.py first."
        )

    for path in corpus_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        merged = _merge_segments(
            raw,
            min_words=min_words,
            max_words=max_words,
            max_duration=max_dur,
        )

        for seg in merged:
            dur   = seg["end"] - seg["start"]
            text  = seg["text"].strip()
            pred  = len(text) / chars_sec   # estimated synth duration

            total_n    += 1
            fit_n      += int(pred <= dur)
            boundary_n += int(bool(text) and text[-1] in SENTENCE_ENDINGS)
            sweet_n    += int(SWEET_MIN <= dur <= SWEET_MAX)
            over_n     += int(dur > OVER_THRESH)

    n = max(total_n, 1)
    fit_r      = fit_n      / n
    boundary_r = boundary_n / n
    sweet_r    = sweet_n    / n
    over_r     = over_n     / n

    S = W_FIT * fit_r + W_BOUNDARY * boundary_r + W_SWEET * sweet_r - W_OVER * over_r

    return {
        "S":        round(S,          5),
        "fit":      round(fit_r,      4),
        "boundary": round(boundary_r, 4),
        "sweet":    round(sweet_r,    4),
        "over":     round(over_r,     4),
        "n_segs":   total_n,
        "n_files":  len(corpus_files),
    }


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

_TSV_FIELDS = [
    "timestamp", "status", "S",
    "fit", "boundary", "sweet", "over", "n_segs",
    "min_words", "max_words", "max_duration", "chars_per_sec",
    "description",
]


def log_result(config: dict, scores: dict, status: str, description: str = "") -> None:
    row = {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "status":      status,
        "S":           scores["S"],
        "fit":         scores["fit"],
        "boundary":    scores["boundary"],
        "sweet":       scores["sweet"],
        "over":        scores["over"],
        "n_segs":      scores["n_segs"],
        "min_words":   config.get("min_words",    4),
        "max_words":   config.get("max_words",    40),
        "max_duration": config.get("max_duration", "null"),
        "chars_per_sec": config.get("chars_per_sec", 14.5),
        "description": description or "",
    }
    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_best_S() -> float | None:
    """Return the best (highest) S among all KEEP rows, or None."""
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


def print_last(n: int = 5) -> None:
    if not RESULTS_PATH.exists():
        print("No results.tsv yet.")
        return
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows[-n:]:
        print(
            f"[{row['timestamp']}] {row['status']:8s} "
            f"S={row['S']:7s}  fit={row['fit']}  boundary={row['boundary']}  "
            f"sweet={row['sweet']}  over={row['over']}  "
            f"n={row['n_segs']}  "
            f"min_w={row['min_words']} max_w={row['max_words']} "
            f"max_dur={row['max_duration']}  "
            f"{row.get('description', '')}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubweave Loop 1 benchmark — score merge_config.json"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Record this run as BASELINE (do this once before the first agent run)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the last 5 results and exit",
    )
    parser.add_argument(
        "--description", "-d",
        default="",
        help="Short description of this experiment (logged to results.tsv)",
    )
    args = parser.parse_args()

    if args.status:
        print_last(10)
        return

    # Load config
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Create it first.", file=sys.stderr)
        sys.exit(1)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))

    # Score
    print(f"Scoring corpus ({CORPUS_DIR})…")
    scores = score_corpus(config)

    # Determine status
    if args.baseline:
        status = "BASELINE"
    else:
        best = read_best_S()
        if best is None:
            print(
                "WARNING: No BASELINE or KEEP row found in results.tsv. "
                "Run with --baseline first.",
                file=sys.stderr,
            )
            status = "?"
        else:
            status = "KEEP" if scores["S"] > best else "DISCARD"

    # Log
    log_result(config, scores, status, description=args.description)

    # Print
    print(
        f"\n  S          = {scores['S']:.5f}"
        f"  {'← NEW BEST' if status == 'KEEP' else ''}"
    )
    print(f"  fit        = {scores['fit']:.4f}   (α={W_FIT})")
    print(f"  boundary   = {scores['boundary']:.4f}   (β={W_BOUNDARY})")
    print(f"  sweet      = {scores['sweet']:.4f}   (γ={W_SWEET})")
    print(f"  over       = {scores['over']:.4f}   (δ={W_OVER}, subtracted)")
    print(f"  n_segs     = {scores['n_segs']}  across {scores['n_files']} files")
    print(f"\n  status     = {status}")
    if status == "DISCARD":
        best = read_best_S()
        print(f"  best so far = {best:.5f}")

    print(f"\n  Logged → {RESULTS_PATH}")


if __name__ == "__main__":
    main()