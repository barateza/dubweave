# Dubweave — Autoresearch Loop 1: Merge Threshold Optimization

## Your role

You are an autonomous research agent optimizing the segment merge parameters
for the Dubweave PT-BR dubbing pipeline. You run experiments, evaluate results,
and keep improvements. You work in a loop without human input until you find
no further improvement or reach 50 experiments.

## What you are optimizing

The file `merge_config.json` controls how raw Whisper segments are merged into
translation utterances before synthesis. You edit this file, run the benchmark,
and decide whether to keep or discard each change.

## The metric

`pixi run python benchmark.py -d "<your hypothesis>"` scores the current config
and prints a composite score S ∈ [0, 1]. Higher is better.

S = 0.40·fit + 0.25·boundary + 0.25·sweet − 0.10·over

- fit:      fraction of segments whose predicted synthesis fits in slot
- boundary: fraction ending on sentence-terminal punctuation (., ?, !, …)
- sweet:    fraction with duration in [2.5, 8.0] seconds
- over:     fraction longer than 12 seconds (penalty)

The benchmark prints KEEP or DISCARD automatically by comparing against the
best S in results.tsv. It also appends a row to results.tsv.

## The mutable file

Only edit `merge_config.json`. Do not touch any other file.

```json
{
  "min_words": <int>,
  "max_words": <int>,
  "max_duration": <float or null>,
  "chars_per_sec": <float>
}
```

Parameter meanings:

- min_words:    minimum word count before a punctuated segment flushes
- max_words:    hard flush regardless of punctuation (prevents run-ons)
- max_duration: flush if merged duration exceeds this (seconds). null = disabled
- chars_per_sec: Kokoro speech rate for timing prediction. LOCKED at 25.0 — do not change.

## Current best (as of last human run)

S = 0.80699  →  min_words=8, max_words=40, max_duration=null, chars_per_sec=25.0

Known from manual sweep:

- chars_per_sec is calibrated. Do not touch it.
- min_words=12 → over spikes, boundary drops → DISCARD
- max_words=30, 35 → boundary drops → DISCARD
- max_duration=7.0, 10.0 → boundary drops → DISCARD
- boundary is sensitive: anything that forces a mid-sentence flush loses β=0.25

## Search strategy

1. Read results.tsv to understand what has been tried.
2. Propose one hypothesis — a specific reason why a change might improve S.
3. Edit merge_config.json with exactly one meaningful change.
4. Run: pixi run python benchmark.py -d "<short hypothesis description>"
5. If KEEP: note what worked and why.
   If DISCARD: revert merge_config.json to the last KEEP values.
6. Repeat from step 1.

## Hypothesis ideas to explore (not exhaustive)

- min_words=6: looser than 8, might recover some boundary without losing sweet
- min_words=9, 10: tighter than 8, worth checking if over stays low
- max_words=45, 50: looser cap, might help long technical utterances
- max_duration=8.5, 9.0: narrow duration cap that avoids mid-sentence flush zone
- Combinations: min_words=6 + max_words=35

## Rules

- Always include a hypothesis in the -d description. Example:
  -d "min_words=6: looser flush, hypothesis: boundary holds, sweet improves"
- Never change chars_per_sec.
- Never edit any file other than merge_config.json.
- After a DISCARD, always restore the last KEEP config before the next experiment.
- Stop after 50 experiments or if 5 consecutive experiments DISCARD with S < best − 0.005.
- When done, print a summary: best config found, S achieved, and key findings.

## How to check current best config

```powershell
pixi run python benchmark.py --status
```

## How to restore last KEEP config

Read results.tsv, find the last KEEP row, and set merge_config.json to those values.
