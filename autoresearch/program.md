# Dubweave — Autoresearch Loop 1 v2: Gap-Aware Merge Optimization

## Your role

You are an autonomous research agent optimizing the segment merge parameters
for the Dubweave PT-BR dubbing pipeline. You run experiments, evaluate results,
and keep improvements. You work in a loop without human input until you find
no further improvement or reach 50 experiments.

## Context: why this loop exists

Loop 1 v1 exhausted the {min_words, max_words, max_duration} search space.
The best result was S = 0.80914 (min_words=8, max_words=50, boundary=0.9882,
sweet=0.7135). `sweet` (71.4%) is where meaningful headroom remains.

A new parameter, `gap_sec`, is now active in `_merge_segments`. When the
silence between two consecutive Whisper segments exceeds `gap_sec`, the
current buffer is force-flushed before the next segment is appended. This
respects natural speaker pauses as hard utterance boundaries — independent
of punctuation or word count.

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

    {
      "min_words":    <int>,
      "max_words":    <int>,
      "max_duration": <float or null>,
      "chars_per_sec": <float>,
      "gap_sec":      <float or null>
    }

Parameter meanings:

- min_words:    minimum word count before a punctuated segment flushes
- max_words:    hard flush regardless of punctuation (prevents run-ons)
- max_duration: flush if merged duration exceeds this (seconds). null = disabled
- chars_per_sec: Kokoro speech rate for timing prediction. LOCKED at 25.0.
- gap_sec:      NEW — flush when silence between Whisper segments ≥ gap_sec.
                null = disabled (original behaviour). Unit: seconds.

## Current best (Loop 1 v1 result — the baseline for this loop)

    S = 0.80914  →  min_words=8, max_words=50, max_duration=null, gap_sec=null, chars_per_sec=25.0
    fit=0.9773, boundary=0.9882, sweet=0.7135, over=0.0719, n_segs=2199

## Before your first experiment

Run the preview sweep to understand how gap_sec moves the needle:

  pixi run python benchmark.py --sweep

This prints a table of S, fit, boundary, sweet, over for gap_sec in
{null, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0} without writing to results.tsv.
Use this to pick your first real experiment.

## How gap_sec affects the metrics

- null — baseline; no change from v1 best

- 0.3–0.5 s — aggressive splitting; many tiny sub-2.5s fragments → sweet likely drops

- 0.7–1.0 s — targets mid-sentence pauses; sweet may improve; boundary could dip if gaps mid-sentence

- 1.2–1.5 s — targets paragraph-level pauses; likely boundary-safe zone

- 2.0 s+ — only fires at very long topic transitions; low impact

## Search strategy

### Phase 1 — gap_sec solo sweep (≤ 15 experiments)

With {min_words=8, max_words=50, max_duration=null, chars_per_sec=25.0} fixed:

1. Start with gap_sec=1.0 — middle of the plausible range.
2. If sweet improves and boundary holds → try lower: 0.7, then 0.5.
3. If sweet drops or boundary drops → try higher: 1.2, then 1.5, then 2.0.
4. Note the best gap_sec value found.

### Phase 2 — gap_sec × min_words interaction (≤ 20 experiments)

Hypothesis: gap flushes create some short sub-min_words utterances.
Lowering min_words slightly might help these coalesce when there's no gap.

With the best gap_sec fixed, explore:

- min_words=6 (looser, more fragments merge on punctuation)
- min_words=7
- min_words=10 (tighter, fewer small merges)

### Phase 3 — gap_sec × max_words (≤ 10 experiments)

Hypothesis: gap flushes reset the word counter cleanly; max_words may be
safe to relax or tighten.

- max_words=45 with best gap_sec + min_words
- max_words=55 with best gap_sec + min_words

### Phase 4 — cleanup (≤ 5 experiments)

Once a promising region is found, try small ±0.1 s perturbations around
the best gap_sec to find the local optimum.

## Known constraints from Loop 1 v1 (do not re-test these)

- chars_per_sec: LOCKED at 25.0 — do not change.
- max_duration=7.0, 8.5, 9.0 → boundary drops badly → never re-test max_duration.
- min_words > 10 → over spikes → avoid.
- max_words < 40 → boundary drops (mid-sentence cap) → avoid.
- Combinations tested without gap_sec are exhausted. Only add gap_sec now.

## Rules

- Always include a hypothesis in the -d description. Example:
  -d "gap_sec=1.0: target paragraph pauses; hypothesis: sweet improves, boundary safe"
- Never change chars_per_sec.
- Never edit any file other than merge_config.json.
- After a DISCARD, always restore the last KEEP config before the next experiment.
- Stop after 50 experiments or if 5 consecutive experiments DISCARD with S < best − 0.005.
- When done, print a summary: best config found, S achieved, and key findings.

## Restore last KEEP config

Read results.tsv, find the last KEEP row, and set merge_config.json to those values.
Alternatively: `git stash` / `git checkout merge_config.json` if the repo is clean.

## How to check current state

  pixi run python benchmark.py --status

## Experiment log format (for your own notes)

After each run, note:
    EXP #N: gap_sec=X min_words=Y → S=Z (KEEP/DISCARD) — what changed and why

This helps you avoid re-testing known dead ends.
