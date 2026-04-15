# Dubweave — Autoresearch Loop 1 v2: Edge-TTS Merge Sweep

## Your role

You are an autonomous research agent whose job is to finish the Edge-TTS merge optimization that stalled in Loop 1. Loops 2 and 3 are already clean, the handoff data matches every number in the TSVs, Hermetic translation normalization is at 90% 4+-score coverage, and the Kokoro baseline (Loop 1 v2) is documented and stable with gap_sec=3.0, max_words=100 and chars_per_sec=25.0. What remains is a repeatable way to merge segments so the newly calibrated Edge-TTS voices (Francisca/Antonio @ 11.1 cps and Thalita @ 13.1 cps) actually fit into the original Kokoro slots.

## Why this loop exists now

- The Edge-TTS merge tuner stopped with `merge_config.json` still pointing at the discarded `max_duration=10.0` experiment and `chars_per_sec=18.3`. That experiment produced misleading `fit` numbers because it pretended the Edge voices were faster than they really are. Production Edge runs still send `MERGE_CONFIGS["edge"]` through Kokoro-like parameters, so heavy trimming and truncation happen on every run.
- At the real production rate (11.1 cps, the calibrated number stored in `VOICE_CALIBRATION` inside `app.py` for Francisca and Antonio), the benchmark reports `fit≈0.086`. In plain terms, roughly 91% of the Kokoro-style merged segments are too long for Edge-TTS to synthesize into their slots. That is not a bug in the benchmark—it is the exact pain point we need to fix.

## True baseline you must restore before running experiments

Reset `merge_config.json` to the values that match the Edge-TTS production routing in `app.py`. This file is the only mutable artifact for this loop. The baseline configuration is:

```
{
  "min_words":    10,
  "max_words":    100,
  "max_duration": null,
  "chars_per_sec": 11.1,
  "gap_sec":      2.0,
  "voice":        "pt-BR-FranciscaNeural"
}
```

- `min_words=10` reflects the Kokoro baseline that already produces readable utterances. Keep it locked unless a future sweep proves otherwise.
- `max_duration=null` means we rely on word counts and silence gaps instead of a hard second-based cap.
- `chars_per_sec` must stay at 11.1 while working on the Francisca/Antonio branch. The Thalita calibration (13.1 cps) is noted in `VOICE_CALIBRATION`, but the current delivery pipeline still routes via Francisca, so focus there.
- `gap_sec=2.0` is the current production gap. You will sweep it tighter to see where silence-triggered flushes help your fit.
- The `voice` field exists so the benchmark logs which routing path you are tuning. Keep it set to Francisca while you are establishing the new merge strategy.

After every DISCARD, restore this snippet before running the next experiment. That keeps your baseline clean and keeps the downstream pipeline aligned with what `MERGE_CONFIGS["edge"]` should eventually mirror.

## Benchmarks and metrics

We still target the composite score `S = fit · 0.40 + boundary · 0.25 + sweet · 0.25 − over · 0.10`. With the 11.1 cps baseline, `fit` starts at ~0.086, which is the immediate limitation Edge synthesis faces; sweet/boundary/over numbers are not terrible, they simply sit behind a `fit` wall.

Your job is to raise `fit` by shortening utterances via `max_words` and intelligent gap flushing. If `fit` improves, the composite `S` should rise enough to flip KEEP on the benchmark. The benchmark will automatically append each row to `results.tsv` and print KEEP/DISCARD for you.

## Search strategy

1. **Max words sweep (long → short).** Starting from the baseline `max_words=100`, iteratively edit `merge_config.json` to try the tighter caps {40, 35, 30, 25, 20}. Run a benchmark for each value while keeping `gap_sec=2.0`. This shrinks utterances by forcing earlier flushes, which is the main tool for raising `fit` at 11.1 cps.
2. **Gap_sec refinement.** Once you have signal from the max_words sweep, run the same downward sweep while tightening `gap_sec` in {1.5, 1.0}. The idea is to flush at shorter silences so you break long Kokoro clusters into smaller Edge-friendly chunks. For each gap value, re-evaluate the max_words series (so you end up exploring the matrix gap × max_words in a targeted way). Always keep `min_words=10`, `max_duration=null`, and `chars_per_sec=11.1`.
3. **CPS sanity.** Never change `chars_per_sec` from 11.1 while tuning Edge-Francisca/Antonio. If you eventually tune Thalita, spin up a new loop referencing 13.1 cps.

The goal is to find the shortest `max_words` that still keeps polite transitions, then confirm what `gap_sec` prevents re-joining excessively long spans. Keep track of the best `max_words`/`gap_sec` pair and make sure to update `merge_config.json` when you find a KEEP configuration that beats the current best `S`.

## Running experiments

1. Edit `merge_config.json` to one of the target parameter sets (max_words/gap combinations listed above). `min_words`, `max_duration`, `chars_per_sec`, and `voice` stay constant.

2. Run the benchmark with a hypothesis-laden description:

```text
pixi run python benchmark.py -d "gap_sec=1.5 max_words=30: hypothesis lower gap + tighter max_words raises fit for 11.1 cps"
```

3. After the run, inspect the console output. The benchmark prints `fit`, `boundary`, `sweet`, `over`, and `S`, and whether the config is `KEEP` or `DISCARD`. Record the results using your own log format (e.g., `EXP #N: gap_sec=1.5 max_words=30 → S=0.xxx (KEEP) — what changed`).

4. If the run is a DISCARD, restore the baseline snippet before making the next edit. If it is a KEEP and the `S` beats the previous best, leave the new values in `merge_config.json` so that future runs start from the improved baseline.

5. You may run `pixi run python benchmark.py --status` at any time to check how many experiments have been logged and where the current best `S` sits. The `--sweep` flag is helpful for previewing `gap_sec` behavior before writing to `results.tsv`.

## Rules

- Only edit `merge_config.json`. The rest of the repo—including `app.py` and any TSV results—is treated as immutable evidence for this loop.
- Always include a hypothesis in the `-d` description. That makes your reasoning searchable in the benchmark logs.
- After each DISCARD, restore the baseline snippet (min_words=10, max_words=100, max_duration=null, chars_per_sec=11.1, gap_sec=2.0, voice=pt-BR-FranciscaNeural). A DISCARD followed by un-restored parameters makes the next run invalid from a comparison standpoint.
- Stop if you reach 50 experiments or if you hit five DISCARDs in a row with `S` more than 0.005 below the best-known config.
- When done, report the best config found, its `S`, and the key insight (e.g., "gap_sec=1.0, max_words=30 gave the first fit > 0.6 at 11.1 cps"). That summary closes the loop for whoever follows you.

## After you find an improvement

- Update `MERGE_CONFIGS["edge"]` in `app.py` so production uses the merged parameters that actually fit 11.1 cps. The `merge_config.json` file is where you experiment, and `MERGE_CONFIGS["edge"]` is where the pipeline reads the result. Keeping both aligned prevents future regressions and the "Edge runs are truncated" symptom from reappearing.
- If you ever tune Thalita separately, treat it as a new loop with `chars_per_sec=13.1` and the appropriate voice routing.
