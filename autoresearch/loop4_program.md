# Dubweave — Autoresearch Loop 4: Kokoro Speech Rate Calibration

## Your role

You are an autonomous research agent calibrating the `chars_per_sec` constant used
by `apply_timing_budget` in the Dubweave PT-BR dubbing pipeline.

You work in a loop without human input until you find the optimal value or run
50 experiments.

---

## The problem

`apply_timing_budget` in `app.py` predicts whether a translated segment will
overflow its time slot by computing:

```
predicted_duration = len(text) / XTTS_CHARS_PER_SEC
```

If the constant is wrong, the timing budget either over-prunes (constant too low)
or under-prunes (constant too high). Loop 1 found `25.0` empirically from merge
scoring. Loop 4 measures the **real** synthesis duration and finds the value that
minimises prediction error.

---

## The metric

**MAE (mean absolute error) in seconds — lower is better.**

```
mae_all = mean over all voices and sentences of |predicted_duration - actual_duration|
```

where `predicted_duration = len(sentence_chars) / chars_per_sec`.

The benchmark also reports per-voice MAE and **bias**:
- **Positive bias**: predicted duration > real → constant is too high → try lower
- **Negative bias**: predicted duration < real → constant is too low → try higher
- **|bias| < 0.05 s**: well calibrated for this voice

---

## Setup (one-time)

Before starting the agent loop, run:

```powershell
pixi run python calibrate_tts.py --measure
```

This synthesises 40 PT-BR sentences with all 3 Kokoro voices (`pf_dora`,
`pm_alex`, `pm_santa`) and caches the real durations to
`corpus/loop4_durations.json`. Takes ~60–120 seconds. **Never re-run --measure**
once caching is done — the corpus must be fixed for results to be comparable.

Then establish the baseline:

```powershell
pixi run python calibrate_tts.py --baseline -d "Loop 1 empirical value"
```

---

## Optional: find a starting point fast

```powershell
pixi run python calibrate_tts.py --find-best
```

This prints a full MAE table for `chars_per_sec` in [15.0, 35.0] at 0.5-step
intervals without writing to results.tsv. Use it to identify the rough optimum
before fine-grained search.

---

## The mutable file

Only edit `loop4_config.json`:

```json
{
  "chars_per_sec": <float>
}
```

Do not touch any other file.

---

## Scoring a config

```powershell
pixi run python calibrate_tts.py -d "<hypothesis>"
```

The benchmark prints KEEP or DISCARD automatically by comparing `mae_all`
against the best value in `loop4_results.tsv`. It also reports per-voice MAE
and bias so you can detect asymmetric calibration across voices.

---

## Search strategy

1. Run `--find-best` once to identify the rough optimum.
2. Set `loop4_config.json` to the rough optimum and run `--baseline`.
3. Then do fine-grained search:
   - Step 0.5 around the optimum: e.g. ±2.0 in 0.5 increments
   - If a value is KEEP, narrow further: ±0.5 in 0.1 increments
   - If per-voice biases diverge significantly (> 1.0 s difference between voices),
     record each voice's individual optimum — the pipeline may need per-voice constants.

## Key question

**Do all three voices share the same optimal `chars_per_sec`, or do they diverge?**

If `pf_dora`, `pm_alex`, and `pm_santa` each minimise MAE at different values,
report the per-voice optima. The agent notes in its final summary whether a single
constant is adequate or whether per-voice calibration is warranted.

---

## Rules

- Always include a hypothesis in every `-d` description. Example:
  `-d "cps=26.5: dora bias was +0.3 → predicting long → try lower"`
- Never edit any file except `loop4_config.json`.
- After a DISCARD, always restore the last KEEP value before the next experiment.
- Stop after 50 experiments or 5 consecutive DISCARDs within 0.5 of the optimum.
- At the end, print:
  - The optimal `chars_per_sec` (single value or per-voice)
  - The achieved `mae_all`
  - Whether per-voice calibration is recommended
  - The recommended `XTTS_CHARS_PER_SEC` value to patch into `app.py`

---

## Applying the result to the pipeline

After finding the optimum, update `app.py`:

1. Find the line: `XTTS_CHARS_PER_SEC = 18.0` (or whatever current value)
2. Replace with the measured value, e.g.: `XTTS_CHARS_PER_SEC = 26.5`
3. Add a comment:
   ```python
   XTTS_CHARS_PER_SEC = 26.5  # Loop 4 calibrated: MAE=0.XXXs on 40-sentence PT-BR corpus
   ```

If per-voice calibration is warranted, create a dict instead:
```python
KOKORO_CPS_BY_VOICE = {
    "pf_dora":  26.5,
    "pm_alex":  25.0,
    "pm_santa": 24.0,
}
XTTS_CHARS_PER_SEC = 25.5  # fallback for XTTS / unknown voice
```

---

## Status commands

```powershell
# Print last 10 rows
pixi run python calibrate_tts.py --status

# Restore last KEEP config
# Read loop4_results.tsv, find last KEEP row, set loop4_config.json accordingly
```
