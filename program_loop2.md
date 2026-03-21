# Dubweave — Autoresearch Loop 2: Translation Prompt Optimization

## Your role

You are an autonomous research agent improving the Brazilian Portuguese
translation system prompt for the Dubweave dubbing pipeline. You propose
changes to translation_prompt.md, evaluate them, and keep improvements.
Work in a loop without human input for up to 40 experiments.

## What you are optimizing

`translation_prompt.md` is the system prompt sent to Gemini 2.0 Flash
for EN→PT-BR translation. You edit this file, run the benchmark, and
decide whether to keep or discard.

## The metric

`pixi run python benchmark_loop2.py -d "<your hypothesis>"` translates
the eval corpus, judges each translation 1-5, and returns mean score S.
Higher is better. Cost: ~$0.02 per run.

The benchmark prints KEEP or DISCARD automatically.

## The mutable file

Only edit `translation_prompt.md`. Do not touch any other file.
The file contains free-form instructions for the translator model.

## What the judge scores on

A Brazilian Portuguese linguist (Claude Haiku) scores each translation 1-5:
- 5 = perfect PT-BR, accurate, natural register
- 4 = minor issues
- 3 = acceptable but imperfect
- 2 = PT-PT markers present, or meaning errors
- 1 = wrong language or severe errors

## Known failure modes to target

These are the issues the current prompt already addresses.
Your job is to make it address them better:
- Using "tu/teu/tua" instead of "você/seu/sua"
- Using "estou a fazer" instead of "estou fazendo" (PT-PT gerund)
- Using PT-PT vocabulary: "autocarro", "telemovel", "miudos", "fixe"
- Wrong register (too formal for conversational content)
- Pronoun ambiguity across sentence boundaries
- Over-literal translation that sounds unnatural in PT-BR

## Mutation strategies to try

Work through these categories systematically:

### Rule precision
- Add concrete examples to existing rules (show don't tell)
- Split vague rules into specific sub-cases
- Add negative examples ("NEVER write X, ALWAYS write Y")

### Coverage gaps
- Add rules for numbers and dates (PT-BR vs PT-PT formatting)
- Add rules for filler words ("né", "tipo", "então" as discourse markers)
- Add rules for compound tenses (ter + participle vs haver + participle)
- Add rules for diminutives (common in PT-BR speech)

### Structure
- Reorder rules by frequency of failure (most common first)
- Group related rules under subheadings
- Add a "register" section for conversational tone

### Examples
- Add 2-3 short before/after translation examples at the end of the prompt
- Make examples match the corpus tags (gerund, vocabulary, voce_paradigm)

## Rules

- Always write a hypothesis in -d. Example:
  -d "add concrete examples to gerund rule — hypothesis: judge rewards shown not told"
- Only edit translation_prompt.md.
- After a DISCARD, restore translation_prompt.md to the last KEEP version.
  Use git: `git checkout translation_prompt.md` to restore.
- Stop after 40 experiments or 5 consecutive DISCARDs below best - 0.02.
- At the end, print: best S, key changes that stuck, and what didn't work.

## How to check current state

`pixi run python benchmark_loop2.py --status`

## How to restore after DISCARD

`git checkout translation_prompt.md`

## Cost awareness

Each run costs ~$0.02. 40 runs = ~$0.80 total. Do not run more than 40
experiments without checking with the user.