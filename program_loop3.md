# Dubweave — Autoresearch Loop 3: PT-BR Normalizer Rule Discovery

## Your role

You are an autonomous research agent improving the Brazilian Portuguese normalizer
for the Dubweave dubbing pipeline. You add, remove, or refine regex rules in
`normalizer_rules.json`, evaluate them, and keep improvements.
Work in a loop without human input for up to 60 experiments.

## What you are optimizing

`normalizer_rules.json` contains a list of regex substitution rules applied
after every translation pass (NLLB or LLM) to convert PT-PT constructions
to PT-BR equivalents. You edit this file, run the benchmark, and decide
whether to keep or discard.

## The metric

`pixi run python benchmark_loop3.py -d "<your hypothesis>"` scores the current
rule set against two corpora and prints composite score S.

```text
S = detection_rate - 0.50 * false_positive_rate
```

- **detection_rate**: fraction of 50 injected PT-PT sentences where at least
  one correction was made (higher is better)
- **false_positive_rate**: fraction of 200 native PT-BR sentences incorrectly
  modified (lower is better, heavily penalized)
- **perfect_rate**: fraction where normalized output exactly matches reference
  (diagnostic only, not in S)

The benchmark prints KEEP or DISCARD automatically.
Cost: zero — pure Python regex, no API calls, ~1 second per run.

## The mutable file

Only edit `normalizer_rules.json`. Do not touch any other file.

### Rule format

```json
{"pattern": "\\btu\\b", "replacement": "você", "note": "pronoun"}
```

- `pattern`: Python regex (remember to double-escape backslashes in JSON)
- `replacement`: the PT-BR replacement string
- `note`: short human-readable label for the rule
- Case is always handled: the rule preserves the capitalisation of the first letter

### Gerund rule (special type)

The gerund rule has `"type": "gerund"` and a `"verbs"` list. The benchmark
converts PT-PT `a + infinitive` → PT-BR gerund automatically for all verbs
in the list. To add a new verb, add it to the `"verbs"` array.

## Current baseline

Run `pixi run python benchmark_loop3.py --status` to see the current best.

## What the corpus covers

The 50-sentence injection corpus tests these PT-PT phenomena:

- `tu/teu/tua/teus/tuas/vós` — pronoun forms
- `estás/gostas/fazes/podes/queres/sabes/tens/vens/dizes/vês/vais/ficas` — 2nd person verbs
- `a fazer/a ver/a estar` — infinitive constructions (gerund rules)
- `autocarro/comboio/telemóvel/miúdos/fixe/giro/casa de banho/passeio/perceber` — vocabulary
- `somente/apenas/imensamente` — register words

The 200-sentence false positive corpus is native PT-BR (phonetically balanced
sentences). Rules must NOT change these sentences.

## What to try

Work through these categories systematically. Always try ONE hypothesis at a time.

### Uncovered vocabulary (highest impact)

These PT-PT terms are not in the current rules — add them:

- `apanhar` → `pegar` (common PT-PT verb, e.g. "apanhar o autocarro")
- `conduzir` → `dirigir` (driving)
- `frigorífico` → `geladeira`
- `elevador` → `elevador` (same — verify this is NOT a false positive risk)
- `pequeno-almoço` → `café da manhã`
- `talho` → `açougue`
- `cinema` → same in both, skip
- `cão` → `cachorro` (dog — risky, verify false positive first)
- `rapaz/rapariga` → `menino/menina` or `cara/garota`
- `andar` (floor/storey meaning) → careful, `andar` also means "to walk" in PT-BR

### Uncovered verb forms

Current rules miss some 2nd person forms:

- `trazes` → `traz`
- `ouves` → `ouve`
- `sentes` → `sente`
- `moves` → `move`
- `serves` → `serve`
- `vivias` → `vivia` (imperfect)
- `estavas` → `estava`
- `ias` → `ia` (imperfect of ir)
- `tinhas` → `tinha`

### Gerund coverage gaps

Add missing verbs to the gerund `"verbs"` list:

- `fumar`, `comer`, `beber`, `escrever`, `ler`, `subir`, `sair`, `abrir`,
  `conduzir`, `sorrir`, `construir`, `produzir`, `introduzir`

### Pattern refinement

- `apenas` → `só` may be too aggressive (context-dependent).
  Test removing it: does fp_rate drop without hurting detection?
- `passeio` → `calçada` may false-positive on "dar um passeio" (take a walk).
  Consider adding word boundary context: `\\bpasseio\\b` only when not preceded by "dar um"

## Rules

- Always write a hypothesis in -d. Example:
  -d "add apanhar→pegar: common PT-PT verb missing from rules"
- Only edit normalizer_rules.json.
- After a DISCARD, restore normalizer_rules.json using:
  `git checkout normalizer_rules.json`
- After a KEEP, commit:
  `git add normalizer_rules.json && git commit -m "KEEP: <description>"`
- Stop after 60 experiments or 5 consecutive DISCARDs below best - 0.01.
- Use `--verbose` flag to see exactly which sentences are missed or wrong:
  `pixi run python benchmark_loop3.py --verbose -d "<hypothesis>"`

## How to read the verbose output

**Missed detections** = PT-PT sentences the rules didn't catch.
These show you what new rules to write.

**Imperfect corrections** = rules caught something but the output doesn't
exactly match the reference. This is less critical — partial correction is
still better than none.

## How to check current state

```powershell
pixi run python benchmark_loop3.py --status
pixi run python benchmark_loop3.py --verbose -d "diagnostic"
```

## One-time setup

Before the first agent run, record the baseline:

```powershell
pixi run python benchmark_loop3.py --baseline -d "original 30-rule set"
```

Also ensure the clean corpus file exists:

```text
corpus/ptbr_clean_corpus.txt
```

This should contain the 1000 phonetically balanced PT-BR sentences,
one per line, formatted as "0001. Sentence here."
If it doesn't exist, the false positive test is skipped (fp_rate = 0.0).
