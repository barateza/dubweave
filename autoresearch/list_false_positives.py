from benchmark_loop3 import apply_rules
import json, re
from pathlib import Path

RULES = Path('normalizer_rules.json')
CLEAN = Path('corpus') / 'ptbr_clean_corpus.txt'

cfg = json.loads(RULES.read_text(encoding='utf-8'))
rules = cfg.get('rules', [])

if not CLEAN.exists():
    print('Clean corpus not found')
    raise SystemExit(1)

lines = [re.sub(r"^\d+\.\s*", '', l).strip() for l in CLEAN.read_text(encoding='utf-8').splitlines() if l.strip()]

changed = []
for i, s in enumerate(lines[:200], start=1):
    out = apply_rules(s, rules)
    if out.strip() != s.strip():
        changed.append((i, s, out))

print(f'Found {len(changed)} changed clean sentences (false positives):')
for idx, orig, out in changed:
    print(f'[{idx}]')
    print('  before:', orig)
    print('  after: ', out)
