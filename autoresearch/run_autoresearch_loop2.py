#!/usr/bin/env python3
"""Autoresearch loop runner for Dubweave — loop2

This script mutates `translation_prompt.md`, runs `benchmark_loop2.py`,
and keeps or discards changes according to the benchmark's KEEP/DISCARD.

Usage: python run_autoresearch_loop2.py
"""
import csv
import subprocess
import sys
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "translation_prompt.md"
RESULTS = ROOT / "results_loop2.tsv"
BENCH = [sys.executable, str(ROOT / "benchmark_loop2.py")]

MAX_EXPERIMENTS = 40
MAX_DISCARD_STREAK = 5

import re


def replace_rule_block(text: str, rule_num: int, new_block: str) -> str:
    """Replace numbered rule `rule_num` block with `new_block` preserving surrounding structure."""
    lines = text.splitlines()
    start = None
    end = None
    pattern = re.compile(rf"^{rule_num}\.\s")
    next_pattern = re.compile(r"^\d+\.\s")
    for i, ln in enumerate(lines):
        if start is None and pattern.match(ln):
            start = i
            continue
        if start is not None and i > start and next_pattern.match(ln):
            end = i
            break
    if start is None:
        return text
    if end is None:
        end = len(lines)
    new_lines = lines[:start] + new_block.strip().splitlines() + lines[end:]
    return "\n".join(new_lines) + "\n"


def mut_refine_voce(text: str) -> str:
    block = (
        "2. Always use 'você' for second person singular. NEVER use 'tu', 'teu', 'tua', 'vós'.\n"
        "   - CORRECT: 'Você fez isso' / 'Você está bem?'\n"
        "   - WRONG:   'Tu fizeste isso' / 'Tu estás bem?'\n"
        "   - NOTE: When the source uses informal contractions, keep informal register but still use 'você' forms.\n"
        "   - NEGATIVE EXAMPLES (NEVER): avoid 'tu', avoid 'teu/tua' except for quoted dialectal speech.\n"
    )
    return replace_rule_block(text, 2, block)


def mut_refine_gerund(text: str) -> str:
    block = (
        "3. Use gerund forms for ongoing actions in PT-BR. NEVER use European infinitive constructions except in literal quotations.\n"
        "   - CORRECT: 'estou fazendo', 'ele está vendo', 'fica reclamando'\n"
        "   - WRONG:   'estou a fazer', 'ele está a ver', 'fica a reclamar'\n"
        "   - EXCEPTIONS: For certain stative verbs or idioms where PT-BR prefers infinitive, follow natural usage (examples in later section).\n"
    )
    return replace_rule_block(text, 3, block)


def mut_refine_vocab(text: str) -> str:
    block = (
        "4. Use Brazilian vocabulary and provide explicit negative examples.\n"
        "   - PREFERRED: 'ônibus' not 'autocarro'; 'celular' not 'telemóvel'; 'trem' not 'comboio'; 'banheiro' not 'casa de banho'\n"
        "   - NEGATIVE EXAMPLES (NEVER): do NOT use 'autocarro','telemóvel','miúdos','fixe' unless quoting PT-PT.\n"
        "   - CONTEXT: If the source implies a regional term, prefer the neutral PT-BR equivalent.\n"
    )
    return replace_rule_block(text, 4, block)


def mut_refine_register(text: str) -> str:
    block = (
        "6. Register: match the original's formality but with PT-BR naturalness.\n"
        "   - For conversational dialogue: prefer informal constructions, contractions, and natural discourse markers (e.g., 'né', 'tipo', 'então') where appropriate.\n"
        "   - For formal text: keep formal grammar and avoid colloquialisms.\n"
        "   - EXAMPLE: EN dialog 'You okay?' → PT-BR: 'Você tá bem?' (informal) not 'Você está bem?' when source is casual.\n"
    )
    return replace_rule_block(text, 6, block)


def mut_numbers_dates(text: str) -> str:
    # insert as rule 7 replacement to include numbers/dates
    block = (
        "7. Preserve punctuation and formatting; use PT-BR numeric/date formats where explicitly present in source context.\n"
        "   - Dates: DD/MM/YYYY. Numbers: use ',' as decimal separator and '.' as thousand separator when formatting is required.\n"
        "   - Preserve segment numbering and punctuation exactly unless transformation is requested.\n"
    )
    return replace_rule_block(text, 7, block)


MUTATIONS = [
    ("refine_voce", mut_refine_voce),
    ("refine_gerund", mut_refine_gerund),
    ("refine_vocab", mut_refine_vocab),
    ("refine_register", mut_refine_register),
    ("numbers_dates_inplace", mut_numbers_dates),
]


def read_best_S():
    if not RESULTS.exists():
        return None
    best = None
    with open(RESULTS, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('status') in ('KEEP', 'BASELINE'):
                try:
                    s = float(row['S'])
                    if best is None or s > best:
                        best = s
                except Exception:
                    pass
    return best


def git(cmd_args, check=True):
    return subprocess.run(['git'] + cmd_args, cwd=ROOT, check=check, capture_output=True, text=True)


def ensure_git_config():
    try:
        git(['rev-parse', '--is-inside-work-tree'])
    except Exception:
        print('ERROR: not a git repository.', file=sys.stderr)
        sys.exit(1)
    # set local user if missing
    try:
        git(['config', 'user.email', 'autobot@example.com'])
        git(['config', 'user.name', 'autobot'])
    except Exception:
        pass


def run_benchmark(description, baseline=False):
    cmd = BENCH + (["--baseline"] if baseline else []) + ["-d", description]
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def last_result():
    if not RESULTS.exists():
        return None
    rows = []
    with open(RESULTS, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows:
        return None
    return rows[-1]


def mutate_prompt(base_text: str, mutation) -> str:
    name, func = mutation
    try:
        return func(base_text)
    except Exception:
        return base_text


def main():
    ensure_git_config()

    if 'OPENROUTER_API_KEY' not in os.environ:
        print('ERROR: OPENROUTER_API_KEY not set in environment.', file=sys.stderr)
        sys.exit(1)

    # Ensure baseline exists
    best = read_best_S()
    if best is None:
        print('No baseline found. Committing current prompt as BASELINE and running baseline...')
        # commit current prompt as baseline
        subprocess.run(['git', 'add', str(PROMPT)], cwd=ROOT)
        subprocess.run(['git', 'commit', '-m', 'BASELINE: initial prompt'], cwd=ROOT)
        run_benchmark('BASELINE initial', baseline=True)
        best = read_best_S()
        print('Baseline S =', best)

    experiments = 0
    discard_streak = 0
    mutation_index = 0

    while experiments < MAX_EXPERIMENTS and discard_streak < MAX_DISCARD_STREAK:
        experiments += 1
        mutation = MUTATIONS[mutation_index % len(MUTATIONS)]
        mutation_index += 1

        base_text = PROMPT.read_text(encoding='utf-8')
        new_text = mutate_prompt(base_text, mutation)
        # write candidate prompt
        PROMPT.write_text(new_text, encoding='utf-8')

        desc = f"exp#{experiments} {mutation[0]}"
        rc = run_benchmark(desc)
        time.sleep(1)

        res = last_result()
        if not res:
            print('No result logged; aborting.')
            break

        status = res.get('status', '')
        S = res.get('S')
        print(f"Experiment {experiments}: status={status} S={S} desc={res.get('description')}")

        if status == 'KEEP' or status == 'BASELINE':
            # commit the kept prompt so git checkout restores to this
            subprocess.run(['git', 'add', str(PROMPT)], cwd=ROOT)
            subprocess.run(['git', 'commit', '-m', f'KEEP: {desc}'], cwd=ROOT)
            discard_streak = 0
            best = read_best_S()
        else:
            # DISCARD -> restore last committed version
            print('DISCARD — restoring last KEEP with git checkout')
            subprocess.run(['git', 'checkout', '--', str(PROMPT)], cwd=ROOT)
            discard_streak += 1

    print('\nRun finished.')
    best = read_best_S()
    print('Best S =', best)
    # print last 5 results
    if RESULTS.exists():
        with open(RESULTS, encoding='utf-8') as f:
            print('\nLast results:')
            lines = f.read().strip().splitlines()[-6:]
            for l in lines:
                print(l)


if __name__ == '__main__':
    main()
