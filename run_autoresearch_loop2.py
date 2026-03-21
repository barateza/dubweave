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

MUTATIONS = [
    ("explicit_você_rule", "Always use 'você' instead of 'tu' or 'teu' — examples: \n- 'Você fez' NOT 'Tu fizeste'\n"),
    ("avoid_pt-pt_vocab", "Avoid PT-PT vocabulary: replace 'autocarro'→'ônibus', 'telemovel'→'celular', 'miudos'→'crianças'\n"),
    ("conversational_register", "Register: use conversational, informal Brazilian Portuguese for dialogues; prefer 'você' and colloquial contractions where natural.\n"),
    ("gerund_examples", "Gerund rule: prefer 'estou fazendo' over 'estou a fazer'. Examples:\n- EN: I'm doing it. → PT-BR: Estou fazendo.\n"),
    ("diminutives_rule", "Diminutives: use '-inho/inha' where natural for friendly tone, e.g., 'carrinho' for small car in informal contexts.\n"),
    ("numbers_dates", "Numbers and dates: use PT-BR formatting (DD/MM/YYYY, ',' for decimals).\n"),
    ("add_before_after_examples", "Examples (before → after):\n- 'He is eating.' → 'Ele está comendo.' (not 'Ele está a comer.')\n- 'The bus arrived.' → 'O ônibus chegou.'\n"),
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


def mutate_prompt(base_text: str, mutation: tuple) -> str:
    name, body = mutation
    # Simple mutation: append a small rule section with the mutation name
    new = base_text.strip() + "\n\n" + f"### Mutation: {name}\n" + body + "\n"
    return new


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
