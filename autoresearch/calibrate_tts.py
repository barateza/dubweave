"""
calibrate_tts.py — Dubweave autoresearch Loop 4
-------------------------------------------------
Measures actual Kokoro speech rate by synthesizing a fixed 40-sentence
PT-BR test corpus across all 3 voices and comparing predicted vs actual
duration using the chars_per_sec value in loop4_config.json.

Metric: MAE in seconds — lower is better.

Two-phase design:
  1. --measure  (slow, run once)  Synthesizes every sentence with every
                                   voice, caches real durations to
                                   corpus/loop4_durations.json.
  2. Normal run (fast, ~0.1 s)    Loads the cache, scores loop4_config.json,
                                   logs to loop4_results.tsv, prints KEEP/DISCARD.

Usage:
    pixi run python calibrate_tts.py --measure          # synthesize corpus (once)
    pixi run python calibrate_tts.py --baseline         # record as BASELINE
    pixi run python calibrate_tts.py                    # score current config
    pixi run python calibrate_tts.py --find-best        # grid search [15..35] step 0.5
    pixi run python calibrate_tts.py --status           # print last 10 rows
    pixi run python calibrate_tts.py -d "hypothesis"    # attach description to log row
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed test corpus — 40 PT-BR sentences, never changed after creation.
# Coverage: short declaratives, medium sentences, questions, technical vocab,
# long complex utterances, numbers, mixed punctuation.
# ---------------------------------------------------------------------------

SENTENCES: list[str] = [
    # Short declaratives (< 30 chars)
    "Sim.",
    "Exatamente.",
    "Não sei.",
    "Com certeza.",
    "Muito obrigado.",
    "Até logo.",
    "Tudo bem.",
    "Que bom.",
    # Medium declaratives (30–80 chars)
    "Ele chegou na cidade ontem à tarde.",
    "O problema foi resolvido rapidamente.",
    "Você precisa verificar o arquivo de configuração.",
    "A reunião foi cancelada por causa do tempo.",
    "Eu não consigo entender o que está acontecendo.",
    "Vamos tentar de novo amanhã cedo.",
    "Isso não faz sentido para mim agora.",
    "Precisamos de mais informações antes de decidir.",
    # Questions
    "O que você acha disso?",
    "Quando você vai chegar?",
    "Por que isso está acontecendo?",
    "Você pode explicar melhor?",
    "Qual é o próximo passo?",
    "Isso já foi testado antes?",
    # Technical vocabulary
    "O servidor não está respondendo às requisições.",
    "Atualize o driver de rede e reinicie o sistema.",
    "A configuração do banco de dados precisa ser revisada.",
    "Execute o comando de diagnóstico no terminal.",
    "Verifique os logs do sistema para mais detalhes.",
    "O certificado SSL expirou e precisa ser renovado.",
    # Long complex sentences (> 80 chars)
    "Quando o processo de instalação for concluído, reinicie o computador e verifique se tudo está funcionando.",
    "A análise dos dados mostra que houve uma melhora significativa no desempenho após a última atualização.",
    "É importante entender que a configuração padrão pode não ser adequada para todos os casos de uso.",
    "Após revisar todos os documentos, chegamos à conclusão de que precisamos de uma abordagem diferente.",
    "O modelo foi treinado com milhões de exemplos para garantir resultados mais precisos e confiáveis.",
    # Numbers and mixed
    "São três horas da tarde.",
    "O preço total é de cento e cinquenta reais.",
    "Foram encontrados quarenta e sete erros no relatório.",
    "A velocidade máxima é de cento e vinte quilômetros por hora.",
    "O prazo final é dia quinze de abril de dois mil e vinte e cinco.",
    # Prosody variety
    "Isso é incrível! Nunca vi nada assim antes.",
    "Que tragédia... não consigo acreditar no que aconteceu.",
]

assert len(SENTENCES) == 40, f"Corpus must have exactly 40 sentences, got {len(SENTENCES)}"

VOICES = ["pf_dora", "pm_alex", "pm_santa"]
SAMPLE_RATE = 24_000  # Kokoro output sample rate (Hz)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR      = Path("corpus")
DURATIONS_PATH  = CORPUS_DIR / "loop4_durations.json"
CONFIG_PATH     = Path("loop4_config.json")
RESULTS_PATH    = Path("loop4_results.tsv")

RESULTS_HEADER = [
    "timestamp", "status", "chars_per_sec",
    "mae_all", "mae_pf_dora", "mae_pm_alex", "mae_pm_santa",
    "bias_pf_dora", "bias_pm_alex", "bias_pm_santa",
    "n_sentences", "description",
]

# ---------------------------------------------------------------------------
# Synthesis helpers (only imported when --measure is requested)
# ---------------------------------------------------------------------------


def _load_kokoro(voice: str):
    """Load a Kokoro KPipeline for PT-BR. Heavy — call once per voice."""
    from kokoro import KPipeline  # noqa: PLC0415
    return KPipeline(lang_code="p", repo_id="hexgrad/Kokoro-82M")


def _synthesize_duration(pipeline, text: str, voice: str) -> float:
    """
    Synthesize `text` with `voice` and return the real audio duration in seconds.
    Kokoro yields (graphemes, phonemes, audio_np) tuples at SAMPLE_RATE.
    We concatenate all chunks and measure len / sample_rate.
    """
    import numpy as np  # noqa: PLC0415

    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

    if not chunks:
        return 0.0

    full = np.concatenate(chunks)
    return float(len(full)) / SAMPLE_RATE


def measure_corpus() -> None:
    """
    Synthesize every sentence with every voice and cache durations.
    This is the slow step (~60 s total). Run once; results are reused by all
    scoring runs.
    """
    CORPUS_DIR.mkdir(exist_ok=True)

    if DURATIONS_PATH.exists():
        print(f"[loop4] Duration cache already exists: {DURATIONS_PATH}")
        print("        Delete it and re-run --measure to re-synthesize.")
        return

    print(f"[loop4] Measuring synthesis durations for {len(SENTENCES)} sentences × {len(VOICES)} voices")
    print("        This will take approximately 60–120 seconds.\n")

    # durations[voice][sentence_index] = seconds
    durations: dict[str, list[float]] = {v: [] for v in VOICES}

    for voice in VOICES:
        print(f"  Loading Kokoro for voice: {voice}")
        pipeline = _load_kokoro(voice)
        print(f"  Synthesizing {len(SENTENCES)} sentences…")

        for i, sentence in enumerate(SENTENCES):
            dur = _synthesize_duration(pipeline, sentence, voice)
            durations[voice].append(dur)
            chars = len(sentence)
            ratio = chars / dur if dur > 0 else 0.0
            print(f"    [{i+1:02d}/{len(SENTENCES)}] {chars:4d} chars  {dur:.3f}s  → {ratio:.1f} cps  {sentence[:50]!r}")

        # Free the model before loading the next voice
        del pipeline

    # Save cache
    cache = {
        "sentences": SENTENCES,
        "voices": VOICES,
        "sample_rate": SAMPLE_RATE,
        "durations": durations,
    }
    DURATIONS_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[loop4] Durations saved → {DURATIONS_PATH}")

    # Quick per-voice summary
    print("\nPer-voice observed chars/sec:")
    for voice in VOICES:
        rates = [
            len(SENTENCES[i]) / d
            for i, d in enumerate(durations[voice])
            if d > 0
        ]
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        print(f"  {voice:12s}  mean observed cps = {mean_rate:.2f}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def load_durations() -> dict[str, list[float]]:
    """Load the cached duration measurements."""
    if not DURATIONS_PATH.exists():
        print(
            f"ERROR: {DURATIONS_PATH} not found.\n"
            "Run --measure first to synthesize the corpus.",
            file=sys.stderr,
        )
        sys.exit(1)

    cache = json.loads(DURATIONS_PATH.read_text(encoding="utf-8"))

    # Validate the cached sentences match the embedded corpus
    if cache.get("sentences") != SENTENCES:
        print(
            "WARNING: cached sentences differ from the embedded SENTENCES list.\n"
            f"         Cache has {len(cache['sentences'])} sentences; embedded has {len(SENTENCES)}.\n"
            "         Delete corpus/loop4_durations.json and re-run --measure.",
            file=sys.stderr,
        )
        sys.exit(1)

    return cache["durations"]


def score_config(chars_per_sec: float) -> dict:
    """
    Load cached durations and compute MAE + bias for each voice.

    Returns dict with keys:
        mae_all         — composite MAE across all voices (the optimisation target)
        mae_<voice>     — per-voice MAE
        bias_<voice>    — mean signed error (positive = model too slow, negative = too fast)
        n_sentences
    """
    durations = load_durations()

    voice_mae: dict[str, float] = {}
    voice_bias: dict[str, float] = {}

    for voice in VOICES:
        real_durs = durations[voice]
        errors = []
        signed_errors = []
        for sentence, real_dur in zip(SENTENCES, real_durs):
            if real_dur <= 0:
                continue
            predicted = len(sentence) / chars_per_sec
            err = abs(predicted - real_dur)
            errors.append(err)
            signed_errors.append(predicted - real_dur)

        voice_mae[voice] = sum(errors) / len(errors) if errors else 0.0
        voice_bias[voice] = sum(signed_errors) / len(signed_errors) if signed_errors else 0.0

    mae_all = sum(voice_mae.values()) / len(voice_mae)

    result: dict = {"mae_all": mae_all, "n_sentences": len(SENTENCES)}
    for voice in VOICES:
        key = voice.replace("-", "_")
        result[f"mae_{key}"] = voice_mae[voice]
        result[f"bias_{key}"] = voice_bias[voice]

    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_result(chars_per_sec: float, scores: dict, status: str, description: str = "") -> None:
    """Append one row to loop4_results.tsv."""
    first_write = not RESULTS_PATH.exists()
    row = {
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
        "status":       status,
        "chars_per_sec": f"{chars_per_sec:.2f}",
        "mae_all":      f"{scores['mae_all']:.5f}",
        "mae_pf_dora":  f"{scores['mae_pf_dora']:.5f}",
        "mae_pm_alex":  f"{scores['mae_pm_alex']:.5f}",
        "mae_pm_santa": f"{scores['mae_pm_santa']:.5f}",
        "bias_pf_dora": f"{scores['bias_pf_dora']:.5f}",
        "bias_pm_alex": f"{scores['bias_pm_alex']:.5f}",
        "bias_pm_santa":f"{scores['bias_pm_santa']:.5f}",
        "n_sentences":  scores["n_sentences"],
        "description":  description,
    }
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        if first_write:
            writer.writeheader()
        writer.writerow(row)


def read_best_mae() -> float | None:
    """Return the lowest mae_all from KEEP or BASELINE rows. None if no rows."""
    if not RESULTS_PATH.exists():
        return None
    best: float | None = None
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") in ("KEEP", "BASELINE"):
                try:
                    v = float(row["mae_all"])
                    if best is None or v < best:
                        best = v
                except (ValueError, KeyError):
                    pass
    return best


def print_last(n: int = 10) -> None:
    if not RESULTS_PATH.exists():
        print("No loop4_results.tsv yet.")
        return
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows[-n:]:
        print(
            f"[{row['timestamp']}]  {row['status']:8s}  "
            f"cps={row['chars_per_sec']:5s}  mae_all={row['mae_all']}  "
            f"dora={row['mae_pf_dora']}  alex={row['mae_pm_alex']}  "
            f"santa={row['mae_pm_santa']}  "
            f"bias(d/a/s)={row['bias_pf_dora']}/{row['bias_pm_alex']}/{row['bias_pm_santa']}  "
            f"{row.get('description', '')}"
        )


# ---------------------------------------------------------------------------
# Grid search helper
# ---------------------------------------------------------------------------


def find_best_cps() -> None:
    """
    Grid search over chars_per_sec in [10.0, 35.0] at 0.5-step intervals.
    Prints per-voice MAE table. Does NOT write to results.tsv.
    """
    print(f"\n{'cps':>6}  {'mae_all':>8}  {'dora':>8}  {'alex':>8}  {'santa':>8}  "
          f"{'bias_d':>8}  {'bias_a':>8}  {'bias_s':>8}")
    print("-" * 80)

    cps = 14.0
    best_cps = cps
    best_mae = float("inf")

    while cps <= 16.01:
        s = score_config(cps)
        marker = " ← best" if s["mae_all"] < best_mae else ""
        if s["mae_all"] < best_mae:
            best_mae = s["mae_all"]
            best_cps = cps
        print(
            f"{cps:6.1f}  {s['mae_all']:8.5f}  "
            f"{s['mae_pf_dora']:8.5f}  {s['mae_pm_alex']:8.5f}  {s['mae_pm_santa']:8.5f}  "
            f"{s['bias_pf_dora']:+8.4f}  {s['bias_pm_alex']:+8.4f}  {s['bias_pm_santa']:+8.4f}"
            f"{marker}"
        )
        cps = round(cps + 0.1, 1)

    print(f"\nOptimal chars_per_sec: {best_cps:.1f}  (mae_all={best_mae:.5f})")
    print(f"\nTo set this as the current config:")
    print(f'  echo \'{{"chars_per_sec": {best_cps}}}\' > loop4_config.json')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubweave Loop 4 — calibrate Kokoro chars/sec via real synthesis"
    )
    parser.add_argument(
        "--measure", action="store_true",
        help="Synthesize the full corpus and cache durations (slow, run once)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Record this run as BASELINE (run once before agent loop)",
    )
    parser.add_argument(
        "--find-best", action="store_true",
        help="Grid search chars_per_sec over [15, 35] and print MAE table",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print the last 10 results",
    )
    parser.add_argument(
        "--description", "-d", default="",
        help="Short hypothesis description (logged to loop4_results.tsv)",
    )
    args = parser.parse_args()

    if args.measure:
        measure_corpus()
        return

    if args.status:
        print_last(10)
        return

    if args.find_best:
        find_best_cps()
        return

    # --- Score current config ---
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.", file=sys.stderr)
        print('Create it: echo \'{"chars_per_sec": 25.0}\' > loop4_config.json', file=sys.stderr)
        sys.exit(1)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    cps = float(config["chars_per_sec"])

    print(f"[loop4] Scoring chars_per_sec={cps:.2f} against {DURATIONS_PATH}")
    scores = score_config(cps)

    # Determine status
    if args.baseline:
        status = "BASELINE"
    else:
        best = read_best_mae()
        if best is None:
            print(
                "WARNING: No BASELINE or KEEP row found. Run with --baseline first.",
                file=sys.stderr,
            )
            status = "?"
        else:
            # Lower MAE is better (opposite of Loop 1's S score)
            status = "KEEP" if scores["mae_all"] < best else "DISCARD"

    log_result(cps, scores, status, description=args.description)

    # Print report
    print(f"\n  chars_per_sec = {cps:.2f}")
    print(f"\n  mae_all       = {scores['mae_all']:.5f} s  {'← NEW BEST' if status == 'KEEP' else ''}")
    print()
    print(f"  {'Voice':12s}  {'MAE (s)':>8}  {'Bias (s)':>9}  {'Interpretation'}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*30}")
    for voice in VOICES:
        key = voice.replace("-", "_")
        mae  = scores[f"mae_{key}"]
        bias = scores[f"bias_{key}"]
        # Positive bias: predicted > real → model is slower than predicted → cps too HIGH
        # Negative bias: predicted < real → model is faster than predicted → cps too LOW
        if abs(bias) < 0.05:
            interp = "well calibrated"
        elif bias > 0:
            interp = f"predicted {bias:.3f}s long → try lower cps"
        else:
            interp = f"predicted {abs(bias):.3f}s short → try higher cps"
        print(f"  {voice:12s}  {mae:8.5f}  {bias:+9.5f}  {interp}")

    print(f"\n  n_sentences   = {scores['n_sentences']}")
    print(f"  status        = {status}")
    if status == "DISCARD":
        best = read_best_mae()
        print(f"  best so far   = {best:.5f} s")
    print(f"\n  Logged → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
