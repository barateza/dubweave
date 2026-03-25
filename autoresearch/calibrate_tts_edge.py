"""
calibrate_tts_edge.py — Dubweave autoresearch Loop 5
-------------------------------------------------
Measures actual Microsoft Edge Neural TTS speech rate by synthesizing
a fixed 40-sentence PT-BR test corpus across all 3 voices.

Metric: MAE in seconds — lower is better.

Usage:
    pixi run python autoresearch/calibrate_tts_edge.py --measure          # synthesize corpus (once)
    pixi run python autoresearch/calibrate_tts_edge.py --find-best        # grid search to find optimal cps
"""

import asyncio
import edge_tts
import json
import numpy as np
import sys
from pathlib import Path
import argparse
import csv
import datetime

# Fixed test corpus — 40 PT-BR sentences (same as Kokoro calibration)
SENTENCES: list[str] = [
    "Sim.", "Exatamente.", "Não sei.", "Com certeza.", "Muito obrigado.",
    "Até logo.", "Tudo bem.", "Que bom.",
    "Ele chegou na cidade ontem à tarde.", "O problema foi resolvido rapidamente.",
    "Você precisa verificar o arquivo de configuração.", "A reunião foi cancelada por causa do tempo.",
    "Eu não consigo entender o que está acontecendo.", "Vamos tentar de novo amanhã cedo.",
    "Isso não faz sentido para mim agora.", "Precisamos de mais informações antes de decidir.",
    "O que você acha disso?", "Quando você vai chegar?", "Por que isso está acontecendo?",
    "Você pode explicar melhor?", "Qual é o próximo passo?", "Isso já foi testado antes?",
    "O servidor não está respondendo às requisições.", "Atualize o driver de rede e reinicie o sistema.",
    "A configuração do banco de dados precisa ser revisada.", "Execute o comando de diagnóstico no terminal.",
    "Verifique os logs do sistema para mais detalhes.", "O certificado SSL expirou e precisa ser renovado.",
    "Quando o processo de instalação for concluído, reinicie o computador e verifique se tudo está funcionando.",
    "A análise dos dados mostra que houve uma melhora significativa no desempenho após a última atualização.",
    "É importante entender que a configuração padrão pode não ser adequada para todos os casos de uso.",
    "Após revisar todos os documentos, chegamos à conclusão de que precisamos de uma abordagem diferente.",
    "O modelo foi treinado com milhões de exemplos para garantir resultados mais precisos e confiáveis.",
    "São três horas da tarde.", "O preço total é de cento e cinquenta reais.",
    "Foram encontrados quarenta e sete erros no relatório.", "A velocidade máxima é de cento e vinte quilômetros por hora.",
    "O prazo final é dia quinze de abril de dois mil e vinte e cinco.",
    "Isso é incrível! Nunca vi nada assim antes.", "Que tragédia... não consigo acreditar no que aconteceu.",
]

VOICES = ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural", "pt-BR-ThalitaNeural"]
CORPUS_DIR = Path("corpus")
DURATIONS_PATH = CORPUS_DIR / "loop5_durations.json"

async def _get_duration(text: str, voice: str) -> float:
    """Synthesize text and return duration via a mock run or full synthesis."""
    import io
    import soundfile as sf
    
    communicate = edge_tts.Communicate(text, voice)
    data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    
    if not data:
        return 0.0
    
    with io.BytesIO(data) as f:
        audio_data, sample_rate = sf.read(f)
        return len(audio_data) / sample_rate

async def measure_corpus():
    CORPUS_DIR.mkdir(exist_ok=True)
    if DURATIONS_PATH.exists():
        print(f"Loading existing durations from {DURATIONS_PATH}")
        return

    print(f"Measuring {len(SENTENCES)} sentences for {len(VOICES)} voices...")
    durations = {v: [] for v in VOICES}
    
    for voice in VOICES:
        print(f"Voice: {voice}")
        for i, sentence in enumerate(SENTENCES):
            dur = await _get_duration(sentence, voice)
            durations[voice].append(dur)
            print(f"  [{i+1}/{len(SENTENCES)}] {len(sentence)} chars -> {dur:.3f}s")
            
    cache = {
        "sentences": SENTENCES,
        "voices": VOICES,
        "durations": durations,
    }
    DURATIONS_PATH.write_text(json.dumps(cache, indent=2))
    print(f"Saved to {DURATIONS_PATH}")

def find_best_cps():
    if not DURATIONS_PATH.exists():
        print("Run --measure first.")
        return
    
    cache = json.loads(DURATIONS_PATH.read_text())
    durations = cache["durations"]
    
    print(f"{'Voice':<25} | {'Mean CPS':>10} | {'Found CPS':>10}")
    print("-" * 50)
    
    voice_cps = {}
    for voice in VOICES:
        rates = []
        for text, dur in zip(SENTENCES, durations[voice]):
            if dur > 0:
                rates.append(len(text) / dur)
        
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        voice_cps[voice] = round(mean_rate, 2)
        print(f"{voice:<25} | {mean_rate:10.2f} | {voice_cps[voice]:10.2f}")
    
    return voice_cps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--find-best", action="store_true")
    args = parser.parse_args()
    
    if args.measure:
        asyncio.run(measure_corpus())
    elif args.find_best:
        find_best_cps()
    else:
        # Default to finding best if durations exist
        if DURATIONS_PATH.exists():
            find_best_cps()
        else:
            asyncio.run(measure_corpus())
            find_best_cps()
