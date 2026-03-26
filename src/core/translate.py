import os
import re
import json
from pathlib import Path
from typing import Any, Callable, cast
from src.config import (
    ROOT_DIR, NLLB_MODEL, NLLB_SRC_LANG, NLLB_TGT_LANG,
    OPENROUTER_MODEL, OPENROUTER_BASE, OPENROUTER_CHUNK_SIZE, OPENROUTER_CONTEXT_SIZE
)
from src.utils.helpers import log, retry_with_backoff

class PipelineError(Exception):
    """User-facing pipeline error with stage context."""
    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage
        self.message = message
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")

# ── PT-PT → PT-BR Normalizer ────────────────────────────────────────────────

_PTPT_TO_PTBR = [
    (r"\btu\b", "você"), (r"\bte\b", "te"), (r"\bteu\b", "seu"), (r"\btua\b", "sua"),
    (r"\bteus\b", "seus"), (r"\btuas\b", "suas"), (r"\bvós\b", "vocês"),
    (r"\bestás\b", "está"), (r"\bgostavas\b", "gostava"), (r"\bgostas\b", "gosta"),
    (r"\bfazes\b", "faz"), (r"\bpodes\b", "pode"), (r"\bqueres\b", "quer"),
    (r"\bsabes\b", "sabe"), (r"\btens\b", "tem"), (r"\bvens\b", "vem"),
    (r"\bdizes\b", "diz"), (r"\bvês\b", "vê"), (r"\bvais\b", "vai"),
    (r"\bficas\b", "fica"), (r"\bperceber\b", "entender"),
    (
        r"\ba (verificar|fazer|dizer|ir|ter|ser|estar|ver|vir|dar|saber|poder|querer|ficar|falar|pensar|olhar|ouvir|sentir|aprender|entender|perceber|mostrar|colocar|pedir|deixar|ajudar|começar|continuar|precisar|tentar|achar|trazer|levar|passar|parecer|acontecer|escolher|cuidar|gostar|amar|crescer|brincar|rir|chorar|correr|andar|esperar|trabalhar|estudar|viver|morrer|ganhar|perder|mudar|criar|usar|encontrar|conhecer|acreditar|lembrar|esquecer|chamar|jogar)\b",
        lambda m: (
            m.group(1)[:-2] + "ando" if m.group(1).endswith("ar")
            else m.group(1)[:-2] + "endo" if m.group(1).endswith("er")
            else m.group(1)[:-2] + "indo" if m.group(1).endswith("ir")
            else m.group(1) + "ndo"
        ),
    ),
    (r"\bmiúdos\b", "crianças"), (r"\bfixe\b", "legal"), (r"\bgiro\b", "bonito"),
    (r"\bchato\b", "chato"), (r"\bpropriamente\b", "corretamente"),
    (r"\bsempre que\b", "sempre que"), (r"\bcertamente\b", "certamente"),
    (r"\bapenas\b", "só"), (r"\bsomente\b", "só"), (r"\bimensamente\b", "muito"),
    (r"\bimenso\b", "enorme"), (r"\bautocarro\b", "ônibus"), (r"\bcomboio\b", "trem"),
    (r"\btelemovel\b", "celular"), (r"\btelemóvel\b", "celular"),
    (r"\bpasseio\b", "calçada"), (r"\bpetróleos\b", "petróleo"),
    (r"\bcasas de banho\b", "banheiros"), (r"\bcasa de banho\b", "banheiro"),
    (r"\bsaneamento\b", "saneamento"), (r"\bfutebol\b", "futebol"),
]

def ptpt_to_ptbr(text: str) -> str:
    """Apply PT-PT → PT-BR lexical substitutions."""
    rules_path = ROOT_DIR / "normalizer_rules.json"
    if rules_path.exists():
        try:
            cfg = json.loads(rules_path.read_text(encoding="utf-8"))
            rules = cfg.get("rules", [])
            for rule in rules:
                if rule.get("type") == "gerund":
                    verbs = rule.get("verbs", [])
                    if not verbs: continue
                    verb_pattern = "|".join(re.escape(v) for v in verbs)
                    pattern = rf"\ba ({verb_pattern})\b"
                    def _replace_gerund(m: re.Match[str]) -> str:
                        v = m.group(1)
                        if v.endswith("ar"): return v[:-2] + "ando"
                        if v.endswith("er"): return v[:-2] + "endo"
                        if v.endswith("ir"): return v[:-2] + "indo"
                        return v + "ndo"
                    text = str(re.sub(pattern, _replace_gerund, text, flags=re.IGNORECASE))
                else:
                    pattern = rule.get("pattern")
                    replacement = rule.get("replacement")
                    if not pattern: continue
                    def _replace_preserve_case(m: re.Match[str], repl: str = replacement) -> str:
                        if m.group(0) and m.group(0)[0].isupper():
                            return repl[0].upper() + repl[1:]
                        return repl
                    try:
                        text = str(re.sub(pattern, _replace_preserve_case, text, flags=re.IGNORECASE))
                    except re.error: continue
            return text
        except Exception: pass

    for pattern, replacement in _PTPT_TO_PTBR:
        if callable(replacement):
            text = str(re.sub(pattern, cast(Callable[[re.Match[str]], str], replacement), text, flags=re.IGNORECASE))
        else:
            def _replace(m: re.Match[str], repl: str = cast(str, replacement)) -> str:
                if m.group(0)[0].isupper(): return repl[0].upper() + repl[1:]
                return repl
            text = str(re.sub(pattern, _replace, text, flags=re.IGNORECASE))
    return text

# ── NLLB-200 ────────────────────────────────────────────────────────────────

_nllb_pipeline = None

def get_nllb_pipeline(logs: list):
    global _nllb_pipeline
    if _nllb_pipeline is not None: return _nllb_pipeline, logs
    from transformers import pipeline as hf_pipeline
    import torch
    log(f"🧠 Loading NLLB-200 ({NLLB_MODEL})…", logs)
    device = 0 if torch.cuda.is_available() else -1
    _nllb_pipeline = hf_pipeline("translation", model=NLLB_MODEL, src_lang=NLLB_SRC_LANG, tgt_lang=NLLB_TGT_LANG, device=device, max_length=512)
    log(f"   NLLB-200 loaded on {'GPU' if device == 0 else 'CPU'}", logs)
    return _nllb_pipeline, logs

def translate_nllb(texts: list[str], logs: list) -> tuple[list[str], list]:
    pipe, logs = get_nllb_pipeline(logs)
    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = cast(list, pipe(batch, batch_size=min(8, len(batch))))
        results.extend(o["translation_text"] for o in outputs)
    results = [ptpt_to_ptbr(t) for t in results]
    return results, logs

# ── OpenRouter ──────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    prompt_path = ROOT_DIR / "translation_prompt.md"
    if prompt_path.exists(): return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "You are a professional translator specialising in Brazilian Portuguese (PT-BR). "
        "CRITICAL RULES:\n1. Output ONLY in Brazilian Portuguese (PT-BR). NEVER use European Portuguese (PT-PT).\n"
        "2. Use 'voce' for second person singular. NEVER use 'tu', 'teu', 'tua', 'vos'.\n"
        "3. Use gerund forms: 'estao fazendo', 'estou vendo'. NEVER use 'estao a fazer', 'estou a ver'.\n"
        "4. Use Brazilian vocabulary: 'onibus' not 'autocarro', 'celular' not 'telemovel', 'trem' not 'comboio', 'banheiro' not 'casa de banho', 'legal' not 'fixe', 'criancas' not 'miudos', 'entender' not 'perceber'.\n"
        "5. Use 3rd person verb conjugations with 'voce'.\n6. Keep informal register.\n7. Preserve all punctuation."
    )

SYSTEM_PROMPT = load_system_prompt()

def call_openrouter(texts: list[str], api_key: str, context: list[str] | None = None) -> list[str]:
    import urllib.request
    ctx_block = ""
    if context:
        ctx_lines = "\n".join(f"  {t}" for t in context)
        ctx_block = f"[CONTEXT — already translated, do NOT include in output]\n{ctx_lines}\n[END CONTEXT]\n\n"
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = f"{ctx_block}Translate these {len(texts)} numbered utterances to Brazilian Portuguese (PT-BR).\nOutput ONLY the numbered translations.\n\n{numbered}"
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(f"{OPENROUTER_BASE}/chat/completions", data=payload, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dubweave", "X-Title": "Dubweave",
    }, method="POST")
    data = retry_with_backoff(lambda: json.loads(urllib.request.urlopen(req, timeout=180).read()))
    raw = data["choices"][0]["message"]["content"].strip()
    result = []
    for line in raw.splitlines():
        clean = re.sub(r"^\d+[\.)\s]+", "", line.strip()).strip()
        if clean: result.append(clean)
    return result

def translate_openrouter(texts: list[str], api_key: str, logs: list) -> tuple[list[str], list]:
    log(f"🌐 Translating via OpenRouter ({OPENROUTER_MODEL})…", logs)
    all_translated = []
    total_chunks = (len(texts) + OPENROUTER_CHUNK_SIZE - 1) // OPENROUTER_CHUNK_SIZE
    for chunk_i in range(total_chunks):
        chunk = texts[chunk_i * OPENROUTER_CHUNK_SIZE : (chunk_i + 1) * OPENROUTER_CHUNK_SIZE]
        ctx = all_translated[-OPENROUTER_CONTEXT_SIZE:] if all_translated else []
        log(f"   Chunk {chunk_i+1}/{total_chunks}…", logs)
        result = call_openrouter(chunk, api_key, context=ctx)
        if len(result) != len(chunk):
            while len(result) < len(chunk): result.append(chunk[len(result)])
            result = result[:len(chunk)]
        for i, translation in enumerate(result):
            if not translation.strip() or len(translation.strip()) < 2: result[i] = chunk[i]
        all_translated.extend(result)
    all_translated = [ptpt_to_ptbr(t) for t in all_translated]
    log(f"✅ OpenRouter translated {len(all_translated)} utterances", logs)
    return all_translated, logs

# ── Segment Merging ──────────────────────────────────────────────────────────

MERGE_CONFIGS = {
    "kokoro": {"min_words": 8, "max_words": 100, "gap_sec": 3.0, "max_duration": None},
    "francisca": {"min_words": 10, "max_words": 100, "gap_sec": 2.0, "max_duration": None, "chars_per_sec": 24.0},
    "thalita": {"min_words": 10, "max_words": 100, "gap_sec": 2.0, "max_duration": None, "chars_per_sec": 26.0},
    "antonio": {"min_words": 12, "max_words": 100, "gap_sec": 1.5, "max_duration": None, "chars_per_sec": 24.0},
    "default": {"min_words": 8, "max_words": 50, "gap_sec": 2.0, "max_duration": None},
}

def get_merge_config(engine: str) -> dict:
    if "Francisca" in engine: return MERGE_CONFIGS["francisca"]
    if "Antonio" in engine: return MERGE_CONFIGS["antonio"]
    if "Thalita" in engine: return MERGE_CONFIGS["thalita"]
    if "Kokoro" in engine: return MERGE_CONFIGS["kokoro"]
    return MERGE_CONFIGS["default"]

def group_for_synthesis(segments: list, *, min_words: int = 4, max_words: int = 40, gap_sec: float | None = None, max_duration: float | None = None) -> list:
    if not segments: return []
    _SENTENCE_ENDINGS = frozenset('.?!…—"\'')
    merged, buf, buf_children = [], [], []
    def _flush():
        if not buf: return
        merged.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": " ".join(s["text"].strip() for s in buf), "children": buf_children.copy()})
        buf.clear(); buf_children.clear()
    for idx, seg in enumerate(segments):
        if gap_sec is not None and buf and (seg["start"] - buf[-1]["end"]) >= gap_sec: _flush()
        buf.append(seg); buf_children.append(idx)
        text = " ".join(s["text"].strip() for s in buf)
        word_count = len(text.split())
        duration = buf[-1]["end"] - buf[0]["start"]
        if word_count >= max_words or (max_duration and duration >= max_duration): _flush()
        elif word_count >= min_words and text.rstrip() and text.rstrip()[-1] in _SENTENCE_ENDINGS: _flush()
    _flush()
    return merged

def expand_merged(merged_translated: list, original_segments: list) -> list:
    result = []
    for utt in merged_translated:
        children = utt["children"]
        if len(children) == 1:
            result.append({"start": original_segments[children[0]]["start"], "end": original_segments[children[0]]["end"], "text": utt["text"]})
            continue
        words = utt["text"].split()
        total_dur = sum(original_segments[c]["end"] - original_segments[c]["start"] for c in children)
        if total_dur <= 0:
            for j, c in enumerate(children):
                result.append({"start": original_segments[c]["start"], "end": original_segments[c]["end"], "text": utt["text"] if j == 0 else "…"})
            continue
        word_cursor = 0
        for j, c in enumerate(children):
            seg_dur = original_segments[c]["end"] - original_segments[c]["start"]
            proportion = seg_dur / total_dur
            if j == len(children) - 1: word_slice = words[word_cursor:]
            else:
                n_words = max(1, round(len(words) * proportion))
                word_slice = words[word_cursor : word_cursor + n_words]
                word_cursor += n_words
            result.append({"start": original_segments[c]["start"], "end": original_segments[c]["end"], "text": " ".join(word_slice) if word_slice else "…"})
    return result

def translate_segments(segments: list, logs: list, openrouter_key: str = "", merge_config: dict | None = None) -> tuple[list, list]:
    m_cfg = merge_config or MERGE_CONFIGS["default"]
    merged = group_for_synthesis(segments, **m_cfg)
    log(f"   Merged {len(segments)} segments → {len(merged)} utterances", logs)
    merged_texts = [u["text"] for u in merged]
    translated_texts, primary_error = None, None
    if openrouter_key.strip():
        try: translated_texts, logs = translate_openrouter(merged_texts, openrouter_key.strip(), logs)
        except Exception as e:
            primary_error = str(e); log(f"   ⚠️  OpenRouter failed: {primary_error[:120]}", logs)
    if translated_texts is None:
        try: translated_texts, logs = translate_nllb(merged_texts, logs)
        except Exception as e: raise RuntimeError(f"All translators failed. NLLB error: {e}")
    for i, (utt, txt) in enumerate(zip(merged, translated_texts)):
        merged[i] = {**utt, "text": txt.strip() if txt and txt.strip() else utt["text"]}
    translated = expand_merged(merged, segments)
    log(f"✅ Translated + re-expanded to {len(translated)} segments", logs)
    return translated, logs
