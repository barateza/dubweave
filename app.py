"""
YT Dubber — YouTube → Brazilian Portuguese Dubbing Pipeline
Uses: yt-dlp → Whisper → NLLB-200 (local PT-BR) → XTTS v2 (GPU) → FFmpeg
Fallback translation: OpenRouter API (configurable model)
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Generator, Any, cast

import warnings
import gradio as gr

# Suppress torch.load pickle warnings from TTS/XTTS internals.
# These are known, safe, third-party model files — not a security concern here.
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS")
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

# ── Lazy imports (installed at runtime) ──────────────────────────────────────
def lazy_import():
    global yt_dlp, whisper, torch, TTS
    import yt_dlp
    import whisper
    import torch
    from TTS.api import TTS
    return True


# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR = Path(tempfile.gettempdir()) / "yt_dubber"
WORK_DIR.mkdir(exist_ok=True)
# Always resolve relative to this script file, not the shell working directory.
# pixi run may cd anywhere — a relative path is unreliable.
OUTPUT_DIR = Path(__file__).parent.resolve() / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PROJECTS_DIR = Path(__file__).parent.resolve() / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

WHISPER_MODEL  = "large-v3-turbo"  # swap to "large-v3" for max accuracy on noisy audio
XTTS_MODEL     = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_LANG    = "pt"               # XTTS v2 uses "pt" for all Portuguese; BR accent comes from the voice reference clip

# Kokoro config
KOKORO_LANG    = "p"                # PT-BR lang_code in Kokoro
KOKORO_VOICE   = "pf_dora"          # pf_dora (F), pm_alex (M), pm_santa (M)
KOKORO_SPEED   = 1.0                # 1.0 = natural rate; increase to fit tight slots
# Translation config
NLLB_MODEL      = "facebook/nllb-200-distilled-600M"  # ~2.4GB, runs on GPU, true PT-BR
NLLB_SRC_LANG   = "eng_Latn"
NLLB_TGT_LANG   = "por_Latn"   # Brazilian Portuguese in NLLB's FLORES-200 code

OPENROUTER_MODEL = "google/gemini-2.0-flash-001"  # $0.10/M input, $0.40/M output — ~$0.002 per 10min video
OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

JOB_MAX_AGE_H  = 2       # hours before a stale job folder is eligible for cleanup


# ── Step helpers ──────────────────────────────────────────────────────────────

def log(msg: str, logs: list) -> list:
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    logs.append(entry)
    return logs


def download_video(url: str, job_dir: Path, logs: list, browser: str = "none", cookies_file: str | None = None):
    """
    Download video + audio with a self-healing format cascade.

    YouTube's JS challenge solver breaks periodically, which makes DASH-only
    formats unavailable. The cascade tries progressively more compatible
    formats, ending with format 18 (360p muxed mp4) which is always a
    progressive stream and never requires challenge solving.

    Cascade — video track:
      1. bestvideo[ext=mp4]          — best MP4 DASH video (ideal)
      2. bestvideo                   — best video any container
      3. best[ext=mp4]               — best muxed mp4 (e.g. format 18)
      4. best                        — anything available
      5. 18                          — hardcoded 360p muxed mp4 (nuclear fallback)

    Cascade — audio track (separate pass for higher quality):
      1. bestaudio[ext=m4a]          — best M4A (140: 129kbps AAC)
      2. bestaudio[ext=webm]         — best WebM (251: 130kbps Opus)
      3. bestaudio                   — any best audio
      4. 140                         — hardcoded medium M4A
      5. 139                         — hardcoded low M4A
      6. extract from video file     — last resort: split muxed download

    If a separate video+audio download succeeded, they are kept as separate
    files for clean muxing. If we fell back to a muxed format, the audio is
    extracted from it via ffmpeg so the transcription still gets clean audio.
    """
    import yt_dlp as yt

    logs = log("📥 Downloading video with yt-dlp…", logs)

    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio_orig.wav"

    # ── Detect aria2c and deno availability ──────────────────────────────────
    import shutil as _shutil
    _aria2c = _shutil.which("aria2c")
    _deno   = _shutil.which("deno")

    if _aria2c:
        logs = log(f"   aria2c found — using for accelerated download", logs)
    if _deno:
        logs = log(f"   deno found — YouTube JS challenge solving enabled", logs)
    else:
        logs = log("   ⚠️  deno not found — some YouTube formats may be missing (run setup.bat)", logs)

    BASE_OPTS = {
        "outtmpl": str(job_dir / "%(id)s_%(format_id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
        "ignoreerrors": False,
        # Use cached EJS solver script for YouTube n-challenge (downloaded by setup.bat)
        # Cookie auth: cookies.txt takes priority over browser extraction.
        # cookies.txt = Netscape format exported from browser extension.
        # browser     = yt-dlp reads directly from browser profile (may fail if Chrome is open).
        # neither     = anonymous download (may trigger PO token / JS challenge errors).
        **({"cookiefile": cookies_file} if cookies_file
           else {"cookiesfrombrowser": (browser, None, None, None)} if browser != "none"
           else {}),
        **({"remote_components": ["ejs:github"]} if _deno else {}),
        # ── aria2c: multi-connection download via your local RPC server ───────
        # Falls back to yt-dlp's built-in downloader if aria2c is not in PATH.
        **({"external_downloader": "aria2c",
            "external_downloader_args": {
                "aria2c": [
                    "--rpc-save-upload-metadata=false",
                    "--file-allocation=none",        # faster start, skip prealloc
                    "--optimize-concurrent-downloads=true",
                    "--max-connection-per-server=4", # YouTube allows ~4 per URL
                    "--min-split-size=5M",
                    "--split=4",
                    "--max-tries=5",
                    "--retry-wait=3",
                ]
            }} if _aria2c else {}),
    }

    # ── Step 1: probe title/duration without downloading ─────────────────────
    title, duration = "video", 0
    try:
        with yt.YoutubeDL(cast(Any, {**BASE_OPTS, "skip_download": True})) as ydl:
            info = ydl.extract_info(url, download=False)
            title    = info.get("title", "video")
            duration = info.get("duration", 0)
            video_id = info.get("id", "video")
    except Exception as e:
        logs = log(f"⚠️  Probe failed ({e}), continuing anyway…", logs)
        video_id = "video"

    # ── Step 2: download video track ─────────────────────────────────────────
    VIDEO_FORMATS = [
        "bestvideo[ext=mp4]",
        "bestvideo",
        "best[ext=mp4]",
        "best",
        "18",
    ]

    muxed_fallback = False   # True when video already contains audio
    raw_video_file = None

    for fmt in VIDEO_FORMATS:
        try:
            opts = {**BASE_OPTS, "format": fmt,
                    "outtmpl": str(job_dir / f"video_raw.%(ext)s")}
            with yt.YoutubeDL(cast(Any, opts)) as ydl:
                ydl.extract_info(url, download=True)
            candidates = list(job_dir.glob("video_raw.*"))
            if candidates:
                raw_video_file = candidates[0]
                muxed_fallback = fmt in ("best[ext=mp4]", "best", "18")
                logs = log(f"   Video format '{fmt}' ✓  ({raw_video_file.name})", logs)
                break
        except Exception as e:
            logs = log(f"   Video format '{fmt}' failed: {e!s:.80}", logs)

    if raw_video_file is None:
        raise RuntimeError("All video format fallbacks exhausted — cannot download video.")

    shutil.move(str(raw_video_file), str(video_path))

    # ── Step 3: download audio track ─────────────────────────────────────────
    # If we got a muxed file, extract audio from it directly — no second
    # download needed and avoids re-triggering any challenge issues.
    if muxed_fallback:
        logs = log("   Muxed fallback — extracting audio from video file…", logs)
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "44100", "-ac", "2", "-f", "wav",
            str(audio_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extract failed: {result.stderr[-300:]}")
    else:
        AUDIO_FORMATS = [
            "bestaudio[ext=m4a]",
            "bestaudio[ext=webm]",
            "bestaudio",
            "140",
            "139",
        ]
        raw_audio_file = None
        for fmt in AUDIO_FORMATS:
            try:
                opts = {
                    **BASE_OPTS,
                    "format": fmt,
                    "outtmpl": str(job_dir / "audio_raw.%(ext)s"),
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "0",
                    }],
                }
                with yt.YoutubeDL(cast(Any, opts)) as ydl:
                    ydl.extract_info(url, download=True)
                candidates = list(job_dir.glob("audio_raw.*"))
                if candidates:
                    raw_audio_file = candidates[0]
                    logs = log(f"   Audio format '{fmt}' ✓  ({raw_audio_file.name})", logs)
                    break
            except Exception as e:
                logs = log(f"   Audio format '{fmt}' failed: {e!s:.80}", logs)

        if raw_audio_file is None:
            # absolute last resort: extract from whatever video we have
            logs = log("   All audio formats failed — extracting from video file…", logs)
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ar", "44100", "-ac", "2", "-f", "wav",
                str(audio_path)
            ], capture_output=True, check=True)
        else:
            shutil.move(str(raw_audio_file), str(audio_path))

    logs = log(f"✅ Downloaded: \"{title}\" ({duration}s)", logs)
    return video_path, audio_path, title, duration, logs


def transcribe_audio(audio_path: Path, logs: list, model_name: str = WHISPER_MODEL) -> tuple[list, list]:
    """Transcribe audio with Whisper, return segments with timestamps."""
    import whisper
    logs = log(f"🎙️ Transcribing with Whisper ({model_name})…", logs)

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        verbose=False,
    )

    segments = cast(list, result["segments"])
    logs = log(f"✅ Transcribed {len(segments)} segments", logs)
    return segments, logs


# ── PT-PT → PT-BR normalizer (post-processing, runs on ALL translators) ─────

# These are the most common European Portuguese markers that NLLB and other
# models default to. Replacing them with Brazilian equivalents covers ~90% of
# the perceptible difference in everyday spoken content.
_PTPT_TO_PTBR = [
    # Pronouns / address
    (r"tu",            "você"),
    (r"te",            "te"),          # keep — both use "te" but context helps
    (r"teu",           "seu"),
    (r"tua",           "sua"),
    (r"teus",          "seus"),
    (r"tuas",          "suas"),
    (r"vós",           "vocês"),
    # Verb forms — 2nd person → 3rd person (você paradigm)
    (r"estás",         "está"),
    (r"gostavas",      "gostava"),
    (r"gostas",        "gosta"),
    (r"fazes",         "faz"),
    (r"podes",         "pode"),
    (r"queres",        "quer"),
    (r"sabes",         "sabe"),
    (r"tens",          "tem"),
    (r"vens",          "vem"),
    (r"dizes",         "diz"),
    (r"vês",           "vê"),
    (r"vais",          "vai"),
    (r"ficas",         "fica"),
    (r"perceber",      "entender"),
    # Gerund — PT-PT uses infinitive constructions, PT-BR uses gerund
    # "a verificar" → "verificando", "a fazer" → "fazendo" etc.
    (r"a (verificar|fazer|dizer|ir|ter|ser|estar|ver|vir|dar|saber|poder|querer|ficar|falar|pensar|olhar|ouvir|sentir|aprender|entender|perceber|mostrar|colocar|pedir|deixar|ajudar|começar|continuar|precisar|tentar|achar|trazer|levar|passar|parecer|acontecer|escolher|cuidar|gostar|amar|crescer|brincar|rir|chorar|correr|andar|esperar|trabalhar|estudar|viver|morrer|ganhar|perder|mudar|criar|usar|encontrar|conhecer|acreditar|lembrar|esquecer|chamar|jogar)",
     lambda m: m.group(1) + "ndo"),
    # Specific common phrases
    (r"miúdos",        "crianças"),
    (r"fixe",          "legal"),
    (r"giro",          "bonito"),
    (r"chato",         "chato"),   # same but keep
    (r"propriamente",  "corretamente"),
    (r"sempre que",    "sempre que"),
    (r"certamente",    "certamente"),
    (r"apenas",        "só"),
    (r"somente",       "só"),
    (r"imensamente",   "muito"),
    (r"imenso",        "enorme"),
    (r"autocarro",     "ônibus"),
    (r"comboio",       "trem"),
    (r"telemovel",     "celular"),
    (r"telemovel",     "celular"),
    (r"telemóvel",     "celular"),
    (r"passeio",       "calçada"),
    (r"petróleos",     "petróleo"),
    (r"casas de banho","banheiros"),
    (r"casa de banho", "banheiro"),
    (r"saneamento",    "saneamento"),
    (r"futebol",       "futebol"),  # same
]

def _ptpt_to_ptbr(text: str) -> str:
    """Apply PT-PT → PT-BR lexical substitutions."""
    import re
    for pattern, replacement in _PTPT_TO_PTBR:
        if callable(replacement):
            text = re.sub(pattern, cast(Any, replacement), text, flags=re.IGNORECASE)
        else:
            # Preserve capitalisation of the first letter
            def _replace(m: re.Match[str], repl: str = replacement) -> str:
                if m.group(0)[0].isupper():
                    return repl[0].upper() + repl[1:]
                return repl
            text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return text


# ── NLLB-200 translation (primary) ───────────────────────────────────────────

_nllb_pipeline = None  # module-level cache — load once, reuse across jobs

def _get_nllb_pipeline(logs: list):
    """Load NLLB-200 translation pipeline, cached after first load."""
    global _nllb_pipeline
    if _nllb_pipeline is not None:
        return _nllb_pipeline, logs

    from transformers import pipeline as hf_pipeline
    import torch

    logs = log(f"🧠 Loading NLLB-200 ({NLLB_MODEL})…", logs)
    device = 0 if torch.cuda.is_available() else -1
    _nllb_pipeline = hf_pipeline(
        "translation",
        model=NLLB_MODEL,
        src_lang=NLLB_SRC_LANG,
        tgt_lang=NLLB_TGT_LANG,
        device=device,
        max_length=512,
    )  # tgt_lang already sets forced_bos_token_id internally; do not pass it again
    logs = log(f"   NLLB-200 loaded on {'GPU' if device == 0 else 'CPU'}", logs)
    return _nllb_pipeline, logs


def _translate_nllb(texts: list[str], logs: list) -> tuple[list[str], list]:
    """Batch-translate EN→PT-BR via NLLB-200, then normalise PT-PT markers."""
    pipe, logs = _get_nllb_pipeline(logs)

    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = cast(list[dict[str, str]], pipe(batch, batch_size=min(8, len(batch))))
        results.extend(o["translation_text"] for o in outputs)

    # Post-process: convert remaining PT-PT markers to PT-BR
    results = [_ptpt_to_ptbr(t) for t in results]
    return results, logs


# ── OpenRouter translation (fallback) ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional translator specialising in Brazilian Portuguese (PT-BR). "
    "CRITICAL RULES:\n"
    "1. Output ONLY in Brazilian Portuguese (PT-BR). NEVER use European Portuguese (PT-PT).\n"
    "2. Use 'voce' for second person singular. NEVER use 'tu', 'teu', 'tua', 'vos'.\n"
    "3. Use gerund forms: 'estao fazendo', 'estou vendo'. NEVER use 'estao a fazer', 'estou a ver'.\n"
    "4. Use Brazilian vocabulary: 'onibus' not 'autocarro', 'celular' not 'telemovel', "
    "'trem' not 'comboio', 'banheiro' not 'casa de banho', 'legal' not 'fixe', "
    "'criancas' not 'miudos', 'entender' not 'perceber' (when meaning to understand).\n"
    "5. Use 3rd person verb conjugations with 'voce': 'voce esta' not 'voce estas'.\n"
    "6. Keep informal, conversational register as in the original.\n"
    "7. Preserve all punctuation and segment numbering exactly."
)


def _call_openrouter(texts: list[str], api_key: str, context: list[str] | None = None) -> list[str]:
    """Single API call: translate a chunk of texts, return list of strings.

    context: previously translated utterances prepended as read-only context
    so the model can resolve pronouns and maintain register across chunk boundaries.
    """
    import urllib.request
    import re as _re

    ctx_block = ""
    if context:
        ctx_lines = "\n".join(f"  {t}" for t in context)
        ctx_block = (
            f"[CONTEXT — already translated, do NOT include in output]\n"
            f"{ctx_lines}\n"
            f"[END CONTEXT]\n\n"
        )

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = (
        f"{ctx_block}"
        f"Translate these {len(texts)} numbered utterances to Brazilian Portuguese (PT-BR).\n"
        "Output ONLY the numbered translations — same count, same order, nothing else.\n\n"
        f"{numbered}"
    )

    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dubweave",
            "X-Title": "Dubweave",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())

    raw = data["choices"][0]["message"]["content"].strip()

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    result = []
    for line in lines:
        clean = _re.sub(r"^\d+[\.)\s]+", "", line).strip()
        if clean:
            result.append(clean)
    return result


def _translate_openrouter(texts: list[str], api_key: str, logs: list) -> tuple[list[str], list]:
    """
    Translate via OpenRouter in chunks of 60 utterances with a 3-utterance
    context window prepended to each chunk.

    Context window rationale: without preceding context, the model translates
    each chunk as if it starts a new document. Pronouns, topics, and register
    established earlier are invisible. Prepending the last 3 utterances of the
    previous chunk as [CONTEXT] (not to be translated) gives the model enough
    coherence to resolve anaphora and maintain register across boundaries.
    """
    logs = log(f"🌐 Translating via OpenRouter ({OPENROUTER_MODEL})…", logs)

    CHUNK   = 60   # smaller than 80 to leave room for context tokens
    CONTEXT = 3    # preceding utterances to include as read-only context
    all_translated = []
    total_chunks = (len(texts) + CHUNK - 1) // CHUNK

    for chunk_i in range(total_chunks):
        chunk = texts[chunk_i * CHUNK:(chunk_i + 1) * CHUNK]
        ctx   = all_translated[-CONTEXT:] if all_translated else []

        logs = log(f"   Chunk {chunk_i+1}/{total_chunks} ({len(chunk)} utterances, {len(ctx)} context)…", logs)

        result = _call_openrouter(chunk, api_key, context=ctx)

        # Safety: pad with originals if count mismatches
        if len(result) != len(chunk):
            logs = log(f"   ⚠️  Got {len(result)} back for {len(chunk)} sent — padding with originals", logs)
            while len(result) < len(chunk):
                result.append(chunk[len(result)])
            result = result[:len(chunk)]

        all_translated.extend(result)

    # Final PT-PT safety pass
    all_translated = [_ptpt_to_ptbr(t) for t in all_translated]
    logs = log(f"✅ OpenRouter translated {len(all_translated)} utterances", logs)
    return all_translated, logs


# ── Public translate_segments (NLLB primary, OpenRouter fallback) ─────────────

# ── Segment merging ──────────────────────────────────────────────────────────

def _merge_segments(segments: list) -> list:
    """
    Merge short/incomplete Whisper segments into utterance-sized units.

    Whisper segments are acoustic boundaries, not semantic ones. Short segments
    (< 4 words) and segments without terminal punctuation are mid-utterance
    fragments. Translating them in isolation loses context and produces broken
    output. We merge them into utterances, translate the utterances, then
    re-expand back to original segment granularity by duration proportion.

    Returns:
        merged   — list of {start, end, text, children: [original indices]}
        index_map — merged_idx → [original_seg_indices]
    """
    import re
    TERMINAL = re.compile(r'[.!?]$')
    MIN_WORDS = 4

    merged = []
    current_text = ""
    current_start = None
    current_children = []

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue

        if current_start is None:
            current_start = seg["start"]

        current_text = (current_text + " " + text).strip()
        current_children.append(i)

        word_count = len(current_text.split())
        has_terminal = bool(TERMINAL.search(current_text))

        # Flush if: terminal punctuation AND enough words, or too long (safety)
        if (has_terminal and word_count >= MIN_WORDS) or word_count >= 40:
            merged.append({
                "start":    current_start,
                "end":      seg["end"],
                "text":     current_text,
                "children": current_children[:],
            })
            current_text = ""
            current_start = None
            current_children = []

    # Flush any trailing fragment
    if current_text:
        last_end = segments[current_children[-1]]["end"]
        merged.append({
            "start":    current_start,
            "end":      last_end,
            "text":     current_text,
            "children": current_children[:],
        })

    return merged


def _expand_merged(merged_translated: list, original_segments: list) -> list:
    """
    Re-expand translated utterances back to original segment granularity.

    Each merged utterance may cover multiple original segments. We distribute
    the translated text across children proportionally by original duration.
    This is an approximation — the invariant (acoustic/semantic/synthesis
    boundary mismatch) means perfect alignment is impossible — but it is
    better than translating fragments blind.
    """
    result = []
    for utt in merged_translated:
        children = utt["children"]
        if len(children) == 1:
            result.append({
                "start": original_segments[children[0]]["start"],
                "end":   original_segments[children[0]]["end"],
                "text":  utt["text"],
            })
            continue

        # Distribute translated text by word proportion across children
        translated_words = utt["text"].split()
        total_dur = sum(
            original_segments[c]["end"] - original_segments[c]["start"]
            for c in children
        )
        if total_dur <= 0:
            # Degenerate: give all text to first child
            for j, c in enumerate(children):
                result.append({
                    "start": original_segments[c]["start"],
                    "end":   original_segments[c]["end"],
                    "text":  utt["text"] if j == 0 else "…",
                })
            continue

        word_cursor = 0
        for j, c in enumerate(children):
            seg_dur = original_segments[c]["end"] - original_segments[c]["start"]
            proportion = seg_dur / total_dur
            if j == len(children) - 1:
                # Last child gets remainder to avoid off-by-one word loss
                word_slice = translated_words[word_cursor:]
            else:
                n_words = max(1, round(len(translated_words) * proportion))
                word_slice = translated_words[word_cursor:word_cursor + n_words]
                word_cursor += n_words

            result.append({
                "start": original_segments[c]["start"],
                "end":   original_segments[c]["end"],
                "text":  " ".join(word_slice) if word_slice else "…",
            })

    return result


def translate_segments(segments: list, logs: list, openrouter_key: str = "") -> tuple[list, list]:
    """
    Translate segments EN→PT-BR.

    Pipeline:
      1. Merge short/incomplete Whisper segments into utterances.
         Rationale: Whisper segments are acoustic units; translating fragments
         blind loses context and produces broken output at boundaries.
      2. Translate merged utterances (OpenRouter primary, NLLB fallback).
         Each chunk is sent with 2 preceding utterances as read-only context.
      3. Re-expand translated utterances back to original segment granularity
         by duration proportion.
      4. Apply PT-PT → PT-BR normalizer as final safety pass.
    """
    # ── Step 1: merge ─────────────────────────────────────────────────────────
    merged = _merge_segments(segments)
    logs = log(f"   Merged {len(segments)} Whisper segments → {len(merged)} utterances for translation", logs)
    merged_texts = [u["text"] for u in merged]

    # ── Step 2: translate ─────────────────────────────────────────────────────
    translated_texts = None
    primary_error = None

    if openrouter_key.strip():
        try:
            translated_texts, logs = _translate_openrouter(
                merged_texts, openrouter_key.strip(), logs
            )
        except Exception as e:
            primary_error = str(e)
            logs = log(f"   ⚠️  OpenRouter failed: {primary_error[:120]}", logs)
            logs = log("   Falling back to NLLB-200 local…", logs)

    if translated_texts is None:
        try:
            logs = log("🌐 Translating EN → PT-BR (NLLB-200 local)…", logs)
            translated_texts, logs = _translate_nllb(merged_texts, logs)
            logs = log(f"✅ Translated {len(translated_texts)} utterances (NLLB-200)", logs)
        except Exception as e:
            raise RuntimeError(
                ("All translators failed.\n"
                + (f"OpenRouter error: {primary_error}\n" if primary_error else "")
                + f"NLLB error: {e}")
            )

    # Empty-translation safety: fall back to original English per utterance
    empty_count = 0
    for i, (utt, txt) in enumerate(zip(merged, translated_texts)):
        clean = txt.strip() if txt else ""
        if not clean:
            clean = utt["text"]
            empty_count += 1
        merged[i] = {**utt, "text": clean}
    if empty_count:
        logs = log(f"   ⚠️  {empty_count} empty translation(s) — kept original English", logs)

    # ── Step 3: re-expand to original segment granularity ────────────────────
    translated = _expand_merged(merged, segments)
    logs = log(f"✅ Translated + re-expanded to {len(translated)} segments", logs)
    return translated, logs


# ── Backward constraint: source rate → text budget ───────────────────────────

# XTTS v2 synthesizes Portuguese at roughly this many characters per second
# at natural speaking rate. Measured empirically on a range of texts.
# Used to estimate output duration from character count before synthesis.
XTTS_CHARS_PER_SEC = 18.0   # conservative estimate; real range is 15–22

# PT-BR is structurally ~15% longer than English in character count after
# translation (more syllables, obligatory pronouns, verbal inflection).
# This is the systematic bias the pipeline cannot see without prediction.
PTBR_EXPANSION_FACTOR = 1.15

# Maximum compression we will apply via atempo without quality loss.
# Matches the cap in synthesize_segments.
MAX_ATEMPO = 1.6


def _estimate_synth_duration(text: str) -> float:
    """Estimate XTTS synthesis duration from character count."""
    return len(text.strip()) / XTTS_CHARS_PER_SEC


def _trim_to_budget(text: str, budget_secs: float, openrouter_key: str) -> str:
    """
    Shorten text to fit within budget_secs at natural speaking rate.

    Strategy:
      1. Estimate duration from character count.
      2. If within budget × MAX_ATEMPO, return as-is (atempo can handle it).
      3. Otherwise, truncate to the last complete word that fits the budget,
         then — if an OpenRouter key is available — ask the LLM to rephrase
         to the same meaning in fewer words rather than hard-truncating.

    This is the backward constraint edge: synthesis constraints flow back
    to the text representation before any audio is generated.
    """
    effective_budget = budget_secs * MAX_ATEMPO
    estimated = _estimate_synth_duration(text)

    if estimated <= effective_budget:
        return text  # fits, no action needed

    # Hard truncation fallback: keep words up to character budget
    char_budget = int(effective_budget * XTTS_CHARS_PER_SEC)
    truncated = text[:char_budget].rsplit(" ", 1)[0].rstrip(".,;:")

    if not openrouter_key.strip():
        return truncated  # no LLM available, use hard truncation

    # LLM rephrase: ask for same meaning in fewer words
    try:
        import urllib.request as _ur
        prompt = (
            f"Rephrase this Brazilian Portuguese text to express the same meaning "
            f"in at most {char_budget} characters. "
            f"Output ONLY the rephrased text, nothing else.\n\n{text}"
        )
        payload = json.dumps({
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a Brazilian Portuguese editor. Shorten text while preserving meaning. Never add explanations."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }).encode()
        req = _ur.Request(
            f"{OPENROUTER_BASE}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {openrouter_key.strip()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/dubweave",
                "X-Title": "Dubweave",
            },
            method="POST",
        )
        with _ur.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        rephrased = result["choices"][0]["message"]["content"].strip()
        # Only use rephrased if it's actually shorter
        if len(rephrased) < len(text):
            return rephrased
    except Exception:
        pass  # LLM rephrase failed — fall back to hard truncation

    return truncated


def apply_timing_budget(
    segments: list,
    logs: list,
    openrouter_key: str = "",
) -> tuple[list, list]:
    """
    Pre-flight pass: measure source speaking rate, predict synthesis duration,
    and shorten translated text that will provably overflow its time slot.

    This is the single backward constraint edge in the pipeline. Without it,
    every stage is a one-way valve: synthesis discovers overflow only after
    rendering, then atempo degrades quality to compensate. With it, the text
    is shortened before synthesis runs, so atempo operates within its quality
    range on all segments.

    Source WPM is measured per-segment to handle variable-rate speakers
    (a presenter who slows down for emphasis vs. speaks rapidly in argument).
    """
    trimmed_count = 0
    overflow_count = 0

    for i, seg in enumerate(segments):
        slot_dur = seg["end"] - seg["start"]
        if slot_dur <= 0.1:
            continue

        text = seg.get("text", "").strip()
        if not text:
            continue

        estimated = _estimate_synth_duration(text)
        effective_budget = slot_dur * MAX_ATEMPO

        if estimated > effective_budget:
            overflow_count += 1
            original_len = len(text)
            text = _trim_to_budget(text, slot_dur, openrouter_key)
            segments[i] = {**seg, "text": text}
            if len(text) < original_len:
                trimmed_count += 1

    if overflow_count:
        logs = log(
            f"   ⏱️  {overflow_count} segments predicted to overflow slot — "
            f"{trimmed_count} shortened pre-synthesis, "
            f"{overflow_count - trimmed_count} within atempo range",
            logs
        )
    else:
        logs = log("   ⏱️  All segments within timing budget", logs)

    return segments, logs


import re

def _sanitize_for_tts(text: str) -> str:
    """
    Clean text before passing to XTTS.
    Removes spelled-out punctuation (ponto, virgula) and characters
    XTTS reads aloud instead of treating as prosody cues.
    """
    spelled = [
        (r"\bponto e v\u00edrgula\b",   ","),
        (r"\bponto e virgula\b",         ","),
        (r"\bdois pontos\b",             ","),
        (r"\bponto final\b",             ""),
        (r"\bponto\b",                   ""),
        (r"\bv\u00edrgula\b",            ","),
        (r"\bvirgula\b",                 ","),
        (r"\bexclama\u00e7\u00e3o\b",    "!"),
        (r"\bexclamacao\b",              "!"),
        (r"\binterroga\u00e7\u00e3o\b",  "?"),
        (r"\binterrogacao\b",            "?"),
        (r"\babre par\u00eanteses\b",    ""),
        (r"\bfecha par\u00eanteses\b",   ""),
        (r"\baspas\b",                   ""),
        (r"\btravess\u00e3o\b",          ","),
        (r"\bperiod\b",                  ""),
        (r"\bcomma\b",                   ","),
        (r"\bsemicolon\b",               ","),
        (r"\bcolon\b",                   ","),
        (r"\bexclamation mark\b",        "!"),
        (r"\bquestion mark\b",           "?"),
    ]
    for pattern, replacement in spelled:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"[;:]",     ",",  text)
    text = re.sub(r"\.{2,}",   ",",  text)
    text = re.sub(r"\.",        "",   text)
    text = re.sub(r"[()\[\]{}]", "", text)
    text = re.sub(r'[\u201c\u201d\u2018\u2019"\'`]', "", text)
    text = re.sub(r"[-\u2013\u2014]{2,}", ",", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "\u2026"


def synthesize_segments_kokoro(
    segments: list,
    job_dir: Path,
    logs: list,
    voice: str = KOKORO_VOICE,
    speed: float = KOKORO_SPEED,
):
    """
    Synthesize PT-BR speech using Kokoro-82M.

    Advantages over XTTS v2:
    - 82M params vs ~1B — loads in <2s, uses ~500MB VRAM
    - 24kHz output, natural prosody, no voice-cloning overhead
    - Native PT-BR support (lang_code='p', voices: pf_dora/pm_alex/pm_santa)
    - Returns numpy arrays directly — no subprocess needed

    Requires: pip install kokoro soundfile
              espeak-ng installed system-wide (Windows: espeak-ng MSI)
    """
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline  # type: ignore

    KOKORO_SR = 24000  # Kokoro always outputs at 24kHz

    logs = log(f"🔊 Loading Kokoro-82M (lang=pt-br, voice={voice})…", logs)
    pipeline = KPipeline(lang_code=KOKORO_LANG)
    logs = log("   Kokoro ready", logs)

    # Sanitize and filter empty segments
    empty = [i for i, s in enumerate(segments) if not s.get("text", "").strip()]
    if empty:
        logs = log(f"   ⚠️  {len(empty)} empty segment(s) — skipping", logs)
    segments = [s for s in segments if s.get("text", "").strip()]

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    logs = log(f"🎤 Synthesizing {len(segments)} segments (Kokoro PT-BR, voice={voice})…", logs)

    timed_clips = []
    for i, seg in enumerate(segments):
        out_raw  = seg_dir / f"seg_{i:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{i:04d}.wav"

        text = _sanitize_for_tts(seg["text"].strip())

        # KPipeline returns a generator of (graphemes, phonemes, audio_array)
        # For short segments it yields one chunk; collect and concatenate.
        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio)

        if not chunks:
            logs = log(f"   ⚠️  Kokoro returned no audio for segment {i} — skipping", logs)
            continue

        audio_np = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_raw), audio_np, KOKORO_SR)

        # Timing adjustment — same logic as XTTS path
        orig_dur = seg["end"] - seg["start"]
        if orig_dur > 0.1:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "json", str(out_raw)
            ], capture_output=True, text=True)
            synth_dur = float(json.loads(probe.stdout)["format"]["duration"])

            ratio = synth_dur / orig_dur
            ratio = max(0.8, min(ratio, 1.6))

            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_raw),
                "-filter:a", f"atempo={ratio:.4f}",
                "-ar", "44100", str(out_clip)
            ], capture_output=True)
        else:
            # Resample to 44100 for consistency with the numpy assembler
            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_raw),
                "-ar", "44100", str(out_clip)
            ], capture_output=True)

        timed_clips.append({
            "path":  str(out_clip),
            "start": seg["start"],
            "end":   seg["end"],
        })

        if i % 10 == 0:
            logs = log(f"   Segment {i+1}/{len(segments)}…", logs)

    logs = log("✅ All segments synthesized (Kokoro)", logs)
    return timed_clips, logs


def synthesize_segments(
    segments: list,
    audio_orig: Path,
    job_dir: Path,
    logs: list,
    speaker_wav: str | None = None,
):
    """
    Generate PT-BR speech for each segment using XTTS v2.
    Voice clone from original audio if no speaker_wav provided.
    Stretches/compresses each clip to match original segment duration.
    """
    import torch
    from TTS.api import TTS

    logs = log("🔊 Loading XTTS v2 on GPU…", logs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logs = log(f"   Device: {device.upper()}", logs)

    tts = TTS(XTTS_MODEL).to(device)

    # Use first 30s of original audio as voice clone reference if none provided
    ref_wav = speaker_wav or str(audio_orig)
    if not speaker_wav:
        # Trim to 30s for XTTS reference clip
        ref_wav = str(job_dir / "ref_30s.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_orig),
            "-t", "30", "-ar", "22050", "-ac", "1",
            ref_wav
        ], capture_output=True)

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # Sanitize: empty text crashes XTTS with a misleading "reference_wav" error.
    # This happens when the translation parser drops a segment's content.
    # Log each empty segment so the cause is visible, then skip it.
    empty = [i for i, s in enumerate(segments) if not s.get("text", "").strip()]
    if empty:
        logs = log(f"   ⚠️  {len(empty)} empty segment(s) after translation (indices: {empty[:10]}{'…' if len(empty)>10 else ''}) — skipping", logs)
    segments = [s for s in segments if s.get("text", "").strip()]

    logs = log(f"🎤 Synthesizing {len(segments)} segments (voice clone)…", logs)

    timed_clips = []
    for i, seg in enumerate(segments):
        out_raw  = seg_dir / f"seg_{i:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{i:04d}.wav"

        text = _sanitize_for_tts(seg["text"].strip())

        tts.tts_to_file(
            text=text,
            speaker_wav=ref_wav,
            language=TARGET_LANG,
            file_path=str(out_raw),
        )

        # Timing adjustment: compress or stretch to fit original segment duration.
        #
        # Hard cap at 1.6× speed — beyond this speech becomes unintelligible.
        # If synthesized audio is longer than 1.6× the slot allows, we let it
        # run over into the following silence rather than destroying clarity.
        # Stretching (ratio < 1.0) is always safe; we allow down to 0.8×.
        orig_dur = seg["end"] - seg["start"]
        if orig_dur > 0.1:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "json", str(out_raw)
            ], capture_output=True, text=True)
            synth_dur = float(json.loads(probe.stdout)["format"]["duration"])

            ratio = synth_dur / orig_dur  # >1 = too long, need to speed up

            # Cap: never compress beyond 1.6× (intelligibility limit)
            # Never stretch beyond 0.8× (sounds unnaturally slow)
            ratio = max(0.8, min(ratio, 1.6))

            atempo = f"atempo={ratio:.4f}"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_raw),
                "-filter:a", atempo,
                "-ar", "44100", str(out_clip)
            ], capture_output=True)
        else:
            shutil.copy(str(out_raw), str(out_clip))

        timed_clips.append({
            "path":  str(out_clip),
            "start": seg["start"],
            "end":   seg["end"],
        })

        if i % 10 == 0:
            logs = log(f"   Segment {i+1}/{len(segments)}…", logs)

    logs = log("✅ All segments synthesized", logs)
    return timed_clips, logs


def assemble_dubbed_video(
    video_path: Path,
    timed_clips: list,
    duration: float,
    job_dir: Path,
    title: str,
    logs: list,
):
    """Build silence timeline, overlay each PT-BR segment, mux with video."""
    logs = log("🎬 Assembling final video…", logs)

    def run_ffmpeg(cmd: list, step: str):
        """Run ffmpeg, raise with full stderr on failure."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed at step [{step}]\n"
                f"CMD: {' '.join(cmd[:6])}…\n"
                f"STDERR (last 600 chars):\n{result.stderr[-600:]}"
            )
        return result

    # Audio assembly: place each segment at its timestamp using sox.
    # 
    # All previous amix-based approaches divided amplitude by input count,
    # producing audio 20-34 dB below full scale regardless of weights or
    # normalization passes applied afterward.
    #
    # The correct primitive for this task is NOT mixing (amix) — it is
    # timeline placement. sox `splice` or a Python numpy buffer writes each
    # segment's samples directly at the correct offset. No averaging occurs.
    # Amplitude is preserved exactly as XTTS produced it.
    #
    # We use a numpy buffer: pre-allocate silence, write each clip's samples
    # at its timestamp offset. One read per clip, one write total. O(N) time,
    # O(duration) memory (~80MB for a 13-min mono 44100Hz float32 buffer).

    import numpy as np
    import wave

    SR = 44100
    total_samples = int((duration + 2) * SR)
    buffer = np.zeros(total_samples, dtype=np.float32)

    logs = log("🔊 Placing segments into audio timeline…", logs)
    for clip in timed_clips:
        offset = int(clip["start"] * SR)
        # Read clip WAV into numpy
        try:
            with wave.open(clip["path"], "rb") as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                clip_sr = wf.getframerate()
                n_ch = wf.getnchannels()
                sampwidth = wf.getsampwidth()

            # Convert raw bytes → float32 [-1, 1]
            if sampwidth == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            # Mix down to mono if stereo
            if n_ch == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)

            # Resample if needed (shouldn't happen — all clips are 44100Hz)
            if clip_sr != SR:
                factor = SR / clip_sr
                new_len = int(len(samples) * factor)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, new_len),
                    np.arange(len(samples)),
                    samples
                ).astype(np.float32)

            end = min(offset + len(samples), total_samples)
            buffer[offset:end] += samples[:end - offset]

        except Exception as e:
            logs = log(f"   ⚠️  Could not read clip {clip['path']}: {e}", logs)

    # Peak-normalize to -1 dBFS so the output is loud without clipping
    peak = np.max(np.abs(buffer))
    if peak > 0.001:
        buffer = buffer * (0.891 / peak)  # 0.891 ≈ -1 dBFS
    else:
        logs = log("   ⚠️  Audio buffer is nearly silent — synthesis may have failed", logs)

    # Convert to stereo int16 WAV
    stereo = np.stack([buffer, buffer], axis=1)
    stereo_int16 = (stereo * 32767).clip(-32768, 32767).astype(np.int16)

    mixed_audio = job_dir / "dubbed_audio.wav"
    with wave.open(str(mixed_audio), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo_int16.tobytes())

    logs = log("✅ Audio timeline assembled", logs)

    # 3. Mux video + dubbed audio
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
    output_path = OUTPUT_DIR / f"{safe_title}_PT-BR.mp4"

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(mixed_audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(output_path)
    ], "final mux")

    # Verify the file actually landed on disk
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError(
            f"Output file missing or empty after mux: {output_path}\n"
            f"Check that OUTPUT_DIR is writable: {OUTPUT_DIR}"
        )

    abs_path = str(output_path.resolve())
    logs = log(f"✅ Done! Saved to:", logs)
    logs = log(f"   {abs_path}", logs)
    return abs_path, logs


# ── Cleanup helpers ───────────────────────────────────────────────────────────

def cleanup_stale_jobs(logs: list) -> list:
    """
    Remove job folders older than JOB_MAX_AGE_H hours from WORK_DIR.

    Python does NOT auto-clean tempfiles — processes killed mid-run, browser
    tab closes, or Gradio cancellations all leave orphaned folders behind.
    This runs at the start of every new job so stale work is swept before
    disk fills up. The current job_dir is not yet created when this runs,
    so there is no risk of self-deletion.

    Three sources of orphaned jobs:
      1. Exception mid-pipeline (finally block handles this, but SIGKILL won't)
      2. User closes browser tab — Gradio generator is abandoned, finally may
         not run if the process is forcibly torn down
      3. Previous app crash — WORK_DIR survives across restarts
    """
    if not WORK_DIR.exists():
        return logs

    now = time.time()
    max_age_s = JOB_MAX_AGE_H * 3600
    cleaned = 0

    for entry in WORK_DIR.iterdir():
        if not entry.is_dir():
            continue
        try:
            age = now - entry.stat().st_mtime
            if age > max_age_s:
                shutil.rmtree(str(entry), ignore_errors=True)
                cleaned += 1
        except OSError:
            pass  # already gone or permission issue — skip silently

    if cleaned:
        logs = log(f"🧹 Cleaned {cleaned} stale job folder(s) (>{JOB_MAX_AGE_H}h old)", logs)
    return logs



# ── Project persistence ───────────────────────────────────────────────────────

STAGES = ["download", "transcribe", "translate", "synthesize", "assemble"]

def project_dir(name: str) -> Path:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in " _-")[:60].strip()
    if not safe:
        safe = "project"
    d = PROJECTS_DIR / safe
    d.mkdir(exist_ok=True)
    return d


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(
        d.name for d in PROJECTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def project_status(name: str) -> dict:
    """Return which stage outputs exist for a project."""
    d = project_dir(name)
    return {
        "download":   (d / "video.mp4").exists() and (d / "audio_orig.wav").exists(),
        "transcribe": (d / "segments.json").exists(),
        "translate":  (d / "translated.json").exists(),
        "synthesize": (d / "timed_clips.json").exists(),
        "assemble":   any((d / "outputs").glob("*.mp4")) if (d / "outputs").exists() else False,
    }


def save_project_stage(name: str, stage: str, data):
    """Persist a stage output to the project directory."""
    d = project_dir(name)
    if stage == "download":
        # data = (video_path, audio_path, title, duration)
        # Files are already in job_dir; copy them to project dir
        video_src, audio_src, title, duration = data
        shutil.copy2(str(video_src), str(d / "video.mp4"))
        shutil.copy2(str(audio_src), str(d / "audio_orig.wav"))
        meta = {"title": title, "duration": duration}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    elif stage == "transcribe":
        (d / "segments.json").write_text(json.dumps(data), encoding="utf-8")
    elif stage == "translate":
        (d / "translated.json").write_text(json.dumps(data), encoding="utf-8")
    elif stage == "synthesize":
        # data = timed_clips list; WAV files already in job_dir/segments/
        # Copy segment WAVs to project dir
        seg_dst = d / "segments"
        seg_dst.mkdir(exist_ok=True)
        updated = []
        for clip in data:
            src_path = Path(clip["path"])
            dst_path = seg_dst / src_path.name
            if src_path.exists() and src_path != dst_path:
                shutil.copy2(str(src_path), str(dst_path))
            updated.append({**clip, "path": str(dst_path)})
        (d / "timed_clips.json").write_text(json.dumps(updated), encoding="utf-8")
    elif stage == "assemble":
        # data = output_path string
        out_dst = d / "outputs"
        out_dst.mkdir(exist_ok=True)
        dst = out_dst / Path(data).name
        if Path(data) != dst:
            shutil.copy2(data, str(dst))


def load_project_stage(name: str, stage: str) -> Any:
    """Load a previously saved stage output from project directory."""
    d = project_dir(name)
    try:
        if stage == "download":
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            return d / "video.mp4", d / "audio_orig.wav", meta["title"], meta["duration"]
        elif stage == "transcribe":
            return json.loads((d / "segments.json").read_text(encoding="utf-8"))
        elif stage == "translate":
            return json.loads((d / "translated.json").read_text(encoding="utf-8"))
        elif stage == "synthesize":
            return json.loads((d / "timed_clips.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    url: str,
    speaker_wav_path: str | None,
    whisper_model: str,
    browser: str,
    cookies_file: str | None,
    openrouter_key: str,
    project_name: str,
    resume_from: str,
    tts_engine: str = "XTTS v2 (voice clone)",
    kokoro_voice: str = KOKORO_VOICE,
    progress=gr.Progress(),
):
    logs = []
    proj = project_name.strip() or "default"
    pdir = project_dir(proj)

    # Stage order for resume logic
    stage_order = {s: i for i, s in enumerate(STAGES)}
    resume_idx  = stage_order.get(resume_from, 0)

    # job_dir: temp workspace for files generated this run
    # (segment WAVs etc). Saved stages are persisted to pdir.
    job_id  = str(int(time.time()))
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    model_to_use = whisper_model.strip() if whisper_model.strip() else WHISPER_MODEL

    try:
        lazy_import()
        logs = cleanup_stale_jobs(logs)
        logs = log(f"📁 Project: {proj}  |  Resume from: {resume_from}", logs)
        yield None, "\n".join(logs)

        # ── Download ──────────────────────────────────────────────────────────
        if resume_idx <= stage_order["download"]:
            if cookies_file:
                cookie_msg = f"cookies.txt ({Path(cookies_file).name})"
            elif browser != "none":
                cookie_msg = f"browser cookies ({browser})"
            else:
                cookie_msg = "anonymous (no cookies)"
            logs = log(f"🍪 Download mode: {cookie_msg}", logs)
            yield None, "\n".join(logs)

            progress(0.05, desc="Downloading…")
            video_path, audio_path, title, duration, logs = download_video(
                url, job_dir, logs, browser=browser, cookies_file=cookies_file
            )
            save_project_stage(proj, "download", (video_path, audio_path, title, duration))
            yield None, "\n".join(logs)
        else:
            logs = log("⏭️  Skipping download (loaded from project)", logs)
            res = load_project_stage(proj, "download")
            if not res:
                raise RuntimeError(f"Project '{proj}' download stage not found — cannot resume.")
            video_path, audio_path, title, duration = res
            logs = log(f"   📹 {title} ({duration}s)", logs)
            yield None, "\n".join(logs)

        # ── Transcribe ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["transcribe"]:
            progress(0.2, desc="Transcribing…")
            segments, logs = transcribe_audio(audio_path, logs, model_name=model_to_use)
            save_project_stage(proj, "transcribe", segments)
            yield None, "\n".join(logs)
        else:
            logs = log("⏭️  Skipping transcription (loaded from project)", logs)
            segments = cast(list, load_project_stage(proj, "transcribe"))
            if segments is None:
                raise RuntimeError(f"Project '{proj}' transcription stage not found — cannot resume.")
            logs = log(f"   📝 {len(segments)} segments loaded", logs)
            yield None, "\n".join(logs)

        # ── Translate ─────────────────────────────────────────────────────────
        if resume_idx <= stage_order["translate"]:
            progress(0.4, desc="Translating…")
            translated, logs = translate_segments(segments, logs, openrouter_key=openrouter_key)
            progress(0.5, desc="Checking timing budget…")
            translated, logs = apply_timing_budget(translated, logs, openrouter_key=openrouter_key)
            save_project_stage(proj, "translate", translated)
            yield None, "\n".join(logs)
        else:
            logs = log("⏭️  Skipping translation (loaded from project)", logs)
            translated = cast(list, load_project_stage(proj, "translate"))
            if translated is None:
                raise RuntimeError(f"Project '{proj}' translation stage not found — cannot resume.")
            logs = log(f"   🌐 {len(translated)} translated segments loaded", logs)
            yield None, "\n".join(logs)

        # ── Synthesize ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["synthesize"]:
            progress(0.55, desc="Synthesizing voice…")
            if tts_engine == "Kokoro (fast, PT-BR native)":
                timed_clips, logs = synthesize_segments_kokoro(
                    translated, job_dir, logs, voice=kokoro_voice,
                )
            else:
                timed_clips, logs = synthesize_segments(
                    translated, audio_path, job_dir, logs,
                    speaker_wav=speaker_wav_path,
                )
            save_project_stage(proj, "synthesize", timed_clips)
            yield None, "\n".join(logs)
        else:
            logs = log("⏭️  Skipping synthesis (loaded from project)", logs)
            timed_clips = cast(list, load_project_stage(proj, "synthesize"))
            if timed_clips is None:
                raise RuntimeError(f"Project '{proj}' synthesis stage not found — cannot resume.")
            logs = log(f"   🔊 {len(timed_clips)} audio clips loaded", logs)
            yield None, "\n".join(logs)

        # ── Assemble ──────────────────────────────────────────────────────────
        progress(0.85, desc="Assembling video…")
        output_path, logs = assemble_dubbed_video(
            video_path, timed_clips, float(duration or 0), job_dir, title or "video", logs
        )
        save_project_stage(proj, "assemble", output_path)
        progress(1.0, desc="Done!")
        yield output_path, "\n".join(logs)

    except Exception as e:
        import traceback
        logs = log(f"❌ Error: {e}\n{traceback.format_exc()}", logs)
        yield None, "\n".join(logs)
    finally:
        shutil.rmtree(str(job_dir), ignore_errors=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #12121a;
    --card:    #1a1a26;
    --border:  #2a2a3e;
    --accent:  #00e5a0;
    --accent2: #7c6dff;
    --text:    #e8e8f0;
    --muted:   #6b6b8a;
    --danger:  #ff4f6e;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container { max-width: 900px !important; margin: 0 auto !important; padding: 32px 24px !important; }

/* Hero header */
#header {
    text-align: center;
    padding: 48px 0 40px;
    position: relative;
}
#header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(0,229,160,0.08) 0%, transparent 70%);
    pointer-events: none;
}
#header h1 {
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #00e5a0 0%, #7c6dff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#header p {
    color: var(--muted);
    font-size: 1rem;
    margin: 10px 0 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* Pipeline steps */
#steps {
    display: flex;
    justify-content: center;
    gap: 0;
    margin-bottom: 36px;
}
.step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--border);
    flex-shrink: 0;
}
.step.active .step-dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.step-arrow { color: var(--border); margin: 0 8px; }

/* Panels */
.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,160,0.3), transparent);
}
.panel-label {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Inputs */
input[type="text"], textarea, .gr-textbox input, .gr-textbox textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,229,160,0.08) !important;
}

/* Labels */
label, .gr-form label, .block span {
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Upload area */
.upload-area {
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    transition: border-color 0.2s !important;
}
.upload-area:hover { border-color: var(--accent2) !important; }

/* Run button */
#run-btn {
    width: 100% !important;
    background: linear-gradient(135deg, #00e5a0, #7c6dff) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 16px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    text-transform: uppercase !important;
}
#run-btn:hover { opacity: 0.92 !important; transform: translateY(-1px) !important; }
#run-btn:active { transform: translateY(0) !important; }

/* Log output */
#log-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #06060c !important;
    border-color: var(--border) !important;
    color: var(--accent) !important;
    line-height: 1.7 !important;
}

/* Video output */
video { border-radius: 12px !important; }

/* Info chips */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,109,255,0.1);
    border: 1px solid rgba(124,109,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #a99dff;
    margin: 4px;
}
.chip.green {
    background: rgba(0,229,160,0.08);
    border-color: rgba(0,229,160,0.2);
    color: var(--accent);
}
#chips { margin-bottom: 32px; text-align: center; }

/* Accordion */
.gr-accordion { background: var(--surface) !important; border-color: var(--border) !important; border-radius: 12px !important; }
"""

def build_ui():
    with gr.Blocks(title="Dubweave — PT-BR") as demo:

        gr.HTML("""
        <div id="header">
          <h1>DUBWEAVE</h1>
          <p>youtube → dubbing → português brasileiro</p>
        </div>
        <div id="chips">
          <span class="chip green">⚡ XTTS v2 · GPU</span>
          <span class="chip green">🎙️ Voice Clone</span>
          <span class="chip">🌐 Argos Translate · Local</span>
          <span class="chip">🎬 FFmpeg Mux</span>
          <span class="chip">🔊 Whisper Transcription</span>
        </div>
        <div id="steps">
          <span class="step active"><span class="step-dot"></span>Download</span>
          <span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Transcribe</span>
          <span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Translate</span>
          <span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Synthesize</span>
          <span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Mux</span>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-label">01 · Project</div>')
                with gr.Row():
                    project_name_input = gr.Textbox(
                        placeholder="my-video (letters, numbers, hyphens)",
                        label="Project name",
                        lines=1,
                        scale=2,
                    )
                    resume_from_input = gr.Dropdown(
                        choices=["download", "transcribe", "translate", "synthesize", "assemble"],
                        value="download",
                        label="Resume from stage",
                        scale=1,
                    )
                project_status_html = gr.HTML("<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#6b6b8a;margin-top:6px;'>Enter a project name to see its status.</div>")

        def refresh_status(name):
            name = name.strip()
            if not name:
                return "<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#6b6b8a;'>Enter a project name to see its status.</div>"
            status = project_status(name)
            icons = {True: "<span style='color:#00e5a0'>✓</span>", False: "<span style='color:#3a3a58'>○</span>"}
            parts = " &nbsp;·&nbsp; ".join(
                f"{icons[v]} {s}" for s, v in status.items()
            )
            return f"<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#6b6b8a;margin-top:6px;'>{parts}</div>"

        project_name_input.change(fn=refresh_status, inputs=project_name_input, outputs=project_status_html)

        gr.HTML('<div style="height:8px"></div>')
        gr.HTML('<div class="panel-label">02 · Input</div>')
        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    placeholder="https://youtube.com/watch?v=…",
                    label="YouTube URL (required for download stage, ignored otherwise)",
                    lines=1,
                )

        gr.HTML('<div style="height:12px"></div>')

        with gr.Accordion("🎙️ Custom Voice Reference (optional)", open=False):
            gr.HTML("""
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#6b6b8a;margin:0 0 12px;">
            Upload a WAV/MP3 of the voice you want to clone (10–30s ideal).<br>
            Leave empty to auto-clone from the original video's speaker.
            </p>
            """)
            speaker_input = gr.Audio(
                label="Voice reference clip",
                type="filepath",
                sources=["upload"],
            )

        gr.HTML('<div style="height:8px"></div>')

        with gr.Accordion("⚙️ Transcription Model", open=False):
            gr.HTML("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#6b6b8a;margin:0 0 16px;line-height:1.7;">
              <strong style="color:#e8e8f0;">large-v3-turbo</strong> — Recommended for most videos.<br>
              Distilled from large-v3: ~8× faster, near-identical accuracy on clean audio.<br>
              Uses ~3 GB VRAM. Best choice for YouTube videos with clear speech.<br>
              <br>
              <strong style="color:#e8e8f0;">large-v3</strong> — Maximum accuracy.<br>
              Use this when the video has heavy background music, strong accents,<br>
              technical jargon, or poor audio quality. ~3× slower, uses ~6 GB VRAM.
            </div>
            """)
            whisper_model_input = gr.Radio(
                choices=["large-v3-turbo", "large-v3"],
                value="large-v3-turbo",
                label="Whisper model",
                elem_id="whisper-radio",
            )

        gr.HTML('<div style="height:8px"></div>')

        with gr.Accordion("🔑 OpenRouter API Key (fallback translation)", open=False):
            gr.HTML("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#6b6b8a;margin:0 0 16px;line-height:1.7;">
              <strong style="color:#00e5a0;">If a key is provided</strong>, OpenRouter is used <strong style="color:#e8e8f0;">first</strong> — it follows explicit PT-BR instructions and produces the best output.<br>
              Model: <code style="color:#a99dff;">google/gemini-2.0-flash-001</code>.<br>
              A 10-min video costs roughly <strong style="color:#00e5a0;">~$0.002</strong> via OpenRouter.<br>
              <br>
              If no key is given (or OpenRouter fails), <strong style="color:#e8e8f0;">NLLB-200</strong> runs locally on your GPU as fallback.<br>
              Leave empty to use NLLB-200 only.
            </div>
            """)
            openrouter_key_input = gr.Textbox(
                placeholder="sk-or-v1-…",
                label="OpenRouter API key",
                type="password",
                lines=1,
            )

        gr.HTML('<div style="height:8px"></div>')

        with gr.Accordion("🍪 YouTube Account (optional)", open=False):
            gr.HTML("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#6b6b8a;margin:0 0 16px;line-height:1.7;">
              Logged-in cookies give yt-dlp access to YouTube's most reliable clients,<br>
              avoiding PO token errors and JS challenge failures.<br>
              <br>
              <strong style="color:#e8e8f0;">Option A — Browser</strong>: yt-dlp reads cookies directly from a running browser profile.<br>
              Chrome may fail if the profile is locked. Use Edge as an alternative.<br>
              <br>
              <strong style="color:#e8e8f0;">Option B — cookies.txt</strong>: Export cookies using a browser extension
              (e.g. <em>Get cookies.txt LOCALLY</em> for Chrome) and upload the file here.<br>
              This is the most reliable method and works regardless of which browser you use.<br>
              <br>
              If both are provided, <strong style="color:#e8e8f0;">cookies.txt takes priority</strong>.
              Select browser <strong style="color:#e8e8f0;">none</strong> and leave file empty to download anonymously.
            </div>
            """)
            with gr.Row():
                browser_input = gr.Radio(
                    choices=["none", "chrome", "firefox", "edge", "brave"],
                    value="none",
                    label="Option A · Browser cookies",
                    elem_id="browser-radio",
                )
            cookies_file_input = gr.File(
                label="Option B · cookies.txt file (Netscape format)",
                file_types=[".txt"],
                type="filepath",
            )

        gr.HTML('<div style="height:8px"></div>')

        with gr.Accordion("🔊 TTS Engine", open=True):
            gr.HTML("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#6b6b8a;margin:0 0 16px;line-height:1.7;">
              <strong style="color:#00e5a0;">Kokoro</strong> — recommended. 82M params, loads in &lt;2s, native PT-BR voices,
              no voice cloning. Extremely fast on RTX 4070 Super.<br>
              Voices: <code>pf_dora</code> (female) · <code>pm_alex</code> (male) · <code>pm_santa</code> (male)<br>
              Requires: <code>pip install kokoro soundfile</code> + espeak-ng installed.<br>
              <br>
              <strong style="color:#e8e8f0;">XTTS v2</strong> — clones the original speaker's voice. Slower, uses ~4GB VRAM.
              Best when matching the original speaker matters more than speed.
            </div>
            """)
            tts_engine_input = gr.Radio(
                choices=["Kokoro (fast, PT-BR native)", "XTTS v2 (voice clone)"],
                value="Kokoro (fast, PT-BR native)",
                label="TTS engine",
            )
            kokoro_voice_input = gr.Dropdown(
                choices=["pf_dora", "pm_alex", "pm_santa"],
                value="pf_dora",
                label="Kokoro voice (ignored when using XTTS v2)",
            )

        gr.HTML('<div style="height:16px"></div>')

        run_btn = gr.Button("▶  DUB THIS VIDEO", elem_id="run-btn")

        gr.HTML('<div style="height:20px"></div>')
        gr.HTML('<div class="panel-label">03 · Progress</div>')
        log_output = gr.Textbox(
            label="Pipeline log",
            lines=14,
            max_lines=14,
            interactive=False,
            elem_id="log-box",
        )

        gr.HTML('<div style="height:4px"></div>')
        gr.HTML('<div class="panel-label">04 · Output</div>')
        video_output = gr.Video(label="Dubbed video (PT-BR)")

        gr.HTML("""
        <div style="text-align:center;margin-top:40px;padding-top:24px;border-top:1px solid #2a2a3e;">
          <p style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3a3a58;">
            XTTS v2 · Whisper · NLLB-200 / Gemini · yt-dlp · FFmpeg · RTX 4070 Super
          </p>
        </div>
        """)

        run_btn.click(
            fn=run_pipeline,
            inputs=[url_input, speaker_input, whisper_model_input, browser_input, cookies_file_input, openrouter_key_input, project_name_input, resume_from_input, tts_engine_input, kokoro_voice_input],
            outputs=[video_output, log_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=3)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )