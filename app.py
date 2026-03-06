"""
YT Dubber — YouTube → Brazilian Portuguese Dubbing Pipeline
Uses: yt-dlp → Whisper → Argos Translate (local) → XTTS v2 (GPU) → FFmpeg
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
from typing import Generator

import warnings
import gradio as gr

# Suppress torch.load pickle warnings from TTS/XTTS internals.
# These are known, safe, third-party model files — not a security concern here.
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS")
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

# ── Lazy imports (installed at runtime) ──────────────────────────────────────
def lazy_import():
    global yt_dlp, whisper, torch, TTS, argostranslate
    import yt_dlp
    import whisper
    import torch
    from TTS.api import TTS
    import argostranslate.package
    import argostranslate.translate
    return True


# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR = Path(tempfile.gettempdir()) / "yt_dubber"
WORK_DIR.mkdir(exist_ok=True)
# Always resolve relative to this script file, not the shell working directory.
# pixi run may cd anywhere — a relative path is unreliable.
OUTPUT_DIR = Path(__file__).parent.resolve() / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WHISPER_MODEL  = "large-v3-turbo"  # swap to "large-v3" for max accuracy on noisy audio
XTTS_MODEL     = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_LANG    = "pt"               # XTTS v2 uses "pt" for all Portuguese; BR accent comes from the voice reference clip
ARGOS_FROM     = "en"
ARGOS_TO       = "pt"

JOB_MAX_AGE_H  = 2       # hours before a stale job folder is eligible for cleanup


# ── Step helpers ──────────────────────────────────────────────────────────────

def log(msg: str, logs: list) -> list:
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    logs.append(entry)
    return logs


def download_video(url: str, job_dir: Path, logs: list):
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
        "extractor_args": {"youtube": {"player_client": ["web", "android"]}},
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
        with yt.YoutubeDL({**BASE_OPTS, "skip_download": True}) as ydl:
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
            with yt.YoutubeDL(opts) as ydl:
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
                with yt.YoutubeDL(opts) as ydl:
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


def transcribe_audio(audio_path: Path, logs: list, model_name: str = WHISPER_MODEL):
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

    segments = result["segments"]
    logs = log(f"✅ Transcribed {len(segments)} segments", logs)
    return segments, logs


def install_argos_language_pair():
    """
    Ensure the best available EN→PT package is installed.

    Argos Translate's package index uses plain "pt" for its EN→PT model —
    there is no separate pt_BR entry. The "pt" model was trained on mixed
    Portuguese corpora and leans European in some lexical choices, but it
    is the only local option. XTTS v2 with language="pt-br" corrects the
    accent and prosody at the synthesis stage, which is where the Brazilian
    identity actually lives. Translation quality differences between PT-PT
    and PT-BR are minor for spoken content (contractions, some vocabulary);
    synthesis accent is the dominant perceptual factor.

    Preference order: en→pt_br > en→pt > any en→pt* (future-proofing)
    """
    import argostranslate.package
    import argostranslate.translate

    available = argostranslate.translate.get_installed_languages()
    codes = [l.code for l in available]

    # Check if we already have any usable PT variant installed
    pt_variants = [c for c in codes if c.startswith("pt")]
    if ARGOS_FROM in codes and pt_variants:
        return

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Prefer pt_br → pt → any pt* 
    def pkg_priority(p):
        if p.from_code != ARGOS_FROM: return 99
        if p.to_code == "pt_br":      return 0
        if p.to_code == "pt":         return 1
        if p.to_code.startswith("pt"): return 2
        return 99

    candidates = [p for p in available_packages if p.from_code == ARGOS_FROM
                  and p.to_code.startswith("pt")]
    if not candidates:
        raise RuntimeError("No EN→PT Argos package found in index.")

    best = sorted(candidates, key=pkg_priority)[0]
    argostranslate.package.install_from_path(best.download())
    return best.to_code   # returns actual installed code so translate_segments can use it


def translate_segments(segments: list, logs: list):
    """Translate each segment EN→PT using Argos Translate (fully local).
    
    Note on Brazilian Portuguese: Argos handles the text/lexical layer.
    The accent, intonation, and prosody of PT-BR are enforced by XTTS v2
    via language="pt-br" in synthesize_segments — that is the correct layer
    for accent control, not the translation layer.
    """
    import argostranslate.translate

    logs = log("🌐 Translating EN → PT-BR (local Argos Translate)…", logs)
    install_argos_language_pair()

    langs = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in langs if l.code == ARGOS_FROM), None)
    if from_lang is None:
        raise RuntimeError(f"Argos: source language '{ARGOS_FROM}' not found after install.")

    # Find best PT target: prefer pt_br, fall back to pt, then any pt*
    to_lang = (
        next((l for l in langs if l.code == "pt_br"), None) or
        next((l for l in langs if l.code == "pt"), None) or
        next((l for l in langs if l.code.startswith("pt")), None)
    )
    if to_lang is None:
        raise RuntimeError("Argos: no PT target language found after install.")

    logs = log(f"   Argos translation model: {ARGOS_FROM}→{to_lang.code}", logs)
    translation = from_lang.get_translation(to_lang)

    translated = []
    for seg in segments:
        pt_text = translation.translate(seg["text"].strip())
        translated.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  pt_text,
        })

    logs = log(f"✅ Translated {len(translated)} segments", logs)
    return translated, logs


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

    logs = log(f"🎤 Synthesizing {len(segments)} segments (voice clone)…", logs)

    timed_clips = []
    for i, seg in enumerate(segments):
        out_raw  = seg_dir / f"seg_{i:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{i:04d}.wav"

        tts.tts_to_file(
            text=seg["text"],
            speaker_wav=ref_wav,
            language=TARGET_LANG,
            file_path=str(out_raw),
        )

        # Stretch/compress to match original duration
        orig_dur = seg["end"] - seg["start"]
        if orig_dur > 0.1:
            # Get synthesized duration
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "json", str(out_raw)
            ], capture_output=True, text=True)
            synth_dur = float(json.loads(probe.stdout)["format"]["duration"])

            # atempo supports 0.5–2.0; chain filters for extremes
            ratio = synth_dur / orig_dur
            ratio = max(0.4, min(ratio, 3.5))  # safety clamp

            if 0.5 <= ratio <= 2.0:
                atempo = f"atempo={ratio:.4f}"
            elif ratio < 0.5:
                atempo = f"atempo={max(ratio**0.5, 0.5):.4f},atempo={ratio/max(ratio**0.5,0.5):.4f}"
            else:  # > 2.0
                atempo = f"atempo=2.0,atempo={ratio/2.0:.4f}"

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

    # 1. Build a silent base track at full duration
    silent_base = job_dir / "silent_base.wav"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(duration + 2),
        str(silent_base)
    ], "silent base")

    # 2. Mix each segment onto the silent base with adelay
    inputs = ["-i", str(silent_base)]
    filter_parts = []

    for i, clip in enumerate(timed_clips):
        inputs += ["-i", clip["path"]]
        delay_ms = int(clip["start"] * 1000)
        filter_parts.append(
            f"[{i+1}]adelay={delay_ms}|{delay_ms}[d{i}]"
        )

    mix_inputs = "[0]" + "".join(f"[d{i}]" for i in range(len(timed_clips)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(timed_clips)+1}:normalize=0[aout]"
    )
    filter_complex = ";".join(filter_parts)

    mixed_audio = job_dir / "dubbed_audio.wav"
    run_ffmpeg(
        ["ffmpeg", "-y"] + inputs +
        ["-filter_complex", filter_complex, "-map", "[aout]", str(mixed_audio)],
        "audio mix"
    )

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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(url: str, speaker_wav_path: str | None, whisper_model: str, progress=gr.Progress()):
    logs = []
    job_id = str(int(time.time()))
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Resolve model: UI value overrides global default
    model_to_use = whisper_model.strip() if whisper_model.strip() else WHISPER_MODEL

    try:
        lazy_import()

        # Sweep stale jobs from previous runs before starting
        logs = cleanup_stale_jobs(logs)
        yield None, "\n".join(logs)

        progress(0.05, desc="Downloading…")
        video_path, audio_path, title, duration, logs = download_video(url, job_dir, logs)
        yield None, "\n".join(logs)

        progress(0.2, desc="Transcribing…")
        segments, logs = transcribe_audio(audio_path, logs, model_name=model_to_use)
        yield None, "\n".join(logs)

        progress(0.4, desc="Translating…")
        translated, logs = translate_segments(segments, logs)
        yield None, "\n".join(logs)

        progress(0.55, desc="Synthesizing voice…")
        timed_clips, logs = synthesize_segments(
            translated, audio_path, job_dir, logs,
            speaker_wav=speaker_wav_path
        )
        yield None, "\n".join(logs)

        progress(0.85, desc="Assembling video…")
        output_path, logs = assemble_dubbed_video(
            video_path, timed_clips, duration, job_dir, title, logs
        )
        progress(1.0, desc="Done!")
        yield output_path, "\n".join(logs)

    except Exception as e:
        import traceback
        logs = log(f"❌ Error: {e}\n{traceback.format_exc()}", logs)
        yield None, "\n".join(logs)
    finally:
        # Always clean current job dir — runs on success, exception, AND
        # graceful Gradio cancellation. Does NOT run on SIGKILL (OS kill),
        # which is why cleanup_stale_jobs() handles those on next startup.
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
                gr.HTML('<div class="panel-label">01 · Input</div>')
                url_input = gr.Textbox(
                    placeholder="https://youtube.com/watch?v=…",
                    label="YouTube URL",
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

        gr.HTML('<div style="height:16px"></div>')

        run_btn = gr.Button("▶  DUB THIS VIDEO", elem_id="run-btn")

        gr.HTML('<div style="height:20px"></div>')
        gr.HTML('<div class="panel-label">02 · Progress</div>')
        log_output = gr.Textbox(
            label="Pipeline log",
            lines=14,
            max_lines=14,
            interactive=False,
            elem_id="log-box",
        )

        gr.HTML('<div style="height:4px"></div>')
        gr.HTML('<div class="panel-label">03 · Output</div>')
        video_output = gr.Video(label="Dubbed video (PT-BR)")

        gr.HTML("""
        <div style="text-align:center;margin-top:40px;padding-top:24px;border-top:1px solid #2a2a3e;">
          <p style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3a3a58;">
            XTTS v2 · Whisper · Argos Translate · yt-dlp · FFmpeg · RTX 4070 Super
          </p>
        </div>
        """)

        run_btn.click(
            fn=run_pipeline,
            inputs=[url_input, speaker_input, whisper_model_input],
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