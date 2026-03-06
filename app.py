"""
Dubweave — YouTube → Brazilian Portuguese Dubbing Pipeline
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

import gradio as gr

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
WORK_DIR = Path(tempfile.gettempdir()) / "dubweave"
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

WHISPER_MODEL  = "medium"      # change to "large-v3" for max quality
XTTS_MODEL     = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_LANG    = "pt"          # Brazilian Portuguese
ARGOS_FROM     = "en"
ARGOS_TO       = "pt"


# ── Step helpers ──────────────────────────────────────────────────────────────

def log(msg: str, logs: list) -> list:
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    logs.append(entry)
    return logs


def download_video(url: str, job_dir: Path, logs: list):
    """Download best video+audio streams separately via yt-dlp."""
    logs = log("📥 Downloading video with yt-dlp…", logs)

    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio_orig.wav"

    # Download video (no audio)
    ydl_opts_video = {
        "format": "bestvideo[ext=mp4]/bestvideo",
        "outtmpl": str(job_dir / "video_raw.%(ext)s"),
        "quiet": True,
    }
    with __import__("yt_dlp").YoutubeDL(ydl_opts_video) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")
        duration = info.get("duration", 0)

    # Find downloaded video file
    raw_video = next(job_dir.glob("video_raw.*"))
    shutil.move(str(raw_video), str(video_path))

    # Download audio only → WAV
    ydl_opts_audio = {
        "format": "bestaudio/best",
        "outtmpl": str(job_dir / "audio_raw.%(ext)s"),
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
    }
    with __import__("yt_dlp").YoutubeDL(ydl_opts_audio) as ydl:
        ydl.extract_info(url, download=True)

    raw_audio = next(job_dir.glob("audio_raw.*"))
    shutil.move(str(raw_audio), str(audio_path))

    logs = log(f"✅ Downloaded: \"{title}\" ({duration}s)", logs)
    return video_path, audio_path, title, duration, logs


def transcribe_audio(audio_path: Path, logs: list):
    """Transcribe audio with Whisper, return segments with timestamps."""
    import whisper
    logs = log(f"🎙️ Transcribing with Whisper ({WHISPER_MODEL})…", logs)

    model = whisper.load_model(WHISPER_MODEL)
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
    """Ensure EN→PT language pack is installed."""
    import argostranslate.package
    import argostranslate.translate

    available = argostranslate.translate.get_installed_languages()
    codes = [l.code for l in available]
    if ARGOS_FROM in codes and ARGOS_TO in codes:
        return  # already installed

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available_packages if p.from_code == ARGOS_FROM and p.to_code == ARGOS_TO),
        None
    )
    if pkg:
        argostranslate.package.install_from_path(pkg.download())


def translate_segments(segments: list, logs: list):
    """Translate each segment EN→PT-BR using Argos Translate (fully local)."""
    import argostranslate.translate

    logs = log("🌐 Translating EN → PT-BR (local Argos Translate)…", logs)
    install_argos_language_pair()

    langs = argostranslate.translate.get_installed_languages()
    from_lang = next(l for l in langs if l.code == ARGOS_FROM)
    translation = from_lang.get_translation(
        next(l for l in langs if l.code == ARGOS_TO)
    )

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

    # 1. Build a silent base track at full duration
    silent_base = job_dir / "silent_base.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", str(duration + 2),
        str(silent_base)
    ], capture_output=True)

    # 2. Mix each segment onto the silent base with adelay
    # Build complex filter for amix
    inputs = ["-i", str(silent_base)]
    filter_parts = []

    for i, clip in enumerate(timed_clips):
        inputs += ["-i", clip["path"]]
        delay_ms = int(clip["start"] * 1000)
        filter_parts.append(
            f"[{i+1}]adelay={delay_ms}|{delay_ms}[d{i}]"
        )

    # Chain amix
    mix_inputs = "[0]" + "".join(f"[d{i}]" for i in range(len(timed_clips)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(timed_clips)+1}:normalize=0[aout]"
    )
    filter_complex = ";".join(filter_parts)

    mixed_audio = job_dir / "dubbed_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y"] + inputs +
        ["-filter_complex", filter_complex, "-map", "[aout]", str(mixed_audio)],
        capture_output=True
    )

    # 3. Mux video + dubbed audio
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
    output_path = OUTPUT_DIR / f"{safe_title}_PT-BR.mp4"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(mixed_audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(output_path)
    ], capture_output=True)

    logs = log(f"✅ Done! → {output_path}", logs)
    return str(output_path), logs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(url: str, speaker_wav_path: str | None, progress=gr.Progress()):
    logs = []
    job_id = str(int(time.time()))
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        lazy_import()

        progress(0.05, desc="Downloading…")
        video_path, audio_path, title, duration, logs = download_video(url, job_dir, logs)
        yield None, "\n".join(logs)

        progress(0.2, desc="Transcribing…")
        segments, logs = transcribe_audio(audio_path, logs)
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
        # Cleanup job dir (keep output)
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
    with gr.Blocks(css=CSS, title="Dubweave — PT-BR") as demo:

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

        gr.HTML('<div style="height:4px"></div>')

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
            inputs=[url_input, speaker_input],
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
    )
