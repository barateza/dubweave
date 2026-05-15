from __future__ import annotations

import os
import re
import shutil
import subprocess
import json
from pathlib import Path

from src.config import (
    SUPERTONIC_AUTO_DOWNLOAD,
    SUPERTONIC_ASSETS_DIR,
    SUPERTONIC_DEFAULT_LANG,
    SUPERTONIC_DEFAULT_VOICE,
    SUPERTONIC_INTRA_OP_THREADS,
    SUPERTONIC_INTER_OP_THREADS,
)
from src.core.translate import PipelineError
from src.utils.helpers import log

ALLOWED_TAGS = {"laugh", "breath", "sigh"}
DEFAULT_SUPERTONIC_SAMPLE_RATE = 24000
DEFAULT_SUPERTONIC_CPS = 16.0


def _sanitize_supertonic_text(text: str) -> str:
    """Keep allowed expressive tags and remove unsupported markup for Supertonic."""
    if not text:
        return "…"

    allowed_pattern = "|".join(ALLOWED_TAGS)
    clean = text
    clean = re.sub(
        rf"<(?!/?(?:{allowed_pattern})\b)[^>]+>", "", clean, flags=re.IGNORECASE
    )
    clean = re.sub(
        rf"</?(?:{allowed_pattern})(?:\s*/?>)",
        lambda m: m.group(0).lower(),
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"[\u201c\u201d\u2018\u2019\"`]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean if clean else "…"


def get_supertonic_model_dir() -> Path | None:
    if SUPERTONIC_ASSETS_DIR:
        return Path(SUPERTONIC_ASSETS_DIR).expanduser().resolve()
    return None


def ensure_supertonic_assets_or_raise() -> None:
    model_dir = get_supertonic_model_dir()
    if model_dir and model_dir.exists():
        return
    if SUPERTONIC_AUTO_DOWNLOAD:
        return
    raise PipelineError(
        "Validation",
        "Supertonic assets are missing and SUPERTONIC_AUTO_DOWNLOAD is false. Set SUPERTONIC_ASSETS_DIR or enable auto-download.",
    )


def build_supertonic_tts():
    try:
        from supertonic import TTS
    except ImportError as exc:
        raise PipelineError("Synthesize", f"supertonic not installed: {exc}")

    kwargs: dict[str, object] = {
        "auto_download": SUPERTONIC_AUTO_DOWNLOAD,
    }
    model_dir = get_supertonic_model_dir()
    if model_dir is not None:
        kwargs["model_dir"] = str(model_dir)
    if SUPERTONIC_INTRA_OP_THREADS > 0:
        kwargs["intra_op_num_threads"] = SUPERTONIC_INTRA_OP_THREADS
    if SUPERTONIC_INTER_OP_THREADS > 0:
        kwargs["inter_op_num_threads"] = SUPERTONIC_INTER_OP_THREADS

    tts = TTS(**kwargs)
    return tts


def get_provider_label() -> str:
    try:
        import onnxruntime as ort
    except ImportError:
        return "cpu (onnxruntime missing)"

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "gpu"
    if "DmlExecutionProvider" in providers:
        return "gpu"
    if "CoreMLExecutionProvider" in providers:
        return "gpu"
    return "cpu"


def synthesize_segments_supertonic(
    segments: list,
    job_dir: Path,
    logs: list,
    voice: str = SUPERTONIC_DEFAULT_VOICE,
    lang: str = SUPERTONIC_DEFAULT_LANG,
    total_steps: int = 5,
    speed: float = 1.0,
):
    import numpy as np
    import soundfile as sf

    voice = (voice or SUPERTONIC_DEFAULT_VOICE).strip().upper()
    lang = (lang or SUPERTONIC_DEFAULT_LANG).strip().lower()
    ensure_supertonic_assets_or_raise()
    provider = get_provider_label()
    if provider == "gpu":
        log("🔊 Loading Supertonic v3 (GPU preferred)…", logs)
    elif provider.startswith("cpu"):
        log("🔊 Loading Supertonic v3 (CPU)…", logs)
    else:
        log(f"🔊 Loading Supertonic v3 ({provider})…", logs)

    try:
        tts = build_supertonic_tts()
    except Exception as exc:
        raise PipelineError("Synthesize", f"Supertonic initialization failed: {exc}")

    try:
        voice_style = tts.get_voice_style(voice_name=voice)
    except Exception as exc:
        raise PipelineError("Synthesize", f"Invalid Supertonic voice '{voice}': {exc}")

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    clean_segments = [s for s in segments if s.get("text", "").strip()]
    timed_clips = []

    for idx, seg in enumerate(clean_segments):
        out_raw = seg_dir / f"seg_{idx:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{idx:04d}.wav"
        text = _sanitize_supertonic_text(seg["text"].strip())

        try:
            wav, duration = tts.synthesize(
                text,
                voice_style=voice_style,
                lang=lang,
                total_steps=total_steps,
                speed=speed,
                verbose=False,
            )
            tts.save_audio(wav, str(out_raw))

            synth_dur = (
                float(duration[0])
                if hasattr(duration, "__getitem__")
                else float(duration)
            )
            orig_dur = seg["end"] - seg["start"]
            if orig_dur > 0.1 and synth_dur > 0:
                ratio = max(0.5, min(synth_dur / orig_dur, 1.6))
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(out_raw),
                        "-filter:a",
                        f"atempo={ratio:.4f}",
                        "-ar",
                        "44100",
                        str(out_clip),
                    ],
                    capture_output=True,
                )
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(out_raw), "-ar", "44100", str(out_clip)],
                    capture_output=True,
                )
        except Exception as exc:
            log(f"   ⚠️  Supertonic failed (segment {idx}): {exc}", logs)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=44100:cl=stereo",
                    "-t",
                    "0.5",
                    str(out_clip),
                ],
                capture_output=True,
            )

        timed_clips.append(
            {"path": str(out_clip), "start": seg["start"], "end": seg["end"]}
        )
        if idx % 10 == 0:
            log(f"   Segment {idx + 1}/{len(clean_segments)}…", logs)

    return timed_clips, logs
