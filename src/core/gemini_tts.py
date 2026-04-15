from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from src.config import GEMINI_TTS_MODEL
from src.core.synthesis import _clamp_atempo_ratio
from src.core.translate import PipelineError
from src.utils.helpers import log


RETRYABLE_ERROR_MARKERS = (
    "http 500",
    "internal",
    "temporar",
    "unavailable",
    "text token",
    "deadline exceeded",
    "timed out",
)



def _looks_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in RETRYABLE_ERROR_MARKERS)



def decode_inline_audio_data(data: Any) -> bytes:
    """Normalize Gemini inline audio payload to bytes."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        return base64.b64decode(data)
    raise ValueError(f"Unsupported inline audio type: {type(data)!r}")



def select_speaker_for_segment(
    text: str,
    idx: int,
    speaker1_name: str,
    speaker2_name: str,
    assignment_mode: str,
) -> tuple[str, str]:
    """Resolve a speaker label and content text for a single segment."""
    clean_text = text.strip()
    mode = (assignment_mode or "alternate").strip().lower()

    if mode == "prefix":
        for speaker in (speaker1_name, speaker2_name):
            prefix = f"{speaker}:"
            if clean_text.lower().startswith(prefix.lower()):
                return speaker, clean_text[len(prefix) :].strip()

    if mode == "alternate":
        return (speaker1_name, clean_text) if idx % 2 == 0 else (speaker2_name, clean_text)

    return speaker1_name, clean_text



def _build_single_speaker_prompt(text: str) -> str:
    return (
        "Synthesize natural Brazilian Portuguese speech. "
        "Speak only the transcript content exactly once.\n\n"
        f"TRANSCRIPT:\n{text}"
    )



def _build_multi_speaker_prompt(speaker_name: str, text: str) -> str:
    return (
        "Synthesize speech for the conversation line below. "
        "Speak only the line content after the speaker label.\n\n"
        f"{speaker_name}: {text}"
    )



def _write_pcm_wave(file_path: Path, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)



def _generate_with_retry(callable_fn, max_retries: int = 3, base_delay: float = 1.5):
    for attempt in range(max_retries + 1):
        try:
            return callable_fn()
        except Exception as exc:
            if attempt == max_retries or not _looks_retryable(exc):
                raise
            time.sleep(base_delay * (2**attempt))



def synthesize_segments_gemini_tts(
    segments: list,
    job_dir: Path,
    logs: list,
    api_key: str,
    model: str = GEMINI_TTS_MODEL,
    single_voice: str = "Kore",
    multi_speaker: bool = False,
    speaker1_name: str = "Speaker1",
    speaker1_voice: str = "Kore",
    speaker2_name: str = "Speaker2",
    speaker2_voice: str = "Puck",
    assignment_mode: str = "alternate",
):
    """Synthesize segments using Gemini 3.1 Flash TTS Preview."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise PipelineError("Synthesize", f"google-genai not installed: {exc}")

    if not api_key.strip():
        raise PipelineError("Synthesize", "GEMINI_TTS_API_KEY missing.")

    client = genai.Client(api_key=api_key.strip())

    log(
        (
            f"🔊 Gemini TTS ready (model={model}, "
            f"mode={'multi-speaker' if multi_speaker else 'single-speaker'})"
        ),
        logs,
    )

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    clean_segments = [s for s in segments if s.get("text", "").strip()]
    timed_clips = []

    for idx, seg in enumerate(clean_segments):
        out_raw = seg_dir / f"seg_{idx:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{idx:04d}.wav"

        text = seg["text"].strip()

        if multi_speaker:
            speaker_name, speaker_text = select_speaker_for_segment(
                text,
                idx,
                speaker1_name,
                speaker2_name,
                assignment_mode,
            )
            prompt = _build_multi_speaker_prompt(speaker_name, speaker_text)
            speech_config = types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=speaker1_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=speaker1_voice,
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker=speaker2_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=speaker2_voice,
                                )
                            ),
                        ),
                    ]
                )
            )
        else:
            prompt = _build_single_speaker_prompt(text)
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=single_voice)
                )
            )

        def _call_gemini():
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config,
                ),
            )

        try:
            response = _generate_with_retry(_call_gemini)
            data = response.candidates[0].content.parts[0].inline_data.data
            pcm_bytes = decode_inline_audio_data(data)
            _write_pcm_wave(out_raw, pcm_bytes)

            orig_dur = seg["end"] - seg["start"]
            if orig_dur > 0.1:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "json",
                        str(out_raw),
                    ],
                    capture_output=True,
                    text=True,
                )
                synth_dur = float(json.loads(probe.stdout)["format"]["duration"])
                ratio = _clamp_atempo_ratio(synth_dur / orig_dur)
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
            log(f"   ⚠️  Gemini TTS failed (segment {idx}): {exc}", logs)
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

        timed_clips.append({"path": str(out_clip), "start": seg["start"], "end": seg["end"]})
        if idx % 10 == 0:
            log(f"   Segment {idx+1}/{len(clean_segments)}…", logs)

    return timed_clips, logs
