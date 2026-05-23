import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from src.config import (
    EDGE_TTS_VOICE_NAME,
    OUTPUT_DIR,
    OPENROUTER_MODEL,
    OPENROUTER_BASE,
)
from src.utils.helpers import log, retry_with_backoff
from src.core.translate import PipelineError

# ── Timing & Budget ──────────────────────────────────────────────────────────

VOICE_CALIBRATION: dict[str, float] = {
    "pt-BR-FranciscaNeural": 11.1,
    "pt-BR-AntonioNeural": 11.1,
    "pt-BR-ThalitaNeural": 13.1,
    "default": 15.1,
}

MIN_ATEMPO = 0.5
MAX_ATEMPO = 1.6


def _clamp_atempo_ratio(ratio: float) -> float:
    return max(MIN_ATEMPO, min(ratio, MAX_ATEMPO))


def _estimate_synth_duration(text: str, cps: float = 15.1) -> float:
    return len(text.strip()) / cps


def get_cps_for_voice(engine: str, voice: str) -> float:
    return VOICE_CALIBRATION.get(voice, VOICE_CALIBRATION["default"])


def _trim_to_budget(
    text: str, budget_secs: float, openrouter_key: str, cps: float = 15.1
) -> str:
    effective_budget = budget_secs * MAX_ATEMPO
    estimated = _estimate_synth_duration(text, cps=cps)
    if estimated <= effective_budget:
        return text
    char_budget = int(effective_budget * cps)
    truncated = text[:char_budget].rsplit(" ", 1)[0].rstrip(".,;:")
    if not openrouter_key.strip():
        return truncated
    try:
        import urllib.request

        prompt = f"Rephrase this text to express the same meaning in at most {char_budget} characters. Output ONLY the rephrased text.\n\n{text}"
        payload = json.dumps(
            {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a text editor. Shorten text while preserving meaning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            }
        ).encode()
        req = urllib.request.Request(
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
        data = retry_with_backoff(
            lambda: json.loads(urllib.request.urlopen(req, timeout=30).read())
        )
        rephrased = data["choices"][0]["message"]["content"].strip()
        if len(rephrased) < len(text):
            return rephrased
    except Exception:
        pass
    return truncated


def apply_timing_budget(
    segments: list, logs: list, openrouter_key: str = "", cps: float = 15.1
) -> tuple[list, list]:
    trimmed_count, overflow_count = 0, 0
    for i, seg in enumerate(segments):
        slot_dur = seg["end"] - seg["start"]
        if slot_dur <= 0.1 or not seg.get("text", "").strip():
            continue
        text = seg["text"].strip()
        estimated = _estimate_synth_duration(text, cps=cps)
        if estimated > (slot_dur * MAX_ATEMPO):
            overflow_count += 1
            original_len = len(text)
            text = _trim_to_budget(text, slot_dur, openrouter_key, cps=cps)
            segments[i] = {**seg, "text": text}
            if len(text) < original_len:
                trimmed_count += 1
    if overflow_count:
        log(
            f"   ⏱️  {overflow_count} segments predicted to overflow slot — {trimmed_count} shortened",
            logs,
        )
    else:
        log("   ⏱️  All segments within timing budget", logs)
    return segments, logs


# ── Sanitization ─────────────────────────────────────────────────────────────


def _sanitize_for_tts(text: str) -> str:
    spelled = [
        (r"\bponto e v\u00edrgula\b", ","),
        (r"\bponto e virgula\b", ","),
        (r"\bdois pontos\b", ","),
        (r"\bponto final\b", ""),
        (r"\bponto\b", ""),
        (r"\bv\u00edrgula\b", ","),
        (r"\bvirgula\b", ","),
        (r"\bexclama\u00e7\u00e3o\b", "!"),
        (r"\bexclamacao\b", "!"),
        (r"\binterroga\u00e7\u00e3o\b", "?"),
        (r"\binterrogacao\b", "?"),
        (r"\babre par\u00eanteses\b", ""),
        (r"\bfecha par\u00eanteses\b", ""),
        (r"\baspas\b", ""),
        (r"\btravess\u00e3o\b", ","),
        (r"\bperiod\b", ""),
        (r"\bcomma\b", ","),
        (r"\bsemicolon\b", ","),
        (r"\bcolon\b", ","),
        (r"\bexclamation mark\b", "!"),
        (r"\bquestion mark\b", "?"),
    ]
    for pattern, replacement in spelled:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[;:]", ",", text)
    text = re.sub(r"\.{2,}", ",", text)
    text = re.sub(r"\.(?=\s)", ",", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"[()\[\]{}]", "", text)
    text = re.sub(r'[\u201c\u201d\u2018\u2019"\'`]', "", text)
    text = re.sub(r"[-\u2013\u2014]{2,}", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "\u2026"


# ── Edge TTS ─────────────────────────────────────────────────────────────────


def synthesize_segments_edge_tts(
    segments: list,
    job_dir: Path,
    logs: list,
    voice: str = EDGE_TTS_VOICE_NAME,
    speed: float = 1.0,
):
    import asyncio, io

    try:
        import edge_tts
    except ImportError:
        raise PipelineError("Synthesize", "edge-tts not installed.")
    rate_str = f"{round((speed - 1.0) * 100):+d}%"
    log(f"🔊 Edge TTS ready (voice={voice}, rate={rate_str})", logs)
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    segments = [s for s in segments if s.get("text", "").strip()]
    results, errors = [None] * len(segments), []

    async def _stream_mp3(text):
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                buf.write(chunk["data"])
        return buf.getvalue()

    def _synthesize_one(idx, seg):
        out_raw, out_clip = (
            seg_dir / f"seg_{idx:04d}_raw.wav",
            seg_dir / f"seg_{idx:04d}.wav",
        )
        text = _sanitize_for_tts(seg["text"].strip())
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            mp3_bytes = loop.run_until_complete(_stream_mp3(text))
            loop.close()
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "pipe:0",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(out_raw),
                ],
                input=mp3_bytes,
                capture_output=True,
                check=True,
            )
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
                shutil.copy(str(out_raw), str(out_clip))
        except Exception as exc:
            errors.append(f"   ⚠️  Edge TTS failed: {exc}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=44100:cl=mono",
                    "-t",
                    "0.5",
                    str(out_clip),
                ],
                capture_output=True,
            )
        return {"path": str(out_clip), "start": seg["start"], "end": seg["end"]}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(_synthesize_one, i, seg): i
            for i, seg in enumerate(segments)
        }
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    for msg in errors:
        log(msg, logs)
    return results, logs


# ── Assembly ─────────────────────────────────────────────────────────────────


def assemble_dubbed_video(
    video_path: Path,
    timed_clips: list,
    duration: float,
    job_dir: Path,
    title: str,
    logs: list,
):
    import numpy as np, wave

    log("🎬 Assembling final video…", logs)
    SR, total_samples = 44100, int((duration + 2) * 44100)
    buffer = np.zeros(total_samples, dtype=np.float32)
    for clip in timed_clips:
        offset = int(clip["start"] * SR)
        try:
            with wave.open(clip["path"], "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                samples = np.frombuffer(
                    raw,
                    dtype=np.int16 if sw == 2 else np.int32 if sw == 4 else np.uint8,
                ).astype(np.float32)
                if sw == 2:
                    samples /= 32768.0
                elif sw == 4:
                    samples /= 2147483648.0
                else:
                    samples = (samples / 128.0) - 1.0
                if n_ch == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                end = min(offset + len(samples), total_samples)
                buffer[offset:end] += samples[: end - offset]
        except Exception as e:
            log(f"   ⚠️  Clip error {clip['path']}: {e}", logs)

    peak = np.max(np.abs(buffer))
    if peak > 0.001:
        buffer *= 0.891 / peak
    stereo_int16 = (
        (np.stack([buffer, buffer], axis=1) * 32767)
        .clip(-32768, 32767)
        .astype(np.int16)
    )
    mixed_audio = job_dir / "dubbed_audio.wav"
    with wave.open(str(mixed_audio), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo_int16.tobytes())

    output_path = (
        OUTPUT_DIR
        / f"{''.join(c for c in title if c.isalnum() or c in ' _-')[:50]}_PT-BR.mp4"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(mixed_audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )
    return str(output_path.resolve()), logs
