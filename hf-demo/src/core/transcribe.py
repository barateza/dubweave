from pathlib import Path
from src.config import WHISPER_MODEL
from src.utils.helpers import log

def transcribe_audio(
    audio_path: Path,
    logs: list,
    model_name: str = WHISPER_MODEL,
    language: str | None = None,
):
    """Transcribe audio with Whisper, return segments with timestamps."""
    import whisper
    log(f"🎙️ Transcribing with Whisper ({model_name})…", logs)
    model = whisper.load_model(model_name)

    if language:
        log(f"   Using language hint: {language}", logs)
    else:
        log("   Auto-detecting language…", logs)

    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        verbose=False,
    )
    detected_lang = result.get("language") or language or "unknown"
    log(f"   Detected language: {detected_lang}", logs)
    segments = result["segments"]
    log(f"✅ Transcribed {len(segments)} segments", logs)
    return segments, logs, detected_lang
