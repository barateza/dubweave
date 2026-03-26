from pathlib import Path
from src.config import WHISPER_MODEL
from src.utils.helpers import log

def transcribe_audio(audio_path: Path, logs: list, model_name: str = WHISPER_MODEL):
    """Transcribe audio with Whisper, return segments with timestamps."""
    import whisper
    log(f"🎙️ Transcribing with Whisper ({model_name})…", logs)
    model = whisper.load_model(model_name)
    
    log("   Detecting language...", logs)
    detection_result = model.transcribe(
        str(audio_path),
        language=None,
        word_timestamps=False,
        verbose=False,
        task="detect_language"
    )
    detected_lang = detection_result.get("detected_language", "en")
    log(f"   Detected language: {detected_lang}", logs)
    
    result = model.transcribe(
        str(audio_path),
        language=detected_lang,
        word_timestamps=True,
        verbose=False,
    )
    segments = result["segments"]
    log(f"✅ Transcribed {len(segments)} segments", logs)
    return segments, logs
