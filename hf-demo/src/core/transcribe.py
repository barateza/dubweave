import sys
import time
from pathlib import Path
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from src.config import WHISPER_MODEL
from src.utils.helpers import log

def format_tqdm(s: str) -> str:
    # Retained as a pass-through for backwards compatibility with the pipeline log handlers
    return s

def transcribe_audio(
    audio_path: Path,
    logs: list,
    model_name: str = WHISPER_MODEL,
    language: str | None = None,
):
    """Transcribe audio with faster-whisper, yielding real-time logs."""
    try:
        # Get total duration of the WAV audio file for accurate progress reporting
        total_duration = 1.0
        try:
            info = sf.info(str(audio_path))
            total_duration = info.duration
        except Exception:
            pass

        # Detect device and select best compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"

        yield ("log", f"   Loading faster-whisper model '{model_name}' on {device.upper()} ({compute_type})…")
        
        # Load the model
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        yield ("log", "   Transcribing audio segments…")

        # Run transcription (word_timestamps=True is highly optimized in faster-whisper)
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            beam_size=5
        )

        detected_lang = info.language
        yield ("log", f"   Detected language: {detected_lang} (probability: {info.language_probability:.2f})")

        converted_segments = []
        for seg in segments:
            # Yield real-time progress update
            progress_pct = min(100, int((seg.end / total_duration) * 100))
            yield ("log", f"   ⏳ Transcribing: {progress_pct}% ({seg.end:.1f}s / {total_duration:.1f}s)")

            # Convert Segment namedtuple/class to standard dict compatible with the downstream pipeline
            words_list = []
            if seg.words is not None:
                for w in seg.words:
                    words_list.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    })

            converted_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words_list
            })

        yield ("done", (converted_segments, detected_lang))

    except Exception as e:
        yield ("log", f"❌ faster-whisper error: {e}")
        raise e
