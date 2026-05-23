import sys
import queue
import threading
import time
from pathlib import Path
from src.config import WHISPER_MODEL
from src.utils.helpers import log

def format_tqdm(s: str) -> str:
    if "%|" in s:
        parts = s.split('%|')
        pct = parts[0].strip()
        info_parts = parts[1].split('|')
        if len(info_parts) > 1:
            details = info_parts[1].strip()
            return f"   ⏳ Transcribing: {pct}% ({details})"
    return s

def transcribe_audio_thread(audio_path, model_name, language, result_queue, log_queue):
    class StderrRedirector:
        def __init__(self, original_stderr, log_q):
            self.original_stderr = original_stderr
            self.log_q = log_q
            
        def write(self, s):
            self.original_stderr.write(s)
            self.original_stderr.flush()
            parts = s.split('\r')
            for part in parts:
                clean = part.strip()
                if clean:
                    self.log_q.put(clean)
                    
        def flush(self):
            self.original_stderr.flush()

    original_stderr = sys.stderr
    sys.stderr = StderrRedirector(original_stderr, log_queue)
    
    try:
        import whisper
        import torch
        model = whisper.load_model(model_name)
        result = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=torch.cuda.is_available()
        )
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", e))
    finally:
        sys.stderr = original_stderr

def transcribe_audio(
    audio_path: Path,
    logs: list,
    model_name: str = WHISPER_MODEL,
    language: str | None = None,
):
    """Transcribe audio with Whisper in a separate thread, yielding real-time logs."""
    log_q = queue.Queue()
    res_q = queue.Queue()
    
    thread = threading.Thread(
        target=transcribe_audio_thread,
        args=(audio_path, model_name, language, res_q, log_q)
    )
    thread.start()
    
    while thread.is_alive() or not res_q.empty():
        while not log_q.empty():
            yield ("log", log_q.get())
        time.sleep(0.5)
        
    status, res = res_q.get()
    if status == "error":
        raise res
        
    segments = res["segments"]
    detected_lang = res.get("language") or language or "unknown"
    yield ("done", (segments, detected_lang))
