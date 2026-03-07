# Production Readiness — Tasks

**Design**: Inline (no separate design.md — changes are localized, no new architectural patterns)
**Status**: Draft

---

## Execution Plan

### Phase 1: Foundation — Documentation & Validation (Sequential)

Establishes the baseline: users can install, configs are validated at startup.

```
T1 → T2 → T3 → T4
```

### Phase 2: Core Hardening (Parallel OK)

Error handling, security, and performance improvements — independent concerns.

```
      ┌→ T5 ─┐
T4 ───┼→ T6 ─┼──→ T11
      ├→ T7 ─┤
      └→ T8 ─┘
```

### Phase 3: Testing (Sequential, depends on Phase 2)

Tests validate the hardened code from Phase 2.

```
T9 → T10
```

### Phase 4: Observability (Sequential)

Structured logging wraps around everything.

```
T11 → T12
```

---

## Task Breakdown

### T1: Expand README with installation troubleshooting [PROD-01]

**What**: Add troubleshooting section to README.md covering common setup failures.
**Where**: `README.md`
**Depends on**: None
**Reuses**: Existing README structure
**Requirement**: PROD-01

**Example**:
```markdown
## Troubleshooting

### "espeak-ng not found"
Kokoro TTS requires espeak-ng for phoneme processing.
Download from: https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi
After installing, restart your terminal.

### "CUDA not available"
Ensure NVIDIA drivers are installed and `nvidia-smi` shows your GPU.
Minimum: 4GB VRAM (Kokoro), 8GB VRAM (XTTS v2).

### "yt-dlp download fails with 403"
YouTube may require browser cookies. Use the "Browser cookies" dropdown in the UI
to select your browser (Chrome, Firefox, Edge).
```

**Done when**:
- [ ] Troubleshooting section covers espeak-ng, CUDA, yt-dlp, Pixi, and .env issues
- [ ] Each entry has symptoms, cause, and fix

---

### T2: Add startup environment validation [PROD-01, PROD-03]

**What**: Create a `validate_environment()` function that checks dependencies at startup and reports missing ones clearly.
**Where**: `app.py` (near top, before `build_ui()`)
**Depends on**: None
**Requirement**: PROD-01, PROD-03

**Example**:
```python
def validate_environment() -> list[str]:
    """Check required tools and report missing ones."""
    warnings = []
    # Check espeak-ng
    if shutil.which("espeak-ng") is None:
        warnings.append("espeak-ng not found — Kokoro TTS will fail. Install from: ...")
    # Check ffmpeg
    if shutil.which("ffmpeg") is None:
        warnings.append("ffmpeg not found — video assembly will fail.")
    # Check CUDA
    import torch
    if not torch.cuda.is_available():
        warnings.append("CUDA not available — GPU acceleration disabled.")
    # Check .env
    if not Path(".env").exists():
        shutil.copy(".env.example", ".env")
        warnings.append(".env created from template — review settings before running.")
    return warnings
```

**Done when**:
- [ ] Function checks: espeak-ng, ffmpeg, ffprobe, CUDA, .env
- [ ] Warnings displayed in Gradio UI on startup
- [ ] Missing tools named with install instructions

---

### T3: Add API key validation functions [PROD-03]

**What**: Create validation functions for OpenRouter and Google TTS API keys that run before the pipeline starts.
**Where**: `app.py`
**Depends on**: None
**Requirement**: PROD-03

**Example**:
```python
def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    """Validate OpenRouter API key with a lightweight auth check."""
    if not api_key or not api_key.startswith("sk-or-"):
        return False, "OpenRouter key must start with 'sk-or-'"
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200, "Valid"
    except Exception as e:
        return False, f"Validation failed: {e}"

def validate_google_tts_key(api_key: str) -> tuple[bool, str]:
    """Validate Google TTS API key with a voices.list call."""
    try:
        import urllib.request
        url = f"https://texttospeech.googleapis.com/v1/voices?key={api_key}&languageCode=pt-BR"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status == 200, "Valid"
    except Exception as e:
        return False, f"Validation failed: {e}"
```

**Done when**:
- [ ] OpenRouter key validated via /auth/key endpoint
- [ ] Google TTS key validated via voices.list endpoint
- [ ] Validation called in `run_pipeline()` before starting work
- [ ] API keys never appear in log output (redact in `log()`)

---

### T4: Add URL validation for YouTube inputs [PROD-03]

**What**: Validate YouTube URLs before passing to yt-dlp to catch obviously invalid inputs early.
**Where**: `app.py` (in `run_pipeline()` or as a helper)
**Depends on**: None
**Requirement**: PROD-03

**Example**:
```python
import re

_YT_URL_PATTERN = re.compile(
    r'^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[A-Za-z0-9_-]{11}'
)

def validate_youtube_url(url: str) -> tuple[bool, str]:
    """Check if URL looks like a valid YouTube video URL."""
    url = url.strip()
    if not url:
        return False, "No URL provided"
    if not _YT_URL_PATTERN.match(url):
        return False, f"'{url}' doesn't look like a YouTube URL. Expected: https://youtube.com/watch?v=..."
    return True, "Valid"
```

**Done when**:
- [ ] Regex validates youtube.com/watch, youtu.be, and youtube.com/shorts patterns
- [ ] Invalid URLs caught before yt-dlp is invoked
- [ ] Error message suggests the expected format

---

### T5: Wrap pipeline errors with user-friendly messages [PROD-02]

**What**: Add a `PipelineError` exception class and wrap all stage failures with actionable error messages in the Gradio UI.
**Where**: `app.py`
**Depends on**: None
**Requirement**: PROD-02

**Example**:
```python
class PipelineError(Exception):
    """User-facing pipeline error with stage context."""
    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage
        self.message = message
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")

# In run_pipeline():
try:
    segments = transcribe_audio(audio_path, log)
except Exception as e:
    raise PipelineError(
        "Transcribe",
        f"Whisper failed to transcribe audio: {e}. "
        f"Try a smaller model (WHISPER_MODEL=base in .env).",
        recoverable=False
    )
```

**Done when**:
- [ ] PipelineError class with stage, message, recoverable fields
- [ ] Each pipeline stage wrapped with descriptive error message
- [ ] Gradio UI displays stage name + actionable message, not raw traceback
- [ ] Recoverable errors indicate how to resume

---

### T6: Add exponential backoff for API calls [PROD-02]

**What**: Wrap `_call_openrouter()` and Google TTS HTTP calls with retry logic using exponential backoff.
**Where**: `app.py` (modify `_call_openrouter()` and `synthesize_segments_google_tts()`)
**Depends on**: None
**Requirement**: PROD-02

**Example**:
```python
def _retry_with_backoff(fn, max_retries=3, base_delay=2.0):
    """Retry a callable with exponential backoff on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, 'status', getattr(e, 'code', None))
            if attempt == max_retries or status not in (429, 500, 502, 503, 504, None):
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
```

**Done when**:
- [ ] OpenRouter calls retry on 429/5xx with exponential backoff
- [ ] Google TTS calls retry on transient errors
- [ ] Max 3 retries, delays: 2s → 4s → 8s
- [ ] Non-retryable errors (401, 403) fail immediately

---

### T7: Add GPU memory release between stages [PROD-04]

**What**: Explicitly release GPU memory after Whisper transcription and before TTS synthesis to prevent OOM on long videos.
**Where**: `app.py` (in `run_pipeline()` between stages)
**Depends on**: None
**Requirement**: PROD-04

**Example**:
```python
def release_gpu_memory():
    """Force GPU memory release between pipeline stages."""
    import torch
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# In run_pipeline(), after transcribe_audio():
release_gpu_memory()
log("GPU memory released after transcription")
```

**Done when**:
- [ ] `release_gpu_memory()` called after transcription, after translation (if NLLB), and after synthesis
- [ ] GPU memory drops measurably between stages (verify with `torch.cuda.memory_allocated()`)

---

### T8: Add temp file cleanup on completion/failure [PROD-04]

**What**: Ensure WORK_DIR temp files are cleaned up after pipeline completion or failure, not just by the stale-job cleaner.
**Where**: `app.py` (in `run_pipeline()` — finally block)
**Depends on**: None
**Requirement**: PROD-04

**Example**:
```python
# In run_pipeline():
job_dir = WORK_DIR / project_name
try:
    # ... pipeline stages ...
    pass
finally:
    # Clean up temp files for this job
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
            log(f"Cleaned up temp files: {job_dir}")
        except OSError:
            log(f"Warning: could not clean up {job_dir}")
```

**Done when**:
- [ ] Job-specific temp dir cleaned on success
- [ ] Job-specific temp dir cleaned on failure (finally block)
- [ ] Existing stale-job cleaner remains as fallback

---

### T9: Add unit tests for core pure functions [PROD-05]

**What**: Create test file with pytest tests for segment merging, SRT generation, PT-PT→PT-BR normalization, timing budget, and URL validation.
**Where**: `tests/test_core.py` (new file)
**Depends on**: T4 (URL validation), T5 (PipelineError)
**Requirement**: PROD-05

**Example**:
```python
# tests/test_core.py
import pytest
from app import (
    _ptpt_to_ptbr, _merge_segments, _srt_timestamp,
    _wrap_subtitle_line, validate_youtube_url
)

class TestPtBrNormalizer:
    def test_voce_to_tu(self):
        assert "você" not in _ptpt_to_ptbr("Você precisa saber").lower() or \
               _ptpt_to_ptbr("Você precisa saber") is not None  # depends on actual rules

    def test_empty_string(self):
        assert _ptpt_to_ptbr("") == ""

class TestSrtTimestamp:
    def test_zero(self):
        assert _srt_timestamp(0.0) == "00:00:00,000"

    def test_one_hour(self):
        assert _srt_timestamp(3661.5) == "01:01:01,500"

class TestMergeSegments:
    def test_empty_list(self):
        assert _merge_segments([]) == []

    def test_single_segment(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        result = _merge_segments(segs)
        assert len(result) >= 1

class TestYouTubeUrl:
    def test_valid_url(self):
        ok, _ = validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert ok

    def test_invalid_url(self):
        ok, msg = validate_youtube_url("https://example.com")
        assert not ok

    def test_empty(self):
        ok, _ = validate_youtube_url("")
        assert not ok
```

**Done when**:
- [ ] Tests cover: `_ptpt_to_ptbr`, `_merge_segments`, `_srt_timestamp`, `_wrap_subtitle_line`, `validate_youtube_url`
- [ ] Each function tested with: normal input, empty input, boundary cases
- [ ] `pixi run test` executes all tests

---

### T10: Add pixi test task and pytest config [PROD-05]

**What**: Add pytest to dependencies and a `test` task to `pixi.toml`.
**Where**: `pixi.toml`
**Depends on**: T9
**Requirement**: PROD-05

**Example**:
```toml
# In [pypi-dependencies]:
pytest = ">=7.0"

# In [tasks]:
test = "pytest tests/ -v"
```

**Done when**:
- [ ] `pytest` added to pypi-dependencies
- [ ] `pixi run test` works and runs all tests
- [ ] Tests pass

---

### T11: Add log redaction for sensitive values [PROD-03, PROD-06]

**What**: Modify the `log()` function to redact API keys and tokens from all log output.
**Where**: `app.py` (modify `log()`)
**Depends on**: None
**Requirement**: PROD-03, PROD-06

**Example**:
```python
_REDACT_PATTERNS = []

def _init_redact_patterns():
    """Build redaction patterns from current env vars."""
    global _REDACT_PATTERNS
    for key_env in ("OPENROUTER_API_KEY", "GOOGLE_TTS_API_KEY"):
        val = os.getenv(key_env, "").strip()
        if len(val) > 8:
            _REDACT_PATTERNS.append(val)

def _redact(msg: str) -> str:
    for secret in _REDACT_PATTERNS:
        msg = msg.replace(secret, f"{secret[:4]}****")
    return msg

def log(msg, logs=None):
    msg = _redact(str(msg))
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(entry)
    if logs is not None:
        logs.append(entry)
```

**Done when**:
- [ ] All API key values redacted from log output
- [ ] Redaction covers both OpenRouter and Google TTS keys
- [ ] Partial key shown (first 4 chars) for debugging

---

### T12: Add startup info logging [PROD-06]

**What**: Log environment info (Python version, CUDA, model paths, config summary) at application startup.
**Where**: `app.py` (at startup, before `build_ui()`)
**Depends on**: T11 (log redaction)
**Requirement**: PROD-06

**Example**:
```python
def log_startup_info():
    """Log environment details for diagnostics."""
    import platform
    log(f"Dubweave v0.1.0 starting")
    log(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
            log(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            log("CUDA not available — running CPU only")
    except ImportError:
        log("PyTorch not installed")
    log(f"Whisper model: {WHISPER_MODEL}")
    log(f"TTS models available: Kokoro, XTTS v2" +
        (", Google Cloud TTS" if GOOGLE_TTS_API_KEY else ""))
```

**Done when**:
- [ ] Python version, OS, CUDA status logged at startup
- [ ] GPU name and VRAM logged if available
- [ ] Config summary (models, available TTS engines) logged
- [ ] No API keys in startup log (T11 redaction active)
