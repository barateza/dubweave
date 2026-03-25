"""
Dubweave — Video → Brazilian Portuguese Dubbing Pipeline
Uses: yt-dlp → Whisper → NLLB-200 (local PT-BR) → XTTS v2 (GPU) → FFmpeg
Fallback translation: OpenRouter API (configurable model)
Supports any URL handled by yt-dlp, or direct video file upload.
"""

__version__ = "0.1.0"

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Generator, cast

import warnings
import gradio as gr
from dotenv import load_dotenv

# Suppress torch.load pickle warnings from TTS/XTTS internals.
# These are known, safe, third-party model files — not a security concern here.
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS")
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*dropout option.*", category=UserWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
load_dotenv()  # Load environment variables from .env file


# ── Lazy imports (installed at runtime) ──────────────────────────────────────
def lazy_import():
    global yt_dlp, whisper, torch, TTS
    import yt_dlp
    import whisper
    import torch
    from TTS.api import TTS

    return True


# ── Config ────────────────────────────────────────────────────────────────────
WORK_DIR = Path(tempfile.gettempdir()) / "yt_dubber"
WORK_DIR.mkdir(exist_ok=True)
# Always resolve relative to this script file, not the shell working directory.
# pixi run may cd anywhere — a relative path is unreliable.
OUTPUT_DIR = Path(__file__).parent.resolve() / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PROJECTS_DIR = Path(__file__).parent.resolve() / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

# Load config from .env, fallback to defaults if not set
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"  # Not configurable
TARGET_LANG = "pt"  # XTTS v2 uses "pt" for all Portuguese; BR accent comes from voice ref

# Kokoro config (from .env)
KOKORO_LANG = os.getenv("KOKORO_LANG", "p")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "pf_dora")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

# Translation config (from .env)
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_SRC_LANG = os.getenv("NLLB_SRC_LANG", "eng_Latn")
NLLB_TGT_LANG = os.getenv("NLLB_TGT_LANG", "por_Latn")

# OpenRouter config (from .env)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")


def _int_env(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


OPENROUTER_CHUNK_SIZE = max(1, _int_env("OPENROUTER_CHUNK_SIZE", 120))
OPENROUTER_CONTEXT_SIZE = max(0, _int_env("OPENROUTER_CONTEXT_SIZE", 8))

# Google Cloud TTS config (from .env)
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
GOOGLE_TTS_LANGUAGE_CODE = os.getenv("GOOGLE_TTS_LANGUAGE_CODE", "pt-BR")
GOOGLE_TTS_VOICE_TYPE = os.getenv("GOOGLE_TTS_VOICE_TYPE", "Neural2")
GOOGLE_TTS_VOICE_NAME = os.getenv("GOOGLE_TTS_VOICE_NAME", "pt-BR-Neural2-A")

# Edge TTS config (from .env)
EDGE_TTS_VOICE_NAME = os.getenv("EDGE_TTS_VOICE_NAME", "pt-BR-FranciscaNeural")

# Known PT-BR Neural voices available via edge-tts (no API key required).
# FranciscaNeural — female, warm/conversational (best default for dubbed content)
# AntonioNeural   — male,   natural/neutral
# ThalitaNeural   — female, energetic/younger cadence
EDGE_TTS_PT_BR_VOICES: list[str] = [
    "pt-BR-FranciscaNeural",
    "pt-BR-AntonioNeural",
    "pt-BR-ThalitaNeural",
]

# Known PT-BR voices per Google Cloud TTS model family.
# Used to populate the UI dropdowns when a valid API key is present.
GOOGLE_TTS_VOICE_CATALOG: dict[str, list[str]] = {
    "Chirp3 HD": [
        "pt-BR-Chirp3-HD-Achernar",
        "pt-BR-Chirp3-HD-Achird",
        "pt-BR-Chirp3-HD-Algenib",
        "pt-BR-Chirp3-HD-Algieba",
        "pt-BR-Chirp3-HD-Alnilam",
        "pt-BR-Chirp3-HD-Aoede",
        "pt-BR-Chirp3-HD-Autonoe",
        "pt-BR-Chirp3-HD-Callirrhoe",
        "pt-BR-Chirp3-HD-Charon",
        "pt-BR-Chirp3-HD-Despina",
        "pt-BR-Chirp3-HD-Enceladus",
        "pt-BR-Chirp3-HD-Erinome",
        "pt-BR-Chirp3-HD-Fenrir",
        "pt-BR-Chirp3-HD-Gacrux",
        "pt-BR-Chirp3-HD-Iapetus",
        "pt-BR-Chirp3-HD-Kore",
        "pt-BR-Chirp3-HD-Laomedeia",
        "pt-BR-Chirp3-HD-Leda",
        "pt-BR-Chirp3-HD-Orus",
        "pt-BR-Chirp3-HD-Puck",
        "pt-BR-Chirp3-HD-Pulcherrima",
        "pt-BR-Chirp3-HD-Rasalgethi",
        "pt-BR-Chirp3-HD-Sadachbia",
        "pt-BR-Chirp3-HD-Sadaltager",
        "pt-BR-Chirp3-HD-Schedar",
        "pt-BR-Chirp3-HD-Sulafat",
        "pt-BR-Chirp3-HD-Umbriel",
        "pt-BR-Chirp3-HD-Vindemiatrix",
        "pt-BR-Chirp3-HD-Zephyr",
        "pt-BR-Chirp3-HD-Zubenelgenubi",
    ],
    "WaveNet": [
        "pt-BR-Wavenet-A",
        "pt-BR-Wavenet-B",
        "pt-BR-Wavenet-C",
        "pt-BR-Wavenet-D",
        "pt-BR-Wavenet-E",
    ],
    "Standard": [
        "pt-BR-Standard-A",
        "pt-BR-Standard-B",
        "pt-BR-Standard-C",
        "pt-BR-Standard-D",
        "pt-BR-Standard-E",
    ],
    "Studio": [
        "pt-BR-Studio-B",
        "pt-BR-Studio-C",
    ],
    "Neural2": [
        "pt-BR-Neural2-A",
        "pt-BR-Neural2-B",
        "pt-BR-Neural2-C",
    ],
    # No dedicated PT-BR Polyglot voices exist; the entry is kept so the
    # voice type is selectable and the name from .env is used as-is.
    "Polyglot (Preview)": [],
}

JOB_MAX_AGE_H = 2  # hours before a stale job folder is eligible for cleanup (not configurable)


# ── Log redaction (T11) ───────────────────────────────────────────────────────

_REDACT_PATTERNS: list[str] = []


def _init_redact_patterns() -> None:
    """Build redaction list from current API key env vars."""
    global _REDACT_PATTERNS
    _REDACT_PATTERNS = []
    for env_var in ("OPENROUTER_API_KEY", "GOOGLE_TTS_API_KEY"):
        val = os.getenv(env_var, "").strip()
        if len(val) > 8:
            _REDACT_PATTERNS.append(val)


def _redact(msg: str) -> str:
    """Replace any known secret value with a masked version."""
    for secret in _REDACT_PATTERNS:
        msg = msg.replace(secret, f"{secret[:4]}****")
    return msg


_init_redact_patterns()


# ── Step helpers ──────────────────────────────────────────────────────────────


def log(msg: str, logs: list) -> list:
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {_redact(str(msg))}"
    print(entry)
    logs.append(entry)
    return logs


# ── Pipeline error (T5) ───────────────────────────────────────────────────────


class PipelineError(Exception):
    """User-facing pipeline error with stage context."""

    def __init__(self, stage: str, message: str, recoverable: bool = False):
        self.stage = stage
        self.message = message
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}")


# ── Video source validation (T4) ──────────────────────────────────────────────

import re as _re_module

# Kept for backward compatibility with existing tests
_YT_URL_PATTERN = _re_module.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?[^\s]*v=[A-Za-z0-9_-]{11}"
    r"|youtu\.be/[A-Za-z0-9_-]{11}"
    r"|youtube\.com/shorts/[A-Za-z0-9_-]{11})"
)


def validate_youtube_url(url: str) -> tuple[bool, str]:
    """Return (True, 'Valid') or (False, reason) for a YouTube video URL.

    Kept for backward compatibility. The pipeline now uses validate_video_source.
    """
    url = url.strip()
    if not url:
        return False, "No URL provided."
    if not _YT_URL_PATTERN.match(url):
        return (
            False,
            f"'{url}' doesn't look like a YouTube URL. "
            "Expected: https://youtube.com/watch?v=… or https://youtu.be/…",
        )
    return True, "Valid"


def validate_video_source(url: str, upload_path: str | None) -> tuple[bool, str]:
    """Validate that at least one video source (URL or uploaded file) is provided.

    Returns (True, 'url'|'file') or (False, reason).
    Uploaded file takes priority over URL when both are provided.
    """
    has_file = bool(upload_path and upload_path.strip() and Path(upload_path.strip()).is_file())
    has_url = bool(url and url.strip())

    if has_file:
        return True, "file"
    if has_url:
        return True, "url"
    return False, "No video source provided. Paste a URL or upload a video file."


# ── Local file ingestion ──────────────────────────────────────────────────────

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".ts", ".m4v"}


def ingest_local_file(
    upload_path: str,
    job_dir: Path,
    logs: list,
) -> tuple[Path, Path, str, float, list]:
    """Copy/re-encode an uploaded video file into the pipeline's expected format.

    Returns (video_path, audio_path, title, duration, logs) — same shape as
    download_video so the rest of the pipeline is source-agnostic.
    """
    src = Path(upload_path.strip())
    logs = log(f"📂 Ingesting uploaded file: {src.name}", logs)

    if not src.exists():
        raise PipelineError("Ingest", f"Uploaded file not found: {src}", recoverable=False)
    if src.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise PipelineError(
            "Ingest",
            f"Unsupported file type '{src.suffix}'. "
            f"Expected video file ({', '.join(sorted(_VIDEO_EXTENSIONS))}).",
            recoverable=False,
        )

    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio_orig.wav"

    # Probe duration and title
    title = src.stem
    duration = 0.0
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(src),
            ],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception as e:
        logs = log(f"⚠️  Could not probe duration: {e}", logs)

    # Re-encode to mp4 if not already mp4, or copy if it is
    if src.suffix.lower() == ".mp4":
        logs = log("   File is mp4 — copying directly", logs)
        shutil.copy2(str(src), str(video_path))
    else:
        logs = log(f"   Re-encoding {src.suffix} → mp4 for pipeline compatibility…", logs)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(video_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise PipelineError(
                "Ingest",
                f"ffmpeg re-encode failed: {result.stderr[-300:]}",
                recoverable=False,
            )

    # Extract audio
    logs = log("   Extracting audio…", logs)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "44100", "-ac", "2", "-f", "wav",
            str(audio_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise PipelineError(
            "Ingest",
            f"Audio extraction failed: {result.stderr[-300:]}",
            recoverable=False,
        )

    if not audio_path.exists() or audio_path.stat().st_size < 1024:
        raise PipelineError(
            "Ingest",
            "Audio extraction produced an empty file. "
            "The uploaded video may have no audio track.",
            recoverable=False,
        )

    logs = log(f'✅ Ingested: "{title}" ({duration:.0f}s)', logs)
    return video_path, audio_path, title, duration, logs


# ── API key validation (T3) ───────────────────────────────────────────────────


def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    """Validate an OpenRouter API key via a lightweight /auth/key call."""
    import urllib.request
    import urllib.error

    api_key = api_key.strip()
    if not api_key:
        return False, "No OpenRouter API key provided."
    if not api_key.startswith("sk-or-"):
        return False, "OpenRouter key must start with 'sk-or-'."
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid"
            return False, f"OpenRouter returned HTTP {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"OpenRouter key invalid: HTTP {e.code}."
    except Exception as e:
        return False, f"OpenRouter key validation failed: {e}"


def validate_google_tts_key(api_key: str) -> tuple[bool, str]:
    """Validate a Google Cloud TTS API key via a voices.list call."""
    import urllib.request
    import urllib.error

    api_key = api_key.strip()
    if not api_key:
        return False, "No Google TTS API key provided."
    try:
        url = f"https://texttospeech.googleapis.com/v1/voices?key={api_key}&languageCode=pt-BR"
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid"
            return False, f"Google TTS returned HTTP {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"Google TTS key invalid: HTTP {e.code}."
    except Exception as e:
        return False, f"Google TTS key validation failed: {e}"


# ── Environment validation (T2) ───────────────────────────────────────────────


def validate_environment() -> list[str]:
    """Check required tools at startup; return list of warning strings."""
    warnings_list: list[str] = []

    if shutil.which("espeak-ng") is None:
        warnings_list.append(
            "⚠️  espeak-ng not found — Kokoro TTS will fail. "
            "Install from: https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi "
            "then restart your terminal."
        )

    if shutil.which("ffmpeg") is None:
        warnings_list.append(
            "⚠️  ffmpeg not found — video assembly will fail. "
            "Run setup.bat to install all dependencies."
        )

    if shutil.which("ffprobe") is None:
        warnings_list.append(
            "⚠️  ffprobe not found — timing detection will fail. "
            "Ensure ffmpeg is installed (ffprobe is bundled with it)."
        )

    try:
        import torch

        if not torch.cuda.is_available():
            warnings_list.append(
                "⚠️  CUDA not available — GPU acceleration disabled. "
                "Ensure NVIDIA drivers are installed and `nvidia-smi` shows your GPU."
            )
    except ImportError:
        warnings_list.append("⚠️  PyTorch not installed — GPU acceleration unavailable.")

    env_path = Path(__file__).parent / ".env"
    env_example_path = Path(__file__).parent / ".env.example"
    if not env_path.exists():
        if env_example_path.exists():
            shutil.copy(str(env_example_path), str(env_path))
            warnings_list.append(
                "ℹ️  .env created from .env.example — review settings before running."
            )
        else:
            warnings_list.append(
                "⚠️  .env not found — using built-in defaults. "
                "Create a .env file to configure API keys and model choices."
            )

    return warnings_list


# ── Retry with exponential backoff (T6) ──────────────────────────────────────

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _retry_with_backoff(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry *fn* (a zero-argument callable) with exponential backoff.

    Retries on transient HTTP errors (429/5xx) and network exceptions.
    Raises immediately on non-retryable errors (401, 403, 404).
    """
    import urllib.error

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if attempt == max_retries or e.code not in _RETRYABLE_STATUS_CODES:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)
        except Exception:
            if attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)


# ── GPU memory release (T7) ───────────────────────────────────────────────────


def release_gpu_memory() -> None:
    """Force GPU memory release between pipeline stages."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# ── Startup info logging (T12) ────────────────────────────────────────────────


def log_startup_info() -> None:
    """Print environment diagnostics to stdout at application startup."""
    import platform

    print(f"[startup] Dubweave v{__version__} starting")
    print(f"[startup] Python {platform.python_version()} on {platform.system()} {platform.release()}")
    try:
        import torch

        if torch.cuda.is_available():
            import torch.version as _torch_version
            print(f"[startup] CUDA {_torch_version.cuda} — {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[startup] VRAM: {vram_gb:.1f} GB")
        else:
            print("[startup] CUDA not available — running CPU only")
    except ImportError:
        print("[startup] PyTorch not installed")

    print(f"[startup] Whisper model: {WHISPER_MODEL}")
    tts_engines = "Kokoro, XTTS v2"
    if GOOGLE_TTS_API_KEY:
        tts_engines += ", Google Cloud TTS"
    print(f"[startup] TTS engines available: {tts_engines}")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        print(f"[startup] OpenRouter: configured ({_redact(openrouter_key)})")
    else:
        print("[startup] OpenRouter: not configured (local NLLB-200 only)")

    env_warnings = validate_environment()
    for w in env_warnings:
        print(f"[startup] {w}")


def download_video(
    url: str,
    job_dir: Path,
    logs: list,
    browser: str = "none",
    cookies_file: str | None = None,
):
    """
    Download video + audio with a self-healing format cascade.

    The format cascade tries progressively more compatible formats,
    ending with broadly available fallbacks. This works for any site
    supported by yt-dlp (YouTube, Vimeo, Twitter/X, TikTok, etc.).

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
    _deno = _shutil.which("deno")

    if _aria2c:
        logs = log(f"   aria2c found — using for accelerated download", logs)
    if _deno:
        logs = log(f"   deno found — YouTube JS challenge solving enabled", logs)
    else:
        logs = log(
            "   ⚠️  deno not found — some YouTube formats may be missing (run setup.bat)",
            logs,
        )

    BASE_OPTS: dict[str, Any] = {
        "outtmpl": str(job_dir / "%(id)s_%(format_id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
        "ignoreerrors": False,
        # Use cached EJS solver script for YouTube n-challenge (downloaded by setup.bat)
        # Cookie auth: cookies.txt takes priority over browser extraction.
        # cookies.txt = Netscape format exported from browser extension.
        # browser     = yt-dlp reads directly from browser profile (may fail if Chrome is open).
        # neither     = anonymous download (may trigger PO token / JS challenge errors).
        **(
            {"cookiefile": cookies_file}
            if cookies_file
            else (
                {"cookiesfrombrowser": (browser, None, None, None)}
                if browser != "none"
                else {}
            )
        ),
        **({"remote_components": ["ejs:github"]} if _deno else {}),
        # ── aria2c: multi-connection download via your local RPC server ───────
        # Falls back to yt-dlp's built-in downloader if aria2c is not in PATH.
        **(
            {
                "external_downloader": "aria2c",
                "external_downloader_args": {
                    "aria2c": [
                        "--rpc-save-upload-metadata=false",
                        "--file-allocation=none",  # faster start, skip prealloc
                        "--optimize-concurrent-downloads=true",
                        "--max-connection-per-server=4",  # YouTube allows ~4 per URL
                        "--min-split-size=5M",
                        "--split=4",
                        "--max-tries=5",
                        "--retry-wait=3",
                    ]
                },
            }
            if _aria2c
            else {}
        ),
    }

    # ── Step 1: probe title/duration without downloading ─────────────────────
    title, duration = "video", 0
    try:
        with yt.YoutubeDL(cast(Any, {**BASE_OPTS, "skip_download": True})) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "video")
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

    muxed_fallback = False  # True when video already contains audio
    raw_video_file = None

    for fmt in VIDEO_FORMATS:
        try:
            opts = cast(
                Any,
                {
                    **BASE_OPTS,
                    "format": fmt,
                    "outtmpl": str(job_dir / f"video_raw.%(ext)s"),
                },
            )
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
        raise RuntimeError(
            "All video format fallbacks exhausted — cannot download video."
        )

    shutil.move(str(raw_video_file), str(video_path))

    # ── Step 3: download audio track ─────────────────────────────────────────
    # If we got a muxed file, extract audio from it directly — no second
    # download needed and avoids re-triggering any challenge issues.
    if muxed_fallback:
        logs = log("   Muxed fallback — extracting audio from video file…", logs)
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-f",
                "wav",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
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
                opts = cast(
                    Any,
                    {
                        **BASE_OPTS,
                        "format": fmt,
                        "outtmpl": str(job_dir / "audio_raw.%(ext)s"),
                        "postprocessors": [
                            {
                                "key": "FFmpegExtractAudio",
                                "preferredcodec": "wav",
                                "preferredquality": "0",
                            }
                        ],
                    },
                )
                with yt.YoutubeDL(opts) as ydl:
                    ydl.extract_info(url, download=True)
                candidates = list(job_dir.glob("audio_raw.*"))
                if candidates:
                    raw_audio_file = candidates[0]
                    logs = log(
                        f"   Audio format '{fmt}' ✓  ({raw_audio_file.name})", logs
                    )
                    break
            except Exception as e:
                logs = log(f"   Audio format '{fmt}' failed: {e!s:.80}", logs)

        if raw_audio_file is None:
            # absolute last resort: extract from whatever video we have
            logs = log(
                "   All audio formats failed — extracting from video file…", logs
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-f",
                    "wav",
                    str(audio_path),
                ],
                capture_output=True,
                check=True,
            )
        else:
            shutil.move(str(raw_audio_file), str(audio_path))

    logs = log(f'✅ Downloaded: "{title}" ({duration}s)', logs)
    return video_path, audio_path, title, duration, logs


def transcribe_audio(audio_path: Path, logs: list, model_name: str = WHISPER_MODEL):
    """Transcribe audio with Whisper, return segments with timestamps."""
    import whisper

    logs = log(f"🎙️ Transcribing with Whisper ({model_name})…", logs)

    model = whisper.load_model(model_name)
    
    # HIGH SEVERITY FIX 2: Add language detection for non-English source videos
    # First, detect language to handle non-English sources
    logs = log("   Detecting language...", logs)
    detection_result = model.transcribe(
        str(audio_path),
        language=None,  # Let Whisper auto-detect
        word_timestamps=False,
        verbose=False,
        task="detect_language"
    )
    
    detected_lang = detection_result.get("detected_language", "en")
    logs = log(f"   Detected language: {detected_lang}", logs)
    
    # Use detected language for actual transcription
    result = model.transcribe(
        str(audio_path),
        language=detected_lang,
        word_timestamps=True,
        verbose=False,
    )

    segments = result["segments"]
    logs = log(f"✅ Transcribed {len(segments)} segments", logs)
    return segments, logs


# ── PT-PT → PT-BR normalizer (post-processing, runs on ALL translators) ─────

# These are the most common European Portuguese markers that NLLB and other
# models default to. Replacing them with Brazilian equivalents covers ~90% of
# the perceptible difference in everyday spoken content.
_PTPT_TO_PTBR = [
    # Pronouns / address
    (r"\btu\b", "você"),
    (r"\bte\b", "te"),  # keep — both use "te" but context helps
    (r"\bteu\b", "seu"),
    (r"\btua\b", "sua"),
    (r"\bteus\b", "seus"),
    (r"\btuas\b", "suas"),
    (r"\bvós\b", "vocês"),
    # Verb forms — 2nd person → 3rd person (você paradigm)
    (r"\bestás\b", "está"),
    (r"\bgostavas\b", "gostava"),
    (r"\bgostas\b", "gosta"),
    (r"\bfazes\b", "faz"),
    (r"\bpodes\b", "pode"),
    (r"\bqueres\b", "quer"),
    (r"\bsabes\b", "sabe"),
    (r"\btens\b", "tem"),
    (r"\bvens\b", "vem"),
    (r"\bdizes\b", "diz"),
    (r"\bvês\b", "vê"),
    (r"\bvais\b", "vai"),
    (r"\bficas\b", "fica"),
    (r"\bperceber\b", "entender"),
    # Gerund — PT-PT uses infinitive constructions, PT-BR uses gerund
    # "a verificar" → "verificando", "a fazer" → "fazendo" etc.
    (
        r"\ba (verificar|fazer|dizer|ir|ter|ser|estar|ver|vir|dar|saber|poder|querer|ficar|falar|pensar|olhar|ouvir|sentir|aprender|entender|perceber|mostrar|colocar|pedir|deixar|ajudar|começar|continuar|precisar|tentar|achar|trazer|levar|passar|parecer|acontecer|escolher|cuidar|gostar|amar|crescer|brincar|rir|chorar|correr|andar|esperar|trabalhar|estudar|viver|morrer|ganhar|perder|mudar|criar|usar|encontrar|conhecer|acreditar|lembrar|esquecer|chamar|jogar)\b",
        lambda m: (
            m.group(1)[:-2] + "ando"
            if m.group(1).endswith("ar")
            else m.group(1)[:-2] + "endo"
            if m.group(1).endswith("er")
            else m.group(1)[:-2] + "indo"
            if m.group(1).endswith("ir")
            else m.group(1) + "ndo"
        ),
    ),
    # Specific common phrases
    (r"\bmiúdos\b", "crianças"),
    (r"\bfixe\b", "legal"),
    (r"\bgiro\b", "bonito"),
    (r"\bchato\b", "chato"),  # same but keep
    (r"\bpropriamente\b", "corretamente"),
    (r"\bsempre que\b", "sempre que"),
    (r"\bcertamente\b", "certamente"),
    (r"\bapenas\b", "só"),
    (r"\bsomente\b", "só"),
    (r"\bimensamente\b", "muito"),
    (r"\bimenso\b", "enorme"),
    (r"\bautocarro\b", "ônibus"),
    (r"\bcomboio\b", "trem"),
    (r"\btelemovel\b", "celular"),
    (r"\btelemovel\b", "celular"),
    (r"\btelemóvel\b", "celular"),
    (r"\bpasseio\b", "calçada"),
    (r"\bpetróleos\b", "petróleo"),
    (r"\bcasas de banho\b", "banheiros"),
    (r"\bcasa de banho\b", "banheiro"),
    (r"\bsaneamento\b", "saneamento"),
    (r"\bfutebol\b", "futebol"),  # same
]


def _ptpt_to_ptbr(text: str) -> str:
    """Apply PT-PT → PT-BR lexical substitutions."""
    import re
    from pathlib import Path

    # Prefer canonical rules file when present
    rules_path = Path(__file__).parent / "normalizer_rules.json"
    if rules_path.exists():
        try:
            cfg = json.loads(rules_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        rules = cfg.get("rules", [])

        for rule in rules:
            if rule.get("type") == "gerund":
                verbs = rule.get("verbs", [])
                if not verbs:
                    continue
                verb_pattern = "|".join(re.escape(v) for v in verbs)
                pattern = rf"\ba ({verb_pattern})\b"

                def _replace_gerund(m: re.Match[str]) -> str:
                    v = m.group(1)
                    if v.endswith("ar"):
                        return v[:-2] + "ando"
                    if v.endswith("er"):
                        return v[:-2] + "endo"
                    if v.endswith("ir"):
                        return v[:-2] + "indo"
                    return v + "ndo"

                text = str(re.sub(pattern, _replace_gerund, text, flags=re.IGNORECASE))
            else:
                pattern = rule.get("pattern")
                replacement = rule.get("replacement")
                if not pattern:
                    continue

                def _replace_preserve_case(m: re.Match[str], repl: str = replacement) -> str:
                    if m.group(0) and m.group(0)[0].isupper():
                        return repl[0].upper() + repl[1:]
                    return repl

                try:
                    text = str(re.sub(pattern, _replace_preserve_case, text, flags=re.IGNORECASE))
                except re.error:
                    # If the pattern from JSON is invalid, skip it gracefully
                    continue
        return text

    # Fallback: use hardcoded tuple list
    for pattern, replacement in _PTPT_TO_PTBR:
        if callable(replacement):
            text = str(
                re.sub(
                    pattern,
                    cast(Callable[[re.Match[str]], str], replacement),
                    text,
                    flags=re.IGNORECASE,
                )
            )
        else:
            # Preserve capitalisation of the first letter
            def _replace(m: re.Match[str], repl: str = cast(str, replacement)) -> str:
                if m.group(0)[0].isupper():
                    return repl[0].upper() + repl[1:]
                return repl

            text = str(re.sub(pattern, _replace, text, flags=re.IGNORECASE))
    return text


# ── NLLB-200 translation (primary) ───────────────────────────────────────────

_nllb_pipeline = None  # module-level cache — load once, reuse across jobs


def _get_nllb_pipeline(logs: list):
    """Load NLLB-200 translation pipeline, cached after first load."""
    global _nllb_pipeline
    if _nllb_pipeline is not None:
        return _nllb_pipeline, logs

    from transformers import pipeline as hf_pipeline
    import torch

    logs = log(f"🧠 Loading NLLB-200 ({NLLB_MODEL})…", logs)
    device = 0 if torch.cuda.is_available() else -1
    _nllb_pipeline = hf_pipeline(
        "translation",
        model=NLLB_MODEL,
        src_lang=NLLB_SRC_LANG,
        tgt_lang=NLLB_TGT_LANG,
        device=device,
        max_length=512,
    )  # tgt_lang already sets forced_bos_token_id internally; do not pass it again
    logs = log(f"   NLLB-200 loaded on {'GPU' if device == 0 else 'CPU'}", logs)
    return _nllb_pipeline, logs


def _translate_nllb(texts: list[str], logs: list) -> tuple[list[str], list]:
    """Batch-translate EN→PT-BR via NLLB-200, then normalise PT-PT markers."""
    pipe, logs = _get_nllb_pipeline(logs)

    results = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = cast(list, pipe(batch, batch_size=min(8, len(batch))))
        results.extend(o["translation_text"] for o in outputs)

    # Post-process: convert remaining PT-PT markers to PT-BR
    results = [_ptpt_to_ptbr(t) for t in results]
    return results, logs


# ── OpenRouter translation (fallback) ─────────────────────────────────────────

def _load_system_prompt() -> str:
    """Load translation system prompt from file, fall back to hardcoded default."""
    prompt_path = Path(__file__).parent / "translation_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    # Hardcoded fallback — only used if translation_prompt.md is missing
    return (
        "You are a professional translator specialising in Brazilian Portuguese (PT-BR). "
        "CRITICAL RULES:\n"
        "1. Output ONLY in Brazilian Portuguese (PT-BR). NEVER use European Portuguese (PT-PT).\n"
        "2. Use 'voce' for second person singular. NEVER use 'tu', 'teu', 'tua', 'vos'.\n"
        "3. Use gerund forms: 'estao fazendo', 'estou vendo'. NEVER use 'estao a fazer', 'estou a ver'.\n"
        "4. Use Brazilian vocabulary: 'onibus' not 'autocarro', 'celular' not 'telemovel', "
        "'trem' not 'comboio', 'banheiro' not 'casa de banho', 'legal' not 'fixe', "
        "'criancas' not 'miudos', 'entender' not 'perceber' (when meaning to understand).\n"
        "5. Use 3rd person verb conjugations with 'voce': 'voce esta' not 'voce estas'.\n"
        "6. Keep informal, conversational register as in the original.\n"
        "7. Preserve all punctuation and segment numbering exactly."
    )

SYSTEM_PROMPT = _load_system_prompt()


def _call_openrouter(
    texts: list[str], api_key: str, context: list[str] | None = None
) -> list[str]:
    """Single API call: translate a chunk of texts, return list of strings.

    context: previously translated utterances prepended as read-only context
    so the model can resolve pronouns and maintain register across chunk boundaries.
    """
    import urllib.request
    import re as _re

    ctx_block = ""
    if context:
        ctx_lines = "\n".join(f"  {t}" for t in context)
        ctx_block = (
            f"[CONTEXT — already translated, do NOT include in output]\n"
            f"{ctx_lines}\n"
            f"[END CONTEXT]\n\n"
        )

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = (
        f"{ctx_block}"
        f"Translate these {len(texts)} numbered utterances to Brazilian Portuguese (PT-BR).\n"
        "Output ONLY the numbered translations — same count, same order, nothing else.\n\n"
        f"{numbered}"
    )

    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
        }
    ).encode()

    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dubweave",
            "X-Title": "Dubweave",
        },
        method="POST",
    )

    def _do_request():
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())

    data = _retry_with_backoff(_do_request)
    if data is None:
        raise RuntimeError("_retry_with_backoff returned None — request failed")

    raw = data["choices"][0]["message"]["content"].strip()

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    result = []
    for line in lines:
        clean = _re.sub(r"^\d+[\.)\s]+", "", line).strip()
        if clean:
            result.append(clean)
    return result


def _translate_openrouter(
    texts: list[str], api_key: str, logs: list
) -> tuple[list[str], list]:
    """
    Translate via OpenRouter in configurable chunks with a configurable
    context window prepended to each chunk.

    Context window rationale: without preceding context, the model translates
    each chunk as if it starts a new document. Pronouns, topics, and register
    established earlier are invisible. Prepending the last few utterances of the
    previous chunk as [CONTEXT] (not to be translated) gives the model enough
    coherence to resolve anaphora and maintain register across boundaries.
    """
    logs = log(f"🌐 Translating via OpenRouter ({OPENROUTER_MODEL})…", logs)

    chunk_size = OPENROUTER_CHUNK_SIZE
    context_size = OPENROUTER_CONTEXT_SIZE
    all_translated = []
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size

    for chunk_i in range(total_chunks):
        chunk = texts[chunk_i * chunk_size : (chunk_i + 1) * chunk_size]
        ctx = all_translated[-context_size:] if all_translated else []

        logs = log(
            f"   Chunk {chunk_i+1}/{total_chunks} (size={len(chunk)}, context={len(ctx)})…",
            logs,
        )

        result = _call_openrouter(chunk, api_key, context=ctx)

        # HIGH SEVERITY FIX 3: Add better validation for OpenRouter translation responses
        # Validate response quality and count
        if len(result) != len(chunk):
            logs = log(
                f"   ⚠️  Got {len(result)} back for {len(chunk)} sent — padding with originals",
                logs,
            )
            while len(result) < len(chunk):
                result.append(chunk[len(result)])
            result = result[: len(chunk)]
        
        # Validate that each result is non-empty and contains actual content
        for i, translation in enumerate(result):
            if not translation.strip():
                logs = log(f"   ⚠️  Empty translation for utterance {i+1} — using original", logs)
                result[i] = chunk[i]
            elif len(translation.strip()) < 2:
                logs = log(f"   ⚠️  Suspiciously short translation for utterance {i+1} — using original", logs)
                result[i] = chunk[i]

        all_translated.extend(result)

    # Final PT-PT safety pass
    all_translated = [_ptpt_to_ptbr(t) for t in all_translated]
    logs = log(f"✅ OpenRouter translated {len(all_translated)} utterances", logs)
    return all_translated, logs


# ── Segment Merge Configs (calibrated via Autoresearch Loop 1) ────────────────
# Optimized to group Whisper segments into utterances that translate well
# and fit within synthesis slots for specific TTS engines.

MERGE_CONFIGS = {
    "kokoro": {
        "min_words": 8,
        "max_words": 100,
        "gap_sec": 3.0,
        "max_duration": None,
    },
    "francisca": {
        "min_words": 10,
        "max_words": 100,
        "gap_sec": 2.0,
        "max_duration": None,
        "chars_per_sec": 24.0,
    },
    "thalita": {
        "min_words": 10,
        "max_words": 100,
        "gap_sec": 2.0,
        "max_duration": None,
        "chars_per_sec": 26.0,
    },
    "antonio": {
        "min_words": 10,
        "max_words": 100,
        "gap_sec": 2.0,
        "max_duration": None,
        "chars_per_sec": 24.0,
    },
    "default": {
        "min_words": 8,
        "max_words": 50,
        "gap_sec": 2.0,
        "max_duration": None,
    },
}


def _get_merge_config(engine: str) -> dict:
    """Return the optimized merge parameters for a given TTS engine."""
    if "Francisca" in engine:
        return MERGE_CONFIGS["francisca"]
    if "Antonio" in engine:
        return MERGE_CONFIGS["antonio"]
    if "Thalita" in engine:
        return MERGE_CONFIGS["thalita"]
    if "Kokoro" in engine:
        return MERGE_CONFIGS["kokoro"]
    return MERGE_CONFIGS["default"]


# ── Public translate_segments (NLLB primary, OpenRouter fallback) ─────────────

# ── Segment merging ──────────────────────────────────────────────────────────


def _merge_segments(
    segments: list,
    *,
    min_words: int = 4,
    max_words: int = 40,
    gap_sec: float | None = None,
    max_duration: float | None = None,
) -> list:
    """Merge raw Whisper segments into translation utterances.

    Flush conditions (in priority order):
      1. gap_sec     — long silence before next segment → force flush NOW
                       (fires *before* appending the new segment, so the
                        silent boundary lands between utterances, not inside)
      2. max_words   — hard word-count cap → flush after appending
      3. max_duration — accumulated duration cap → flush after appending
      4. min_words + sentence punctuation → soft flush after appending

    gap_sec interaction with min_words
    ------------------------------------
    A gap flush is unconditional: it fires even if the buffer has fewer than
    min_words words. This is intentional — a long silence IS a sentence
    boundary regardless of how many words preceded it (e.g. a short "Yes."
    followed by 1.5 s of silence should flush). If you want a minimum word
    guard on gap flushes, add `gap_min_words` in a future iteration.
    """
    if not segments:
        return []

    _SENTENCE_ENDINGS = frozenset('.?!…—"\'')

    merged: list = []
    buf: list = []          # segments being accumulated into current utterance
    buf_children: list[int] = []  # original segment indices for re-expansion

    def _flush() -> None:
        if not buf:
            return
        merged.append({
            "start": buf[0]["start"],
            "end":   buf[-1]["end"],
            "text":  " ".join(s["text"].strip() for s in buf),
            "children": buf_children.copy(),
        })
        buf.clear()
        buf_children.clear()

    for seg_idx, seg in enumerate(segments):
        # ── Priority 1: gap flush ──────────────────────────────────────────
        # Check BEFORE appending so the gap boundary falls between utterances.
        if gap_sec is not None and buf:
            silence = seg["start"] - buf[-1]["end"]
            if silence >= gap_sec:
                _flush()

        buf.append(seg)
        buf_children.append(seg_idx)

        # Running stats on the current buffer
        combined_text  = " ".join(s["text"].strip() for s in buf)
        word_count     = len(combined_text.split())
        duration       = buf[-1]["end"] - buf[0]["start"]
        ends_sentence  = (
            bool(combined_text.rstrip())
            and combined_text.rstrip()[-1] in _SENTENCE_ENDINGS
        )

        # ── Priority 2: hard word-count cap ───────────────────────────────
        if word_count >= max_words:
            _flush()
            continue

        # ── Priority 3: duration cap ──────────────────────────────────────
        if max_duration is not None and duration >= max_duration:
            _flush()
            continue

        # ── Priority 4: soft flush on sentence boundary ───────────────────
        if word_count >= min_words and ends_sentence:
            _flush()

    _flush()   # drain any remaining buffer
    return merged


def _group_for_synthesis(translated: list) -> list:
    """
    Group translated segments back into utterances for sentence-level synthesis.

    After expand_merged, segments are broken at acoustic boundaries mid-sentence.
    This re-groups them into complete utterances (terminal punctuation + ≥4 words)
    so each utterance is synthesized as a single natural-sounding TTS call.
    """
    return _merge_segments(translated)


def _expand_merged(merged_translated: list, original_segments: list) -> list:
    """
    Re-expand translated utterances back to original segment granularity.

    Each merged utterance may cover multiple original segments. We distribute
    the translated text across children proportionally by original duration.
    This is an approximation — the invariant (acoustic/semantic/synthesis
    boundary mismatch) means perfect alignment is impossible — but it is
    better than translating fragments blind.
    """
    result = []
    for utt in merged_translated:
        children = utt["children"]
        if len(children) == 1:
            result.append(
                {
                    "start": original_segments[children[0]]["start"],
                    "end": original_segments[children[0]]["end"],
                    "text": utt["text"],
                }
            )
            continue

        # Distribute translated text by word proportion across children
        translated_words = utt["text"].split()
        total_dur = sum(
            original_segments[c]["end"] - original_segments[c]["start"]
            for c in children
        )
        if total_dur <= 0:
            # Degenerate: give all text to first child
            for j, c in enumerate(children):
                result.append(
                    {
                        "start": original_segments[c]["start"],
                        "end": original_segments[c]["end"],
                        "text": utt["text"] if j == 0 else "…",
                    }
                )
            continue

        word_cursor = 0
        for j, c in enumerate(children):
            seg_dur = original_segments[c]["end"] - original_segments[c]["start"]
            proportion = seg_dur / total_dur
            if j == len(children) - 1:
                # Last child gets remainder to avoid off-by-one word loss
                word_slice = translated_words[word_cursor:]
            else:
                n_words = max(1, round(len(translated_words) * proportion))
                word_slice = translated_words[word_cursor : word_cursor + n_words]
                word_cursor += n_words

            result.append(
                {
                    "start": original_segments[c]["start"],
                    "end": original_segments[c]["end"],
                    "text": " ".join(word_slice) if word_slice else "…",
                }
            )

    return result


def translate_segments(
    segments: list,
    logs: list,
    openrouter_key: str = "",
    merge_config: dict | None = None,
) -> tuple[list, list]:
    """
    Translate segments EN→PT-BR.

    Pipeline:
      1. Merge short/incomplete Whisper segments into utterances.
         Rationale: Whisper segments are acoustic units; translating fragments
         blind loses context and produces broken output at boundaries.
      2. Translate merged utterances (OpenRouter primary, NLLB fallback).
         Each chunk is sent with 2 preceding utterances as read-only context.
      3. Re-expand translated utterances back to original segment granularity
         by duration proportion.
      4. Apply PT-PT → PT-BR normalizer as final safety pass.
    """
    # ── Step 1: merge ─────────────────────────────────────────────────────────
    m_cfg = merge_config or MERGE_CONFIGS["default"]
    merged = _merge_segments(segments, **m_cfg)
    logs = log(
        f"   Merged {len(segments)} Whisper segments → {len(merged)} utterances for translation",
        logs,
    )
    merged_texts = [u["text"] for u in merged]

    # ── Step 2: translate ─────────────────────────────────────────────────────
    translated_texts = None
    primary_error = None

    if openrouter_key.strip():
        try:
            translated_texts, logs = _translate_openrouter(
                merged_texts, openrouter_key.strip(), logs
            )
        except Exception as e:
            primary_error = str(e)
            logs = log(f"   ⚠️  OpenRouter failed: {primary_error[:120]}", logs)
            logs = log("   Falling back to NLLB-200 local…", logs)

    if translated_texts is None:
        try:
            logs = log("🌐 Translating EN → PT-BR (NLLB-200 local)…", logs)
            translated_texts, logs = _translate_nllb(merged_texts, logs)
            logs = log(
                f"✅ Translated {len(translated_texts)} utterances (NLLB-200)", logs
            )
        except Exception as e:
            raise RuntimeError(
                (
                    "All translators failed.\n"
                    + (f"OpenRouter error: {primary_error}\n" if primary_error else "")
                    + f"NLLB error: {e}"
                )
            )

    # Empty-translation safety: fall back to original English per utterance
    empty_count = 0
    for i, (utt, txt) in enumerate(zip(merged, translated_texts)):
        clean = txt.strip() if txt else ""
        if not clean:
            clean = utt["text"]
            empty_count += 1
        merged[i] = {**utt, "text": clean}
    if empty_count:
        logs = log(
            f"   ⚠️  {empty_count} empty translation(s) — kept original English", logs
        )

    # ── Step 3: re-expand to original segment granularity ────────────────────
    translated = _expand_merged(merged, segments)
    logs = log(f"✅ Translated + re-expanded to {len(translated)} segments", logs)
    return translated, logs


# ── Backward constraint: source rate → text budget ───────────────────────────

# PT-BR synthesis capacity (chars/sec) calibrated via Loop 4/5.
# Higher = speaker is faster; lower = speaker needs more time per char.
VOICE_CALIBRATION: dict[str, float] = {
    # Kokoro (calibrated 2026-03-21)
    "pf_dora": 13.3,
    "pm_alex": 13.1,
    "pm_santa": 12.9,
    # Edge-TTS (calibrated 2026-03-25)
    "pt-BR-FranciscaNeural": 11.1,
    "pt-BR-AntonioNeural": 11.1,
    "pt-BR-ThalitaNeural": 13.1,
    # Default/Fallback
    "default": 15.1,  # Previous XTTS v2 baseline
}

MAX_ATEMPO = 1.6  # Upper `atempo` ratio supported in timing-adjustment pipelines


def _estimate_synth_duration(text: str, cps: float = 15.1) -> float:
    """Estimate synthesis duration from character count and capacity."""
    return len(text.strip()) / cps


def _get_cps_for_voice(engine: str, voice: str) -> float:
    """Return the calibrated chars/sec for a given engine + voice."""
    lookup = voice
    if engine.startswith("Kokoro"):
        # kokoro voices are keys themselves
        pass
    elif engine.startswith("Edge"):
        # edge voices are keys themselves
        pass
    else:
        lookup = "default"
    
    return VOICE_CALIBRATION.get(lookup, VOICE_CALIBRATION["default"])


def _trim_to_budget(text: str, budget_secs: float, openrouter_key: str, cps: float = 15.1) -> str:
    """
    Shorten text to fit within budget_secs at natural speaking rate.

    Strategy:
      1. Estimate duration from character count.
      2. If within budget × MAX_ATEMPO, return as-is (atempo can handle it).
      3. Otherwise, truncate to the last complete word that fits the budget,
         then — if an OpenRouter key is available — ask the LLM to rephrase
         to the same meaning in fewer words rather than hard-truncating.

    This is the backward constraint edge: synthesis constraints flow back
    to the text representation before any audio is generated.
    """
    effective_budget = budget_secs * MAX_ATEMPO
    estimated = _estimate_synth_duration(text)

    if estimated <= effective_budget:
        return text  # fits, no action needed

    # Hard truncation fallback: keep words up to character budget
    char_budget = int(effective_budget * cps)
    truncated = text[:char_budget].rsplit(" ", 1)[0].rstrip(".,;:")

    if not openrouter_key.strip():
        return truncated  # no LLM available, use hard truncation

    # LLM rephrase: ask for same meaning in fewer words
    try:
        import urllib.request as _ur

        prompt = (
            f"Rephrase this Brazilian Portuguese text to express the same meaning "
            f"in at most {char_budget} characters. "
            f"Output ONLY the rephrased text, nothing else.\n\n{text}"
        )
        payload = json.dumps(
            {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Brazilian Portuguese editor. Shorten text while preserving meaning. Never add explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            }
        ).encode()
        req = _ur.Request(
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
        with _ur.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        rephrased = result["choices"][0]["message"]["content"].strip()
        # Only use rephrased if it's actually shorter
        if len(rephrased) < len(text):
            return rephrased
    except Exception:
        pass  # LLM rephrase failed — fall back to hard truncation

    return truncated


def apply_timing_budget(
    segments: list,
    logs: list,
    openrouter_key: str = "",
    cps: float = 15.1,
) -> tuple[list, list]:
    """
    Pre-flight pass: measure source speaking rate, predict synthesis duration,
    and shorten translated text that will provably overflow its time slot.

    This is the single backward constraint edge in the pipeline. Without it,
    every stage is a one-way valve: synthesis discovers overflow only after
    rendering, then atempo degrades quality to compensate. With it, the text
    is shortened before synthesis runs, so atempo operates within its quality
    range on all segments.

    Source WPM is measured per-segment to handle variable-rate speakers
    (a presenter who slows down for emphasis vs. speaks rapidly in argument).
    """
    trimmed_count = 0
    overflow_count = 0

    for i, seg in enumerate(segments):
        slot_dur = seg["end"] - seg["start"]
        if slot_dur <= 0.1:
            continue

        text = seg.get("text", "").strip()
        if not text:
            continue

        estimated = _estimate_synth_duration(text, cps=cps)
        effective_budget = slot_dur * MAX_ATEMPO
 
        if estimated > effective_budget:
            overflow_count += 1
            original_len = len(text)
            text = _trim_to_budget(text, slot_dur, openrouter_key, cps=cps)
            segments[i] = {**seg, "text": text}
            if len(text) < original_len:
                trimmed_count += 1

    if overflow_count:
        logs = log(
            f"   ⏱️  {overflow_count} segments predicted to overflow slot — "
            f"{trimmed_count} shortened pre-synthesis, "
            f"{overflow_count - trimmed_count} within atempo range",
            logs,
        )
    else:
        logs = log("   ⏱️  All segments within timing budget", logs)

    return segments, logs


import re


def _sanitize_for_tts(text: str) -> str:
    """
    Clean text before passing to XTTS.
    Removes spelled-out punctuation (ponto, virgula) and characters
    XTTS reads aloud instead of treating as prosody cues.
    """
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
    text = re.sub(r"\.(?=\s)", ",", text)  # interior periods → comma (soft pause, prevents run-ons)
    text = re.sub(r"\.", "", text)         # terminal/remaining periods → nothing (avoids "ponto")
    text = re.sub(r"[()\[\]{}]", "", text)
    text = re.sub(r'[\u201c\u201d\u2018\u2019"\'`]', "", text)
    text = re.sub(r"[-\u2013\u2014]{2,}", ",", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "\u2026"


def synthesize_segments_kokoro(
    segments: list,
    job_dir: Path,
    logs: list,
    voice: str = KOKORO_VOICE,
    speed: float = KOKORO_SPEED,
):
    """
    Synthesize PT-BR speech using Kokoro-82M.

    Advantages over XTTS v2:
    - 82M params vs ~1B — loads in <2s, uses ~500MB VRAM
    - 24kHz output, natural prosody, no voice-cloning overhead
    - Native PT-BR support (lang_code='p', voices: pf_dora/pm_alex/pm_santa)
    - Returns numpy arrays directly — no subprocess needed

    Requires: pip install kokoro soundfile
              espeak-ng installed system-wide (Windows: espeak-ng MSI)
    """
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    KOKORO_SR = 24000  # Kokoro always outputs at 24kHz

    logs = log(f"🔊 Loading Kokoro-82M (lang=pt-br, voice={voice})…", logs)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pipeline = KPipeline(lang_code=KOKORO_LANG, repo_id="hexgrad/Kokoro-82M")
    logs = log("   Kokoro ready", logs)

    # Sanitize and filter empty segments
    empty = [i for i, s in enumerate(segments) if not s.get("text", "").strip()]
    if empty:
        logs = log(f"   ⚠️  {len(empty)} empty segment(s) — skipping", logs)
    segments = [s for s in segments if s.get("text", "").strip()]

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    logs = log(
        f"🎤 Synthesizing {len(segments)} segments (Kokoro PT-BR, voice={voice})…", logs
    )

    timed_clips = []
    for i, seg in enumerate(segments):
        out_raw = seg_dir / f"seg_{i:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{i:04d}.wav"

        text = _sanitize_for_tts(seg["text"].strip())

        # KPipeline returns a generator of (graphemes, phonemes, audio_array)
        # For short segments it yields one chunk; collect and concatenate.
        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio)

        if not chunks:
            logs = log(
                f"   ⚠️  Kokoro returned no audio for segment {i} — skipping", logs
            )
            continue

        audio_np = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_raw), audio_np, KOKORO_SR)

        # Timing adjustment — same logic as XTTS path
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

            ratio = synth_dur / orig_dur
            ratio = max(0.8, min(ratio, 1.6))

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
            # Resample to 44100 for consistency with the numpy assembler
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(out_raw), "-ar", "44100", str(out_clip)],
                capture_output=True,
            )

        timed_clips.append(
            {
                "path": str(out_clip),
                "start": seg["start"],
                "end": seg["end"],
            }
        )

        if i % 10 == 0:
            logs = log(f"   Segment {i+1}/{len(segments)}…", logs)

    logs = log("✅ All segments synthesized (Kokoro)", logs)
    return timed_clips, logs


def synthesize_segments_google_tts(
    segments: list,
    job_dir: Path,
    logs: list,
    api_key: str,
    voice_type: str = "Neural2",
    voice_name: str = "pt-BR-Neural2-A",
    language_code: str = "pt-BR",
):
    """
    Synthesize speech using the Google Cloud Text-to-Speech REST API.

    Supported voice types (passed as voice_type):
      Chirp3 HD  — Latest generation, highest quality (pt-BR-Chirp3-HD-*)
      Neural2    — High-quality neural voices (pt-BR-Neural2-*)
      WaveNet    — High-quality voices (pt-BR-Wavenet-*)
      Studio     — Extremely natural, higher latency (pt-BR-Studio-*)
      Standard   — Fast, low-cost (pt-BR-Standard-*)
      Polyglot   — Multilingual single-speaker (limited PT-BR availability)

    Requests LINEAR16 (WAV) at 24 kHz; timing-adjusts via atempo + resamples
    to 44100 Hz with ffmpeg, identical to the Kokoro and XTTS paths.

    The API key is passed via the x-goog-api-key request header so it never
    appears in URLs or server access logs.
    """
    import base64
    import urllib.request
    import urllib.error

    logs = log(f"🔊 Loading Google Cloud TTS ({voice_type}: {voice_name})…", logs)

    endpoint = "https://texttospeech.googleapis.com/v1/text:synthesize"

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    empty = [i for i, s in enumerate(segments) if not s.get("text", "").strip()]
    if empty:
        logs = log(f"   ⚠️  {len(empty)} empty segment(s) — skipping", logs)
    segments = [s for s in segments if s.get("text", "").strip()]

    logs = log(
        f"🎤 Synthesizing {len(segments)} segments (Google Cloud TTS, up to 16 concurrent)…", logs
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict | None] = [None] * len(segments)  # pre-sized; indexed by segment position
    errors = []

    def _synthesize_one(idx, seg):
        out_raw = seg_dir / f"seg_{idx:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{idx:04d}.wav"

        text = _sanitize_for_tts(seg["text"].strip())

        payload = json.dumps(
            {
                "input": {"text": text},
                "voice": {
                    "languageCode": language_code,
                    "name": voice_name,
                },
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "sampleRateHertz": 24000,
                },
            }
        ).encode()

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                # Header-based key auth keeps the API key out of URLs and logs.
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        # HIGH SEVERITY FIX 4: Add fallback for Google TTS API key failures
        # Use retry with backoff for transient 429/5xx errors.
        data = None
        try:
            def _do_google_tts():
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())

            data = _retry_with_backoff(_do_google_tts)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:300]
            errors.append(f"   ⚠️  Google TTS API error (segment {idx}): HTTP {e.code}\n{err_body}")
            # Create a silent placeholder to maintain timeline
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", "0.5", "-ar", "44100", str(out_raw)
            ], capture_output=True)
            errors.append(f"   ⚠️  Created silent placeholder for segment {idx}")
        except Exception as exc:
            errors.append(f"   ⚠️  Google TTS request failed (segment {idx}): {exc}")
            # Create a silent placeholder to maintain timeline
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", "0.5", "-ar", "44100", str(out_raw)
            ], capture_output=True)
            errors.append(f"   ⚠️  Created silent placeholder for segment {idx}")

        # Only process audio if we have valid data
        if data is not None:
            audio_bytes = base64.b64decode(data["audioContent"])
            out_raw.write_bytes(audio_bytes)

            # Timing adjustment — same atempo logic as Kokoro / XTTS paths.
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
                ratio = max(0.8, min(synth_dur / orig_dur, 1.6))
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
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(out_raw),
                        "-ar",
                        "44100",
                        str(out_clip),
                    ],
                    capture_output=True,
                )
        else:
            # If we created a silent placeholder, copy it to the final clip
            shutil.copy(str(out_raw), str(out_clip))

        return {
            "path": str(out_clip),
            "start": seg["start"],
            "end": seg["end"],
        }

    total = len(segments)
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_idx = {
            executor.submit(_synthesize_one, i, seg): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == total:
                logs = log(f"   Segment {completed}/{total} done…", logs)

    # Flush any per-segment error messages collected from worker threads
    for msg in errors:
        logs = log(msg, logs)

    timed_clips = results

    logs = log("✅ All segments synthesized (Google Cloud TTS)", logs)
    return timed_clips, logs


def synthesize_segments_edge_tts(
    segments: list,
    job_dir: Path,
    logs: list,
    voice: str = EDGE_TTS_VOICE_NAME,
    speed: float = 1.0,
):
    """
    Synthesize PT-BR speech using Microsoft Edge TTS (cloud, no API key, no VRAM).

    Uses the edge-tts package which streams from Microsoft's Neural TTS service.
    Output is MP3 decoded via ffmpeg to WAV, then timing-adjusted with atempo —
    identical final contract to the Kokoro and Google TTS paths.

    Voices: pt-BR-FranciscaNeural (F) · pt-BR-AntonioNeural (M) · pt-BR-ThalitaNeural (F)
    Requires: pip install edge-tts  (add to pixi.toml: edge-tts = "*")
              Internet access during synthesis.
    No espeak-ng, no GPU, no model download.
    """
    import asyncio
    import io

    try:
        import edge_tts  # noqa: PLC0415
    except ImportError:
        raise PipelineError(
            "Synthesize",
            "edge-tts package not installed. Add 'edge-tts = \"*\"' to pixi.toml "
            "under [pypi-dependencies] and run 'pixi install'.",
            recoverable=False,
        )

    # Convert speed multiplier → SSML signed-percent string
    rate_pct = round((speed - 1.0) * 100)
    rate_str = f"{rate_pct:+d}%"

    logs = log(f"🔊 Edge TTS ready (voice={voice}, rate={rate_str})", logs)

    empty = [i for i, s in enumerate(segments) if not s.get("text", "").strip()]
    if empty:
        logs = log(f"   ⚠️  {len(empty)} empty segment(s) — skipping", logs)
    segments = [s for s in segments if s.get("text", "").strip()]

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    logs = log(
        f"🎤 Synthesizing {len(segments)} segments (Edge TTS, up to 16 concurrent)…", logs
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict | None] = [None] * len(segments)
    errors: list[str] = []

    # Edge TTS sometimes force-closes the socket after streaming. Swallow
    # the resulting ConnectionResetError so asyncio does not spam the log.
    def _edge_tts_asyncio_exception_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError):
            return
        loop.default_exception_handler(context)

    async def _stream_mp3(text: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                if data:
                    buf.write(data)
        return buf.getvalue()

    def _run_edge_tts_stream(text: str) -> bytes:
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(_edge_tts_asyncio_exception_handler)
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_stream_mp3(text))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _synthesize_one(idx: int, seg: dict) -> dict:
        out_raw = seg_dir / f"seg_{idx:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{idx:04d}.wav"

        text = _sanitize_for_tts(seg["text"].strip())

        # Synthesize: stream MP3 from Edge TTS, then decode to WAV via ffmpeg.
        # ffmpeg is already a pipeline dependency — no new tools needed.
        try:
            mp3_bytes = _run_edge_tts_stream(text)
            if not mp3_bytes:
                raise RuntimeError("Edge TTS returned empty audio")

            decode = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", "pipe:0",
                    "-ar", "44100",
                    "-ac", "1",
                    str(out_raw),
                    "-loglevel", "quiet",
                ],
                input=mp3_bytes,
                capture_output=True,
            )
            if decode.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg MP3 decode failed: {decode.stderr.decode(errors='replace')[:200]}"
                )

        except Exception as exc:
            errors.append(f"   ⚠️  Edge TTS failed (segment {idx}): {exc}")
            # Silent placeholder keeps the timeline intact
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
                    "-t", "0.5",
                    str(out_raw),
                    "-loglevel", "quiet",
                ],
                capture_output=True,
            )
            shutil.copy(str(out_raw), str(out_clip))
            return {"path": str(out_clip), "start": seg["start"], "end": seg["end"]}

        # Timing adjustment — same atempo logic as Kokoro / Google TTS paths.
        orig_dur = seg["end"] - seg["start"]
        if orig_dur > 0.1:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    str(out_raw),
                ],
                capture_output=True, text=True,
            )
            synth_dur = float(json.loads(probe.stdout)["format"]["duration"])
            ratio = max(0.8, min(synth_dur / orig_dur, 1.6))
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(out_raw),
                    "-filter:a", f"atempo={ratio:.4f}",
                    "-ar", "44100",
                    str(out_clip),
                    "-loglevel", "quiet",
                ],
                capture_output=True,
            )
        else:
            shutil.copy(str(out_raw), str(out_clip))

        return {"path": str(out_clip), "start": seg["start"], "end": seg["end"]}

    total = len(segments)
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_idx = {
            executor.submit(_synthesize_one, i, seg): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == total:
                logs = log(f"   Segment {completed}/{total} done…", logs)

    for msg in errors:
        logs = log(msg, logs)

    logs = log("✅ All segments synthesized (Edge TTS)", logs)
    return results, logs


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
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_orig),
                "-t",
                "30",
                "-ar",
                "22050",
                "-ac",
                "1",
                ref_wav,
            ],
            capture_output=True,
        )

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # Sanitize: empty text crashes XTTS with a misleading "reference_wav" error.
    # This happens when the translation parser drops a segment's content.
    # Log each empty segment so the cause is visible, then skip it.
    empty = [i for i, s in enumerate(segments) if not s.get("text", "").strip()]
    if empty:
        logs = log(
            f"   ⚠️  {len(empty)} empty segment(s) after translation (indices: {empty[:10]}{'…' if len(empty)>10 else ''}) — skipping",
            logs,
        )
    segments = [s for s in segments if s.get("text", "").strip()]

    logs = log(f"🎤 Synthesizing {len(segments)} segments (voice clone)…", logs)

    timed_clips = []
    for i, seg in enumerate(segments):
        out_raw = seg_dir / f"seg_{i:04d}_raw.wav"
        out_clip = seg_dir / f"seg_{i:04d}.wav"

        text = _sanitize_for_tts(seg["text"].strip())

        # HIGH SEVERITY FIX 1: Add intra-stage checkpointing to prevent losing progress
        # Save progress after each segment synthesis
        checkpoint_file = job_dir / "synthesize_checkpoint.json"
        try:
            # Try synthesis
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_wav,
                language=TARGET_LANG,
                file_path=str(out_raw),
            )
        except Exception as e:
            logs = log(f"   ⚠️  XTTS synthesis failed for segment {i}: {e}", logs)
            # Create a silent placeholder to maintain timeline
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", "0.5", "-ar", "44100", str(out_raw)
            ], capture_output=True)
            logs = log(f"   ⚠️  Created silent placeholder for segment {i}", logs)

        # Timing adjustment: compress or stretch to fit original segment duration.
        #
        # Hard cap at 1.6× speed — beyond this speech becomes unintelligible.
        # If synthesized audio is longer than 1.6× the slot allows, we let it
        # run over into the following silence rather than destroying clarity.
        # Stretching (ratio < 1.0) is always safe; we allow down to 0.8×.
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

            ratio = synth_dur / orig_dur  # >1 = too long, need to speed up

            # Cap: never compress beyond 1.6× (intelligibility limit)
            # Never stretch beyond 0.8× (sounds unnaturally slow)
            ratio = max(0.8, min(ratio, 1.6))

            atempo = f"atempo={ratio:.4f}"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(out_raw),
                    "-filter:a",
                    atempo,
                    "-ar",
                    "44100",
                    str(out_clip),
                ],
                capture_output=True,
            )
        else:
            shutil.copy(str(out_raw), str(out_clip))

        timed_clips.append(
            {
                "path": str(out_clip),
                "start": seg["start"],
                "end": seg["end"],
            }
        )

        # Save checkpoint after each segment
        checkpoint_data = {
            "completed_segments": i + 1,
            "total_segments": len(segments),
            "last_segment": i,
            "timed_clips": timed_clips
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")

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

    # Audio assembly: place each segment at its timestamp using sox.
    #
    # All previous amix-based approaches divided amplitude by input count,
    # producing audio 20-34 dB below full scale regardless of weights or
    # normalization passes applied afterward.
    #
    # The correct primitive for this task is NOT mixing (amix) — it is
    # timeline placement. sox `splice` or a Python numpy buffer writes each
    # segment's samples directly at the correct offset. No averaging occurs.
    # Amplitude is preserved exactly as XTTS produced it.
    #
    # We use a numpy buffer: pre-allocate silence, write each clip's samples
    # at its timestamp offset. One read per clip, one write total. O(N) time,
    # O(duration) memory (~80MB for a 13-min mono 44100Hz float32 buffer).

    import numpy as np
    import wave

    SR = 44100
    total_samples = int((duration + 2) * SR)
    buffer = np.zeros(total_samples, dtype=np.float32)

    logs = log("🔊 Placing segments into audio timeline…", logs)
    for clip in timed_clips:
        offset = int(clip["start"] * SR)
        # Read clip WAV into numpy
        try:
            with wave.open(clip["path"], "rb") as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                clip_sr = wf.getframerate()
                n_ch = wf.getnchannels()
                sampwidth = wf.getsampwidth()

            # Convert raw bytes → float32 [-1, 1]
            if sampwidth == 2:
                samples = (
                    np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                )
            elif sampwidth == 4:
                samples = (
                    np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                )
            else:
                samples = (
                    np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                )

            # Mix down to mono if stereo
            if n_ch == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)

            # Resample if needed (shouldn't happen — all clips are 44100Hz)
            if clip_sr != SR:
                factor = SR / clip_sr
                new_len = int(len(samples) * factor)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, new_len),
                    np.arange(len(samples)),
                    samples,
                ).astype(np.float32)

            end = min(offset + len(samples), total_samples)
            buffer[offset:end] += samples[: end - offset]

        except Exception as e:
            logs = log(f"   ⚠️  Could not read clip {clip['path']}: {e}", logs)

    # Peak-normalize to -1 dBFS so the output is loud without clipping
    peak = np.max(np.abs(buffer))
    if peak > 0.001:
        buffer = buffer * (0.891 / peak)  # 0.891 ≈ -1 dBFS
    else:
        logs = log(
            "   ⚠️  Audio buffer is nearly silent — synthesis may have failed", logs
        )

    # Convert to stereo int16 WAV
    stereo = np.stack([buffer, buffer], axis=1)
    stereo_int16 = (stereo * 32767).clip(-32768, 32767).astype(np.int16)

    mixed_audio = job_dir / "dubbed_audio.wav"
    with wave.open(str(mixed_audio), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo_int16.tobytes())

    logs = log("✅ Audio timeline assembled", logs)

    # 3. Mux video + dubbed audio
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
    output_path = OUTPUT_DIR / f"{safe_title}_PT-BR.mp4"

    run_ffmpeg(
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
        "final mux",
    )

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


# ── SRT subtitle generation ───────────────────────────────────────────────────

# Reading-speed constants (Netflix/BBC standard for accessible subtitles)
SRT_CHARS_PER_SEC = 17.0  # comfortable reading pace
SRT_MIN_DURATION = 1.2  # minimum cue display time (seconds)
SRT_MAX_CHARS = 80  # maximum chars per cue before refusing to merge
SRT_MERGE_GAP = 0.5  # merge adjacent segments separated by ≤ this gap (seconds)
SRT_LINE_WIDTH = 42  # wrap to 2 lines when text exceeds this character count


def _srt_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp: HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1_000
    ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _wrap_subtitle_line(text: str) -> str:
    """
    Split text into at most 2 balanced lines broken at the word boundary
    nearest the midpoint. Returns the original string when it fits on one line.
    """
    if len(text) <= SRT_LINE_WIDTH:
        return text
    words = text.split()
    mid = len(text) // 2
    pos = 0
    best_split = max(1, len(words) // 2)
    best_dist = float("inf")
    for i in range(1, len(words)):
        pos += len(words[i - 1]) + 1
        dist = abs(pos - mid)
        if dist < best_dist:
            best_dist = dist
            best_split = i
    return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])


def generate_srt(segments: list, output_path: Path) -> int:
    """
    Convert translated segments to a natural-reading SRT subtitle file.

    Pass 1 — Merge: join consecutive short fragments (gap ≤ SRT_MERGE_GAP,
    combined length ≤ SRT_MAX_CHARS) into complete, readable cues.

    Pass 2 — Timing: extend any cue whose display window is shorter than the
    time needed to read it at SRT_CHARS_PER_SEC (floor: SRT_MIN_DURATION).
    Extensions are capped just before the next cue starts (50 ms gap).

    Pass 3 — Wrap + write: split cues longer than SRT_LINE_WIDTH at the word
    boundary nearest the midpoint, then write standard SRT output.

    Returns the number of subtitle cues written.
    """
    # ── Pass 1: merge ─────────────────────────────────────────────────────────
    cues: list[dict] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start, end = seg["start"], seg["end"]
        if cues:
            prev = cues[-1]
            gap = start - prev["end"]
            combined = prev["text"] + " " + text
            if gap <= SRT_MERGE_GAP and len(combined) <= SRT_MAX_CHARS:
                prev["text"] = combined
                prev["end"] = end
                continue
        cues.append({"start": start, "end": end, "text": text})

    # ── Pass 2: enforce minimum reading time ──────────────────────────────────
    for i, cue in enumerate(cues):
        min_dur = max(SRT_MIN_DURATION, len(cue["text"]) / SRT_CHARS_PER_SEC)
        natural_end = cue["start"] + min_dur
        if cue["end"] < natural_end:
            ceiling = cues[i + 1]["start"] - 0.05 if i + 1 < len(cues) else natural_end
            new_end = min(natural_end, ceiling)
            if new_end > cue["end"]:
                cue["end"] = new_end

    # ── Pass 3: write SRT ─────────────────────────────────────────────────────
    lines: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{_srt_timestamp(cue['start'])} --> {_srt_timestamp(cue['end'])}")
        lines.append(_wrap_subtitle_line(cue["text"]))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return len(cues)


def generate_srt_for_project(project_name: str) -> tuple[str | None, str]:
    """
    Generate an SRT subtitle file from a project's translated.json.

    Saves the SRT to projects/{name}/outputs/{title}_PT-BR.srt and also
    copies it to the global outputs/ folder for easy access.

    Returns (srt_file_path, status_message).
    """
    proj = project_name.strip()
    if not proj:
        return None, "❌ No project name provided."

    d = project_dir(proj)
    translated_file = d / "translated.json"
    if not translated_file.exists():
        return None, (
            f"❌ No translated.json found for project '{proj}'. "
            "Run the translate stage first."
        )

    title = proj
    meta_file = d / "meta.json"
    if meta_file.exists():
        try:
            title = json.loads(meta_file.read_text(encoding="utf-8")).get("title", proj)
        except Exception:
            pass

    segments = json.loads(translated_file.read_text(encoding="utf-8"))
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]

    out_dir = d / "outputs"
    out_dir.mkdir(exist_ok=True)
    srt_path = out_dir / f"{safe_title}_PT-BR.srt"

    n = generate_srt(segments, srt_path)

    global_srt = OUTPUT_DIR / f"{safe_title}_PT-BR.srt"
    shutil.copy2(str(srt_path), str(global_srt))

    return str(srt_path), f"✅ {n} subtitle cues written → {srt_path.name}"


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
        logs = log(
            f"🧹 Cleaned {cleaned} stale job folder(s) (>{JOB_MAX_AGE_H}h old)", logs
        )
    return logs


# ── Project persistence ───────────────────────────────────────────────────────

STAGES = ["download", "transcribe", "translate", "synthesize", "assemble"]


def project_dir(name: str) -> Path:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in " _-")[:60].strip()
    if not safe:
        safe = "project"
    d = PROJECTS_DIR / safe
    d.mkdir(exist_ok=True)
    return d


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in PROJECTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def project_status(name: str) -> dict:
    """Return which stage outputs exist for a project."""
    d = project_dir(name)
    return {
        "download": (d / "video.mp4").exists() and (d / "audio_orig.wav").exists(),
        "transcribe": (d / "segments.json").exists(),
        "translate": (d / "translated.json").exists(),
        "synthesize": (d / "timed_clips.json").exists(),
        "assemble": (
            any((d / "outputs").glob("*.mp4")) if (d / "outputs").exists() else False
        ),
    }


def save_project_stage(name: str, stage: str, data):
    """Persist a stage output to the project directory."""
    d = project_dir(name)
    if stage == "download":
        # data = (video_path, audio_path, title, duration)
        # Files are already in job_dir; copy them to project dir
        video_src, audio_src, title, duration = data
        shutil.copy2(str(video_src), str(d / "video.mp4"))
        shutil.copy2(str(audio_src), str(d / "audio_orig.wav"))
        meta = {"title": title, "duration": duration}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    elif stage == "transcribe":
        (d / "segments.json").write_text(json.dumps(data), encoding="utf-8")
    elif stage == "translate":
        (d / "translated.json").write_text(json.dumps(data), encoding="utf-8")
    elif stage == "synthesize":
        # data = timed_clips list; WAV files already in job_dir/segments/
        # Copy segment WAVs to project dir
        seg_dst = d / "segments"
        seg_dst.mkdir(exist_ok=True)
        updated = []
        for clip in data:
            src_path = Path(clip["path"])
            dst_path = seg_dst / src_path.name
            if src_path.exists() and src_path != dst_path:
                shutil.copy2(str(src_path), str(dst_path))
            updated.append({**clip, "path": str(dst_path)})
        (d / "timed_clips.json").write_text(json.dumps(updated), encoding="utf-8")
    elif stage == "assemble":
        # data = output_path string
        out_dst = d / "outputs"
        out_dst.mkdir(exist_ok=True)
        dst = out_dst / Path(data).name
        if Path(data) != dst:
            shutil.copy2(data, str(dst))


def load_project_stage(name: str, stage: str) -> Any:
    """Load a previously saved stage output from project directory."""
    d = project_dir(name)
    if stage == "download":
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return d / "video.mp4", d / "audio_orig.wav", meta["title"], meta["duration"]
    elif stage == "transcribe":
        return json.loads((d / "segments.json").read_text(encoding="utf-8"))
    elif stage == "translate":
        return json.loads((d / "translated.json").read_text(encoding="utf-8"))
    elif stage == "synthesize":
        return json.loads((d / "timed_clips.json").read_text(encoding="utf-8"))


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(
    url: str,
    video_upload_path: str | None,
    speaker_wav_path: str | None,
    whisper_model: str,
    browser: str,
    cookies_file: str | None,
    project_name: str,
    resume_from: str,
    tts_engine: str = "XTTS v2 (voice clone)",
    kokoro_voice: str = KOKORO_VOICE,
    google_tts_voice_type: str = GOOGLE_TTS_VOICE_TYPE,
    google_tts_voice_name: str = GOOGLE_TTS_VOICE_NAME,
    edge_tts_voice: str = EDGE_TTS_VOICE_NAME,
    progress=gr.Progress(),
):
    logs = []
    # Read API key from environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    proj = project_name.strip() or "default"
    pdir = project_dir(proj)

    # Stage order for resume logic
    stage_order = {s: i for i, s in enumerate(STAGES)}
    resume_idx = stage_order.get(resume_from, 0)

    # job_dir: temp workspace for files generated this run
    # (segment WAVs etc). Saved stages are persisted to pdir.
    job_id = str(int(time.time()))
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    model_to_use = whisper_model.strip() if whisper_model.strip() else WHISPER_MODEL

    try:
        lazy_import()
        logs = cleanup_stale_jobs(logs)
        logs = log(f"📁 Project: {proj}  |  Resume from: {resume_from}", logs)
        yield None, "\n".join(logs)

        # ── Pre-flight: video source validation (T4) ─────────────────────────
        source_mode = None  # 'url' or 'file'
        if resume_idx <= stage_order["download"]:
            src_ok, src_result = validate_video_source(url, video_upload_path)
            if not src_ok:
                raise PipelineError(
                    "Validation",
                    src_result,
                    recoverable=False,
                )
            source_mode = src_result  # 'url' or 'file'

        # ── Pre-flight: API key validation (T3) ──────────────────────────────
        if openrouter_key:
            logs = log("🔑 Validating OpenRouter API key…", logs)
            yield None, "\n".join(logs)
            ok, msg = validate_openrouter_key(openrouter_key)
            if not ok:
                raise PipelineError(
                    "Validation",
                    f"OpenRouter API key is invalid: {msg}. "
                    "Fix OPENROUTER_API_KEY in .env, or remove it to use local NLLB-200.",
                    recoverable=False,
                )
            logs = log("   ✅ OpenRouter key valid", logs)
            yield None, "\n".join(logs)

        if tts_engine == "Google Cloud TTS":
            if not GOOGLE_TTS_API_KEY:
                raise PipelineError(
                    "Validation",
                    "Google Cloud TTS selected but GOOGLE_TTS_API_KEY is not set in .env.",
                    recoverable=False,
                )
            logs = log("🔑 Validating Google TTS API key…", logs)
            yield None, "\n".join(logs)
            ok, msg = validate_google_tts_key(GOOGLE_TTS_API_KEY)
            if not ok:
                raise PipelineError(
                    "Validation",
                    f"Google TTS API key is invalid: {msg}. "
                    "Fix GOOGLE_TTS_API_KEY in .env or switch to a different TTS engine.",
                    recoverable=False,
                )
            logs = log("   ✅ Google TTS key valid", logs)
            yield None, "\n".join(logs)

        # ── Download / Ingest ─────────────────────────────────────────────────
        if resume_idx <= stage_order["download"]:
            if source_mode == "file":
                # Local file upload path
                progress(0.05, desc="Ingesting uploaded file…")
                try:
                    video_path, audio_path, title, duration, logs = ingest_local_file(
                        video_upload_path, job_dir, logs  # type: ignore[arg-type]
                    )
                except PipelineError:
                    raise
                except Exception as e:
                    raise PipelineError(
                        "Ingest",
                        f"Failed to process uploaded file: {e}",
                        recoverable=False,
                    ) from e
            else:
                # URL download path (any yt-dlp supported site)
                if cookies_file:
                    cookie_msg = f"cookies.txt ({Path(cookies_file).name})"
                elif browser != "none":
                    cookie_msg = f"browser cookies ({browser})"
                else:
                    cookie_msg = "anonymous (no cookies)"
                logs = log(f"🍪 Download mode: {cookie_msg}", logs)
                yield None, "\n".join(logs)

                progress(0.05, desc="Downloading…")
                try:
                    video_path, audio_path, title, duration, logs = download_video(
                        url, job_dir, logs, browser=browser, cookies_file=cookies_file
                    )
                except Exception as e:
                    raise PipelineError(
                        "Download",
                        f"yt-dlp failed: {e}. "
                        "Check the URL, try browser cookies, or verify the video is publicly available.",
                        recoverable=False,
                    ) from e
            save_project_stage(
                proj, "download", (video_path, audio_path, title, duration)
            )
            yield None, "\n".join(logs)
        else:
            logs = log("⏭️  Skipping download (loaded from project)", logs)
            video_path, audio_path, title, duration = load_project_stage(
                proj, "download"
            )
            logs = log(f"   📹 {title} ({duration}s)", logs)
            yield None, "\n".join(logs)

        # ── Transcribe ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["transcribe"]:
            progress(0.2, desc="Transcribing…")
            try:
                segments, logs = transcribe_audio(audio_path, logs, model_name=model_to_use)
            except Exception as e:
                raise PipelineError(
                    "Transcribe",
                    f"Whisper failed: {e}. "
                    "Try a smaller model (set WHISPER_MODEL=base in .env) or check available VRAM.",
                    recoverable=False,
                ) from e
            save_project_stage(proj, "transcribe", segments)
            yield None, "\n".join(logs)
            # Release Whisper GPU memory before loading translation model (T7)
            release_gpu_memory()
            logs = log("   🧹 GPU memory released after transcription", logs)
        else:
            logs = log("⏭️  Skipping transcription (loaded from project)", logs)
            segments = cast(list, load_project_stage(proj, "transcribe"))
            logs = log(f"   📝 {len(segments)} segments loaded", logs)
            yield None, "\n".join(logs)

        # ── Translate ─────────────────────────────────────────────────────────
        if resume_idx <= stage_order["translate"]:
            progress(0.4, desc="Translating…")
            try:
                # Select merge config based on engine
                m_cfg = _get_merge_config(tts_engine)
                
                translated, logs = translate_segments(
                    cast(list, segments),
                    logs,
                    openrouter_key=openrouter_key,
                    merge_config=m_cfg,
                )
                progress(0.5, desc="Checking timing budget…")
                # Detect current active voice for calibration
                active_voice = "default"
                if tts_engine.startswith("Kokoro"):
                    active_voice = kokoro_voice
                elif tts_engine.startswith("Edge"):
                    active_voice = edge_tts_voice
                elif tts_engine.startswith("Google"):
                    active_voice = google_tts_voice_name
                
                cps = _get_cps_for_voice(tts_engine, active_voice)
                
                translated, logs = apply_timing_budget(
                    translated, logs, openrouter_key=openrouter_key, cps=cps
                )
            except PipelineError:
                raise
            except Exception as e:
                raise PipelineError(
                    "Translate",
                    f"Translation failed: {e}. "
                    "Check your OpenRouter key (if set) or verify the NLLB model is downloaded.",
                    recoverable=True,
                ) from e
            save_project_stage(proj, "translate", translated)
            yield None, "\n".join(logs)
            # Release NLLB GPU memory before synthesis (T7)
            release_gpu_memory()
            logs = log("   🧹 GPU memory released after translation", logs)
        else:
            logs = log("⏭️  Skipping translation (loaded from project)", logs)
            translated = cast(list, load_project_stage(proj, "translate"))
            logs = log(f"   🌐 {len(translated)} translated segments loaded", logs)
            yield None, "\n".join(logs)

        # ── Synthesize ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["synthesize"]:
            progress(0.55, desc="Synthesizing voice…")
            utterances = _group_for_synthesis(translated)
            logs = log(
                f"   Grouped {len(translated)} segments → {len(utterances)} utterances for sentence-level synthesis",
                logs,
            )
            try:
                if tts_engine == "Kokoro (fast, PT-BR native)":
                    timed_clips, logs = synthesize_segments_kokoro(
                        utterances,
                        job_dir,
                        logs,
                        voice=kokoro_voice,
                    )
                elif tts_engine == "Google Cloud TTS":
                    timed_clips, logs = synthesize_segments_google_tts(
                        utterances,
                        job_dir,
                        logs,
                        api_key=GOOGLE_TTS_API_KEY,
                        voice_type=google_tts_voice_type,
                        voice_name=google_tts_voice_name,
                        language_code=GOOGLE_TTS_LANGUAGE_CODE,
                    )
                elif tts_engine == "Edge TTS (cloud, no key)":
                    timed_clips, logs = synthesize_segments_edge_tts(
                        utterances,
                        job_dir,
                        logs,
                        voice=edge_tts_voice,
                    )
                else:
                    timed_clips, logs = synthesize_segments(
                        utterances,
                        audio_path,
                        job_dir,
                        logs,
                        speaker_wav=speaker_wav_path,
                    )
            except PipelineError:
                raise
            except Exception as e:
                raise PipelineError(
                    "Synthesize",
                    f"TTS synthesis failed: {e}. "
                    "Check available VRAM. For Kokoro, ensure espeak-ng is installed. "
                    "Resume from 'synthesize' after fixing the issue.",
                    recoverable=True,
                ) from e
            save_project_stage(proj, "synthesize", timed_clips)
            yield None, "\n".join(logs)
            # Release TTS GPU memory before assembly (T7)
            release_gpu_memory()
            logs = log("   🧹 GPU memory released after synthesis", logs)
        else:
            logs = log("⏭️  Skipping synthesis (loaded from project)", logs)
            timed_clips = load_project_stage(proj, "synthesize")
            logs = log(f"   🔊 {len(timed_clips)} audio clips loaded", logs)
            yield None, "\n".join(logs)

        # ── Assemble ──────────────────────────────────────────────────────────
        progress(0.85, desc="Assembling video…")
        try:
            output_path, logs = assemble_dubbed_video(
                video_path,
                timed_clips,
                float(duration or 0),
                job_dir,
                title or "video",
                logs,
            )
        except Exception as e:
            raise PipelineError(
                "Assemble",
                f"FFmpeg assembly failed: {e}. "
                "Ensure ffmpeg is installed. Resume from 'assemble' after fixing.",
                recoverable=True,
            ) from e
        save_project_stage(proj, "assemble", output_path)

        # Auto-generate SRT subtitles from the translated segments
        try:
            _srt_path, _srt_msg = generate_srt_for_project(proj)
            logs = log(f"📄 {_srt_msg}", logs)
        except Exception as _srt_err:
            logs = log(f"⚠️  SRT generation failed: {_srt_err}", logs)

        progress(1.0, desc="Done!")
        yield output_path, "\n".join(logs)

    except PipelineError as e:
        resume_hint = (
            f" You can resume from stage '{e.stage.lower()}' after fixing the issue."
            if e.recoverable
            else ""
        )
        logs = log(f"❌ [{e.stage}] {e.message}{resume_hint}", logs)
        yield None, "\n".join(logs)
    except Exception as e:
        import traceback

        logs = log(f"❌ Unexpected error: {e}\n{traceback.format_exc()}", logs)
        yield None, "\n".join(logs)
    finally:
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
    --muted:   #b0b0cc;
    --danger:  #ff4f6e;
}

* { box-sizing: border-box; }

/* Skip link */
.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: #0a0a0f;
    color: #00e5a0;
    padding: 8px 16px;
    z-index: 9999;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    border: 1px solid #00e5a0;
    border-radius: 0 0 8px 0;
    text-decoration: none;
}
.skip-link:focus {
    top: 0;
    outline: 2px solid #00e5a0;
    outline-offset: 2px;
}

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
    font-size: 0.75rem;
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
    font-size: 0.75rem;
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
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(0,229,160,0.20) !important;
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
#run-btn:focus-visible { outline: 2px solid #00e5a0 !important; outline-offset: 3px !important; }

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
    font-size: 0.75rem;
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

/* High Contrast Mode: restore H1 visibility when gradient clip is suppressed */
@media (forced-colors: active) {
    #header h1 {
        -webkit-text-fill-color: revert;
        color: ButtonText;
        background: none;
    }
}

/* Respect reduced-motion preference */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
"""


def build_ui():
    # Collect environment warnings to display in the UI (T2)
    _env_warnings = validate_environment()

    with gr.Blocks(title="Dubweave — PT-BR") as demo:

        gr.HTML("""
        <a href="#main-content" class="skip-link">Skip to main content</a>
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

        if _env_warnings:
            warning_items = "".join(
                f"<li style='margin-bottom:6px;'>{w}</li>" for w in _env_warnings
            )
            gr.HTML(
                f"""<div style="background:rgba(255,79,110,0.08);border:1px solid rgba(255,79,110,0.3);"""
                f"""border-radius:10px;padding:14px 18px;margin-bottom:16px;"""
                f"""font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#ff4f6e;">"""
                f"""<strong>⚠️ Setup Warnings</strong>"""
                f"""<ul style='margin:8px 0 0;padding-left:20px;line-height:1.8;'>{warning_items}</ul>"""
                f"""</div>"""
            )

        with gr.Row(elem_id="main-content"):
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-label">01 · Project</div>')
                with gr.Row():
                    project_name_input = gr.Textbox(
                        placeholder="my-video (letters, numbers, hyphens)",
                        label="Project name",
                        lines=1,
                        scale=2,
                    )
                    resume_from_input = gr.Dropdown(
                        choices=[
                            "download",
                            "transcribe",
                            "translate",
                            "synthesize",
                            "assemble",
                        ],
                        value="download",
                        label="Resume from stage",
                        scale=1,
                    )
                project_status_html = gr.HTML(
                    "<div aria-live='polite' aria-atomic='true'><div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>Enter a project name to see its status.</div></div>"
                )

        def refresh_status(name):
            name = name.strip()
            if not name:
                return "<div aria-live='polite' aria-atomic='true'><div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;'>Enter a project name to see its status.</div></div>"
            status = project_status(name)
            icons = {
                True: "<span style='color:#00e5a0'>✓</span>",
                False: "<span style='color:#9494b2'>·</span>",
            }
            parts = " &nbsp;·&nbsp; ".join(f"{icons[v]} {s}" for s, v in status.items())
            return f"<div aria-live='polite' aria-atomic='true'><div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>{parts}</div></div>"

        project_name_input.change(
            fn=refresh_status, inputs=project_name_input, outputs=project_status_html
        )

        gr.HTML('<div style="height:8px"></div>')
        gr.HTML('<div class="panel-label">02 · Input</div>')
        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    placeholder="https://youtube.com/watch?v=…  or any video URL",
                    label="Video URL (any site supported by yt-dlp)",
                    lines=1,
                )
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("""
                <p style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#9494b2;margin:0 0 4px;text-align:center;">
                — or —
                </p>
                """)
                video_upload_input = gr.File(
                    label="Upload a video file (mp4, mkv, webm, avi, mov)",
                    file_types=[".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".ts", ".m4v"],
                    type="filepath",
                )
                gr.HTML("""
                <p style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#6a6a8e;margin:4px 0 0;">
                If both URL and file are provided, the uploaded file takes priority.
                </p>
                """)

        gr.HTML('<div style="height:12px"></div>')

        with gr.Accordion("🎙️ Custom Voice Reference (optional)", open=False):
            gr.HTML("""
            <p style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#9494b2;margin:0 0 12px;">
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
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#9494b2;margin:0 0 16px;line-height:1.7;">
              <strong style="color:#e8e8f0;">large-v3-turbo</strong> — Recommended for most videos.<br>
              Distilled from large-v3: ~8× faster, near-identical accuracy on clean audio.<br>
              Uses ~3 GB VRAM. Best choice for videos with clear speech.<br>
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

        gr.HTML('<div style="height:8px"></div>')

        with gr.Accordion("🍪 Browser Cookies (optional, for URL downloads)", open=False):
            gr.HTML("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#9494b2;margin:0 0 16px;line-height:1.7;">
              Logged-in cookies give yt-dlp access to more reliable download clients,<br>
              avoiding PO token errors and JS challenge failures.<br>
              <br>
              <strong style="color:#e8e8f0;">Option A — Browser</strong>: yt-dlp reads cookies directly from a running browser profile.<br>
              Chrome may fail if the profile is locked. Use Edge as an alternative.<br>
              <br>
              <strong style="color:#e8e8f0;">Option B — cookies.txt</strong>: Export cookies using a browser extension
              (e.g. <em>Get cookies.txt LOCALLY</em> for Chrome) and upload the file here.<br>
              This is the most reliable method and works regardless of which browser you use.<br>
              <br>
              If both are provided, <strong style="color:#e8e8f0;">cookies.txt takes priority</strong>.
              Select browser <strong style="color:#e8e8f0;">none</strong> and leave file empty to download anonymously.
            </div>
            """)
            with gr.Row():
                browser_input = gr.Radio(
                    choices=["none", "chrome", "firefox", "edge", "brave"],
                    value="none",
                    label="Option A · Browser cookies",
                    elem_id="browser-radio",
                )
            cookies_file_input = gr.File(
                label="Option B · cookies.txt file (Netscape format)",
                file_types=[".txt"],
                type="filepath",
            )

        gr.HTML('<div style="height:8px"></div>')

        with gr.Accordion("🔊 TTS Engine", open=True):
            _google_note = ""
            if GOOGLE_TTS_API_KEY:
                _google_note = (
                    "<br><strong style='color:#7c6dff;'>Google Cloud TTS</strong> "
                    "— cloud-based, multiple model families. "
                    "Voice type and name are set in <code>.env</code> and "
                    "can be overridden here. Requires a valid "
                    "<code>GOOGLE_TTS_API_KEY</code> in <code>.env</code>."
                )
            gr.HTML(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#9494b2;margin:0 0 16px;line-height:1.7;">
              <strong style="color:#00e5a0;">Kokoro</strong> — recommended. 82M params, loads in &lt;2s, native PT-BR voices,
              no voice cloning. Extremely fast on RTX 4070 Super.<br>
              Voices: <code>pf_dora</code> (female) · <code>pm_alex</code> (male) · <code>pm_santa</code> (male)<br>
              Requires: <code>pip install kokoro soundfile</code> + espeak-ng installed.<br>
              <br>
              <strong style="color:#7c6dff;">Edge TTS</strong> — cloud Neural voices, no API key, no VRAM.
              Quality above Kokoro; requires internet during synthesis.<br>
              Voices: <code>FranciscaNeural</code> (F) · <code>AntonioNeural</code> (M) · <code>ThalitaNeural</code> (F)<br>
              Requires: <code>edge-tts = "*"</code> in pixi.toml + internet.<br>
              <br>
              <strong style="color:#e8e8f0;">XTTS v2</strong> — clones the original speaker's voice. Slower, uses ~4GB VRAM.
              Best when matching the original speaker matters more than speed.{_google_note}
            </div>
            """)

            _tts_choices = ["Kokoro (fast, PT-BR native)", "Edge TTS (cloud, no key)", "XTTS v2 (voice clone)"]
            if GOOGLE_TTS_API_KEY:
                _tts_choices.append("Google Cloud TTS")

            tts_engine_input = gr.Radio(
                choices=_tts_choices,
                value="Kokoro (fast, PT-BR native)",
                label="TTS engine",
            )
            kokoro_voice_input = gr.Dropdown(
                choices=["pf_dora", "pm_alex", "pm_santa"],
                value="pf_dora",
                label="Kokoro voice",
                visible=True,
            )
            edge_voice_input = gr.Dropdown(
                choices=EDGE_TTS_PT_BR_VOICES,
                value=EDGE_TTS_VOICE_NAME,
                label="Edge TTS voice",
                visible=False,
            )

            # ── Google Cloud TTS controls (shown only when Google TTS is selected) ──
            _gtypes = list(GOOGLE_TTS_VOICE_CATALOG.keys())
            _gtype_default = (
                GOOGLE_TTS_VOICE_TYPE
                if GOOGLE_TTS_VOICE_TYPE in _gtypes
                else "Neural2"
            )
            _gvoices = list(GOOGLE_TTS_VOICE_CATALOG.get(_gtype_default, []))
            if GOOGLE_TTS_VOICE_NAME and GOOGLE_TTS_VOICE_NAME not in _gvoices:
                _gvoices.insert(0, GOOGLE_TTS_VOICE_NAME)
            if not _gvoices:
                _gvoices = [GOOGLE_TTS_VOICE_NAME or "pt-BR-Neural2-A"]
            _gvoice_default = (
                GOOGLE_TTS_VOICE_NAME
                if GOOGLE_TTS_VOICE_NAME in _gvoices
                else _gvoices[0]
            )

            with gr.Row(visible=False) as google_tts_row:
                google_voice_type_input = gr.Dropdown(
                    choices=_gtypes,
                    value=_gtype_default,
                    label="Google TTS · Voice type",
                    scale=1,
                )
                google_voice_input = gr.Dropdown(
                    choices=_gvoices,
                    value=_gvoice_default,
                    label="Google TTS · Voice name",
                    scale=2,
                )
                
                # Dynamic voice sample preview
                initial_sample_path = os.path.join("samples", f"{_gvoice_default}.wav")
                google_voice_sample = gr.Audio(
                    value=initial_sample_path if os.path.exists(initial_sample_path) else None,
                    label="Sample",
                    show_label=False,
                    interactive=False,
                    scale=1,
                    elem_id="google-voice-sample",
                )

            def _on_tts_engine_change(engine):
                is_kokoro = engine == "Kokoro (fast, PT-BR native)"
                is_edge   = engine == "Edge TTS (cloud, no key)"
                is_google = engine == "Google Cloud TTS"
                return (
                    gr.update(visible=is_kokoro),
                    gr.update(visible=is_edge),
                    gr.update(visible=is_google),
                )

            tts_engine_input.change(
                fn=_on_tts_engine_change,
                inputs=[tts_engine_input],
                outputs=[kokoro_voice_input, edge_voice_input, google_tts_row],
            )

            def _update_google_voice_sample(voice_name):
                if not voice_name:
                    return None
                sample_path = os.path.join("samples", f"{voice_name}.wav")
                if os.path.exists(sample_path):
                    return sample_path
                return None

            def _on_google_voice_type_change(voice_type):
                voices = list(GOOGLE_TTS_VOICE_CATALOG.get(voice_type, []))
                if not voices:
                    # Polyglot (Preview) or unknown — fall back to env-configured name
                    env_name = GOOGLE_TTS_VOICE_NAME or "pt-BR-Neural2-A"
                    voices = [env_name]
                
                new_voice = voices[0]
                new_sample = _update_google_voice_sample(new_voice)
                return gr.update(choices=voices, value=new_voice), new_sample

            google_voice_type_input.change(
                fn=_on_google_voice_type_change,
                inputs=[google_voice_type_input],
                outputs=[google_voice_input, google_voice_sample],
            )

            google_voice_input.change(
                fn=_update_google_voice_sample,
                inputs=[google_voice_input],
                outputs=[google_voice_sample],
            )

        gr.HTML('<div style="height:16px"></div>')

        run_btn = gr.Button("▶  DUB THIS VIDEO", elem_id="run-btn")

        gr.HTML('<div style="height:20px"></div>')
        gr.HTML('<div class="panel-label">03 · Progress</div>')
        log_output = gr.Textbox(
            label="Pipeline log",
            lines=14,
            max_lines=14,
            interactive=False,
            elem_id="log-box",
        )

        gr.HTML('<div style="height:4px"></div>')
        gr.HTML('<div class="panel-label">04 · Output</div>')
        video_output = gr.Video(label="Dubbed video (PT-BR)")

        gr.HTML('<div style="height:12px"></div>')
        gr.HTML('<div class="panel-label">05 · Subtitles</div>')
        gr.HTML("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#9494b2;margin:0 0 12px;line-height:1.7;">
          Generates an SRT from <code>translated.json</code> — no re-synthesis needed.<br>
          Short fragments are merged, minimum reading time is enforced (17 chars/sec),
          and long lines are wrapped to 2 lines for readability.
        </p>
        """)
        with gr.Row():
            srt_btn = gr.Button("📝  Generate / Download SRT", variant="secondary")
        with gr.Row():
            srt_file_output = gr.File(label="SRT subtitle file")
            srt_status = gr.Textbox(label="Status", lines=1, interactive=False)

        def _run_generate_srt(project_name):
            path, msg = generate_srt_for_project(project_name)
            return path, msg

        srt_btn.click(
            fn=_run_generate_srt,
            inputs=[project_name_input],
            outputs=[srt_file_output, srt_status],
        )

        gr.HTML("""
        <div style="text-align:center;margin-top:40px;padding-top:24px;border-top:1px solid #2a2a3e;">
          <p style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3a3a58;">
            XTTS v2 · Whisper · NLLB-200 / Gemini · yt-dlp · FFmpeg
          </p>
        </div>
        """)

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                url_input,
                video_upload_input,
                speaker_input,
                whisper_model_input,
                browser_input,
                cookies_file_input,
                project_name_input,
                resume_from_input,
                tts_engine_input,
                kokoro_voice_input,
                google_voice_type_input,
                google_voice_input,
                edge_voice_input,
            ],
            outputs=[video_output, log_output],
            show_progress="minimal",
        )

    return demo


if __name__ == "__main__":
    log_startup_info()
    demo = build_ui()
    demo.queue(max_size=3)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
        css=CSS,
    )
