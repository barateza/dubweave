import os
import shutil
import platform
from pathlib import Path

from src.config import (
    __version__,
    WHISPER_MODEL,
    GOOGLE_TTS_API_KEY,
    ELEVENLABS_API_KEY,
    OPENROUTER_API_KEY,
    ROOT_DIR,
)
from src.utils.security import redact


def validate_environment() -> list[str]:
    """Check required tools at startup; return list of warning strings."""
    warnings_list: list[str] = []
    system = platform.system()

    if shutil.which("espeak-ng") is None:
        if system == "Darwin":
            hint = "Install with: brew install espeak-ng"
        elif system == "Windows":
            hint = (
                "Install from: https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi "
                "then restart your terminal."
            )
        else:
            hint = "Install via your package manager (e.g. apt install espeak-ng)"
        warnings_list.append(
            f"⚠️  espeak-ng not found — Kokoro TTS will fail. {hint}"
        )

    if shutil.which("ffmpeg") is None:
        if system == "Darwin":
            hint = "Install with: brew install ffmpeg"
        elif system == "Windows":
            hint = "Run setup.bat to install all dependencies."
        else:
            hint = "Install via your package manager (e.g. apt install ffmpeg)"
        warnings_list.append(
            f"⚠️  ffmpeg not found — video assembly will fail. {hint}"
        )

    if shutil.which("ffprobe") is None:
        warnings_list.append(
            "⚠️  ffprobe not found — timing detection will fail. "
            "Ensure ffmpeg is installed (ffprobe is bundled with it)."
        )

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        warnings_list.append(
            "⚠️  onnxruntime not installed — Supertonic TTS will fail."
        )

    try:
        import supertonic  # noqa: F401
    except ImportError:
        warnings_list.append(
            "⚠️  supertonic not installed — Supertonic local TTS is unavailable."
        )

    try:
        import torch

        if not torch.cuda.is_available():
            if system == "Darwin":
                # macOS Apple Silicon — CUDA never available; check MPS instead
                if torch.backends.mps.is_available():
                    pass  # MPS is available, no need to warn
                else:
                    warnings_list.append(
                        "⚠️  MPS (Metal GPU) not available — running CPU only. "
                        "Ensure you're on Apple Silicon with macOS 12.3+ and PyTorch >= 1.12."
                    )
            else:
                warnings_list.append(
                    "⚠️  CUDA not available — GPU acceleration disabled. "
                    "Ensure NVIDIA drivers are installed and `nvidia-smi` shows your GPU."
                )
    except ImportError:
        warnings_list.append(
            "⚠️  PyTorch not installed — GPU acceleration unavailable."
        )

    env_path = ROOT_DIR / ".env"
    env_example_path = ROOT_DIR / ".env.example"
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

    try:
        from src.config import SUPERTONIC_ASSETS_DIR

        if SUPERTONIC_ASSETS_DIR:
            assets_path = Path(SUPERTONIC_ASSETS_DIR)
            if not assets_path.exists():
                warnings_list.append(
                    f"⚠️  SUPERTONIC_ASSETS_DIR does not exist: {assets_path}. Supertonic auto-download may still work if enabled."
                )
    except Exception:
        pass

    return warnings_list


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


def log_startup_info() -> None:
    """Print environment diagnostics to stdout at application startup."""
    print(f"[startup] Dubweave v{__version__} starting")
    print(
        f"[startup] Python {platform.python_version()} on {platform.system()} {platform.release()}"
    )
    try:
        import torch

        if torch.cuda.is_available():
            import torch.version as _torch_version

            print(
                f"[startup] CUDA {_torch_version.cuda} — {torch.cuda.get_device_name(0)}"
            )
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[startup] VRAM: {vram_gb:.1f} GB")
        else:
            mps_info = ""
            if torch.backends.mps.is_available():
                mps_info = " — MPS (Metal GPU) available"
            print(f"[startup] CUDA not available — running CPU only{mps_info}")
    except ImportError:
        print("[startup] PyTorch not installed")

    print(f"[startup] Whisper model: {WHISPER_MODEL}")
    tts_engines: list[str] = []
    try:
        from TTS.api import TTS  # noqa: F401

        tts_engines.append("XTTS v2")
    except ImportError:
        pass
    try:
        from kokoro import KPipeline  # noqa: F401

        tts_engines.append("Kokoro")
    except ImportError:
        pass
    try:
        import supertonic  # noqa: F401

        tts_engines.append("Supertonic")
    except ImportError:
        pass
    if GOOGLE_TTS_API_KEY:
        tts_engines.append("Google Cloud TTS")
    if ELEVENLABS_API_KEY:
        tts_engines.append("ElevenLabs TTS")
    print(f"[startup] TTS engines available: {', '.join(tts_engines)}")
    if OPENROUTER_API_KEY:
        print(f"[startup] OpenRouter: configured ({redact(OPENROUTER_API_KEY)})")
    else:
        print("[startup] OpenRouter: not configured (local NLLB-200 only)")

    print(f"[startup] Translation: {translation_engine}", logs) if False else None  # placeholder
    env_warnings = validate_environment()
    for w in env_warnings:
        print(f"[startup] {w}")
