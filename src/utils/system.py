import os
import shutil
import platform
from pathlib import Path
from src.config import __version__, WHISPER_MODEL, GOOGLE_TTS_API_KEY, ROOT_DIR
from src.utils.security import redact

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
        print(f"[startup] OpenRouter: configured ({redact(openrouter_key)})")
    else:
        print("[startup] OpenRouter: not configured (local NLLB-200 only)")

    env_warnings = validate_environment()
    for w in env_warnings:
        print(f"[startup] {w}")
