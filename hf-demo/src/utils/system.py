import os
import shutil
import platform
from pathlib import Path

from src.config import (
    __version__,
    WHISPER_MODEL,
    OPENROUTER_API_KEY,
    ROOT_DIR,
)
from src.utils.security import redact


def validate_environment(demo_mode: bool = False) -> list[str]:
    """Check required tools at startup; return list of warning strings."""
    warnings_list: list[str] = []

    if shutil.which("ffmpeg") is None:
        warnings_list.append(
            "⚠️  ffmpeg not found — video assembly will fail. "
            "Ensure ffmpeg is installed and on your system PATH."
        )

    if shutil.which("ffprobe") is None:
        warnings_list.append(
            "⚠️  ffprobe not found — timing detection will fail. "
            "Ensure ffmpeg is installed (ffprobe is bundled with it)."
        )

    try:
        import torch
    except ImportError:
        warnings_list.append(
            "⚠️  PyTorch not installed — GPU/CPU acceleration unavailable."
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
    print(f"[startup] Dubweave Demo v{__version__} starting")
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
            print("[startup] CUDA not available — running CPU only")
    except ImportError:
        print("[startup] PyTorch not installed")

    print(f"[startup] Whisper model: {WHISPER_MODEL}")
    print("[startup] TTS engines available: Edge TTS")
    if OPENROUTER_API_KEY:
        print(f"[startup] OpenRouter: configured ({redact(OPENROUTER_API_KEY)})")
    else:
        print("[startup] OpenRouter: not configured (local Helsinki-NLP only)")

    env_warnings = validate_environment(demo_mode=True)
    for w in env_warnings:
        print(f"[startup] {w}")
