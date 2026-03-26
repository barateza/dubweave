import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast
from src.config import WORK_DIR
from src.utils.helpers import log
from src.core.translate import PipelineError

_YT_URL_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?[^\s]*v=[A-Za-z0-9_-]{11}"
    r"|youtu\.be/[A-Za-z0-9_-]{11}"
    r"|youtube\.com/shorts/[A-Za-z0-9_-]{11})"
)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".ts", ".m4v"}

def validate_video_source(url: str, upload_path: str | None) -> tuple[bool, str]:
    has_file = bool(upload_path and upload_path.strip() and Path(upload_path.strip()).is_file())
    has_url = bool(url and url.strip())
    if has_file: return True, "file"
    if has_url: return True, "url"
    return False, "No video source provided. Paste a URL or upload a video file."

def ingest_local_file(upload_path: str, job_dir: Path, logs: list) -> tuple[Path, Path, str, float, list]:
    src = Path(upload_path.strip())
    log(f"📂 Ingesting uploaded file: {src.name}", logs)
    if not src.exists(): raise PipelineError("Ingest", f"Uploaded file not found: {src}")
    if src.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise PipelineError("Ingest", f"Unsupported file type '{src.suffix}'.")
    
    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio_orig.wav"
    title, duration = src.stem, 0.0
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)], capture_output=True, text=True)
        if probe.returncode == 0: duration = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception as e: log(f"⚠️  Could not probe duration: {e}", logs)

    if src.suffix.lower() == ".mp4": shutil.copy2(str(src), str(video_path))
    else:
        log(f"   Re-encoding {src.suffix} → mp4…", logs)
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(video_path)], check=True)

    log("   Extracting audio…", logs)
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(audio_path)], check=True)
    log(f'✅ Ingested: "{title}" ({duration:.0f}s)', logs)
    return video_path, audio_path, title, duration, logs

def download_video(url: str, job_dir: Path, logs: list, browser: str = "none", cookies_file: str | None = None):
    import yt_dlp as yt
    log("📥 Downloading video with yt-dlp…", logs)
    video_path, audio_path = job_dir / "video.mp4", job_dir / "audio_orig.wav"
    
    _aria2c = shutil.which("aria2c")
    _deno = shutil.which("deno")
    
    BASE_OPTS: dict[str, Any] = {
        "outtmpl": str(job_dir / "%(id)s_%(format_id)s.%(ext)s"),
        "quiet": True, "no_warnings": False,
        **( {"cookiefile": cookies_file} if cookies_file else ({"cookiesfrombrowser": (browser, None, None, None)} if browser != "none" else {}) ),
        **({"remote_components": ["ejs:github"]} if _deno else {}),
        **({ "external_downloader": "aria2c", "external_downloader_args": {"aria2c": ["--rpc-save-upload-metadata=false", "--file-allocation=none", "--optimize-concurrent-downloads=true", "--max-connection-per-server=4", "--min-split-size=5M", "--split=4", "--max-tries=5", "--retry-wait=3"]} } if _aria2c else {}),
    }

    title, duration = "video", 0
    try:
        with yt.YoutubeDL({**BASE_OPTS, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title, duration = info.get("title", "video"), info.get("duration", 0)
    except Exception as e: log(f"⚠️  Probe failed ({e}), continuing anyway…", logs)

    VIDEO_FORMATS = ["bestvideo[ext=mp4]", "bestvideo", "best[ext=mp4]", "best", "18"]
    muxed_fallback, raw_video_file = False, None
    for fmt in VIDEO_FORMATS:
        try:
            with yt.YoutubeDL({**BASE_OPTS, "format": fmt, "outtmpl": str(job_dir / "video_raw.%(ext)s")}) as ydl: ydl.extract_info(url, download=True)
            candidates = list(job_dir.glob("video_raw.*"))
            if candidates:
                raw_video_file = candidates[0]
                muxed_fallback = fmt in ("best[ext=mp4]", "best", "18")
                break
        except Exception: continue
    if not raw_video_file: raise RuntimeError("All video formats failed.")
    shutil.move(str(raw_video_file), str(video_path))

    if muxed_fallback:
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(audio_path)], check=True)
    else:
        AUDIO_FORMATS = ["bestaudio[ext=m4a]", "bestaudio[ext=webm]", "bestaudio", "140", "139"]
        raw_audio_file = None
        for fmt in AUDIO_FORMATS:
            try:
                with yt.YoutubeDL({**BASE_OPTS, "format": fmt, "outtmpl": str(job_dir / "audio_raw.%(ext)s"), "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}]}) as ydl: ydl.extract_info(url, download=True)
                candidates = list(job_dir.glob("audio_raw.*"))
                if candidates:
                    raw_audio_file = candidates[0]
                    break
            except Exception: continue
        if not raw_audio_file:
            subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(audio_path)], check=True)
        else: shutil.move(str(raw_audio_file), str(audio_path))

    log(f'✅ Downloaded: "{title}" ({duration}s)', logs)
    return video_path, audio_path, title, duration, logs
