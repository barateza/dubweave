import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from src.config import WORK_DIR
from src.utils.helpers import log
from src.core.translate import PipelineError

_YT_URL_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?[^\s]*v=[A-Za-z0-9_-]{11}"
    r"|youtu\.be/[A-Za-z0-9_-]{11}"
    r"|youtube\.com/shorts/[A-Za-z0-9_-]{11})"
)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".ts", ".m4v"}


def _cookie_opts(browser: str, cookies_file: str | None, logs: list) -> dict[str, Any]:
    if cookies_file:
        cookie_path = Path(cookies_file).expanduser()
        if cookie_path.is_file():
            return {"cookiefile": str(cookie_path)}
        log(f"⚠️  cookies.txt not found: {cookies_file}. Falling back to browser/anonymous session.", logs)

    if browser != "none":
        return {"cookiesfrombrowser": (browser, None, None, None)}
    return {}


def _build_yt_download_profiles(job_dir: Path, cookie_opts: dict[str, Any], has_aria2c: bool, has_deno: bool) -> list[tuple[str, dict[str, Any]]]:
    shared_opts: dict[str, Any] = {
        "outtmpl": str(job_dir / "%(id)s_%(format_id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 3,
        "extractor_retries": 3,
        "sleep_interval_requests": 0.25,
        **cookie_opts,
        **({"remote_components": ["ejs:github"]} if has_deno else {}),
    }

    aria2_opts: dict[str, Any] = {}
    if has_aria2c:
        aria2_opts = {
            "external_downloader": "aria2c",
            "external_downloader_args": {
                "aria2c": [
                    "--rpc-save-upload-metadata=false",
                    "--file-allocation=none",
                    "--optimize-concurrent-downloads=true",
                    "--max-connection-per-server=4",
                    "--min-split-size=5M",
                    "--split=4",
                    "--max-tries=5",
                    "--retry-wait=3",
                ]
            },
        }

    yt_client_fallback = {
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "tv", "web"],
            }
        }
    }

    profiles: list[tuple[str, dict[str, Any]]] = []
    if has_aria2c:
        profiles.append(("aria2c", {**shared_opts, **aria2_opts}))
    profiles.append(("builtin-http", {**shared_opts}))
    profiles.append(("youtube-client-fallback", {**shared_opts, **yt_client_fallback}))
    profiles.append(("youtube-client-ipv4", {**shared_opts, **yt_client_fallback, "forceipv4": True}))

    # If authenticated mode fails (stale cookies / locked DB), try one anonymous profile.
    if cookie_opts:
        anon_shared = {k: v for k, v in shared_opts.items() if k not in {"cookiefile", "cookiesfrombrowser"}}
        profiles.append(("anonymous-ipv4", {**anon_shared, **yt_client_fallback, "forceipv4": True}))

    return profiles


def _summarize_exc(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:180] if len(text) > 180 else text

def get_video_metadata(url: str, upload_path: str | None) -> dict:
    """Quickly fetch duration and title without full download/ingest."""
    meta = {"title": "Unknown", "duration": 0.0}
    
    if upload_path and Path(upload_path).is_file():
        src = Path(upload_path)
        meta["title"] = src.stem
        try:
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)], capture_output=True, text=True)
            if probe.returncode == 0:
                meta["duration"] = float(json.loads(probe.stdout)["format"]["duration"])
        except Exception: pass
        return meta

    if url and url.strip():
        import yt_dlp as yt
        try:
            with yt.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                meta["title"] = info.get("title", "YouTube Video")
                meta["duration"] = float(info.get("duration", 0.0))
        except Exception: pass
        return meta

    return meta

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

    has_aria2c = bool(shutil.which("aria2c"))
    has_deno = bool(shutil.which("deno"))
    cookies = _cookie_opts(browser=browser, cookies_file=cookies_file, logs=logs)
    profiles = _build_yt_download_profiles(job_dir=job_dir, cookie_opts=cookies, has_aria2c=has_aria2c, has_deno=has_deno)

    if has_aria2c:
        log("   aria2c detected: enabling fast profile with automatic fallback to built-in downloader.", logs)

    title, duration = "video", 0
    try:
        with yt.YoutubeDL({**profiles[0][1], "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title, duration = info.get("title", "video"), info.get("duration", 0)
    except Exception as e: log(f"⚠️  Probe failed ({e}), continuing anyway…", logs)

    VIDEO_FORMATS = ["bestvideo[ext=mp4]", "bestvideo", "best[ext=mp4]", "best", "18"]
    muxed_fallback, raw_video_file = False, None
    selected_profile_idx = 0
    video_errors: list[str] = []
    for profile_idx, (profile_name, profile_opts) in enumerate(profiles):
        if profile_idx > 0:
            log(f"   Retrying with profile: {profile_name}", logs)

        for fmt in VIDEO_FORMATS:
            for stale in job_dir.glob("video_raw.*"):
                stale.unlink(missing_ok=True)

            try:
                with yt.YoutubeDL({**profile_opts, "format": fmt, "outtmpl": str(job_dir / "video_raw.%(ext)s")}) as ydl:
                    ydl.extract_info(url, download=True)

                candidates = list(job_dir.glob("video_raw.*"))
                if candidates:
                    raw_video_file = candidates[0]
                    muxed_fallback = fmt in ("best[ext=mp4]", "best", "18")
                    selected_profile_idx = profile_idx
                    break
            except Exception as e:
                msg = _summarize_exc(e)
                if msg and msg not in video_errors:
                    video_errors.append(msg)
                continue

        if raw_video_file:
            break

    if not raw_video_file:
        hint = " Use Browser Cookies or a fresh cookies.txt if the video is age/member restricted." if any("403" in e for e in video_errors) else ""
        summary = f" Last errors: {' | '.join(video_errors[:3])}." if video_errors else ""
        raise RuntimeError(f"All video formats failed.{summary}{hint}")
    shutil.move(str(raw_video_file), str(video_path))

    if muxed_fallback:
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(audio_path)], check=True)
    else:
        AUDIO_FORMATS = ["bestaudio[ext=m4a]", "bestaudio[ext=webm]", "bestaudio", "140", "139"]
        raw_audio_file = None
        audio_errors: list[str] = []

        ordered_audio_profiles = profiles[selected_profile_idx:] + profiles[:selected_profile_idx]
        for profile_idx, (profile_name, profile_opts) in enumerate(ordered_audio_profiles):
            if profile_idx > 0:
                log(f"   Audio retry with profile: {profile_name}", logs)

            for fmt in AUDIO_FORMATS:
                for stale in job_dir.glob("audio_raw.*"):
                    stale.unlink(missing_ok=True)

                try:
                    with yt.YoutubeDL({
                        **profile_opts,
                        "format": fmt,
                        "outtmpl": str(job_dir / "audio_raw.%(ext)s"),
                        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}],
                    }) as ydl:
                        ydl.extract_info(url, download=True)

                    candidates = list(job_dir.glob("audio_raw.*"))
                    if candidates:
                        raw_audio_file = candidates[0]
                        break
                except Exception as e:
                    msg = _summarize_exc(e)
                    if msg and msg not in audio_errors:
                        audio_errors.append(msg)
                    continue

            if raw_audio_file:
                break

        if not raw_audio_file:
            if audio_errors:
                log(f"⚠️  Audio stream download failed, extracting from muxed video instead. ({audio_errors[0]})", logs)
            subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(audio_path)], check=True)
        else: shutil.move(str(raw_audio_file), str(audio_path))

    log(f'✅ Downloaded: "{title}" ({duration}s)', logs)
    return video_path, audio_path, title, duration, logs
