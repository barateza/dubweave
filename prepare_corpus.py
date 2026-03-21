"""
prepare_corpus.py — Dubweave autoresearch corpus prep
------------------------------------------------------
Downloads audio for each video in VIDEOS, runs Whisper large-v3-turbo,
and saves the raw segments list to corpus/<slug>_whisper.json.

Usage (from dubweave project root):
    pixi run python prepare_corpus.py

Requirements: yt-dlp, aria2c, whisper — all present in the pixi env.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import whisper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VIDEOS = [
    ("how_to_speak",     "https://www.youtube.com/watch?v=Unzc731iCUY"),
    ("nms_big_problem",    "https://www.youtube.com/watch?v=upOUMD7cDtA"),
    ("pixar_cars",     "https://www.youtube.com/watch?v=RctdcjUMn0w"),
    ("the_seven_warframe",          "https://www.youtube.com/watch?v=9S3fkggRzqM"),
    ("canada_mayor",         "https://www.youtube.com/watch?v=nACJOKV_YYA"),
]

WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
CORPUS_DIR     = Path("corpus")
AUDIO_FORMAT   = "m4a"   # fast, lossless enough for Whisper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_audio(url: str, out_path: Path) -> None:
    """Download best audio track to out_path using yt-dlp + aria2c."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bestaudio[ext=m4a]/bestaudio/best",
        "--extract-audio",
        "--audio-format", AUDIO_FORMAT,
        "--audio-quality", "0",
        "--downloader", "aria2c",
        "--downloader-args", "aria2c:-x4 -s4 -k1M",
        "--output", str(out_path),
        "--no-overwrites",
        url,
    ]
    print(f"  Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # yt-dlp already printed progress; surface the error
        print(result.stderr[-800:] if result.stderr else "(no stderr)")
        raise RuntimeError(f"yt-dlp failed for {url}")


def transcribe(audio_path: Path, model) -> list[dict]:
    """Run Whisper and return the segments list."""
    print(f"  Transcribing: {audio_path.name}")
    result = model.transcribe(str(audio_path), language="en", verbose=False)
    # Keep only the fields benchmark.py uses: start, end, text
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in result["segments"]
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CORPUS_DIR.mkdir(exist_ok=True)

    print(f"\nLoading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)
    print("Model loaded.\n")

    for slug, url in VIDEOS:
        json_path  = CORPUS_DIR / f"{slug}_whisper.json"
        audio_path = CORPUS_DIR / f"{slug}.{AUDIO_FORMAT}"

        # --- Skip if already done ---
        if json_path.exists():
            segs = json.loads(json_path.read_text())
            print(f"[skip] {slug}  ({len(segs)} segments already in {json_path.name})")
            continue

        print(f"\n[{slug}]")

        # --- Download ---
        try:
            download_audio(url, audio_path)
        except RuntimeError as exc:
            print(f"  ERROR: {exc} — skipping.")
            continue

        if not audio_path.exists():
            print(f"  ERROR: audio file not found after download — skipping.")
            continue

        # --- Transcribe ---
        try:
            segments = transcribe(audio_path, model)
        except Exception as exc:
            print(f"  ERROR during transcription: {exc} — skipping.")
            continue

        # --- Save ---
        json_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved {len(segments)} segments → {json_path.name}")

        # --- Clean up audio (keep only the JSON) ---
        audio_path.unlink(missing_ok=True)
        print(f"  Audio removed.")

    # --- Summary ---
    done = sorted(CORPUS_DIR.glob("*_whisper.json"))
    print(f"\nCorpus ready: {len(done)}/{len(VIDEOS)} files in {CORPUS_DIR}/")
    for p in done:
        segs = json.loads(p.read_text())
        dur  = segs[-1]["end"] if segs else 0
        print(f"  {p.name:40s}  {len(segs):4d} segs  {dur/60:.1f} min")

    if len(done) < len(VIDEOS):
        sys.exit(1)


if __name__ == "__main__":
    main()