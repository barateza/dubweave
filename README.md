# Dubweave 🎬

> YouTube → Brazilian Portuguese dubbing pipeline, fully local, GPU-accelerated.

---

## What it does

```
YouTube URL
    ↓  yt-dlp          → downloads video (MP4) + audio (WAV) separately
    ↓  Whisper (GPU)   → transcribes English speech with timestamps
    ↓  Argos Translate → translates EN→PT-BR entirely offline
    ↓  XTTS v2 (GPU)   → synthesizes PT-BR speech cloning original speaker's voice
    ↓  FFmpeg          → time-aligns each audio segment, mixes, muxes with video
    ↓  Output MP4      → dubbed video, ready to watch
```

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Windows 11  | Tested on Win 11 |
| Pixi        | [pixi.sh](https://pixi.sh) — package manager |
| NVIDIA RTX 4070 Super | CUDA 12.1 drivers |
| ~10 GB free disk | XTTS v2 model (~3.5 GB) + Whisper medium (~1.5 GB) |
| Internet (first run) | To download models and language pack |

---

## Setup (once)

1. Open PowerShell and install Pixi: `iwr -useb https://pixi.sh/install.ps1 | iex` (then close and reopen your terminal)
2. Double-click **`setup.bat`**
3. Wait ~10 min for model downloads (3.5 GB XTTS v2 + Whisper medium)

## Launch

Double-click **`start.bat`** → open http://localhost:7860

---

## Usage

1. Paste a YouTube URL
2. Optionally upload a voice reference WAV (10–30 seconds of the voice you want to clone)
   - Leave empty → auto-clones from the video's own speaker (first 30s)
3. Click **DUB THIS VIDEO**
4. Watch the log stream; output MP4 appears when done

---

## Voice Quality

**XTTS v2** is Coqui's state-of-the-art multilingual TTS model:
- Runs fully on your RTX 4070 Super (CUDA)
- Clones a voice from a short reference clip (10–30s)
- Supports Brazilian Portuguese natively (`pt` language code)
- Produces near-human quality speech

With voice cloning from the original speaker, the dubbed voice will sound like the original person speaking Portuguese — not a generic TTS voice.

---

## Translation

**Argos Translate** runs 100% locally — no API key, no cost, no data leaving your machine. Quality is good for most spoken content. For premium quality, you can swap to DeepL API (free tier: 500k chars/month):

```python
# In app.py, replace translate_segments() body:
import deepl
translator = deepl.Translator("YOUR_DEEPL_API_KEY")
result = translator.translate_text(text, target_lang="PT-BR")
```

---

## Timing / Sync

Speech synthesis produces audio at a natural pace, which may differ from the original timing. The pipeline uses FFmpeg `atempo` filter to stretch or compress each segment to match the original duration. This keeps lips/actions roughly in sync. Not perfect — professional dubbing tools use more sophisticated phoneme alignment — but it's very watchable.

---

## GPU Memory

| Model | VRAM |
|-------|------|
| XTTS v2 | ~3–4 GB |
| Whisper medium | ~2 GB |
| **Total** | ~5–6 GB |

RTX 4070 Super has 12 GB VRAM — plenty of headroom.

---

## Troubleshooting

**CUDA out of memory** → Run Whisper and XTTS sequentially (default). Close other GPU apps.

**Argos language pack error** → Run setup.bat again; it re-downloads the pack.

**yt-dlp download fails** → Update yt-dlp: open terminal, run `pixi update yt-dlp`

---

## License

Pipeline code: MIT. Models: XTTS v2 (Coqui Public Model License), Whisper (MIT), Argos Translate (MIT).
