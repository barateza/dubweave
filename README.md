# Dubweave 🎬

> YouTube → Brazilian Portuguese dubbing pipeline, fully local, GPU-accelerated.

---

## What it does

```
YouTube URL
    ↓  yt-dlp (Deno)   → downloads best video + audio track separately
    ↓  Whisper (GPU)   → transcribes English speech (large-v3-turbo)
    ↓  Segment Merging → groups fragments into semantic utterances
    ↓  NLLB-200 (GPU)  → translates EN→PT-BR 100% locally
    ↓  PT-BR Norm      → normalizes European Portuguese markers (tu/te/tuas)
    ↓  XTTS v2 (GPU)   → synthesizes PT-BR speech cloning original speaker's voice
    ↓  FFmpeg          → time-aligns each audio segment, mixes, muxes with video
    ↓  Output MP4      → dubbed video, ready to watch
```

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Windows 11  | Tested on Windows 11 |
| Pixi        | [pixi.sh](https://pixi.sh) — unified environment manager |
| NVIDIA GPU  | 8+ GB VRAM recommended (tested on RTX 4070 Super) |
| ~12 GB free disk | Models: XTTS v2 (~3.5 GB) + NLLB-200 (~2.4 GB) + Whisper large-v3-turbo (~1.5 GB) |
| Internet (setup) | Downloads models, Deno, and YouTube JS solver script |

---

## Setup (once)

1. Open PowerShell and install Pixi: `iwr -useb https://pixi.sh/install.ps1 | iex` (then close and reopen your terminal)
2. Double-click **`setup.bat`**
3. Wait for downloads (~10 min). This fetches:
   - **NLLB-200** (Distilled-600M) translation model
   - **XTTS v2** voice synthesis model
   - **Whisper large-v3-turbo** for transcription
   - **Deno** for resolving YouTube's *n-challenge* (prevents download errors)

## Launch

Double-click **`start.bat`** → open http://localhost:7860

---

## Features

### Coherent Translation
Unlike segment-by-segment translation (Argos), Dubweave merges Whisper fragments into full semantic utterances before translating. This ensures pronouns and context remain consistent.

### PT-PT → PT-BR Normalizer
Most open-source translation models (NLLB) often default to European Portuguese. Dubweave includes a post-processing layer that automatically converts "tu/te/teus" to "você/seu" and fixes gerund forms (e.g., "a fazer" → "fazendo").

### Fallback Translation
If you have an **OpenRouter API Key**, you can enable higher-quality LLM translation (e.g., Gemini 2.0 Flash) in the UI. It uses a 3-utterance sliding context window to maintain narrative flow.

### Self-Healing Downloads
Uses a format cascade via `yt-dlp`. If a high-quality DASH stream is throttled or fails JS challenge solvability, it automatically drops to lower but guaranteed muxed formats (including format 18).

---

## GPU Memory

| Model | VRAM (approx.) |
|-------|------|
| Whisper large-v3-turbo | ~2 GB |
| NLLB-200 | ~1.5 GB |
| XTTS v2 | ~4 GB |
| **Total Overhead** | **~7-8 GB** |

Runs comfortably on 12GB cards (RTX 3060/4070+).

---

## License

Pipeline code: MIT. Models: XTTS v2 (Coqui PLM), NLLB (MIT/Apache 2.0), Whisper (MIT).
