# Dubweave 🎬

> YouTube → Brazilian Portuguese dubbing pipeline, fully local, GPU-accelerated.
>
> _Dedicado à Aline ❤️_

---

## Why I built this

My wife Aline speaks Portuguese, and I kept coming across English content — YouTube essays, documentaries, interviews — that I wanted to share with her but couldn't because of the language barrier. Subtitles help, but watching dubbed video together is a completely different experience: she can follow without reading, and we can just enjoy it side by side.

Beyond that, this project was a deliberate learning exercise. I wanted to understand, end-to-end, how modern AI pipelines actually fit together — not just call an API and get a result, but build the full stack myself: automatic speech recognition with Whisper, neural machine translation with NLLB-200, large-language-model translation with context windows, a text-to-speech voice-cloning system with XTTS v2, a lightweight native TTS model with Kokoro-82M, and media assembly with FFmpeg. Every stage taught something different — how to manage GPU memory across multiple models, how to handle timing constraints when synthesized speech is a different length than the original, how to make download pipelines resilient to platform countermeasures, and how to build a UI that exposes all of this without getting in the way.

The result is a fully local, privacy-preserving dubbing pipeline that runs on a consumer GPU and produces a watchable dubbed MP4 from any YouTube URL. It's personal software built for a personal reason, and that made it easy to care about the details.

---

## What it does

```text
YouTube URL
    ↓  yt-dlp (aria2c + Deno)  → downloads best video + audio track separately
    ↓  Whisper (GPU)            → transcribes English speech (large-v3-turbo)
    ↓  Segment Merging          → groups fragments into semantic utterances
    ↓  NLLB-200 / OpenRouter    → translates EN→PT-BR (local or LLM)
    ↓  PT-BR Norm               → 30+ rules: pronouns, gerunds, Brazilian vocab
    ↓  Timing Budget Pass       → predicts overflow, truncates or rephrases
    ↓  Kokoro-82M / XTTS v2     → synthesizes PT-BR speech (fast or voice-clone)
    ↓  FFmpeg + numpy buffer    → time-aligns, peak-normalizes, muxes with video
    ↓  Output MP4 + SRT         → dubbed video + optional subtitle track
```

---

## Requirements

| Requirement | Notes |
| --- | --- |
| Windows 11 | Tested on Windows 11 |
| Pixi | [pixi.sh](https://pixi.sh) — unified environment manager |
| NVIDIA GPU | 4+ GB VRAM (Kokoro default) · 8+ GB for XTTS v2 voice-clone mode |
| ~12 GB free disk | Models: XTTS v2 (~3.5 GB) + NLLB-200 (~2.4 GB) + Whisper (~1.5 GB) + Kokoro (~0.5 GB) |
| Internet (setup) | Downloads models, Deno, and YouTube JS solver script |
| espeak-ng 1.52 | Required for Kokoro phoneme processing (install from espeak-ng releases: <https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi>) |

---

## Setup (once)

1. Open PowerShell and install Pixi: `iwr -useb https://pixi.sh/install.ps1 | iex` (then close and reopen your terminal)
2. Double-click **`setup.bat`**
3. Wait for downloads (~10 min). This fetches:
   - **NLLB-200** (Distilled-600M) translation model
   - **XTTS v2** voice synthesis model
   - **Kokoro-82M** PT-BR TTS model
   - **Whisper large-v3-turbo** for transcription
   - **Deno** for resolving YouTube's _n-challenge_ (prevents download errors)

## Launch

Double-click **`start.bat`** → open <http://localhost:7860>

---

## Features

### Kokoro-82M TTS (New Default)

Dubweave now ships with **Kokoro-82M** as the default TTS engine — 82 million parameters, loads in under 2 seconds, uses under 500 MB of VRAM, and produces native PT-BR prosody with three built-in voices (`pf_dora` · `pm_alex` · `pm_santa`). No voice cloning overhead. Switch to XTTS v2 in the UI when you need speaker-matched voice cloning from the original audio.

### Project Management & Pipeline Resume

Every run is saved as a named project under `projects/`. You can resume from any stage — **download**, **transcribe**, **translate**, **synthesize**, or **assemble** — without rerunning earlier steps. The UI shows a live status badge for each completed stage, so you can see at a glance what's already been done.

### Coherent Translation with Context Window

Unlike segment-by-segment translation (Argos), Dubweave merges Whisper fragments into full semantic utterances before translating. When using OpenRouter (LLM mode), each request includes the 3 preceding translated utterances as a read-only context window to preserve pronoun references and narrative flow across chunk boundaries.

### PT-PT → PT-BR Normalizer

More than 30 regex-based substitution rules applied after every translation pass — NLLB or LLM. Covers pronouns (`tu` → `você`), verb paradigms (`estás` → `está`, `gostavas` → `gostava`), gerund construction (`a fazer` → `fazendo`), and vocabulary (`autocarro` → `ônibus`, `telemóvel` → `celular`, `miúdos` → `crianças`).

### Timing Budget Pass

Before synthesis, Dubweave estimates the output duration for each segment using a measured PT-BR speech-rate constant. Segments predicted to overflow their time slot are proactively shortened: first by trying a concise LLM rephrase (if OpenRouter is configured), then by hard-truncating to the last complete word that fits. This eliminates desync before it happens.

### LLM Translation via OpenRouter

If you have an **OpenRouter API key**, Gemini 2.0 Flash translates at ~$0.002 per 10-minute video with explicit PT-BR system instructions (Brazilian vocabulary, `você` paradigm, gerund forms). NLLB-200 local translation is used automatically as a fallback if the key is absent or the request fails.

### SRT Subtitle Export

After any run, click **Generate / Download SRT** to export a properly timed `.srt` subtitle file from the translated project — no re-synthesis needed. The file is also saved to `projects/<name>/outputs/`.

### Self-Healing Downloads

A 5-level video + 6-level audio format cascade ensures downloads complete even when YouTube's JS challenge solver is unavailable. If `aria2c` is in your PATH, Dubweave automatically enables 4-connection parallel downloading. Muxed format 18 (360p, never challenge-gated) is the final fallback.

### Cookie Authentication

Three-tier cookie strategy: **cookies.txt** (Netscape format, highest priority) → **browser auto-extraction** (Chrome · Firefox · Edge · Brave) → **anonymous**. Use a cookies.txt export for the most reliable authentication across age-restricted or member-only content.

---

## GPU Memory

| Configuration | VRAM (approx.) |
| --------------- | ---------------- |
| Whisper large-v3-turbo | ~2 GB |
| NLLB-200 | ~1.5 GB |
| **Kokoro-82M (default)** | **~0.5 GB** |
| **Total (Kokoro mode)** | **~4 GB** |
| XTTS v2 (voice-clone mode) | ~4 GB (replaces Kokoro) |
| **Total (XTTS v2 mode)** | **~7-8 GB** |

Kokoro mode runs comfortably on 6 GB cards (RTX 3060 Ti / RTX 4060+). XTTS v2 mode needs 8–12 GB (RTX 3060 12GB / RTX 4070+).

---

## License

Pipeline code: MIT. Models: XTTS v2 (Coqui PLM), NLLB (MIT/Apache 2.0), Whisper (MIT), Kokoro-82M (Apache 2.0).

---

## Acknowledgements

This project bundles and integrates several third-party tools and models. Thank you to the authors and maintainers of the following projects — please review each project's license before redistribution or commercial use.

- **espeak-ng** (v1.52) — compact, open-source text-to-speech kernel used for Kokoro phoneme processing. Releases: <https://github.com/espeak-ng/espeak-ng/releases/latest> · MSI installer: <https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi> · Repo: <https://github.com/espeak-ng/espeak-ng>
- **Kokoro-82M** — PT-BR TTS model (hexgrad/Kokoro-82M)
- **XTTS v2** — Coqui XTTS voice-cloning model
- **NLLB-200** — Facebook NLLB translation models
- **Whisper (large-v3 / large-v3-turbo)** — Open-source transcription model
- **yt-dlp** / **Deno** — video downloads and JS challenge resolution
- **FFmpeg** — audio/video processing and muxing
- **aria2c** — optional download accelerator
- **Pixi** — environment manager used for setup
- **OpenRouter** — optional LLM translation endpoint (user-supplied key)

License compliance note: the pipeline code is MIT. Each bundled model and tool carries its own license (Apache 2.0, MIT, Coqui PLM). If you redistribute binaries, model files, or installers, ensure you comply with each project's license and attribution requirements.
