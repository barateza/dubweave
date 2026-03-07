# Dubweave 🎬

> Any video → Brazilian Portuguese dubbing pipeline, fully local, GPU-accelerated.
>
> _Dedicado à Aline ❤️_

---

## Why I built this

My wife Aline speaks Portuguese, and I kept coming across English content — essays, documentaries, interviews — that I wanted to share with her but couldn't because of the language barrier. Subtitles help, but watching dubbed video together is a completely different experience: she can follow without reading, and we can just enjoy it side by side.

Beyond that, this project was a deliberate learning exercise. I wanted to understand, end-to-end, how modern AI pipelines actually fit together — not just call an API and get a result, but build the full stack myself: automatic speech recognition with Whisper, neural machine translation with NLLB-200, large-language-model translation with context windows, a text-to-speech voice-cloning system with XTTS v2, a lightweight native TTS model with Kokoro-82M, and media assembly with FFmpeg. Every stage taught something different — how to manage GPU memory across multiple models, how to handle timing constraints when synthesized speech is a different length than the original, how to make download pipelines resilient to platform countermeasures, and how to build a UI that exposes all of this without getting in the way.

The result is a fully local, privacy-preserving dubbing pipeline that runs on a consumer GPU and produces a watchable dubbed MP4 from any video source. Every stage taught something different: how to manage GPU memory across multiple models, how to handle timing constraints when synthesized speech is a different length than the original, how to make download and upload pipelines resilient, and how to build a UI that exposes all of this without getting in the way.

The result is a fully local, privacy-preserving dubbing pipeline that runs on a consumer GPU and produces a watchable dubbed MP4 from any video source. It's personal software built for a personal reason, and that made it easy to care about the details.

---

## What it does

```text
Any URL or local file upload
    ↓  yt-dlp / Local Ingest    → downloads from any site or uses local file
    ↓  Whisper (GPU)            → transcribes speech (large-v3-turbo)
    ↓  Segment Merging          → groups fragments into semantic utterances
    ↓  NLLB-200 / OpenRouter    → translates EN→PT-BR (local or LLM)
    ↓  PT-BR Norm               → 30+ rules: pronouns, gerunds, Brazilian vocab
    ↓  Timing Budget Pass       → predicts overflow, truncates or rephrases
    ↓  Kokoro / XTTS / Google   → synthesizes PT-BR speech (fast, clone, or cloud)
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

## Configuration

### Environment Variables (`.env`)

Dubweave reads configuration from a `.env` file in the project root. A default `.env` is created on first run with all settings pre-configured.

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | _(empty)_ | Optional LLM translation via OpenRouter (Gemini 2.0 Flash). Leave empty to use local NLLB-200 only. ~$0.002 per 10-min video. |
| `WHISPER_MODEL` | `large-v3-turbo` | Speech transcription model. Options: `large-v3-turbo` (4GB, faster), `large-v3` (10GB, more accurate) |
| `NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | Local translation model. Options: distilled-600M (2.4GB, fast) or 1.3B (5.4GB, accurate) |
| `KOKORO_VOICE` | `pf_dora` | Kokoro TTS voice. Options: `pf_dora` (F, energetic), `pm_alex` (M, natural), `pm_santa` (M, warm) |
| `KOKORO_SPEED` | `1.0` | Speech rate multiplier. `1.0` = natural, `>1.0` = faster (to fit tight synthesis slots) |
| `GOOGLE_TTS_API_KEY` | _(empty)_ | Optional Google Cloud TTS. Enable the API in Google Cloud Console. |
| `GOOGLE_TTS_VOICE_NAME` | `pt-BR-Neural2-A` | Default Google voice name (overridable in UI if key is set) |
| `GRADIO_SERVER_PORT` | `7860` | Web UI port. Change to `8000`, `8080`, etc. to avoid conflicts |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server host. `0.0.0.0` = network accessible, `127.0.0.1` = localhost only |
| `GRADIO_SHARE` | `false` | Enable public Gradio.live tunnel. Set to `true` for temporary sharing. |

### Setting up OpenRouter (Optional)

To enable LLM-based translation with better context awareness and PT-BR instructions:

1. Sign up at [openrouter.ai](https://openrouter.ai/sign-up) (free tier: $5/month credits)
2. Get your API key from the dashboard
3. Open `.env` and paste your key into `OPENROUTER_API_KEY=sk-or-v1-...`
4. Restart the app

When enabled, OpenRouter is used for translation first; NLLB-200 runs as fallback if the request fails or the key is empty. Cost: ~$0.002 for a 10-minute video.

### Customizing Models & Voice

Edit `.env` to change any setting. Examples:

```ini
# Use larger, more accurate Whisper model
WHISPER_MODEL=large-v3

# Switch to Kokoro male voice
KOKORO_VOICE=pm_alex

# Speed up synthesis to fit tight clips
KOKORO_SPEED=1.2

# Run web UI on port 8080
GRADIO_SERVER_PORT=8080

# Use more accurate (but slower) NLLB model
NLLB_MODEL=facebook/nllb-200-1.3B
```

Restart `start.bat` for changes to take effect.

---

## Features

### Kokoro-82M TTS (New Default)

Dubweave now ships with **Kokoro-82M** as the default TTS engine — 82 million parameters, loads in under 2 seconds, uses under 500 MB of VRAM, and produces native PT-BR prosody with three built-in voices (`pf_dora` · `pm_alex` · `pm_santa`). No voice cloning overhead. Switch to XTTS v2 in the UI when you need speaker-matched voice cloning from the original audio.

### Google Cloud TTS (Premium Cloud Option)

If you have a valid **Google Cloud TTS API Key**, Dubweave supports 40+ high-quality Brazilian Portuguese voices across 6 model families: **Chirp 3 HD**, **Neural2**, **WaveNet**, **Studio**, **Standard**, and **Polyglot (Preview)**. This provides the highest possible audio quality and naturalness for content where cloud dependencies are acceptable. Configuration is done via `.env`, but voice types and names can be changed directly in the UI if the key is active.

### Project Management & Pipeline Resume

Every run is saved as a named project under `projects/`. You can resume from any stage — **download**, **transcribe**, **translate**, **synthesize**, or **assemble** — without rerunning earlier steps. The UI shows a live status badge for each completed stage, so you can see at a glance what's already been done.

### Coherent Translation with Context Window

Unlike segment-by-segment translation (Argos), Dubweave merges Whisper fragments into full semantic utterances before translating. When using OpenRouter (LLM mode), each request includes the 3 preceding translated utterances as a read-only context window to preserve pronoun references and narrative flow across chunk boundaries.

### PT-PT → PT-BR Normalizer

More than 30 regex-based substitution rules applied after every translation pass — NLLB or LLM. Covers pronouns (`tu` → `você`), verb paradigms (`estás` → `está`, `gostavas` → `gostava`), gerund construction (`a fazer` → `fazendo`), and vocabulary (`autocarro` → `ônibus`, `telemóvel` → `celular`, `miúdos` → `crianças`).

### Timing Budget Pass

Before synthesis, Dubweave estimates the output duration for each segment using a measured PT-BR speech-rate constant. Segments predicted to overflow their time slot are proactively shortened: first by trying a concise LLM rephrase (if OpenRouter is configured), then by hard-truncating to the last complete word that fits. This eliminates desync before it happens.

### LLM Translation via OpenRouter

If you set an **OpenRouter API key** in `.env` (see [Configuration](#configuration) section), Gemini 2.0 Flash translates at ~$0.002 per 10-minute video with explicit PT-BR system instructions (Brazilian vocabulary, `você` paradigm, gerund forms). NLLB-200 local translation is used automatically as a fallback if the key is empty or the request fails.

### SRT Subtitle Export

After any run, click **Generate / Download SRT** to export a properly timed `.srt` subtitle file from the translated project — no re-synthesis needed. The file is also saved to `projects/<name>/outputs/`.

### Self-Healing Downloads

A 5-level video + 6-level audio format cascade ensures downloads complete even when YouTube's JS challenge solver is unavailable. If `aria2c` is in your PATH, Dubweave automatically enables 4-connection parallel downloading. Muxed format 18 (360p, never challenge-gated) is the final fallback.

### Cookie Authentication

Three-tier cookie strategy: **cookies.txt** (Netscape format, highest priority) → **browser auto-extraction** (Chrome · Firefox · Edge · Brave) → **anonymous**. Use a cookies.txt export for the most reliable authentication across age-restricted or member-only content.

---

## Accessibility

The Dubweave web UI meets **WCAG 2.1 Level AA** accessibility standards with enhanced contrast exceeding **AAA (7:1 ratio)**. All users, including those with disabilities, can:

- **Keyboard-only**: Tab through the entire UI, focus is always visible, no keyboard traps
- **Screen reader** (NVDA, JAWS, VoiceOver): All buttons, labels, and status updates are announced; skip link bypasses decorative header
- **High Contrast Mode** (Windows): Title and all text remain visible and readable
- **Motion sensitivity** (`prefers-reduced-motion`): Animations are disabled for users with vestibular disorders
- **Zoom to 200%**: All content remains accessible at high magnification

For detailed compliance information, see [WCAG 2.1 Accessibility Compliance](https://github.com/karpetrosyan/dubweave/blob/main/.specs/features/accessibility/WCAG-COMPLIANCE.md).

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

## Troubleshooting

### "espeak-ng not found" / Kokoro fails at startup

Kokoro TTS requires espeak-ng for phoneme processing.

1. Download and install the MSI: <https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi>
2. Close and reopen your terminal (or restart `start.bat`)
3. Verify: `espeak-ng --version` should print a version number

### "CUDA not available" / no GPU detected

1. Install or update NVIDIA drivers: <https://www.nvidia.com/Download/index.aspx>
2. Verify GPU is visible: run `nvidia-smi` in a terminal
3. Minimum VRAM: 4 GB (Kokoro mode), 8 GB (XTTS v2 mode)
4. Ensure `pytorch-cuda` version matches your driver (the `pixi.toml` pins `12.4`)

### "yt-dlp download fails" / 403 / PO token error

YouTube periodically changes its challenge system. Try these options in order:

1. **Use browser cookies** — select your browser in the "YouTube Account" accordion in the UI
2. **Use cookies.txt** — export cookies with the _Get cookies.txt LOCALLY_ browser extension and upload the file in the UI
3. **Re-run setup** to refresh the Deno EJS solver: double-click `setup.bat` again

### "Pixi not found" after installation

Close your terminal completely, reopen it, and try again. Pixi adds itself to `PATH` via a terminal restart, not just a profile reload.

### ".env file not found" / missing configuration

A `.env` is created automatically from `.env.example` on first run. If this doesn't happen:

```bat
copy .env.example .env
```

Open `.env` with a text editor and review all settings before running.

### OpenRouter API key invalid

- Key must start with `sk-or-`
- Obtain a key at <https://openrouter.ai/keys>
- The pipeline validates the key before starting; an invalid key stops execution immediately with a clear error

### Disk space errors

- Models require ~12 GB total; outputs can grow large for long videos
- Clean up the `outputs/` directory periodically
- The pipeline cleans up temporary work files automatically after each run

### App crashes mid-synthesis on a long video

- Switch from XTTS v2 to **Kokoro** (lower VRAM footprint)
- After a crash, resume from the `synthesize` stage in the UI — completed segment checkpoints are preserved in the project directory

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
