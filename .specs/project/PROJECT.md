# Dubweave

**Vision:** A production-ready, fully local YouTube → Brazilian Portuguese dubbing pipeline that runs on a consumer GPU and produces watchable dubbed MP4s from any YouTube URL.

**For:** The project author (personal use) and potential open-source contributors.

**Solves:** Language barrier for sharing English YouTube content with Portuguese-speaking family, while serving as a learning exercise for end-to-end AI pipeline engineering.

## Goals

- [x] Core dubbing pipeline (Download → Transcribe → Translate → Synthesize → Assemble)
- [x] Multiple TTS engines (Kokoro, XTTS v2, Google Cloud TTS)
- [x] Resume-capable project system with stage persistence
- [x] High-severity robustness fixes (checkpointing, language detection, validation, TTS fallback)
- [ ] Production-ready documentation, error handling, security, performance, testing, and deployment

## Tech Stack

**Core:**

- Language: Python 3.10–3.11
- UI: Gradio (web interface on localhost:7860)
- Environment: Pixi (conda + PyPI unified manager)
- GPU: CUDA 12.4 (PyTorch)

**Key dependencies:** Whisper (ASR), NLLB-200 (translation), XTTS v2 / Kokoro / Google Cloud TTS (synthesis), yt-dlp (download), FFmpeg (media assembly), OpenRouter (LLM translation)

## Scope

**v1 (production release) includes:**

- Comprehensive README and setup documentation
- Robust error handling with user-friendly messages
- API key validation and secure configuration
- Performance guardrails for long videos
- Basic test coverage for core functions
- Production logging configuration

**Explicitly out of scope:**

- Multi-language target support (PT-BR only)
- Cloud deployment / containerization (local-only for v1)
- Multi-user concurrent access
- Automated CI/CD pipeline
- Web-accessible public deployment

## Constraints

- Single-file architecture (app.py ~2,720 lines) — refactoring into modules is desirable but not required for v1
- Windows-only (tested on Windows 11 with NVIDIA GPU)
- CUDA-dependent — no CPU-only fallback planned
