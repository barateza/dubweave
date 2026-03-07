# Project State

**Last Updated**: 2026-03-06
**Current Work**: Production readiness planning

## Decisions

- **D1**: Keep single-file architecture (app.py) for v1 — refactor is out of scope
- **D2**: Windows-only for v1 — Linux/macOS support deferred
- **D3**: No CI/CD for v1 — local pytest only
- **D4**: Four high-severity fixes already merged (checkpointing, language detection, validation, TTS fallback)

## Blockers

_None currently._

## Lessons Learned

- OpenRouter response validation is critical — LLMs return variable-length arrays that can silently corrupt segment alignment
- Intra-stage checkpointing in TTS synthesis saved hours of re-synthesis on long videos
- Google TTS API key validation must happen before synthesis starts, not mid-pipeline

## Deferred Ideas

- Module extraction: split app.py into `download.py`, `transcribe.py`, `translate.py`, `synthesize.py`, `assemble.py`
- Rate-limited concurrent segment synthesis
- Progress bar with ETA (requires timing calibration data)
- SRT style options (font size, position, color)
