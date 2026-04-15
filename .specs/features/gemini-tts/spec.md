# Feature Spec - Gemini 3.1 Flash TTS Preview

Status: Draft
Date: 2026-04-15

## Requirements

- GEM-001: System must expose Gemini 3.1 Flash TTS Preview as a selectable TTS engine.
- GEM-002: System must support single-speaker synthesis with selectable prebuilt voice.
- GEM-003: System must support two-speaker synthesis with explicit speaker mapping controls.
- GEM-004: System must validate Gemini API key before synthesis starts.
- GEM-005: System must estimate Gemini costs using published standard and batch rates.
- GEM-006: Estimator must show one final Gemini price and choose cheapest in auto mode.
- GEM-007: Gemini key values must be redacted in logs.
- GEM-008: Existing providers (Kokoro, Edge, Google, XTTS) must remain operational.

## Non-Goals

- Automatic diarization and speaker attribution.
- Runtime Batch API execution path.
- Streaming synthesis.

## Acceptance Criteria

1. Gemini engine is visible in UI when GEMINI_TTS_API_KEY is configured.
2. Gemini single-speaker and multi-speaker modes produce output clips.
3. Estimator displays one final Gemini price for standard, batch, and auto modes.
4. Unit tests cover pricing conversion and helper behavior.
5. Existing test suite remains green.
