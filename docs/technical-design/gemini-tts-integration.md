# TDD - Gemini 3.1 Flash TTS Preview Integration

| Field | Value |
| --- | --- |
| Tech Lead | TBD |
| Team | Dubweave maintainers |
| Status | Draft |
| Created | 2026-04-15 |
| Last Updated | 2026-04-15 |
| Epic/Ticket | Gemini TTS rollout |

## Context

Dubweave already supports Kokoro, Edge TTS, XTTS v2, and Google Cloud TTS.
The synthesis pipeline is stable, but it has no Gemini provider path and no
pricing support for audio-token billing. Gemini 3.1 Flash TTS Preview adds
controllable single-speaker and multi-speaker speech generation with low latency.

The existing estimator in src/ui/layout.py uses hardcoded pricing constants and
mixed units. Gemini pricing introduces text input tokens plus audio output tokens,
where audio tokens map to generated audio duration.

## Problem Statement and Motivation

### Problems to solve

- There is no Gemini TTS synthesis engine in the pipeline.
- There is no explicit 2-speaker mapping UX for Gemini TTS.
- The estimator cannot model Gemini standard/batch pricing.
- Pricing logic is not centralized, making rate updates error-prone.

### Why now

- Gemini 3.1 Flash TTS Preview was released and is now relevant to the product.
- The user requested cheapest-final-price estimator behavior and immediate
  implementation.

### Impact if not solved

- Users cannot select Gemini TTS for generation.
- Cost estimates are incorrect for Gemini usage.
- Multi-speaker scenarios stay unsupported in the UI.

## Scope

### In scope

- Add Gemini TTS engine to UI and pipeline.
- Add explicit 2-speaker mapping controls for Gemini.
- Add centralized pricing module with standard/batch Gemini rates.
- Show one final estimate in UI, auto-selecting cheapest mode when requested.
- Add unit tests for pricing and Gemini helper logic.
- Update README and .env.example.

### Out of scope

- Full provider abstraction rewrite across all engines.
- Automatic diarization from source audio.
- Streaming audio generation.
- Batch API runtime execution path.

## Technical Solution

### Architecture overview

New and changed components:

- src/core/gemini_tts.py:
  - Gemini provider adapter.
  - Single-speaker and multi-speaker request builders.
  - Preview-aware retry behavior for transient failures.
- src/core/pricing.py:
  - Centralized cost constants and conversion math.
  - Gemini auto-cheapest estimator policy.
- src/ui/layout.py:
  - Gemini engine controls.
  - Multi-speaker mapping UX.
  - Pricing output uses pricing module.
- src/pipeline.py:
  - Gemini key preflight validation.
  - Gemini synthesis branch dispatch.

### Data flow

1. User selects Gemini engine and configures voice settings.
2. Pipeline validates Gemini API key in preflight.
3. Translation output is grouped for synthesis as before.
4. Gemini provider synthesizes per-segment audio.
5. Existing atempo and muxing flow remains unchanged.
6. UI estimator shows one final Gemini price estimate.

### Pricing model

Definitions:

- Audio tokens per second: 25.
- Standard: $1.00/1M text input tokens and $20.00/1M audio output tokens.
- Batch: $0.50/1M text input tokens and $10.00/1M audio output tokens.

Estimator policy:

- `standard`: force standard estimate.
- `batch`: force batch estimate.
- `auto`: compute both and show only the cheaper total.

## Risks

1. Preview model intermittency (for example transient 500 responses).
2. Prompt classifier false rejections on weak prompts.
3. Speaker assignment ambiguity without diarization metadata.
4. Pricing table drift as preview rates evolve.

## Security Considerations

- Validate GEMINI_TTS_API_KEY before synthesis when Gemini is selected.
- Redact Gemini key in logs, same as other provider keys.
- Keep env/config parity between runtime constants and .env.example.

## Testing Strategy

Automated tests:

- tests/test_pricing.py
  - audio token conversion at 25 tokens/second.
  - standard vs batch estimate comparison.
  - auto mode cheapest selection.
- tests/test_gemini_tts.py
  - speaker assignment strategy behavior.
  - inline audio payload decoding behavior.

Manual checks:

- Gemini single-speaker generation from UI.
- Gemini two-speaker generation with alternate assignment.
- Prefix-based speaker mapping with "SpeakerName: ..." input lines.
- Cost estimate output for standard, batch, and auto.

## Monitoring and Observability

- Log Gemini key validation result in preflight.
- Log Gemini synthesis failures with segment index.
- Log note that runtime uses real-time endpoint while estimator can show
  batch-cheapest price in auto mode.

## Rollback Plan

1. Remove Gemini engine option from UI.
2. Remove Gemini branch from pipeline dispatch.
3. Keep existing providers unchanged.
4. Remove Gemini pricing entries if needed.

## Dependencies

- google-genai SDK added to pixi.toml.
- Existing ffmpeg pipeline for post-processing and muxing.

## Success Metrics

- Gemini engine selectable and functional in UI.
- Multi-speaker mode available with explicit controls.
- Estimator outputs one final Gemini price estimate.
- Existing engines continue to pass regression tests.
