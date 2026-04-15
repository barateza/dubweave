# Gemini 3.1 Flash TTS - Execution Task Board

Status: Active
Owner: Dubweave maintainers
Last Updated: 2026-04-15

## Board

| PR | Title | Scope | Status |
| --- | --- | --- | --- |
| PR-01 | Add Gemini config and key validation | Config and security wiring | Done |
| PR-02 | Add Gemini provider adapter | Synthesis integration path | Done |
| PR-03 | Centralize pricing estimation | New pricing module and Gemini math | Done |
| PR-04 | Wire UI controls and pipeline dispatch | Engine selection, speaker mapping, dispatch | Done |
| PR-05 | Add focused unit tests | Pricing and helper tests | Done |
| PR-06 | Update docs and env template | README, .env.example, TDD | Done |
| PR-07 | Run regression and fix fallout | Validation and test pass | Pending |
| PR-08 | Final polish and release notes | Changelog and follow-up actions | Pending |

## Atomic PR Tasks with Explicit Test Cases

### PR-01: Add Gemini config and key validation

Goal:

- Add runtime constants for Gemini model, pricing mode, and speaker settings.
- Extend redaction and API-key preflight validation.

Files:

- src/config.py
- src/utils/security.py
- .env.example

Test cases:

1. Unit: invalid GEMINI_TTS_PRICING_MODE falls back to auto.
2. Unit: invalid GEMINI_TTS_SPEAKER_ASSIGNMENT falls back to alternate.
3. Unit: redact() masks GEMINI_TTS_API_KEY value in log strings.
4. Manual: key validation endpoint returns success for valid key and clear error for invalid key.

Done when:

- Config constants exist and are normalized.
- Security redaction includes Gemini key.
- Preflight validation helper exists for Gemini key.

### PR-02: Add Gemini provider adapter

Goal:

- Implement Gemini synthesis adapter with single-speaker and two-speaker paths.
- Handle inline audio data decoding and transient retries.

Files:

- src/core/gemini_tts.py

Test cases:

1. Unit: decode_inline_audio_data handles bytes input.
2. Unit: decode_inline_audio_data handles base64 string input.
3. Unit: speaker assignment alternate mode switches speaker by index parity.
4. Unit: speaker assignment prefix mode respects "SpeakerName: ..." prefixes.
5. Manual: synthesize one short segment in single-speaker mode and validate WAV output exists.
6. Manual: synthesize short two-speaker sample and validate output files exist.

Done when:

- Gemini synthesis function returns timed clips in existing contract.
- Segment-level fallback silence is preserved on errors.

### PR-03: Centralize pricing estimation

Goal:

- Move cost formulas out of UI and add Gemini standard/batch cost model.
- Add auto-cheapest estimator policy.

Files:

- src/core/pricing.py
- src/ui/layout.py

Test cases:

1. Unit: 60 seconds converts to 1500 audio tokens.
2. Unit: batch estimate is cheaper than standard for same duration.
3. Unit: auto mode selects cheaper estimate.
4. Unit: unknown Google voice type falls back to Neural2 pricing.
5. Unit: OpenRouter flash-lite pricing is lower than default flash.

Done when:

- Layout imports pricing helpers and no longer hardcodes Gemini formulas inline.
- UI displays one final Gemini estimate and selected mode.

### PR-04: Wire UI controls and pipeline dispatch

Goal:

- Add Gemini engine choice and controls for multi-speaker mapping.
- Route Gemini branch in pipeline dispatch.

Files:

- src/ui/layout.py
- src/pipeline.py

Test cases:

1. Manual: Gemini engine appears only when GEMINI_TTS_API_KEY is set.
2. Manual: toggling multi-speaker updates visible controls correctly.
3. Manual: run_pipeline dispatches Gemini branch when selected.
4. Manual: preflight validates Gemini key before download/transcribe steps.

Done when:

- Gemini controls are visible and functional.
- Pipeline branch invokes gemini_tts adapter with chosen settings.

### PR-05: Add focused unit tests

Goal:

- Add stable unit tests for pricing and Gemini pure helpers.

Files:

- tests/test_pricing.py
- tests/test_gemini_tts.py

Test cases:

1. All tests in tests/test_pricing.py pass.
2. All tests in tests/test_gemini_tts.py pass.
3. Existing tests in tests/test_core.py remain green.

Done when:

- New tests are committed and deterministic.

### PR-06: Update docs and env template

Goal:

- Document Gemini configuration, estimator behavior, and design rationale.

Files:

- README.md
- .env.example
- docs/technical-design/gemini-tts-integration.md

Test cases:

1. Manual: README config table includes all new GEMINI_* variables.
2. Manual: .env.example and src/config.py constants are in sync.
3. Manual: TDD covers scope, risks, tests, and rollback.

Done when:

- Reader can configure and run Gemini mode from docs only.

### PR-07: Run regression and fix fallout

Goal:

- Validate no regressions for non-Gemini providers.

Files:

- tests/*
- impacted source files from prior PRs

Test cases:

1. Run pytest tests/ -v and confirm passing results.
2. Manual smoke: Kokoro path still synthesizes and assembles.
3. Manual smoke: Edge path still synthesizes and assembles.
4. Manual smoke: Google TTS path still works when key is present.

Done when:

- No new regressions in baseline engines.

### PR-08: Final polish and release notes

Goal:

- Capture release details and follow-up items.

Files:

- README.md or release notes artifact
- .specs/features/gemini-tts/tasks.md

Test cases:

1. Manual: release notes include known limitations and mitigations.
2. Manual: follow-up backlog captures deferred Batch API runtime support.

Done when:

- Feature is release-ready with clear known limitations.

## Deferred Follow-Ups

- Implement true Gemini Batch API runtime path (currently estimator supports batch pricing selection).
- Add diarization-based speaker mapping to avoid alternating fallback for unlabeled transcripts.
- Add integration tests with mocked Gemini SDK responses.
