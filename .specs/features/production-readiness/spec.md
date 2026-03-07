# Production Readiness Specification

## Problem Statement

Dubweave's core dubbing pipeline is fully functional after four high-severity fixes, but lacks production-grade documentation, error handling, security hardening, performance guardrails, test coverage, and operational monitoring. These gaps must be addressed before publishing as an open-source tool.

## Goals

- [ ] Users can install and run Dubweave without hitting undocumented failures
- [ ] All error paths surface actionable, user-friendly messages in the Gradio UI
- [ ] API keys are validated early and secrets are never leaked to logs or UI
- [ ] Long videos (>1h) don't crash from memory exhaustion
- [ ] Core pure functions have automated test coverage
- [ ] Operators can diagnose failures from structured log output

## Out of Scope

| Feature | Reason |
| --- | --- |
| Module extraction / refactor | Architectural improvement, not required for v1 production |
| Multi-language targets | PT-BR only for v1 |
| CI/CD pipeline | No automated deployment needed for local-only tool |
| Docker / containerization | Local CUDA setup required; container adds complexity |
| CPU-only fallback | GPU is a stated requirement |

---

## User Stories

### P1: Documentation & Setup ⭐ MVP

**User Story**: As a first-time user, I want clear installation instructions so that I can get Dubweave running without trial-and-error.

**Why P1**: Users who can't install can't use the tool at all.

**Acceptance Criteria**:

1. WHEN a user follows README setup steps THEN the system SHALL complete setup without undocumented errors
2. WHEN a required dependency (espeak-ng, CUDA, Pixi) is missing THEN the system SHALL print a diagnostic message naming the missing dependency
3. WHEN `.env` is missing THEN the system SHALL create a `.env.example` template with documented defaults

**Independent Test**: Fresh Windows machine with NVIDIA GPU — follow README from scratch, reach `localhost:7860`.

---

### P1: Error Handling & UX ⭐ MVP

**User Story**: As a user processing a video, I want clear error messages so that I can understand and fix problems without reading source code.

**Why P1**: Cryptic tracebacks cause users to abandon the tool.

**Acceptance Criteria**:

1. WHEN a network request fails (yt-dlp, OpenRouter, Google TTS) THEN the system SHALL display a user-friendly message including the HTTP status and retry guidance
2. WHEN an API rate limit is hit THEN the system SHALL wait and retry with exponential backoff (max 3 retries)
3. WHEN a pipeline stage fails THEN the system SHALL indicate which stage failed, what the error was, and whether the project can be resumed
4. WHEN a video cannot be downloaded (geo-blocked, private, removed) THEN the system SHALL display the specific reason from yt-dlp

**Independent Test**: Provide an invalid URL → verify the error message is actionable, not a raw traceback.

---

### P1: Security & Configuration ⭐ MVP

**User Story**: As a user with API keys, I want my credentials validated early and never exposed in logs or the UI.

**Why P1**: Leaked API keys are a security incident; invalid keys waste pipeline time before failing.

**Acceptance Criteria**:

1. WHEN a user provides an OpenRouter API key THEN the system SHALL validate it with a lightweight /auth/key call before starting the pipeline
2. WHEN a user provides a Google TTS API key THEN the system SHALL validate it with a voices.list call before starting synthesis
3. WHEN logging pipeline activity THEN the system SHALL NEVER include API keys, tokens, or credentials in log output
4. WHEN a YouTube URL is provided THEN the system SHALL validate it matches expected YouTube URL patterns before passing to yt-dlp

**Independent Test**: Set an invalid API key → verify the error is caught at validation, not mid-pipeline. Search all log output for key substrings.

---

### P2: Performance & Scalability

**User Story**: As a user processing long videos (30min–2h), I want the pipeline to complete without running out of memory or disk space.

**Why P2**: Core pipeline works for short videos; long videos are a frequent real-world use case but not a blocker.

**Acceptance Criteria**:

1. WHEN processing a video >30 minutes THEN the system SHALL release GPU memory between pipeline stages (Whisper → Translation → TTS)
2. WHEN temporary files accumulate >5GB THEN the system SHALL warn the user and offer cleanup
3. WHEN synthesizing >200 segments THEN the system SHALL process in batches to limit peak memory
4. WHEN the pipeline completes or fails THEN the system SHALL clean up intermediate temp files in WORK_DIR

**Independent Test**: Process a 1-hour video → monitor peak memory stays under 12GB RAM + 8GB VRAM.

---

### P2: Testing & QA

**User Story**: As a developer maintaining Dubweave, I want automated tests so that changes don't silently break core functionality.

**Why P2**: Tests prevent regressions but aren't user-facing; manual testing works for v1.

**Acceptance Criteria**:

1. WHEN running `pixi run test` THEN the system SHALL execute all unit tests and report pass/fail
2. WHEN a core pure function (segment merging, SRT generation, PT-PT→PT-BR normalization, timing budget) is tested THEN tests SHALL cover normal input, empty input, and boundary cases
3. WHEN translated segment count mismatches source THEN translation validation tests SHALL catch it

**Independent Test**: Run `pixi run test` → all tests pass, coverage >80% for targeted modules.

---

### P3: Deployment & Monitoring

**User Story**: As an operator running Dubweave for extended sessions, I want structured logs so that I can diagnose failures without reproducing them.

**Why P3**: Nice-to-have for single-user local tool; becomes critical if shared.

**Acceptance Criteria**:

1. WHEN the application starts THEN the system SHALL log environment info (Python version, CUDA availability, model paths)
2. WHEN a pipeline stage completes THEN the system SHALL log duration, input/output counts, and resource usage
3. WHEN an error occurs THEN the system SHALL log structured JSON with timestamp, stage, error type, and message

**Independent Test**: Run a full pipeline → verify log output is parseable JSON with consistent schema.

---

## Edge Cases

- WHEN YouTube returns a CAPTCHA or bot check THEN system SHALL display a message suggesting browser cookie auth
- WHEN espeak-ng is not installed THEN Kokoro TTS SHALL fail with a clear message naming the dependency
- WHEN disk space is insufficient for output THEN system SHALL detect and report before writing partial files
- WHEN `.env` has malformed values (non-numeric speed, invalid model name) THEN system SHALL validate at startup and report which setting is wrong

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| PROD-01 | P1: Documentation & Setup | Tasks | Pending |
| PROD-02 | P1: Error Handling & UX | Tasks | Pending |
| PROD-03 | P1: Security & Configuration | Tasks | Pending |
| PROD-04 | P2: Performance & Scalability | Tasks | Pending |
| PROD-05 | P2: Testing & QA | Tasks | Pending |
| PROD-06 | P3: Deployment & Monitoring | Tasks | Pending |

**Coverage:** 6 total, 0 mapped to tasks, 6 unmapped ⚠️

---

## Success Criteria

- [ ] A new user can install and dub a 10-minute video on first attempt following only the README
- [ ] No raw Python tracebacks are ever displayed in the Gradio UI
- [ ] API keys are validated before pipeline starts; never appear in logs
- [ ] A 1-hour video completes without OOM crashes
- [ ] `pixi run test` passes with >80% coverage on targeted pure functions
- [ ] Log output is structured and includes stage timing
