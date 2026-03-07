# Universal Video Input Specification

## Problem Statement

Dubweave is currently hardcoded to accept only YouTube URLs, which limits its usefulness and may create legal exposure. yt-dlp already supports thousands of sites — the pipeline should leverage that. Additionally, users who already have a video file on their computer should be able to upload it directly, skipping the download step entirely.

## Goals

- [ ] Accept any URL that yt-dlp supports (not just YouTube)
- [ ] Allow uploading a local video file as an alternative to providing a URL
- [ ] Ensure uploaded/downloaded video is always compatible with the assembly pipeline (mp4 + wav)
- [ ] Update UI to reflect the broader input model

## Out of Scope

| Feature | Reason |
|---------|--------|
| Batch/playlist downloads | Separate feature, complexity |
| Video format conversion UI | Pipeline handles internally |
| Audio-only input (podcast) | Different pipeline shape |

---

## User Stories

### P1: Any yt-dlp URL ⭐ MVP

**User Story**: As a user, I want to paste any video URL (not just YouTube) so that I can dub content from Vimeo, Twitter/X, TikTok, and thousands of other sites.

**Why P1**: Unlocks the full yt-dlp ecosystem and removes YouTube branding risk.

**Acceptance Criteria**:

1. WHEN user pastes a non-YouTube URL (e.g. Vimeo, Twitter) THEN system SHALL accept it and attempt download via yt-dlp
2. WHEN user pastes an empty URL and no file is uploaded THEN system SHALL show "No video source provided"
3. WHEN yt-dlp cannot handle the URL THEN system SHALL show a clear error from yt-dlp (not a regex validation error)
4. WHEN user pastes a YouTube URL THEN system SHALL still work exactly as before

**Independent Test**: Paste a Vimeo URL and verify download starts without validation error.

---

### P1: Local File Upload ⭐ MVP

**User Story**: As a user, I want to upload a video file from my computer so that I can dub existing content without needing a URL.

**Why P1**: Many users have downloaded/recorded content already on disk.

**Acceptance Criteria**:

1. WHEN user uploads a video file THEN system SHALL skip the URL download and use the uploaded file directly
2. WHEN user uploads a non-mp4 file (e.g. mkv, webm, avi, mov) THEN system SHALL re-encode to mp4 for pipeline compatibility
3. WHEN user uploads a file AND provides a URL THEN system SHALL prefer the uploaded file
4. WHEN user uploads a file THEN system SHALL extract audio as WAV for the transcription step
5. WHEN user uploads a corrupt/unreadable file THEN system SHALL show a clear error

**Independent Test**: Upload a local .mkv file and verify the full pipeline completes.

---

## Edge Cases

- WHEN URL is whitespace-only THEN system SHALL treat as empty (no URL)
- WHEN uploaded file has no audio track THEN system SHALL raise a clear error at the audio extraction step
- WHEN uploaded file is very large (>2GB) THEN system SHALL process normally (Gradio handles streaming)
- WHEN resuming from a stage after download THEN system SHALL work regardless of original source (URL or upload)

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|---|---|---|---|
| UVI-01 | P1: Any URL | Implement | Pending |
| UVI-02 | P1: Any URL - empty validation | Implement | Pending |
| UVI-03 | P1: Any URL - yt-dlp error passthrough | Implement | Pending |
| UVI-04 | P1: Any URL - YouTube backward compat | Implement | Pending |
| UVI-05 | P1: Upload - skip download | Implement | Pending |
| UVI-06 | P1: Upload - re-encode non-mp4 | Implement | Pending |
| UVI-07 | P1: Upload - prefer over URL | Implement | Pending |
| UVI-08 | P1: Upload - extract audio | Implement | Pending |
| UVI-09 | P1: Upload - error on corrupt file | Implement | Pending |

---

## Success Criteria

- [ ] Any URL that yt-dlp supports works without validation rejection
- [ ] Local file upload produces a dubbed video identical in quality to URL-sourced
- [ ] All existing YouTube URL tests still pass (backward compatible)
- [ ] UI no longer says "YouTube" in the main input area
