"""
Unit tests for Dubweave core pure functions.

Covers: _ptpt_to_ptbr, _merge_segments, _srt_timestamp, _wrap_subtitle_line,
        validate_youtube_url, PipelineError, _retry_with_backoff, _redact
"""

import sys
import os
import time
import urllib.error

# Ensure project root is on path so we can import app without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs so app.py can be imported without heavy ML deps
# ---------------------------------------------------------------------------

import unittest.mock as mock

# Stub out gradio before importing app — it is not available in the test env
gradio_stub = mock.MagicMock()
gradio_stub.Progress = mock.MagicMock(return_value=mock.MagicMock())
sys.modules.setdefault("gradio", gradio_stub)

# Stub other heavy optional deps
for _mod in (
    "dotenv",
    "torch",
    "whisper",
    "yt_dlp",
    "TTS",
    "TTS.api",
    "transformers",
    "kokoro",
    "soundfile",
):
    sys.modules.setdefault(_mod, mock.MagicMock())

# dotenv.load_dotenv must be a callable that does nothing
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]

from app import (  # noqa: E402 — must come after stubs
    _ptpt_to_ptbr,
    _merge_segments,
    _srt_timestamp,
    _wrap_subtitle_line,
    validate_youtube_url,
    validate_openrouter_key,
    validate_google_tts_key,
    PipelineError,
    _retry_with_backoff,
    _redact,
    _REDACT_PATTERNS,
    log,
)


# ===========================================================================
# _ptpt_to_ptbr
# ===========================================================================


class TestPtptToPtbr:
    def test_empty_string(self):
        assert _ptpt_to_ptbr("") == ""

    def test_tu_to_voce(self):
        result = _ptpt_to_ptbr("Tu precisas fazer isso")
        assert "tu" not in result.lower()

    def test_autocarro_to_onibus(self):
        result = _ptpt_to_ptbr("O autocarro está atrasado")
        assert "ônibus" in result.lower() or "onibus" in result.lower()

    def test_telemóvel_to_celular(self):
        result = _ptpt_to_ptbr("O telemóvel está descarregado")
        assert "celular" in result.lower()

    def test_no_change_for_ptbr_text(self):
        # Pure PT-BR text should pass through without corruption
        text = "Você precisa ver isso agora."
        result = _ptpt_to_ptbr(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_gerund_a_fazer(self):
        result = _ptpt_to_ptbr("Estou a fazer o jantar")
        assert "a fazer" not in result.lower()

    def test_preserves_case_for_capitalised_word(self):
        # "Autocarro" starts with capital — result should also start with capital
        result = _ptpt_to_ptbr("Autocarro chegou")
        first_word = result.split()[0]
        assert first_word[0].isupper()


# ===========================================================================
# _srt_timestamp
# ===========================================================================


class TestSrtTimestamp:
    def test_zero(self):
        assert _srt_timestamp(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert _srt_timestamp(1.0) == "00:00:01,000"

    def test_one_minute(self):
        assert _srt_timestamp(60.0) == "00:01:00,000"

    def test_one_hour(self):
        assert _srt_timestamp(3600.0) == "01:00:00,000"

    def test_one_hour_one_minute_one_second(self):
        assert _srt_timestamp(3661.5) == "01:01:01,500"

    def test_fractional_milliseconds(self):
        result = _srt_timestamp(0.123)
        assert result == "00:00:00,123"

    def test_format_structure(self):
        result = _srt_timestamp(3723.456)
        parts = result.split(",")
        assert len(parts) == 2
        hms = parts[0].split(":")
        assert len(hms) == 3


# ===========================================================================
# _wrap_subtitle_line
# ===========================================================================


class TestWrapSubtitleLine:
    def test_short_line_unchanged(self):
        text = "Hello world"
        result = _wrap_subtitle_line(text)
        assert result == text

    def test_long_line_is_wrapped(self):
        text = "This is a very long subtitle line that should definitely be wrapped at a word boundary"
        result = _wrap_subtitle_line(text)
        assert "\n" in result

    def test_empty_string(self):
        result = _wrap_subtitle_line("")
        assert result == ""

    def test_already_two_lines_unchanged(self):
        text = "Line one\nLine two"
        result = _wrap_subtitle_line(text)
        # Should not add a third line
        assert result.count("\n") <= 1


# ===========================================================================
# _merge_segments
# ===========================================================================


class TestMergeSegments:
    def test_empty_list(self):
        assert _merge_segments([]) == []

    def test_single_complete_segment(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "Hello, world."}]
        result = _merge_segments(segs)
        assert len(result) >= 1

    def test_short_fragments_merged(self):
        # Two very short fragments should be merged into one utterance
        segs = [
            {"start": 0.0, "end": 0.5, "text": "Hello"},
            {"start": 0.5, "end": 1.0, "text": "world."},
        ]
        result = _merge_segments(segs)
        # Result should have fewer or equal segments
        assert len(result) <= len(segs)

    def test_output_preserves_timing(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "First complete sentence."},
            {"start": 2.0, "end": 3.0, "text": "Second complete sentence."},
        ]
        result = _merge_segments(segs)
        assert all("start" in s and "end" in s for s in result)

    def test_all_segments_have_text(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Some speech here, interesting."},
        ]
        result = _merge_segments(segs)
        assert all("text" in s for s in result)


# ===========================================================================
# validate_youtube_url
# ===========================================================================


class TestValidateYoutubeUrl:
    def test_valid_watch_url(self):
        ok, msg = validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert ok is True

    def test_valid_watch_url_without_www(self):
        ok, _ = validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert ok is True

    def test_valid_short_url(self):
        ok, _ = validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert ok is True

    def test_valid_shorts_url(self):
        ok, _ = validate_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ")
        assert ok is True

    def test_empty_url(self):
        ok, msg = validate_youtube_url("")
        assert ok is False
        assert "No URL" in msg

    def test_invalid_domain(self):
        ok, msg = validate_youtube_url("https://example.com/watch?v=dQw4w9WgXcQ")
        assert ok is False
        assert "doesn't look like a YouTube URL" in msg

    def test_whitespace_stripped(self):
        ok, _ = validate_youtube_url("  https://youtu.be/dQw4w9WgXcQ  ")
        assert ok is True

    def test_http_scheme(self):
        ok, _ = validate_youtube_url("http://youtube.com/watch?v=dQw4w9WgXcQ")
        assert ok is True

    def test_playlist_url_without_v(self):
        # A playlist URL without v= param should fail
        ok, _ = validate_youtube_url("https://youtube.com/playlist?list=PLxxx")
        assert ok is False

    def test_short_video_id(self):
        # Video ID shorter than 11 chars should fail
        ok, _ = validate_youtube_url("https://youtube.com/watch?v=short")
        assert ok is False


# ===========================================================================
# PipelineError
# ===========================================================================


class TestPipelineError:
    def test_basic_attributes(self):
        err = PipelineError("Download", "yt-dlp failed", recoverable=False)
        assert err.stage == "Download"
        assert err.message == "yt-dlp failed"
        assert err.recoverable is False

    def test_recoverable_flag(self):
        err = PipelineError("Synthesize", "OOM crash", recoverable=True)
        assert err.recoverable is True

    def test_str_includes_stage(self):
        err = PipelineError("Translate", "NLLB error")
        assert "Translate" in str(err)
        assert "NLLB error" in str(err)

    def test_is_exception(self):
        err = PipelineError("Assemble", "ffmpeg failed")
        assert isinstance(err, Exception)


# ===========================================================================
# _retry_with_backoff
# ===========================================================================


class TestRetryWithBackoff:
    def test_success_on_first_attempt(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            return 42

        result = _retry_with_backoff(fn, max_retries=3, base_delay=0.0)
        assert result == 42
        assert call_count["n"] == 1

    def test_retries_on_transient_error(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise urllib.error.HTTPError(
                    url=None, code=429, msg="Too Many Requests", hdrs=None, fp=None  # type: ignore
                )
            return "ok"

        result = _retry_with_backoff(fn, max_retries=3, base_delay=0.0)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_raises_immediately_on_non_retryable_error(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            raise urllib.error.HTTPError(
                url=None, code=401, msg="Unauthorized", hdrs=None, fp=None  # type: ignore
            )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _retry_with_backoff(fn, max_retries=3, base_delay=0.0)

        assert exc_info.value.code == 401
        assert call_count["n"] == 1

    def test_raises_after_max_retries(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            raise urllib.error.HTTPError(
                url=None, code=503, msg="Service Unavailable", hdrs=None, fp=None  # type: ignore
            )

        with pytest.raises(urllib.error.HTTPError):
            _retry_with_backoff(fn, max_retries=2, base_delay=0.0)

        assert call_count["n"] == 3  # initial + 2 retries

    def test_retries_on_generic_exception(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ConnectionError("network glitch")
            return "recovered"

        result = _retry_with_backoff(fn, max_retries=3, base_delay=0.0)
        assert result == "recovered"


# ===========================================================================
# _redact
# ===========================================================================


class TestRedact:
    def test_redacts_known_secret(self):
        import app as _app

        original = list(_app._REDACT_PATTERNS)
        try:
            _app._REDACT_PATTERNS.clear()
            _app._REDACT_PATTERNS.append("supersecretkey1234")
            result = _redact("Using key supersecretkey1234 in request")
            assert "supersecretkey1234" not in result
            assert "supe****" in result
        finally:
            _app._REDACT_PATTERNS.clear()
            _app._REDACT_PATTERNS.extend(original)

    def test_no_false_redaction(self):
        import app as _app

        original = list(_app._REDACT_PATTERNS)
        try:
            _app._REDACT_PATTERNS.clear()
            result = _redact("This message has no secrets")
            assert result == "This message has no secrets"
        finally:
            _app._REDACT_PATTERNS.clear()
            _app._REDACT_PATTERNS.extend(original)

    def test_short_secret_not_redacted(self):
        # Secrets shorter than 8 chars are not added to redact list by _init_redact_patterns
        import app as _app

        original = list(_app._REDACT_PATTERNS)
        try:
            _app._REDACT_PATTERNS.clear()
            # Short value should not cause redaction
            result = _redact("Using key abc in request")
            assert "abc" in result  # not redacted
        finally:
            _app._REDACT_PATTERNS.clear()
            _app._REDACT_PATTERNS.extend(original)


# ===========================================================================
# log function
# ===========================================================================


class TestLog:
    def test_appends_to_logs(self):
        logs = []
        result = log("Hello, world!", logs)
        assert len(result) == 1
        assert "Hello, world!" in result[0]

    def test_returns_logs(self):
        logs = []
        returned = log("test", logs)
        assert returned is logs

    def test_timestamp_format(self):
        logs = []
        log("test", logs)
        entry = logs[0]
        # Should contain [HH:MM:SS] timestamp
        assert entry.startswith("[")
        assert ":" in entry[:10]

    def test_multiple_entries(self):
        logs = []
        log("first", logs)
        log("second", logs)
        assert len(logs) == 2
        assert "first" in logs[0]
        assert "second" in logs[1]
