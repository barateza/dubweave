from src.core.supertonic_tts import _sanitize_supertonic_text, get_provider_label
from src.core.supertonic_tts import ensure_supertonic_assets_or_raise
from src.core.translate import PipelineError
from src.core.synthesis import get_cps_for_voice


def test_supertonic_sanitizer_preserves_allowed_tags():
    text = "Hello <laugh>world</laugh> <breath/> <sigh></sigh>"
    result = _sanitize_supertonic_text(text)
    assert "<laugh>" in result
    assert "</laugh>" in result
    assert "<breath/>" in result or "<breath>" in result
    assert "<sigh>" in result


def test_supertonic_sanitizer_removes_unknown_tags():
    text = "Hello <foo>world</foo>"
    result = _sanitize_supertonic_text(text)
    assert "<foo>" not in result
    assert "</foo>" not in result


def test_supertonic_provider_label_is_string():
    assert isinstance(get_provider_label(), str)


def test_supertonic_assets_toggle_raises_when_disabled(monkeypatch):
    monkeypatch.setattr("src.core.supertonic_tts.SUPERTONIC_AUTO_DOWNLOAD", False)
    monkeypatch.setattr("src.core.supertonic_tts.SUPERTONIC_ASSETS_DIR", "")
    try:
        ensure_supertonic_assets_or_raise()
        raised = False
    except PipelineError:
        raised = True
    assert raised is True


def test_supertonic_cps_fallback_uses_supertonic_default_rate():
    assert get_cps_for_voice("Supertonic v3 (local)", "unknown") == 16.0
