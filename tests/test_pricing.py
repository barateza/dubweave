from src.core.pricing import (
    estimate_audio_tokens_for_duration,
    estimate_gemini_tts_cost_for_mode,
    estimate_google_tts_cost,
    estimate_openrouter_translation_cost,
    pick_gemini_tts_cost,
)


def test_audio_tokens_conversion_uses_25_per_second():
    assert estimate_audio_tokens_for_duration(60.0) == 1500.0


def test_gemini_batch_is_cheaper_than_standard_for_same_duration():
    standard = estimate_gemini_tts_cost_for_mode(120.0, "standard")
    batch = estimate_gemini_tts_cost_for_mode(120.0, "batch")
    assert batch.total_cost_usd < standard.total_cost_usd


def test_gemini_auto_mode_picks_cheapest():
    auto = pick_gemini_tts_cost(300.0, "auto")
    assert auto.mode == "batch"


def test_google_cost_falls_back_to_neural2_for_unknown_type():
    unknown = estimate_google_tts_cost(60.0, "UnknownFamily")
    neural2 = estimate_google_tts_cost(60.0, "Neural2")
    assert unknown == neural2


def test_openrouter_flash_lite_uses_lite_rates():
    regular = estimate_openrouter_translation_cost(120.0, "google/gemini-2.0-flash-001")
    lite = estimate_openrouter_translation_cost(120.0, "google/gemini-2.0-flash-lite-001")
    assert lite < regular
