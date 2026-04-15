from __future__ import annotations

from dataclasses import dataclass

# Heuristic conversion used by existing UI logic.
WORDS_PER_MINUTE = 150.0
CHARS_PER_WORD = 5.0
TOKENS_PER_MINUTE = 200.0

GOOGLE_TTS_USD_PER_MILLION_CHARS = {
    "Standard": 4.00,
    "WaveNet": 16.00,
    "Neural2": 16.00,
    "Studio": 160.00,
    "Chirp3 HD": 30.00,
    "Polyglot (Preview)": 16.00,
}

OPENROUTER_DEFAULT_INPUT_USD_PER_MILLION = 0.10
OPENROUTER_DEFAULT_OUTPUT_USD_PER_MILLION = 0.40
OPENROUTER_FLASH_LITE_INPUT_USD_PER_MILLION = 0.075
OPENROUTER_FLASH_LITE_OUTPUT_USD_PER_MILLION = 0.30

GEMINI_TTS_AUDIO_TOKENS_PER_SECOND = 25.0

GEMINI_TTS_PRICING = {
    "standard": {
        "input_usd_per_million_text_tokens": 1.00,
        "output_usd_per_million_audio_tokens": 20.00,
    },
    "batch": {
        "input_usd_per_million_text_tokens": 0.50,
        "output_usd_per_million_audio_tokens": 10.00,
    },
}


@dataclass(frozen=True)
class GeminiCostEstimate:
    mode: str
    text_tokens: float
    audio_tokens: float
    total_cost_usd: float



def estimate_text_from_duration(duration_seconds: float) -> tuple[float, float]:
    """Estimate text chars/tokens from media duration using stable heuristics."""
    minutes = max(0.0, duration_seconds) / 60.0
    est_chars = minutes * WORDS_PER_MINUTE * CHARS_PER_WORD
    est_tokens = minutes * TOKENS_PER_MINUTE
    return est_chars, est_tokens



def estimate_openrouter_translation_cost(duration_seconds: float, model_name: str) -> float:
    """Estimate translation cost with model-specific OpenRouter rates."""
    _, est_tokens = estimate_text_from_duration(duration_seconds)
    in_price = OPENROUTER_DEFAULT_INPUT_USD_PER_MILLION
    out_price = OPENROUTER_DEFAULT_OUTPUT_USD_PER_MILLION
    if "flash-lite" in model_name.lower():
        in_price = OPENROUTER_FLASH_LITE_INPUT_USD_PER_MILLION
        out_price = OPENROUTER_FLASH_LITE_OUTPUT_USD_PER_MILLION
    return (est_tokens * (in_price + out_price)) / 1_000_000



def estimate_google_tts_cost(duration_seconds: float, voice_type: str) -> float:
    """Estimate Google Cloud TTS cost by voice family and duration."""
    est_chars, _ = estimate_text_from_duration(duration_seconds)
    rate = GOOGLE_TTS_USD_PER_MILLION_CHARS.get(voice_type, GOOGLE_TTS_USD_PER_MILLION_CHARS["Neural2"])
    return (est_chars * rate) / 1_000_000



def estimate_audio_tokens_for_duration(duration_seconds: float, tokens_per_second: float = GEMINI_TTS_AUDIO_TOKENS_PER_SECOND) -> float:
    return max(0.0, duration_seconds) * tokens_per_second



def estimate_gemini_tts_cost_for_mode(duration_seconds: float, mode: str) -> GeminiCostEstimate:
    """Estimate Gemini TTS synthesis cost for a single mode."""
    _, text_tokens = estimate_text_from_duration(duration_seconds)
    audio_tokens = estimate_audio_tokens_for_duration(duration_seconds)
    price = GEMINI_TTS_PRICING[mode]
    total = (
        text_tokens * price["input_usd_per_million_text_tokens"]
        + audio_tokens * price["output_usd_per_million_audio_tokens"]
    ) / 1_000_000
    return GeminiCostEstimate(
        mode=mode,
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
        total_cost_usd=total,
    )



def pick_gemini_tts_cost(duration_seconds: float, preferred_mode: str = "auto") -> GeminiCostEstimate:
    """Return one final Gemini estimate. Auto mode picks the cheapest option."""
    preferred = (preferred_mode or "auto").strip().lower()
    standard_estimate = estimate_gemini_tts_cost_for_mode(duration_seconds, "standard")
    batch_estimate = estimate_gemini_tts_cost_for_mode(duration_seconds, "batch")

    if preferred == "standard":
        return standard_estimate
    if preferred == "batch":
        return batch_estimate
    if batch_estimate.total_cost_usd <= standard_estimate.total_cost_usd:
        return batch_estimate
    return standard_estimate
