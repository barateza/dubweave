import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

__version__ = "0.1.0"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.resolve()
WORK_DIR = Path(tempfile.gettempdir()) / "yt_dubber"
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
PROJECTS_DIR = ROOT_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

# ── Demo mode ───────────────────────────────────────────────────────────────
DEMO_MODE = _env_flag("DEMO_MODE", False)
DEFAULT_INPUT_LANGUAGE = os.getenv("DEFAULT_INPUT_LANGUAGE", "Auto-detect")
DEFAULT_OUTPUT_LANGUAGE = os.getenv("DEFAULT_OUTPUT_LANGUAGE", "Portuguese (BR)")
EDGE_TTS_DEFAULT_GENDER = os.getenv("EDGE_TTS_DEFAULT_GENDER", "Female")

# ── Demo mode limits ────────────────────────────────────────────────────────
MAX_VIDEO_DURATION_S = int(os.getenv("MAX_VIDEO_DURATION_S", "180"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ── Models ───────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")

# ── Demo language dictionaries ────────────────────────────────────────────────
WHISPER_LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Portuguese": "pt",
    "Spanish": "es",
    "French": "fr",
    "Finnish": "fi",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
}

OUTPUT_LANGUAGES = {
    "Portuguese (BR)": {
        "tgt_lang": "por_Latn",
        "openrouter_hint": "pt-BR",
        "edge_voices": {
            "Female": ["pt-BR-FranciscaNeural", "pt-BR-ThalitaNeural"],
            "Male": ["pt-BR-AntonioNeural"],
        },
    },
    "English": {
        "tgt_lang": "eng_Latn",
        "openrouter_hint": "English",
        "edge_voices": {
            "Female": ["en-US-JennyNeural", "en-US-AriaNeural"],
            "Male": ["en-US-GuyNeural"],
        },
    },
    "Spanish": {
        "tgt_lang": "spa_Latn",
        "openrouter_hint": "Spanish",
        "edge_voices": {
            "Female": ["es-ES-ElviraNeural", "es-ES-LuciaNeural"],
            "Male": ["es-ES-AlvaroNeural"],
        },
    },
    "French": {
        "tgt_lang": "fra_Latn",
        "openrouter_hint": "French",
        "edge_voices": {
            "Female": ["fr-FR-DeniseNeural", "fr-FR-BrigitteNeural"],
            "Male": ["fr-FR-HenriNeural"],
        },
    },
    "Finnish": {
        "tgt_lang": "fin_Latn",
        "openrouter_hint": "Finnish",
        "edge_voices": {
            "Female": ["fi-FI-SelmaNeural", "fi-FI-NooraNeural"],
            "Male": ["fi-FI-HarriNeural"],
        },
    },
    "German": {
        "tgt_lang": "deu_Latn",
        "openrouter_hint": "German",
        "edge_voices": {
            "Female": ["de-DE-KatjaNeural", "de-DE-AmalaNeural"],
            "Male": ["de-DE-ConradNeural", "de-DE-KillianNeural"],
        },
    },
}

INPUT_LANGUAGE_OPTIONS = WHISPER_LANGUAGES
OUTPUT_LANGUAGE_CATALOG = OUTPUT_LANGUAGES


def get_output_language_config(language_name: str) -> dict:
    return OUTPUT_LANGUAGE_CATALOG.get(language_name, OUTPUT_LANGUAGE_CATALOG[DEFAULT_OUTPUT_LANGUAGE])


def get_edge_tts_voice_choices(language_name: str, gender: str | None = None) -> list[str]:
    cfg = get_output_language_config(language_name)
    voices = cfg.get("edge_voices", {})
    if gender:
        selected = voices.get(gender, [])
        if selected:
            return list(selected)
    flattened: list[str] = []
    for gender_name in ("Female", "Male"):
        flattened.extend(voices.get(gender_name, []))
    return flattened


def get_default_edge_tts_voice(language_name: str, gender: str | None = None) -> str:
    choices = get_edge_tts_voice_choices(language_name, gender or EDGE_TTS_DEFAULT_GENDER)
    if choices:
        return choices[0]
    fallback = get_edge_tts_voice_choices(DEFAULT_OUTPUT_LANGUAGE)
    return fallback[0] if fallback else EDGE_TTS_VOICE_NAME

# ── Translation config ───────────────────────────────────────────────────────
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_SRC_LANG = os.getenv("NLLB_SRC_LANG", "eng_Latn")
NLLB_TGT_LANG = os.getenv("NLLB_TGT_LANG", "por_Latn")

# ── OpenRouter config ────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")


def _int_env(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


OPENROUTER_CHUNK_SIZE = max(1, _int_env("OPENROUTER_CHUNK_SIZE", 120))
OPENROUTER_CONTEXT_SIZE = max(0, _int_env("OPENROUTER_CONTEXT_SIZE", 8))

# ── Edge TTS config ──────────────────────────────────────────────────────────
EDGE_TTS_VOICE_NAME = os.getenv("EDGE_TTS_VOICE_NAME", "pt-BR-FranciscaNeural")

JOB_MAX_AGE_H = 2
