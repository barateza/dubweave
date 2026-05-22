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

# ── Models ───────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_LANG = "pt"

# ── Demo language catalog ───────────────────────────────────────────────────
INPUT_LANGUAGE_OPTIONS = {
    "Auto-detect": None,
    "English": "en",
    "Portuguese": "pt",
    "Spanish": "es",
    "French": "fr",
    "Finnish": "fi",
}

OUTPUT_LANGUAGE_CATALOG = {
    "Portuguese (BR)": {
        "whisper_code": "pt",
        "translation_target": "por_Latn",
        "supports_local_fallback": True,
        "edge_voices": {
            "Female": ["pt-BR-FranciscaNeural", "pt-BR-ThalitaNeural"],
            "Male": ["pt-BR-AntonioNeural"],
        },
    },
    "English (US)": {
        "whisper_code": "en",
        "translation_target": "eng_Latn",
        "supports_local_fallback": False,
        "edge_voices": {
            "Female": ["en-US-JennyNeural", "en-US-AriaNeural"],
            "Male": ["en-US-GuyNeural"],
        },
    },
    "Spanish (ES)": {
        "whisper_code": "es",
        "translation_target": "spa_Latn",
        "supports_local_fallback": False,
        "edge_voices": {
            "Female": ["es-ES-ElviraNeural", "es-ES-LuciaNeural"],
            "Male": ["es-ES-AlvaroNeural"],
        },
    },
    "French (FR)": {
        "whisper_code": "fr",
        "translation_target": "fra_Latn",
        "supports_local_fallback": False,
        "edge_voices": {
            "Female": ["fr-FR-DeniseNeural", "fr-FR-BrigitteNeural"],
            "Male": ["fr-FR-HenriNeural"],
        },
    },
    "Finnish": {
        "whisper_code": "fi",
        "translation_target": "fin_Latn",
        "supports_local_fallback": False,
        "edge_voices": {
            "Female": ["fi-FI-NooraNeural", "fi-FI-SelmaNeural"],
            "Male": ["fi-FI-HarriNeural"],
        },
    },
}


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

# ── Kokoro config ────────────────────────────────────────────────────────────
KOKORO_LANG = os.getenv("KOKORO_LANG", "p")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "pf_dora")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

# ── Translation config ───────────────────────────────────────────────────────
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_SRC_LANG = os.getenv("NLLB_SRC_LANG", "eng_Latn")
NLLB_TGT_LANG = os.getenv("NLLB_TGT_LANG", "por_Latn")

# ── OpenRouter config ────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
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

# ── Google Cloud TTS config ──────────────────────────────────────────────────
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "").strip()
GOOGLE_TTS_LANGUAGE_CODE = os.getenv("GOOGLE_TTS_LANGUAGE_CODE", "pt-BR")
GOOGLE_TTS_VOICE_TYPE = os.getenv("GOOGLE_TTS_VOICE_TYPE", "Neural2")
GOOGLE_TTS_VOICE_NAME = os.getenv("GOOGLE_TTS_VOICE_NAME", "pt-BR-Neural2-A")

# ── Gemini TTS config ───────────────────────────────────────────────────────
GEMINI_TTS_API_KEY = os.getenv("GEMINI_TTS_API_KEY", "").strip()
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
GEMINI_TTS_PRICING_MODE = os.getenv("GEMINI_TTS_PRICING_MODE", "auto").strip().lower()
GEMINI_TTS_SINGLE_VOICE = os.getenv("GEMINI_TTS_SINGLE_VOICE", "Kore")
GEMINI_TTS_MULTI_SPEAKER = (
    os.getenv("GEMINI_TTS_MULTI_SPEAKER", "false").strip().lower() == "true"
)
GEMINI_TTS_SPEAKER_ASSIGNMENT = os.getenv("GEMINI_TTS_SPEAKER_ASSIGNMENT", "alternate")
GEMINI_TTS_SPEAKER1_NAME = os.getenv("GEMINI_TTS_SPEAKER1_NAME", "Speaker1")
GEMINI_TTS_SPEAKER1_VOICE = os.getenv("GEMINI_TTS_SPEAKER1_VOICE", "Kore")
GEMINI_TTS_SPEAKER2_NAME = os.getenv("GEMINI_TTS_SPEAKER2_NAME", "Speaker2")
GEMINI_TTS_SPEAKER2_VOICE = os.getenv("GEMINI_TTS_SPEAKER2_VOICE", "Puck")

if GEMINI_TTS_PRICING_MODE not in {"auto", "standard", "batch"}:
    GEMINI_TTS_PRICING_MODE = "auto"
if GEMINI_TTS_SPEAKER_ASSIGNMENT not in {"alternate", "prefix"}:
    GEMINI_TTS_SPEAKER_ASSIGNMENT = "alternate"

# ── Edge TTS config ──────────────────────────────────────────────────────────
EDGE_TTS_VOICE_NAME = os.getenv("EDGE_TTS_VOICE_NAME", "pt-BR-FranciscaNeural")

# ── Supertonic v3 config ───────────────────────────────────────────────────
SUPERTONIC_AUTO_DOWNLOAD = (
    os.getenv("SUPERTONIC_AUTO_DOWNLOAD", "true").strip().lower() != "false"
)
SUPERTONIC_ASSETS_DIR = os.getenv("SUPERTONIC_ASSETS_DIR", "").strip()
SUPERTONIC_MODEL_REPO = os.getenv("SUPERTONIC_MODEL_REPO", "Supertone/supertonic")
SUPERTONIC_MODEL_REVISION = os.getenv("SUPERTONIC_MODEL_REVISION", "").strip()
SUPERTONIC_INTRA_OP_THREADS = _int_env("SUPERTONIC_INTRA_OP_THREADS", 0)
SUPERTONIC_INTER_OP_THREADS = _int_env("SUPERTONIC_INTER_OP_THREADS", 0)
SUPERTONIC_LOG_LEVEL = (
    os.getenv("SUPERTONIC_LOG_LEVEL", "INFO").strip().upper() or "INFO"
)
SUPERTONIC_DEFAULT_VOICE = (
    os.getenv("SUPERTONIC_DEFAULT_VOICE", "M4").strip().upper() or "M4"
)
SUPERTONIC_DEFAULT_LANG = (
    os.getenv("SUPERTONIC_DEFAULT_LANG", "pt").strip().lower() or "pt"
)
SUPERTONIC_ALLOWED_TAGS = ["laugh", "breath", "sigh"]

SUPERTONIC_VOICE_OPTIONS = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
SUPERTONIC_LANGUAGE_OPTIONS = ["en", "ko", "es", "pt", "fr"]

EDGE_TTS_PT_BR_VOICES = [
    "pt-BR-FranciscaNeural",
    "pt-BR-AntonioNeural",
    "pt-BR-ThalitaNeural",
]

GOOGLE_TTS_VOICE_CATALOG = {
    "Chirp3 HD": [
        "pt-BR-Chirp3-HD-Achernar",
        "pt-BR-Chirp3-HD-Achird",
        "pt-BR-Chirp3-HD-Algenib",
        "pt-BR-Chirp3-HD-Algieba",
        "pt-BR-Chirp3-HD-Alnilam",
        "pt-BR-Chirp3-HD-Aoede",
        "pt-BR-Chirp3-HD-Autonoe",
        "pt-BR-Chirp3-HD-Callirrhoe",
        "pt-BR-Chirp3-HD-Charon",
        "pt-BR-Chirp3-HD-Despina",
        "pt-BR-Chirp3-HD-Enceladus",
        "pt-BR-Chirp3-HD-Erinome",
        "pt-BR-Chirp3-HD-Fenrir",
        "pt-BR-Chirp3-HD-Gacrux",
        "pt-BR-Chirp3-HD-Iapetus",
        "pt-BR-Chirp3-HD-Kore",
        "pt-BR-Chirp3-HD-Laomedeia",
        "pt-BR-Chirp3-HD-Leda",
        "pt-BR-Chirp3-HD-Orus",
        "pt-BR-Chirp3-HD-Puck",
        "pt-BR-Chirp3-HD-Pulcherrima",
        "pt-BR-Chirp3-HD-Rasalgethi",
        "pt-BR-Chirp3-HD-Sadachbia",
        "pt-BR-Chirp3-HD-Sadaltager",
        "pt-BR-Chirp3-HD-Schedar",
        "pt-BR-Chirp3-HD-Sulafat",
        "pt-BR-Chirp3-HD-Umbriel",
        "pt-BR-Chirp3-HD-Vindemiatrix",
        "pt-BR-Chirp3-HD-Zephyr",
        "pt-BR-Chirp3-HD-Zubenelgenubi",
    ],
    "WaveNet": [
        "pt-BR-Wavenet-A",
        "pt-BR-Wavenet-B",
        "pt-BR-Wavenet-C",
        "pt-BR-Wavenet-D",
        "pt-BR-Wavenet-E",
    ],
    "Standard": [
        "pt-BR-Standard-A",
        "pt-BR-Standard-B",
        "pt-BR-Standard-C",
        "pt-BR-Standard-D",
        "pt-BR-Standard-E",
    ],
    "Studio": ["pt-BR-Studio-B", "pt-BR-Studio-C"],
    "Neural2": ["pt-BR-Neural2-A", "pt-BR-Neural2-B", "pt-BR-Neural2-C"],
    "Polyglot (Preview)": [],
}

GEMINI_TTS_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]

# ── ElevenLabs TTS config ────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_TTS_MODEL_ID = (
    os.getenv("ELEVENLABS_TTS_MODEL_ID", "eleven_multilingual_v2").strip()
    or "eleven_multilingual_v2"
)
ELEVENLABS_TTS_VOICE_ID = os.getenv("ELEVENLABS_TTS_VOICE_ID", "").strip()

JOB_MAX_AGE_H = 2
