import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

__version__ = "0.1.0"

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.resolve()
WORK_DIR = Path(tempfile.gettempdir()) / "yt_dubber"
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
PROJECTS_DIR = ROOT_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

# ── Models ───────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TARGET_LANG = "pt"

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

# ── Edge TTS config ──────────────────────────────────────────────────────────
EDGE_TTS_VOICE_NAME = os.getenv("EDGE_TTS_VOICE_NAME", "pt-BR-FranciscaNeural")

EDGE_TTS_PT_BR_VOICES = [
    "pt-BR-FranciscaNeural",
    "pt-BR-AntonioNeural",
    "pt-BR-ThalitaNeural",
]

GOOGLE_TTS_VOICE_CATALOG = {
    "Chirp3 HD": [
        "pt-BR-Chirp3-HD-Achernar", "pt-BR-Chirp3-HD-Achird", "pt-BR-Chirp3-HD-Algenib",
        "pt-BR-Chirp3-HD-Algieba", "pt-BR-Chirp3-HD-Alnilam", "pt-BR-Chirp3-HD-Aoede",
        "pt-BR-Chirp3-HD-Autonoe", "pt-BR-Chirp3-HD-Callirrhoe", "pt-BR-Chirp3-HD-Charon",
        "pt-BR-Chirp3-HD-Despina", "pt-BR-Chirp3-HD-Enceladus", "pt-BR-Chirp3-HD-Erinome",
        "pt-BR-Chirp3-HD-Fenrir", "pt-BR-Chirp3-HD-Gacrux", "pt-BR-Chirp3-HD-Iapetus",
        "pt-BR-Chirp3-HD-Kore", "pt-BR-Chirp3-HD-Laomedeia", "pt-BR-Chirp3-HD-Leda",
        "pt-BR-Chirp3-HD-Orus", "pt-BR-Chirp3-HD-Puck", "pt-BR-Chirp3-HD-Pulcherrima",
        "pt-BR-Chirp3-HD-Rasalgethi", "pt-BR-Chirp3-HD-Sadachbia", "pt-BR-Chirp3-HD-Sadaltager",
        "pt-BR-Chirp3-HD-Schedar", "pt-BR-Chirp3-HD-Sulafat", "pt-BR-Chirp3-HD-Umbriel",
        "pt-BR-Chirp3-HD-Vindemiatrix", "pt-BR-Chirp3-HD-Zephyr", "pt-BR-Chirp3-HD-Zubenelgenubi",
    ],
    "WaveNet": ["pt-BR-Wavenet-A", "pt-BR-Wavenet-B", "pt-BR-Wavenet-C", "pt-BR-Wavenet-D", "pt-BR-Wavenet-E"],
    "Standard": ["pt-BR-Standard-A", "pt-BR-Standard-B", "pt-BR-Standard-C", "pt-BR-Standard-D", "pt-BR-Standard-E"],
    "Studio": ["pt-BR-Studio-B", "pt-BR-Studio-C"],
    "Neural2": ["pt-BR-Neural2-A", "pt-BR-Neural2-B", "pt-BR-Neural2-C"],
    "Polyglot (Preview)": [],
}

JOB_MAX_AGE_H = 2
