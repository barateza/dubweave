import os
import json
import urllib.request
import urllib.error

_REDACT_PATTERNS: list[str] = []

def init_redact_patterns() -> None:
    """Build redaction list from current API key env vars."""
    global _REDACT_PATTERNS
    _REDACT_PATTERNS = []
    for env_var in ("OPENROUTER_API_KEY", "GOOGLE_TTS_API_KEY", "GEMINI_TTS_API_KEY", "ELEVENLABS_API_KEY"):
        val = os.getenv(env_var, "").strip()
        if len(val) > 8:
            _REDACT_PATTERNS.append(val)

def redact(msg: str) -> str:
    """Replace any known secret value with a masked version."""
    for secret in _REDACT_PATTERNS:
        msg = msg.replace(secret, f"{secret[:4]}****")
    return msg

def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    """Validate an OpenRouter API key via a lightweight /auth/key call."""
    api_key = api_key.strip()
    if not api_key:
        return False, "No OpenRouter API key provided."
    if not api_key.startswith("sk-or-"):
        return False, "OpenRouter key must start with 'sk-or-'."
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid"
            return False, f"OpenRouter returned HTTP {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"OpenRouter key invalid: HTTP {e.code}."
    except Exception as e:
        return False, f"OpenRouter key validation failed: {e}"

def validate_google_tts_key(api_key: str) -> tuple[bool, str]:
    """Validate a Google Cloud TTS API key via a voices.list call."""
    api_key = api_key.strip()
    if not api_key:
        return False, "No Google TTS API key provided."
    try:
        url = f"https://texttospeech.googleapis.com/v1/voices?key={api_key}&languageCode=pt-BR"
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid"
            return False, f"Google TTS returned HTTP {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"Google TTS key invalid: HTTP {e.code}."
    except Exception as e:
        return False, f"Google TTS key validation failed: {e}"

def validate_gemini_tts_key(api_key: str) -> tuple[bool, str]:
    """Validate a Gemini Developer API key via models.list call."""
    api_key = api_key.strip()
    if not api_key:
        return False, "No Gemini TTS API key provided."
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid"
            return False, f"Gemini API returned HTTP {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"Gemini key invalid: HTTP {e.code}."
    except Exception as e:
        return False, f"Gemini key validation failed: {e}"

def validate_elevenlabs_key(api_key: str) -> tuple[bool, str]:
    """Validate an ElevenLabs API key via a lightweight /v1/user call."""
    api_key = api_key.strip()
    if not api_key:
        return False, "No ElevenLabs API key provided."
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/user",
            headers={"xi-api-key": api_key},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Valid"
            return False, f"ElevenLabs returned HTTP {resp.status}."
    except urllib.error.HTTPError as e:
        return False, f"ElevenLabs key invalid: HTTP {e.code}."
    except Exception as e:
        return False, f"ElevenLabs key validation failed: {e}"

def list_elevenlabs_voices(api_key: str) -> list[tuple[str, str]]:
    """List account voices as (label, voice_id) tuples for UI dropdowns."""
    api_key = api_key.strip()
    if not api_key:
        return []
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
        voices = payload.get("voices", [])
        options: list[tuple[str, str]] = []
        for voice in voices:
            voice_id = (voice.get("voice_id") or "").strip()
            name = (voice.get("name") or "").strip() or voice_id
            if voice_id:
                options.append((f"{name} ({voice_id[:8]}…)", voice_id))
        options.sort(key=lambda item: item[0].lower())
        return options
    except Exception:
        return []

# Auto-init patterns on import
init_redact_patterns()
