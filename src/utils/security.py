import os
import urllib.request
import urllib.error

_REDACT_PATTERNS: list[str] = []

def init_redact_patterns() -> None:
    """Build redaction list from current API key env vars."""
    global _REDACT_PATTERNS
    _REDACT_PATTERNS = []
    for env_var in ("OPENROUTER_API_KEY", "GOOGLE_TTS_API_KEY"):
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

# Auto-init patterns on import
init_redact_patterns()
