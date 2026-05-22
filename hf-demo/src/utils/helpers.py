import time
from src.utils.security import redact

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def retry_with_backoff(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry *fn* (a zero-argument callable) with exponential backoff.
    """
    import urllib.error

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if attempt == max_retries or e.code not in _RETRYABLE_STATUS_CODES:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)
        except Exception:
            if attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)

def log(msg: str, logs: list) -> list:
    """Standardized logging with timestamp and redaction."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {redact(str(msg))}"
    print(entry)
    logs.append(entry)
    return logs
