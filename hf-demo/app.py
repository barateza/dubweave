import os

os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("WHISPER_MODEL", "small")
os.environ.setdefault("TTS_ENGINE", "edge")
os.environ.setdefault("EDGE_TTS_VOICE_NAME", "pt-BR-FranciscaNeural")

import sys
import logging
import warnings
import torch

# Cap PyTorch CPU threads to 2 to prevent thread oversubscription and CPU throttling on HF Spaces (2 vCPUs)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# Monkey-patch gradio_client to fix the "TypeError: argument of type 'bool' is not iterable" crash
# caused by standard Pydantic v2 schemas returning additionalProperties: bool
try:
    import gradio_client.utils as client_utils
    original_get_type = client_utils.get_type
    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return original_get_type(schema)
    client_utils.get_type = patched_get_type

    original_json_schema_to_python_type = getattr(client_utils, "_json_schema_to_python_type", None)
    if original_json_schema_to_python_type:
        def patched_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            return original_json_schema_to_python_type(schema, defs)
        client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
except Exception:
    pass

from src.utils.system import log_startup_info
from src.ui.layout import build_ui

# Suppress torch.load pickle warnings from TTS/XTTS internals.
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS")
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*dropout option.*", category=UserWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

def _configure_asyncio_windows_log_filter() -> None:
    """Suppress noisy, benign Proactor disconnect tracebacks on Windows."""
    if sys.platform != "win32":
        return

    class _AsyncioProactorDisconnectFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "_ProactorBasePipeTransport._call_connection_lost" not in msg:
                return True
            if record.exc_info:
                exc = record.exc_info[1]
                if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
                    return False
            return True

    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.addFilter(_AsyncioProactorDisconnectFilter())

if __name__ == "__main__":
    _configure_asyncio_windows_log_filter()
    log_startup_info()
    demo = build_ui()
    demo.queue(max_size=3)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True
    )
