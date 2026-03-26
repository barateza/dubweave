import os
import sys
import logging
import warnings
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
