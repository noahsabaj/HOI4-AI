# ── src/utils/logger.py ────────────────────────────────────────────
"""
One-stop Rich + File + Timing logger for the entire HOI4 project.

Features
────────
✓ Rich-formatted console output
✓ Rotating file log in ./logs/ (14-day retention)
✓ Per-session run-id so you can grep just the current run
✓ 🔥  logger.timeit decorator / ctx-mgr for painless profiling
✓ Dynamic log-level via env-var  HOI4_DEBUG=1
"""

from __future__ import annotations
import logging, os, time
from contextlib import ContextDecorator
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_tb

# ── globals ────────────────────────────────────────────────────────
_LOG_ROOT   = Path("./logs"); _LOG_ROOT.mkdir(exist_ok=True)
_RUN_ID     = time.strftime("%Y%m%d_%H%M%S")
_CONFIGURED = False


def _configure_root() -> None:
    """Configure root logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    install_rich_tb(show_locals=True, width=120)

    level = logging.DEBUG if os.getenv("HOI4_DEBUG") else logging.INFO
    console_handler = RichHandler(
        console=Console(),
        markup=True,
        rich_tracebacks=True,
        show_time=False, show_level=False, show_path=False,
    )
    file_handler = TimedRotatingFileHandler(
        filename=_LOG_ROOT / f"hoi4_{_RUN_ID}.log",
        when="midnight", backupCount=14, encoding="utf-8", delay=True,
    )

    logging.basicConfig(
        level   = level,
        format  = "%(message)s",
        handlers=[console_handler, file_handler],
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    for noisy in ("urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    """Project-wide logger; first call sets up everything."""
    _configure_root()
    lg = logging.getLogger(name or "hoi4")
    lg.debug(f"Logger ready  (run-id={_RUN_ID})")
    return lg


# ── timing helper ─────────────────────────────────────────────────
class timeit(ContextDecorator):
    """`@timeit("label")` or `with timeit("label"):` → DEBUG duration in ms."""
    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label  = label
        self.logger = logger or get_logger("timer")

    def __enter__(self):
        self.start = time.perf_counter(); return self

    def __exit__(self, *exc):
        ms = (time.perf_counter() - self.start) * 1_000
        self.logger.debug(f"⏱ {self.label}: {ms:6.1f} ms")
        return False  # don’t swallow exceptions
