"""Logging for the command-line scripts.

Progress and problems go to the console and to logs/<script>.log (timestamped, gitignored).
Result tables are still printed by the scripts themselves - they are output, not log lines.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from pkgtriage.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"

# errors a user can fix (missing file, bad data, stale model) - shown as one line, no traceback
EXPECTED_ERRORS: tuple[type[BaseException], ...] = (FileNotFoundError, ValueError, RuntimeError)


def setup_logging(script: str, level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    file = logging.FileHandler(LOG_DIR / f"{script}.log", encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root.addHandler(console)
    root.addHandler(file)
    # these libraries are chatty at INFO
    for noisy in ("httpx", "huggingface_hub", "matplotlib", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(script)


def run(main: Callable[[], None], script: str) -> None:
    """Run a script's main(), turning expected failures into a short message and exit code 1."""
    log = logging.getLogger(script)
    try:
        main()
    except EXPECTED_ERRORS as err:
        log.error("%s", err)
        log.debug("details", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("stopped by user")
        sys.exit(130)
