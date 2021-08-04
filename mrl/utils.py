import logging
import sys
from pathlib import Path
from typing import Literal, Optional


def setup_logging(verbosity: Literal["INFO", "DEBUG"], log_path: Optional[Path] = None,) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        handlers.append(logging.FileHandler(str(log_path), mode="w"))

    logging.basicConfig(
        level=verbosity, format="%(levelname)s:%(asctime)s:%(message)s", handlers=handlers
    )

    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
