"""
Console output for ``verbose=True``.

GEARS library code never calls ``print``: progress messages go through the
standard :mod:`logging` module (``INFO`` level, ``gears.*`` loggers). Entry
points that take a ``verbose`` flag wrap their work in :func:`console_logging`,
which attaches a console handler to the ``gears`` logger for the duration of the
call only — no global logging configuration is touched.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


class _ConsoleHandler(logging.StreamHandler):
    """Marker class so nested ``console_logging`` calls attach one handler only."""


@contextmanager
def console_logging(verbose: bool = True) -> Iterator[None]:
    """Show GEARS ``INFO`` messages on stderr while the block runs if ``verbose``."""
    gears_logger = logging.getLogger("gears")
    if not verbose or any(isinstance(h, _ConsoleHandler) for h in gears_logger.handlers):
        yield
        return

    handler = _ConsoleHandler()
    handler.setFormatter(logging.Formatter("[GEARS] %(message)s"))
    previous_level = gears_logger.level
    gears_logger.addHandler(handler)
    if previous_level == logging.NOTSET or previous_level > logging.INFO:
        gears_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        gears_logger.removeHandler(handler)
        gears_logger.setLevel(previous_level)
