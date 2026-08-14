"""Consistent console logging."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger once for the console application."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
