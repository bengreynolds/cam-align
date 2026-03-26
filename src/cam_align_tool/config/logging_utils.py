from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_event(logger: logging.Logger, stage: str, **fields) -> None:
    payload = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("stage=%s %s", stage, payload)
